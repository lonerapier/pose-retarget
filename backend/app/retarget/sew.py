"""Closed-form SEW-Mimic for OpenArm v2: stages 1-2 (arm) and 3 (wrist).

Given the human's upper arm `u` and forearm `l` as unit vectors in the arm base
frame, recover joints 1-4 directly, without IK optimization steps. Given the
human's hand orientation on top of that, recover joints 5-7 the same way.

Pre-requisite required to make Sew work: Joint 3's axis should be collinear with
the upper arm, and joint 5's axis should be collinear with the forearm. True for
OpenArm V2.

Aligning those two axes with the human's two limb directions *is* stages 1 and
2, and since the axes are `y`, `-x`, `-z`, `-y` at `qpos=0` the general
Subproblem-2 machinery collapses to trigonometry.

The forward relations this inverts, with `sx = -1` on the left arm and `+1` on
the right (joint 1's axis is the only thing mirrored between arms):

    u = (sx*sin(q1)*cos(q2), -sin(q2), -cos(q1)*cos(q2))
    l = R(h1, q1) R(h2, q2) v,  v = (cos(q3)*sin(q4), -sin(q3)*sin(q4), -cos(q4))

Stage 3 collapses the same way, for a different reason.

Policy decisions handle which branch to take, what to do where the pose stops
determining an angle, what to do when the human is out of reach.
"""

from dataclasses import dataclass, field

import numpy as np

# Joints this solves for, in order: shoulder yaw/pitch, shoulder roll, elbow,
# forearm roll, wrist pitch, wrist roll.
STAGE12_JOINTS = (1, 2, 3, 4)
STAGE3_JOINTS = (5, 6, 7)

# TODO: this is heuristically added, and need to be checked
# How observable an angle has to be before it is trusted outright. Two of them
# come out of a ratio that vanishes at a singularity:
#   q1 is divided by |cos q2| -> 0 with the arm raised straight out to the side
#   q3 is divided by |sin q4| -> 0 with the arm straight
# So e degrees of landmark error arrives as e/|cos q2| degrees of joint error,
# and these are where that amplification hits ~3x. Much looser than the "handle
# the exact singularity" epsilons the derivation asks for, because MediaPipe is
# noisy even on a still subject: at 0.15, an arm held 8 degrees off straight
# already swings the roll by +-90 degrees.
OBSERVABLE_COS_Q2 = 0.35
OBSERVABLE_SIN_Q4 = 0.35

# How fast an angle the pose cannot see falls back toward the arm's rest
# posture, per frame at ~30fps. Shrinking toward the *previous* value only slows
# an unobservable angle down, it doesn't anchor it, so noise still random-walks
# it: measured, joint 3 drifted from 1 to 90 degrees and parked against its stop
# while the subject held a straight arm. It needs a restoring term.
#
# Scaled by (1 - trust)**2 at the call site, so it bites where an angle really
# is unobservable and adds under a tenth of a degree everywhere else.
REST_RELAXATION = 0.08

# How much pointing accuracy we will trade to avoid switching branches, in
# radians. The two branches reach the same limb directions through very
# different joints (q1 differs by pi), so a switch is a whole-arm slam, and near
# a joint limit noise alone can flip which one scores better. Inside this
# tolerance the closer branch wins, outside it accuracy does. A bound rather
# than a weight, so the worst error the policy can cost is exactly this number.
BRANCH_SWITCH_TOLERANCE = np.radians(20.0)

# The same trade as BRANCH_SWITCH_TOLERANCE, one stage down: how much end
# effector orientation accuracy we will give up rather than swing the wrist to
# an equivalent pose half a turn away. Tighter than the arm's 20 degrees because
# a wrist slam moves far less mass and is far less alarming to stand next to,
# so there is less reason to suppress a switch that is genuinely worth making.
WRIST_SWITCH_TOLERANCE = np.radians(15.0)

# How many frames the wrist keeps its pose after the hand stops being measurable
# before it starts falling back toward rest. MediaPipe Hands drops out for a
# frame or two constantly — far more than the pose model does, since a hand
# turned edge-on or moving fast simply isn't detected — and decaying on the
# first miss would make the wrist sag and snap back with every flicker. ~0.17s
# at 30fps: long enough to ride out a dropout, short enough that a hand that has
# genuinely left the frame doesn't leave the wrist cocked.
WRIST_HOLD_FRAMES = 5

# Error tolerance at a joint limit
_LIMIT_TOLERANCE = 1e-6

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation by `theta` about a unit `axis` (Rodrigues' formula)."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def _arm_axes(side: str) -> tuple[float, np.ndarray, np.ndarray]:
    sx = -1.0 if side == "left" else 1.0
    return sx, np.array([0.0, -sx, 0.0]), np.array([-1.0, 0.0, 0.0])


def limb_directions_from_joints(
    side: str, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Forward map: the (upper arm, forearm) directions joints 1-4 produce."""
    sx, h1, h2 = _arm_axes(side)
    q1, q2, q3, q4 = (float(v) for v in q)
    upper_arm = np.array([
        sx * np.sin(q1) * np.cos(q2),
        -np.sin(q2),
        -np.cos(q1) * np.cos(q2),
    ])
    v = np.array([
        np.cos(q3) * np.sin(q4),
        -np.sin(q3) * np.sin(q4),
        -np.cos(q4),
    ])
    forearm = axis_rotation(h1, q1) @ axis_rotation(h2, q2) @ v
    return upper_arm, forearm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def _trust(margin: float, threshold: float) -> float:
    """How much of a freshly measured angle to believe, in [0, 1).
    - `margin` is the singularity distance (|cos q2| for q1, |sin q4| for q3).
    - `threshold` is the noise floor. Returns smooth shrinkage factor in [0, 1).
    """
    m2 = margin * margin
    t2 = threshold * threshold
    return float(m2 / (m2 + t2))


def _lerp(start: float, target: float, t: float) -> float:
    """Linearly interpolate from `start` to `target` by fraction `t`."""
    return float(start + t * (target - start))


@dataclass
class Branch:
    """One of the two solutions, and what it cost to make it reachable."""

    q: np.ndarray
    # Per joint: the solution wanted a value outside the joint's range, so the
    # arm can't reproduce the human's pose exactly. Surfaced rather than
    # swallowed — a joint pinned at its stop is the difference between "the
    # retarget is wrong" and "the robot doesn't bend that way".
    clamped: np.ndarray = field(
        default_factory=lambda: np.zeros(len(STAGE12_JOINTS), dtype=bool)
    )
    # Per joint: how observable this frame made the angle, in [0, 1]. Only q1 and
    # q3 have one — the other two are determined by the limb directions
    # everywhere — so q2 and q4 sit at 1. Recorded because it's the one number
    # that says how *hard* the pose is: it goes to 0 exactly at the singularities
    # (arm straight, arm out to the side) where the measured angle stops meaning
    # anything, which is the demo HUD's margin readout.
    trust: np.ndarray = field(
        default_factory=lambda: np.ones(len(STAGE12_JOINTS), dtype=float)
    )


def _nearest_turn(angle: float, lo: float, hi: float, previous: float) -> float:
    """Which full turn of `angle` to command. Does not clamp.

    Joint 1's range is [-3.491, 1.396], wider than half a turn on one side, so
    an angle and the same angle a turn away are not interchangeable: one can be
    reachable and the other not, and `arctan2` only returns the principal value.
    Prefer a representative inside the limits, and among those the one nearest
    the previous frame.
    """
    turns = [angle + 2 * np.pi * k for k in (-1, 0, 1)]
    reachable = [
        a for a in turns if lo - _LIMIT_TOLERANCE <= a <= hi + _LIMIT_TOLERANCE
    ]
    return float(min(reachable or turns, key=lambda a: abs(a - previous)))


def solve_branch(
    side: str,
    upper_arm: np.ndarray,
    forearm: np.ndarray,
    previous: np.ndarray,
    positive_cos_q2: bool,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    rest: np.ndarray | None = None,
    smoothing: bool = True,
) -> Branch:
    """One of the two closed-form solutions for (q1, q2, q3, q4).

    # Arguments

    - `upper_arm` and `forearm` are the derivation's `u` and `l`, in the arm base
    frame.
    - `previous` is the last accepted (q1, q2, q3, q4), which supplies the
    angles this pose doesn't determine. `positive_cos_q2` picks the branch.
    - `lower`/`upper` range for a *reachable* answer.
    - `rest` gives the unobservable angles somewhere to fall back to. Without it
    they merely hold, and noise walks them anywhere.
    - `smoothing=False` returns where this branch is *heading*, with no shrinkage
    and no relaxation. That's what the branch choice has to be made on: scoring
    the smoothed candidates compares two half-converged poses, which look alike
    enough that the arm just settles on whichever it started nearest.
    """
    sx, h1, h2 = _arm_axes(side)
    clamped = np.zeros(len(STAGE12_JOINTS), dtype=bool)
    trust = np.ones(len(STAGE12_JOINTS), dtype=float)

    def measured(angle: float, i: int) -> float:
        """This frame's raw solution for joint `i`, on a commandable turn."""
        if lower is None or upper is None:
            return angle
        return _nearest_turn(
            angle, float(lower[i]), float(upper[i]), float(previous[i])
        )

    def hold(i: int, trust: float) -> float:
        """Where an angle sits before this frame's measurement is mixed in."""
        if rest is None:
            return float(previous[i])
        relax = (1.0 - trust) ** 2 * REST_RELAXATION
        return _lerp(float(previous[i]), float(rest[i]), relax)

    def mix(i: int, angle: float, trust: float) -> float:
        """This frame's measurement, shrunk toward where the joint already is."""
        if not smoothing:
            return angle
        return _lerp(hold(i, trust), angle, trust)

    def commanded(angle: float, i: int) -> float:
        """The final value, clamped, recording whether it hit a stop."""
        if lower is None or upper is None:
            return angle
        limited = float(np.clip(angle, lower[i], upper[i]))
        clamped[i] = not np.isclose(limited, angle, atol=_LIMIT_TOLERANCE)
        return limited

    # --- stage 1: the upper arm fixes q1, q2 ---
    sin_q2 = float(np.clip(-upper_arm[1], -1.0, 1.0))
    cos_q2 = float(np.sqrt(max(0.0, 1.0 - sin_q2**2)))
    if not positive_cos_q2:
        cos_q2 = -cos_q2
    q2 = commanded(measured(float(np.arctan2(sin_q2, cos_q2)), 1), 1)

    # Scaling both arguments by sign(cos q2) instead of dividing by cos q2 gives
    # the same angle with nothing to blow up at the singularity; the vanishing
    # magnitude is carried by the trust weight instead.
    turn = 1.0 if cos_q2 >= 0.0 else -1.0
    q1 = measured(float(np.arctan2(turn * sx * upper_arm[0], -turn * upper_arm[2])), 0)
    trust[0] = _trust(abs(cos_q2), OBSERVABLE_COS_Q2)
    q1 = commanded(mix(0, q1, trust[0]), 0)

    # --- stage 2: the forearm, seen from the upper arm, fixes q3, q4 ---
    v = (axis_rotation(h1, q1) @ axis_rotation(h2, q2)).T @ forearm
    sin_q4 = float(np.linalg.norm(v[:2]))  # = sin(q4), and q4 in [0, pi]

    q3 = measured(float(np.arctan2(-v[1], v[0])), 2)
    trust[2] = _trust(sin_q4, OBSERVABLE_SIN_Q4)
    q3 = commanded(mix(2, q3, trust[2]), 2)

    # Best q4 for whatever q3 ended up being: maximise v . v(q3, q4) over q4.
    # Unconstrained this is exactly arccos(-v[2]); with q3 clamped it recovers
    # the closest reachable forearm direction, instead of leaving q4 solving for
    # a roll the arm never applied.
    q4 = float(np.arctan2(np.cos(q3) * v[0] - np.sin(q3) * v[1], -v[2]))
    q4 = commanded(measured(q4, 3), 3)

    return Branch(q=np.array([q1, q2, q3, q4]), clamped=clamped, trust=trust)


class Stage12Solver:
    """One arm's stage-1/2 solve: both branches, scored, with a memory of the
    previous frame.

    That memory only ever disambiguates — it picks between two branches that
    describe the same limb directions, and supplies angles the current pose
    doesn't determine. Unlike a warm-started iterative solver, nothing
    accumulates: away from a singularity the same limb directions give the same
    joints whatever came before, and at a singularity the memory decays toward
    `rest` instead of holding an arbitrary value forever.
    """

    def __init__(
        self,
        side: str,
        lower: np.ndarray,
        upper: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> None:
        self.side = side
        # Arm joints range
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        # Arm's default rest pose
        self.rest = (
            np.zeros(len(STAGE12_JOINTS))
            if initial is None
            else np.asarray(initial, dtype=float).copy()
        )
        self.previous = self.rest.copy()
        # Which joints the last accepted solve had to pin to a stop for telemetry
        self.last_clamped = np.zeros(len(STAGE12_JOINTS), dtype=bool)
        # That solve's per-joint observability, same purpose.
        self.last_trust = np.ones(len(STAGE12_JOINTS), dtype=float)

    def _pointing_error(
        self, q: np.ndarray, upper_arm: np.ndarray, forearm: np.ndarray
    ) -> float:
        """How far a candidate's limbs end up from the human's, in radians.

        Scoring the achieved *directions* rather than per-joint reachability is
        what stops a saturated joint 3 from condemning an otherwise good branch.
        A solution that pins one joint at a stop and still points nearly the
        right way beats one that stays inside every limit by throwing the whole
        arm to the other branch.
        """
        achieved_upper, achieved_forearm = limb_directions_from_joints(self.side, q)
        return _angle_between(achieved_upper, upper_arm) + _angle_between(
            achieved_forearm, forearm
        )

    def _branch(
        self,
        upper_arm: np.ndarray,
        forearm: np.ndarray,
        positive: bool,
        smoothing: bool,
    ) -> Branch:
        return solve_branch(
            self.side,
            upper_arm,
            forearm,
            self.previous,
            positive_cos_q2=positive,
            lower=self.lower,
            upper=self.upper,
            rest=self.rest,
            smoothing=smoothing,
        )

    def solve(self, upper_arm: np.ndarray, forearm: np.ndarray) -> np.ndarray:
        """Reachable joints (q1, q2, q3, q4) for one frame's limb directions."""
        # Choose on where each branch converges, not on this frame's partial
        # step toward it.
        targets = {
            positive: self._branch(upper_arm, forearm, positive, smoothing=False)
            for positive in (True, False)
        }
        errors = {
            positive: self._pointing_error(branch.q, upper_arm, forearm)
            for positive, branch in targets.items()
        }
        # Anything within the tolerance of the best counts as "as good as";
        # among those the least motion wins. That breaks the tie when both
        # branches are reachable, and holds the current branch through the noise
        # around a joint limit.
        floor = min(errors.values()) + BRANCH_SWITCH_TOLERANCE
        positive = min(
            (p for p in targets if errors[p] <= floor),
            key=lambda p: float(np.linalg.norm(targets[p].q - self.previous)),
        )

        best = self._branch(upper_arm, forearm, positive, smoothing=True)
        self.previous = best.q.copy()
        self.last_clamped = best.clamped.copy()
        self.last_trust = best.trust.copy()
        return self.previous.copy()


def wrist_euler_signs(axes: np.ndarray) -> np.ndarray:
    """Signs relating (q5, q6, q7) to their Z, Y, X rotation angles.

    A joint whose axis is `-z` turns by `q` about `-z`, which is `Rz(-q)`; this
    returns the -1. So `R^{4,7} = Rz(s[0]*q5) Ry(s[1]*q6) Rx(s[2]*q7)`.

    `axes` is the wrist joints' local axes straight out of the model, so this is
    the paper's equations (16)-(18) — "inspect which coordinate axis each joint
    turns about, and read the Euler order off it" — rather than a hardcoded
    convention. Raises if OpenArm ever stops being a Z-Y-X wrist, because the
    decomposition below is written for that order and would otherwise return a
    confident wrong answer.
    """
    signs = np.zeros(len(STAGE3_JOINTS))
    for i, (axis, expected) in enumerate(zip(axes, (2, 1, 0))):  # z, y, x
        nonzero = np.flatnonzero(np.abs(axis) > 1e-9)
        if nonzero.size != 1 or nonzero[0] != expected or abs(axis[expected]) != 1.0:
            raise ValueError(
                f"wrist joint {STAGE3_JOINTS[i]} turns about {axis}, expected a "
                f"unit {'xyz'[expected]} axis; the wrist is no longer Z-Y-X and "
                "solve_wrist() needs re-deriving"
            )
        signs[i] = axis[expected]
    return signs


@dataclass
class Wrist:
    """One frame's wrist solution, and what it cost to make it reachable."""

    q: np.ndarray
    # Same meaning as Branch.clamped: the joint wanted a value outside its
    # range. Much more common here than in stages 1-2 — joint 6 spans only
    # +-45 degrees against a human wrist's ~+-70 of flexion — so it is the
    # number that says "the hand orientation was not the robot's to reach".
    clamped: np.ndarray = field(
        default_factory=lambda: np.zeros(len(STAGE3_JOINTS), dtype=bool)
    )


def wrist_rotation(q: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Forward map: the orientation R^{4,7} that wrist joints (q5, q6, q7) give."""
    return (
        axis_rotation(_Z, signs[0] * q[0])
        @ axis_rotation(_Y, signs[1] * q[1])
        @ axis_rotation(_X, signs[2] * q[2])
    )


def _rotation_error(achieved: np.ndarray, desired: np.ndarray) -> float:
    """Angle of the rotation between two frames, in radians."""
    cos = (np.trace(achieved.T @ desired) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _stops(angle: float, lo: float, hi: float) -> list[float]:
    """What can be commanded for `angle`: itself, or else both stops.

    No `_nearest_turn` counterpart: every wrist range is narrower than half a
    turn, so an angle and its +-2pi twin can never both be reachable and
    arctan2's principal value is the only candidate. That much *is* simpler than
    stages 1-2.

    Returning *both* stops when it doesn't fit is the part that isn't. An angle
    that arctan2 produced wraps, and it wraps while out of range: at the far
    side, `angle` flips between +pi and -pi and the two stops are equally close
    to it and equally wrong. Clamping alone would pick whichever side of the wrap
    it landed on and slam the joint half a turn as noise crossed it — the same
    failure the stage-1/2 branch policy exists to prevent, arriving through a
    different door. So both survive to be scored.
    """
    if lo - _LIMIT_TOLERANCE <= angle <= hi + _LIMIT_TOLERANCE:
        return [float(np.clip(angle, lo, hi))]
    return [float(lo), float(hi)]


def solve_wrist(
    desired: np.ndarray,
    signs: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    previous: np.ndarray | None = None,
) -> Wrist:
    """Joints (q5, q6, q7) putting the end effector as close to `desired` as it
    reaches.

    `desired` is R^{4,7}_des: the orientation the 7th joint's frame should have,
    expressed in the frame at the base of the wrist — the one stages 1-2 already
    fixed. The caller builds it by stripping the constants off the measured hand
    orientation (see `Retargeter._wrist_target`).

    The paper calls the same matrix R^{5,7}, in equation (12). It numbers frames
    so that "frame 5" is the frame joint 5 rotates *in*; MJCF names bodies so
    that the body carrying that frame is `link4`, because joint 5 is declared
    inside `link5`. Same frame, and it has to be this one either way: a rotation
    that excluded q5 would only have two angles left to decompose into three.

    Exact when `desired` is reachable, to floating point, and then `previous` has
    no effect at all. Omit `lower`/`upper` for the raw decomposition, which is
    what the round-trip check validates.

    Solved in the two steps the kinematics force, not as a matter of taste:
    joints 5-6 aim joint 7's axis, then joint 7 rolls about it.
    """
    if lower is None or upper is None:
        lower, upper = np.full(3, -np.inf), np.full(3, np.inf)

    # --- joints 5, 6: point joint 7's axis where the hand needs it ---
    #
    # Joint 7 turns about its own axis, so it cannot contribute here; where that
    # axis ends up is down to joints 5 and 6 alone. Rz(a)Ry(b) takes x to
    # (cos a cos b, sin a cos b, -sin b), so inverting is one arcsin and one
    # arctan2.
    target = signs[2] * (desired @ _X)  # where joint 7's axis must point
    b = float(-np.arcsin(np.clip(target[2], -1.0, 1.0)))
    a = float(np.arctan2(target[1], target[0]))

    # q6 comes from an arcsin, which is monotone over the whole range and has no
    # wrap, so it has exactly one candidate and a plain clamp is right. Clamping
    # it independently of q5 is not provably the best pair under the limits, but
    # measured against a sweep over every reachable q6 with q5 re-solved for
    # each, it costs nothing (over uniform random orientations it was 1.2 deg
    # *better* on average), so the sweep isn't worth its runtime.
    q6 = float(np.clip(signs[1] * b, lower[1], upper[1]))

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    for q5 in _stops(signs[0] * a, lower[0], upper[0]):
        # --- joint 7: the residual roll about that axis ---
        #
        # Solved against the *clamped* q5, q6 rather than read out of the
        # original decomposition. Same reason stage 2 re-solves q4 against a
        # clamped q3: once the aim has been given up on, the roll that best
        # completes the achievable aim is not the roll that would have completed
        # the desired one.
        residual = (
            axis_rotation(_Y, signs[1] * q6).T @ axis_rotation(_Z, signs[0] * q5).T
        ) @ desired
        # The rotation about x closest to `residual`, i.e. the c maximising
        # trace(Rx(c).T @ residual). Reduces to the exact answer when the aim was
        # not clamped.
        c = float(
            np.arctan2(residual[2, 1] - residual[1, 2], residual[1, 1] + residual[2, 2])
        )
        for q7 in _stops(signs[2] * c, lower[2], upper[2]):
            q = np.array([q5, q6, q7])
            clamped = np.array([
                not np.isclose(q5, signs[0] * a, atol=_LIMIT_TOLERANCE),
                not np.isclose(q6, signs[1] * b, atol=_LIMIT_TOLERANCE),
                not np.isclose(q7, signs[2] * c, atol=_LIMIT_TOLERANCE),
            ])
            candidates.append((
                q,
                clamped,
                _rotation_error(wrist_rotation(q, signs), desired),
            ))

    if len(candidates) == 1 or previous is None:
        q, clamped, _ = min(candidates, key=lambda candidate: candidate[2])
        return Wrist(q=q, clamped=clamped)

    # choose the value within tolerance and closest to the previous value
    floor = min(error for _, _, error in candidates) + WRIST_SWITCH_TOLERANCE
    q, clamped, _ = min(
        (candidate for candidate in candidates if candidate[2] <= floor),
        key=lambda candidate: float(np.linalg.norm(candidate[0] - previous)),
    )
    return Wrist(q=q, clamped=clamped)


class WristSolver:
    """One arm's stage-3 solve, with somewhere to go when the hand is lost.

    The paper's Algorithm 3 aligns joint 7's axis with Subproblem 2 on joints 5-6,
    then takes the residual roll with Subproblem 1. OpenArm is the *perpendicular
    wrist* case the paper handles separately (its joint 7 axis is `x` while the
    gripper approaches along `-z`), for which it prescribes Euler decomposition
    instead.

    Like `Stage12Solver`, the memory only ever disambiguates: it picks between
    equally-wrong poses when the hand orientation is out of reach, and gives
    `hold()` something to decay from. Nothing accumulates — a reachable target
    gives the same joints whatever came before, so a dropped hand costs exactly
    the frames it was dropped for and nothing after.
    """

    def __init__(
        self,
        side: str,
        lower: np.ndarray,
        upper: np.ndarray,
        signs: np.ndarray,
        initial: np.ndarray | None = None,
    ) -> None:
        self.side = side
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.rest = (
            np.zeros(len(STAGE3_JOINTS))
            if initial is None
            else np.asarray(initial, dtype=float).copy()
        )
        self.previous = self.rest.copy()
        self.last_clamped = np.zeros(len(STAGE3_JOINTS), dtype=bool)
        # Consecutive frames with no usable hand orientation. Telemetry reads it
        # too: a wrist that is always holding is the signal that the hand model
        # isn't earning its place, which §9.4 leaves as an open measurement.
        self.missing = 0

    def solve(self, desired: np.ndarray) -> np.ndarray:
        """Reachable joints (q5, q6, q7) for one frame's hand orientation."""
        wrist = solve_wrist(
            desired, self.signs, self.lower, self.upper, previous=self.previous
        )
        self.previous = wrist.q.copy()
        self.last_clamped = wrist.clamped.copy()
        self.missing = 0
        return self.previous.copy()

    def hold(self) -> np.ndarray:
        """No usable hand this frame: keep the pose, then ease back to rest.

        Hold through `WRIST_HOLD_FRAMES`, then decay, at the rate unobservable angles
        relax at.
        """
        self.missing += 1
        self.last_clamped[:] = False
        if self.missing > WRIST_HOLD_FRAMES:
            self.previous = np.array([
                _lerp(p, r, REST_RELAXATION) for p, r in zip(self.previous, self.rest)
            ])
        return self.previous.copy()
