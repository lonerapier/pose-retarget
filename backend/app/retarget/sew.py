"""Closed-form SEW-Mimic stages 1-2 for OpenArm v2.

Given the human's upper arm `u` and forearm `l` as unit vectors in the arm base
frame, recover joints 1-4 directly, without IK optimization steps.

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

Policy decisions handle which branch to take, what to do where the pose stops
determining an angle, what to do when the human is out of reach.
"""

from dataclasses import dataclass, field

import numpy as np

# Joints this solves for, in order: shoulder yaw/pitch, shoulder roll, elbow.
STAGE12_JOINTS = (1, 2, 3, 4)

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

# Slack for float error at a joint limit, nothing more.
_LIMIT_TOLERANCE = 1e-6


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
    upper_arm = np.array(
        [
            sx * np.sin(q1) * np.cos(q2),
            -np.sin(q2),
            -np.cos(q1) * np.cos(q2),
        ]
    )
    v = np.array(
        [
            np.cos(q3) * np.sin(q4),
            -np.sin(q3) * np.sin(q4),
            -np.cos(q4),
        ]
    )
    forearm = axis_rotation(h1, q1) @ axis_rotation(h2, q2) @ v
    return upper_arm, forearm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def _trust(scale: float, floor: float) -> float:
    """How much of a freshly measured angle to believe, in [0, 1).

    `scale` is what the pose divides the measurement by (|cos q2| for q1,
    |sin q4| for q3), so variance grows as 1/scale**2 and the matching shrinkage
    weight is scale**2 / (scale**2 + floor**2): 1/2 at the floor, 0 at the
    singularity, ~1 well away from it.

    Smooth on purpose. A hard "hold below this threshold, track above it" rule
    puts a discontinuity exactly where the measurement is noisiest, so noise
    makes the arm cross it over and over — the snap the hold was meant to stop.
    """
    return float(scale * scale / (scale * scale + floor * floor))


def _lerp(previous: float, value: float, weight: float) -> float:
    """Move `weight` of the way from `previous` toward `value`.

    Not along the shortest arc, deliberately. Two of these joints have ranges
    wider than half a turn, and for them the short way round between two
    reachable angles can leave the range while the long way stays inside it. A
    blend that leaves the range gets clamped to a stop, which becomes the next
    frame's starting point, and the joint never comes back. `_nearest_turn` has
    already put both arguments on compatible turns, so plain interpolation is
    the right thing here.
    """
    return float(previous + weight * (value - previous))


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
    the previous frame — a joint that unwinds a whole turn between frames is a
    full-speed spin on the real arm.

    Runs on the raw measurement, before any blending, so everything downstream
    works in one continuous stretch of angle instead of wrapping.
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

    `upper_arm` and `forearm` are the derivation's `u` and `l`, in the arm base
    frame. `previous` is the last accepted (q1, q2, q3, q4), which supplies the
    angles this pose doesn't determine. `positive_cos_q2` picks the branch.

    Pass `lower`/`upper` for a *reachable* answer: each angle is put on the
    right turn and clamped, and q4 is then re-solved against the clamped q3 —
    the part a per-joint clamp can't do — so the forearm still points as close
    to the human's as the roll limit allows. Omit them for the raw geometry,
    which is what `check_sew_closed_form.py` validates.

    Pass `rest` to give the unobservable angles somewhere to fall back to;
    without it they merely hold, and noise walks them anywhere.

    `smoothing=False` returns where this branch is *heading*, with no shrinkage
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
