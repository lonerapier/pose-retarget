"""Closed-form SEW-Mimic stages 1-2 for OpenArm v2 (NEXT_STEPS.md §0.2).

Given the human's two limb directions — upper arm `u` and forearm `l`, unit
vectors in the *arm base frame* — this recovers joints 1-4 in closed form. No
Jacobian, no iteration, no warm start, and therefore no cross-frame state to
random-walk through the null space (§2.1).

It works because two facts hold on this model, both verified numerically
against `mj_forward` by `scripts/check_sew_closed_form.py`:

    joint 3's axis is collinear with the upper arm
    joint 5's axis is collinear with the forearm

so aligning those two axes with the human's two limb directions *is* stages 1
and 2, and because the axes are `y`, `-x`, `-z`, `-y` at `qpos=0` the general
Subproblem-2 machinery collapses to plain trigonometry.

The forward relations everything here inverts, with `sx = -1` on the left arm
and `+1` on the right (joint 1's axis is the one thing mirrored between arms):

    u = (sx*sin(q1)*cos(q2), -sin(q2), -cos(q1)*cos(q2))
    l = R(h1, q1) R(h2, q2) v,  v = (cos(q3)*sin(q4), -sin(q3)*sin(q4), -cos(q4))

The geometry is kept separate from the policy — which of the two valid branches
to take, what to do where the pose stops determining an angle, and what to do
when the human's pose is out of reach — because those are control decisions
rather than facts about the arm.
"""

from dataclasses import dataclass, field

import numpy as np

# Joints this solves for, in order: shoulder yaw/pitch, shoulder roll, elbow.
STAGE12_JOINTS = (1, 2, 3, 4)

# How observable an angle has to be before it is trusted outright. Two angles
# come out of a ratio that vanishes at a singularity (§0.2):
#   q1 is scaled by |cos q2| -> 0 when the upper arm lies along the shoulder
#      axis, i.e. the arm raised straight out to the side (§2.3's missing motion)
#   q3 is scaled by |sin q4| -> 0 when the arm is straight
# A landmark direction error of e degrees therefore lands as e/|cos q2| (resp.
# e/|sin q4|) degrees of joint error, so these are the values at which that
# amplification reaches ~3x. MediaPipe's limb directions carry a few degrees of
# noise even when still, which is why these are much larger than the "handle the
# exact singularity" epsilons the derivation calls for: at 0.15 an arm held 8
# degrees off straight is already amplifying noise into +-90 degrees of roll.
OBSERVABLE_COS_Q2 = 0.35
OBSERVABLE_SIN_Q4 = 0.35

# How fast an angle the pose cannot see falls back toward the arm's rest
# posture, per frame, at ~30fps. Shrinking toward the *previous* value alone
# only slows an unobservable angle down; it does not anchor it, so landmark
# noise still random-walks it — measured, joint 3 wandered from 1 to 90 degrees
# and parked against its stop while the subject held a straight arm. This is the
# same unresolved-null-space failure as §2.1, one joint wide, and it needs the
# same answer: a restoring term toward a defined posture.
#
# Scaled by (1 - trust)**2 where it is applied, so it acts where an angle is
# genuinely unobservable and all but vanishes where it is not — the bias it adds
# to a well-conditioned solve stays under a tenth of a degree.
REST_RELAXATION = 0.08

# How much pointing accuracy may be given up to avoid switching branches, in
# radians. The two branches describe the same limb directions through very
# different joint values (q1 differs by pi), so a switch is a whole-arm slam,
# and near a joint limit noise alone can make the better branch alternate frame
# to frame. Within this tolerance the closer branch wins; beyond it accuracy
# does. Stated as a bound rather than as a weight on distance so the worst
# tracking error the policy can cost is exactly this number and not a function
# of how far the arm happened to move.
BRANCH_SWITCH_TOLERANCE = np.radians(20.0)

# How much a joint may sit outside its declared range and still count as
# reachable. Absorbs float error at a limit, nothing more.
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
    """Forward map: the (upper arm, forearm) directions joints 1-4 produce.

    The inverse of everything else in this module, and cheap enough to call per
    branch per frame, which is what lets the solver score its options by what
    they would actually *point at* rather than by whether each joint happens to
    sit inside its own limits.
    """
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

    `scale` is the factor the pose divides the measurement by (|cos q2| for q1,
    |sin q4| for q3), so measurement variance grows as 1/scale**2. This is the
    corresponding shrinkage weight, scale**2 / (scale**2 + floor**2): it is 1/2
    where the amplification reaches the floor, falls to 0 smoothly at the
    singularity, and approaches 1 well away from it.

    Smoothness is the point. A hard "hold below this threshold, track above it"
    rule puts a discontinuity exactly where the measurement is noisiest, so
    noise alone makes the arm cross it repeatedly — which is the snap the hold
    was supposed to prevent.
    """
    return float(scale * scale / (scale * scale + floor * floor))


def _lerp(previous: float, value: float, weight: float) -> float:
    """Move `weight` of the way from `previous` toward `value`.

    Deliberately not "along the shortest arc". Two of these joints have ranges
    wider than half a turn, so for them the short way round between two
    *reachable* angles can leave the range while the long way stays inside it —
    and a blend that leaves the range gets clamped to a stop, which then becomes
    the next frame's starting point, and the joint never comes back. Both
    arguments are put on compatible turns by `_nearest_turn` before this is
    called, which is what makes plain interpolation the correct thing.
    """
    return float(previous + weight * (value - previous))


@dataclass
class Branch:
    """One of the two solutions, and what it cost to make it reachable."""

    q: np.ndarray
    # Per joint: the solution wanted a value outside the joint's range, so this
    # arm cannot reproduce the human's pose exactly. Worth surfacing rather than
    # swallowing — a joint pinned at its stop is the difference between "the
    # retarget is wrong" and "the robot does not bend that way".
    clamped: np.ndarray = field(
        default_factory=lambda: np.zeros(len(STAGE12_JOINTS), dtype=bool)
    )


def _nearest_turn(angle: float, lo: float, hi: float, previous: float) -> float:
    """Which full turn of `angle` to command. Does not clamp.

    Two of these joints have ranges wider than half a turn on one side (joint 1
    is [-3.491, 1.396]), so an angle and the same angle a turn away are not
    interchangeable: one can be reachable and the other not, and `arctan2` only
    ever returns the principal value. Prefer a representative inside the limits,
    and among those the one nearest the previous frame — a joint that unwinds a
    whole turn between frames is a full-speed spin on the real arm.

    This runs on the raw measurement, before any blending, so that everything
    downstream is working in one continuous stretch of angle rather than
    wrapping across the range.
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

    `upper_arm` and `forearm` are the derivation's `u` and `l`: unit vectors for
    the human's two limbs, in the arm base frame. `previous` is the last
    accepted (q1, q2, q3, q4), which supplies the angles this pose does not
    determine. `positive_cos_q2` selects the branch.

    Pass `lower`/`upper` to get a *reachable* answer: each angle is put on the
    right turn and clamped, and — the part a per-joint clamp cannot do — q4 is
    then re-solved against the clamped q3 so the forearm still points as close
    to the human's as the shoulder roll limit allows. Omit them for the raw
    geometry, which is what `check_sew_closed_form.py` validates.

    Pass `rest` to give the two unobservable angles somewhere to fall back to;
    omit it and they merely hold, which lets noise walk them anywhere.

    `smoothing=False` returns where this branch is *heading* — the measurement
    taken outright, with no shrinkage and no relaxation — which is what the
    branch choice has to be made on. Scoring the smoothed candidates instead
    compares two half-converged poses, and two half-converged poses look alike
    enough that the arm settles on whichever it happened to start nearest.
    """
    sx, h1, h2 = _arm_axes(side)
    clamped = np.zeros(len(STAGE12_JOINTS), dtype=bool)

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

    # Scaling both arguments by sign(cos q2) rather than dividing by cos q2
    # itself is the same angle with no division to blow up at the singularity;
    # the vanishing magnitude is carried by the trust weight instead.
    turn = 1.0 if cos_q2 >= 0.0 else -1.0
    q1 = measured(float(np.arctan2(turn * sx * upper_arm[0], -turn * upper_arm[2])), 0)
    q1 = commanded(mix(0, q1, _trust(abs(cos_q2), OBSERVABLE_COS_Q2)), 0)

    # --- stage 2: the forearm, seen from the upper arm, fixes q3, q4 ---
    v = (axis_rotation(h1, q1) @ axis_rotation(h2, q2)).T @ forearm
    sin_q4 = float(np.linalg.norm(v[:2]))  # = sin(q4), and q4 in [0, pi]

    q3 = measured(float(np.arctan2(-v[1], v[0])), 2)
    q3 = commanded(mix(2, q3, _trust(sin_q4, OBSERVABLE_SIN_Q4)), 2)

    # Best q4 for whatever q3 ended up being: maximise v . v(q3, q4) over q4.
    # With an unconstrained q3 this reduces exactly to arccos(-v[2]); with a
    # clamped one it recovers the closest reachable forearm direction instead of
    # leaving q4 solving for a roll the arm never applied.
    q4 = float(np.arctan2(np.cos(q3) * v[0] - np.sin(q3) * v[1], -v[2]))
    q4 = commanded(measured(q4, 3), 3)

    return Branch(q=np.array([q1, q2, q3, q4]), clamped=clamped)


class Stage12Solver:
    """One arm's stage-1/2 solve: both branches, scored, with a memory of the
    previous frame.

    The memory is *only* used to disambiguate — to choose between two branches
    that describe the same limb directions, and to supply angles the current
    pose does not determine. Unlike a warm-started iterative solver nothing
    accumulates: away from a singularity, the same limb directions give the same
    joints regardless of what came before, and where the pose *is* singular the
    memory decays toward `rest` rather than holding an arbitrary value forever.
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
        # The posture unobservable angles fall back to. The arm's neutral pose
        # is the natural choice: it is where nothing is near a stop, so an angle
        # the human's pose does not pin down parks somewhere the next observable
        # frame can move freely out of.
        self.rest = (
            np.zeros(len(STAGE12_JOINTS))
            if initial is None
            else np.asarray(initial, dtype=float).copy()
        )
        self.previous = self.rest.copy()
        # Which joints the last accepted solve had to pin to a stop. Telemetry
        # only — nothing in the solve reads it back.
        self.last_clamped = np.zeros(len(STAGE12_JOINTS), dtype=bool)

    def _pointing_error(
        self, q: np.ndarray, upper_arm: np.ndarray, forearm: np.ndarray
    ) -> float:
        """How far a candidate's limbs end up from the human's, in radians.

        Scoring on the achieved *directions* rather than on per-joint
        reachability is what stops a saturated joint 3 from condemning an
        otherwise good branch: a solution that pins one joint at a stop and
        still points nearly the right way is a better answer than one that
        stays inside every limit by throwing the arm to the other branch.
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
        # Choose the branch on where each one converges, not on this frame's
        # partially-shrunk step toward it.
        targets = {
            positive: self._branch(upper_arm, forearm, positive, smoothing=False)
            for positive in (True, False)
        }
        errors = {
            positive: self._pointing_error(branch.q, upper_arm, forearm)
            for positive, branch in targets.items()
        }
        # Anything within the tolerance of the best is "as good as"; among those
        # the least motion wins, which both breaks the exact tie when both
        # branches are reachable and holds the current branch through the noise
        # around a joint limit.
        floor = min(errors.values()) + BRANCH_SWITCH_TOLERANCE
        positive = min(
            (p for p in targets if errors[p] <= floor),
            key=lambda p: float(np.linalg.norm(targets[p].q - self.previous)),
        )

        best = self._branch(upper_arm, forearm, positive, smoothing=True)
        self.previous = best.q.copy()
        self.last_clamped = best.clamped.copy()
        return self.previous.copy()
