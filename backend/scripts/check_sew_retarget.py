"""End-to-end check of the SEW retargeting path as wired in.

Run:  just sew-retarget-check

`check_sew_closed_form.py` validates the *math* against MuJoCo; this validates the
*plumbing* around it, which is where the mistakes that survive a correct derivation
live: landmark axes, per-side signs, base-frame rotation, joint addressing.

Method: pick a random reachable pose for joints 1-4, read the robot's own limb
directions off `mj_forward`, synthesise the MediaPipe landmarks a human in that pose
would produce, push them through the real `Retargeter`, and measure how far the
resulting limb directions are from the ones we started with.

Four things get measured, since they trade against each other and one number would
hide which one regressed:

  1. accuracy   - handed a pose the robot can hold, does it recover it? Run on
                  `solve_branch` with the previous frame set to the answer, so the
                  smoothing is a no-op and what's left is the closed form plus the
                  joint-limit projection, with none of the policy on top.
  2. continuity - sweeping through both singularities, does any single frame contain
                  a jump the position servos would turn into a snap?
  3. invariance - does the same pose held while the subject turns or leans give the
                  same joints? This is what the torso frame exists for.
  4. steadiness - held still, with landmark noise, does it stay put? Run at each pose
                  that broke live plus a well-conditioned baseline, since some of the
                  residual is just a noisy input. Regression test for the arm going
                  wild near full extension, where joint 3's roll is unobservable and
                  landmark noise was arriving as ~180 degree flips.

Poses inside the unobservable bands are reported separately in (1) rather than counted
as failures: there the solver deliberately shrinks toward the previous frame instead
of tracking, so a large error is the design, not a bug.
"""

import mujoco
import numpy as np

from app.retarget.mapper import Landmark, PoseFrame
from app.retarget.retargeter import Retargeter
from app.retarget.sew import (
    OBSERVABLE_COS_Q2,
    OBSERVABLE_SIN_Q4,
    STAGE12_JOINTS,
    solve_branch,
)
from app.sim.simulator import Simulator

N_SAMPLES = 300
TOL_DEG = 1e-3

# A 90-frame sweep across a quarter turn moves each joint ~1 deg per frame, so
# anything near this is a branch flip or a singularity snap, not motion.
MAX_STEP_DEG = 10.0

# How long a held pose is repeated before its settled error is measured — every
# smoothing term is a filter toward the measurement, so this is the wait a person
# holding still sees before the arm arrives.
SETTLE_FRAMES = 40

# Worst settled error allowed on an observable pose. Not exact like the closed form:
# the branch policy may keep a branch up to BRANCH_SWITCH_TOLERANCE worse than the
# best, and rest-relaxation biases poorly-conditioned angles toward neutral.
SETTLED_TOL_DEG = 20.0

# Landmark noise as a fraction of limb length. MediaPipe's jitter on a still subject
# is a percent or two of body scale; this is the pessimistic end.
NOISE_SIGMA = 0.04

# A well-conditioned pose jitters too: the input is noisy, and tracking a noisy input
# faithfully means moving. So singular poses are judged against that baseline rather
# than against zero — the property under test is that full stretch is no longer
# *special*, not that it's perfectly still. p95 jitter may reach this multiple of the
# baseline pose's, and no single frame may exceed the cap.
NOISE_JITTER_RATIO = 2.5
MAX_NOISE_STEP_DEG = 40.0

# The well-conditioned pose the others are measured against.
BASELINE_POSE = "comfortable"

# Plausible human limb lengths in metres; any positive pair works, since
# limb_directions() throws the magnitudes away.
UPPER_ARM_LEN = 0.30
FOREARM_LEN = 0.25

# The rest of the synthetic subject, in metres, and only ever used as a shape —
# the torso frame normalizes everything it builds. Shoulders wider than hips, both
# above the hip midpoint, which is where MediaPipe's world landmarks put the origin.
SHOULDER_HALF_WIDTH = 0.18
HIP_HALF_WIDTH = 0.13
SPINE_LENGTH = 0.50

# How far the invariance check turns the subject. Exercises all three axes, and large
# enough that camera-frame retargeting fails loudly: with no torso frame a 30 degree
# lean shows up as ~30 degrees of joint error.
INVARIANCE_ROTATIONS_DEG = (
    ("yaw left", (0.0, 0.0, 35.0)),
    ("yaw right", (0.0, 0.0, -35.0)),
    ("lean back", (0.0, 25.0, 0.0)),
    ("lean forward", (0.0, -25.0, 0.0)),
    ("tilt sideways", (20.0, 0.0, 0.0)),
    ("all three", (15.0, 20.0, -30.0)),
)

# Worst joint difference allowed between a square-on subject and the same subject
# rotated. Not exact: the two runs reach the same pose through different rounding,
# and every angle goes through arctan2 and a per-frame filter. A tenth of a degree
# is well below anything visible and well above the noise floor.
INVARIANCE_TOL_DEG = 0.1


def to_landmark_axes(v: np.ndarray) -> np.ndarray:
    """Inverse of mapper.to_robot_axes, which is [-z, x, -y]."""
    return np.array([v[1], -v[2], -v[0]])


def landmark_at(p: np.ndarray) -> Landmark:
    return Landmark(x=float(p[0]), y=float(p[1]), z=float(p[2]))


class ArmProbe:
    """Reads one arm's limb directions off a live MjData, in world axes."""

    def __init__(self, model, data, side: str):
        self.model, self.data = model, data
        self.jnt = {
            n: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{n}"
            )
            for n in (1, 2, 3, 4, 7)
        }
        self.qadr = {n: model.jnt_qposadr[j] for n, j in self.jnt.items()}
        self.range = {n: model.jnt_range[j] for n, j in self.jnt.items()}

    def set_stage12(self, q: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        for n, v in zip(STAGE12_JOINTS, q):
            self.data.qpos[self.qadr[n]] = v
        mujoco.mj_forward(self.model, self.data)

    def _limb(self, a: int, b: int) -> np.ndarray:
        v = self.data.xanchor[self.jnt[b]] - self.data.xanchor[self.jnt[a]]
        return v / np.linalg.norm(v)

    def limbs(self) -> tuple[np.ndarray, np.ndarray]:
        return self._limb(3, 4), self._limb(4, 7)

    def random_stage12(self, rng) -> np.ndarray:
        return np.array([float(rng.uniform(*self.range[n])) for n in STAGE12_JOINTS])


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def pose_frame(
    limbs: dict[str, tuple[np.ndarray, np.ndarray]],
    body: np.ndarray | None = None,
) -> PoseFrame:
    """Landmarks for a subject whose limbs point along the given robot-axis
    directions, hips included so the torso frame has something to build from.

    `body` rigidly rotates the whole subject — torso and arms together — about the
    hip midpoint, in robot axes. It stands in for the operator turning, leaning or
    tilting relative to the camera, and the retargeted joints must not move when it
    changes: the torso frame rotates with the subject, so the limb directions
    expressed *in* it don't. `check_torso_invariance` measures exactly that.

    Built as world landmarks — metres, origin at the hip midpoint — which is what the
    frontend sends.
    """
    body = np.eye(3) if body is None else body
    fields = {}
    for side in ("left", "right"):
        across = 1.0 if side == "left" else -1.0  # robot +y is the subject's left
        hip = np.array([0.0, across * HIP_HALF_WIDTH, 0.0])
        shoulder = np.array([0.0, across * SHOULDER_HALF_WIDTH, SPINE_LENGTH])
        upper_arm, forearm = limbs[side]
        elbow = shoulder + UPPER_ARM_LEN * upper_arm
        wrist = elbow + FOREARM_LEN * forearm

        for part, point in (
            ("hip", hip),
            ("shoulder", shoulder),
            ("elbow", elbow),
            ("wrist", wrist),
        ):
            fields[f"{side}_{part}"] = landmark_at(to_landmark_axes(body @ point))
    return PoseFrame(**fields)


def rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """A rotation in robot axes: roll about +x (forward), pitch about +y (left),
    yaw about +z (up). Applied in that order, which for a subject in front of a
    camera reads as tilt, then lean, then turn."""

    def about(axis: int, degrees: float) -> np.ndarray:
        c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
        r = np.eye(3)
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        r[i, i] = r[j, j] = c
        r[i, j], r[j, i] = -s, s
        return r

    return about(2, yaw_deg) @ about(1, pitch_deg) @ about(0, roll_deg)


def is_singular(q: np.ndarray) -> bool:
    return (
        abs(np.cos(q[1])) < OBSERVABLE_COS_Q2 or abs(np.sin(q[3])) < OBSERVABLE_SIN_Q4
    )


def check_continuity(sim: Simulator) -> float:
    """Largest single-frame joint step while sweeping through both singularities,
    in degrees.

    The accuracy check teleports the arm, which is the worst possible input for a
    solver that resolves ambiguity by continuity. Real input doesn't teleport, so
    this drives the sweep the actual failure modes live on: the arm raising out to
    the side (|cos q2| -> 0) while the elbow straightens (|sin q4| -> 0), at 30fps.

    The tracking error here isn't the point — inside the singular band the solver is
    deliberately trading accuracy for continuity. What matters is that no frame
    contains a jump the position servos would turn into a snap.
    """
    retargeter = Retargeter(sim, mode="sew")
    frames = 90
    worst_step = 0.0
    previous = None

    for i in range(frames):
        # Upper arm swings from straight down to straight out to the left,
        # passing exactly through the shoulder singularity at the end; the elbow
        # straightens over the same interval, passing through the other one.
        t = i / (frames - 1)
        angle = t * np.pi / 2
        upper_arm = np.array([0.0, np.sin(angle), -np.cos(angle)])
        elbow_bend = (1.0 - t) * np.pi / 2
        forearm = np.cos(elbow_bend) * upper_arm + np.sin(elbow_bend) * np.array(
            [1.0, 0.0, 0.0]
        )
        limbs = {"left": (upper_arm, forearm), "right": (upper_arm, forearm)}

        q = retargeter.solve(pose_frame(limbs))
        if previous is not None:
            worst_step = max(worst_step, float(np.max(np.abs(q - previous))))
        previous = q

    return float(np.degrees(worst_step))


def limbs_pointing(
    direction: np.ndarray, bend_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """(upper arm, forearm) for an arm pointing `direction` with the elbow bent
    `bend_deg` out of straight, in robot axes."""
    upper_arm = unit(direction)
    across = unit(np.cross(upper_arm, [0.0, 0.0, 1.0]))
    bend = np.radians(bend_deg)
    return upper_arm, np.cos(bend) * upper_arm + np.sin(bend) * across


def check_torso_invariance(sim: Simulator) -> dict[str, float]:
    """Worst joint difference, in degrees, between a subject square to the camera and
    the same subject turned, leaning or tilted while holding the same pose *relative
    to their own body*.

    This is the property the torso frame exists for, and the one thing camera-frame
    retargeting can't have: read the limb directions straight out of the camera frame
    and leaning back 25 degrees rotates both by 25 degrees, so the robot's arms swing
    though the operator's didn't move. It's also a prerequisite for SEW-Mimic's
    optimality result, which is stated for a body-centric input.

    Measured on a well-conditioned pose on purpose: inside a singular band both runs
    converge to the same rest posture, reporting perfect invariance while measuring
    nothing.
    """
    limbs = {
        side: limbs_pointing(np.array([0.7, 0.3, -0.6]), 60.0)
        for side in ("left", "right")
    }

    def settled(body: np.ndarray | None) -> np.ndarray:
        # Fresh solver per run: smoothing carries state across frames, so sharing
        # one would let the first run's history bias the second's.
        retargeter = Retargeter(sim, mode="sew")
        frame = pose_frame(limbs, body)
        for _ in range(SETTLE_FRAMES):
            q = retargeter.solve(frame)
        return q

    reference = settled(None)
    return {
        name: float(np.degrees(np.max(np.abs(settled(rotation(*rpy)) - reference))))
        for name, rpy in INVARIANCE_ROTATIONS_DEG
    }


def check_steadiness(sim: Simulator, rng) -> dict[str, tuple[float, float]]:
    """(typical, worst) single-frame joint step in degrees while the subject holds a
    pose and only landmark noise moves, for each pose that broke live.

    Both singular directions are unobservable *rolls*: at full stretch nothing in the
    landmarks says how the forearm is twisted, and with the arm straight out to the
    side nothing says how the upper arm is. A solver that answers anyway turns a few
    degrees of landmark noise into whole revolutions of joint travel.

    Typical is p95 rather than the mean — the failure guarded against is occasional
    violent frames, which a mean hides and a max over-reports.
    """
    retargeter = Retargeter(sim, mode="sew")
    poses = {
        "straight out front": (np.array([1.0, 0.0, 0.0]), 0.0),
        "straight, 5 deg bend": (np.array([1.0, 0.0, 0.0]), 5.0),
        "out to the side": (np.array([0.0, 1.0, 0.0]), 25.0),
        BASELINE_POSE: (np.array([0.7, 0.3, -0.6]), 60.0),
    }
    measured = {}

    for name, (direction, bend_deg) in poses.items():
        upper_arm, forearm = limbs_pointing(direction, bend_deg)

        commanded = []
        for _ in range(200):
            limbs = {
                side: (
                    unit(upper_arm + NOISE_SIGMA * rng.normal(size=3)),
                    unit(forearm + NOISE_SIGMA * rng.normal(size=3)),
                )
                for side in ("left", "right")
            }
            commanded.append(retargeter.solve(pose_frame(limbs)))

        step = np.degrees(np.abs(np.diff(np.array(commanded), axis=0)))
        measured[name] = (float(np.percentile(step, 95)), float(step.max()))

    return measured


def check_accuracy(sim: Simulator, retargeter: Retargeter, rng) -> dict[str, float]:
    """Worst limb-direction error, in degrees, recovering poses the arm can hold.

    Runs on `solve_branch` rather than through `Retargeter`, with the previous frame
    set to the pose being recovered, so the smoothing terms become identities —
    blending from the answer toward the answer. What's left is the closed form, the
    turn selection and the joint-limit projection. Through the live path this would
    instead measure how fast the smoothing converges, which the checks below answer.
    """
    worst = {"left": 0.0, "right": 0.0}
    probe = {side: ArmProbe(sim.model, sim.data, side) for side in ("left", "right")}

    for _ in range(N_SAMPLES):
        for side, arm in probe.items():
            truth = arm.random_stage12(rng)
            arm.set_stage12(truth)
            upper_arm, forearm = arm.limbs()

            geometry = retargeter.left if side == "left" else retargeter.right
            best = np.inf
            for positive in (True, False):
                branch = solve_branch(
                    side,
                    geometry.base_R.T @ upper_arm,
                    geometry.base_R.T @ forearm,
                    previous=truth,
                    positive_cos_q2=positive,
                    lower=geometry.stage12_range[:, 0],
                    upper=geometry.stage12_range[:, 1],
                )
                arm.set_stage12(branch.q)
                best = min(
                    best,
                    max(
                        angle_between(a, b)
                        for a, b in zip((upper_arm, forearm), arm.limbs())
                    ),
                )
            worst[side] = max(worst[side], best)

    return worst


def check_tracking(sim: Simulator, rng) -> tuple[float, float, int]:
    """Error after settling on a held pose, through the whole live path.

    Every smoothing term is a filter toward the measurement, so a pose held still
    converges; this measures *where*, which is what a person holding a position
    actually sees. Poses whose singular angles are unobservable are counted
    separately — there the solver converges to `rest` by design, not to the human's
    arbitrary roll.
    """
    retargeter = Retargeter(sim, mode="sew")
    probe = {side: ArmProbe(sim.model, sim.data, side) for side in ("left", "right")}
    worst = worst_unobservable = 0.0
    n_unobservable = 0

    for _ in range(N_SAMPLES // 5):
        truth_q, truth_limbs = {}, {}
        for side, arm in probe.items():
            q = arm.random_stage12(rng)
            arm.set_stage12(q)
            truth_q[side], truth_limbs[side] = q, arm.limbs()

        frame = pose_frame(truth_limbs)
        for _ in range(SETTLE_FRAMES):
            solved = retargeter.solve(frame)

        unobservable = any(is_singular(q) for q in truth_q.values())
        n_unobservable += unobservable
        for side, arm in probe.items():
            geometry = retargeter.left if side == "left" else retargeter.right
            q = solved[
                [
                    int(np.where(sim.actuator_qposadr == a)[0][0])
                    for a in geometry.stage12_qposadr
                ]
            ]
            arm.set_stage12(q)
            error = max(
                angle_between(a, b) for a, b in zip(truth_limbs[side], arm.limbs())
            )
            if unobservable:
                worst_unobservable = max(worst_unobservable, error)
            else:
                worst = max(worst, error)

    return worst, worst_unobservable, n_unobservable


def main() -> int:
    sim = Simulator()
    retargeter = Retargeter(sim, mode="sew")
    rng = np.random.default_rng(0)

    worst = check_accuracy(sim, retargeter, rng)
    settled, settled_unobservable, n_unobservable = check_tracking(sim, rng)

    worst_step = check_continuity(sim)
    steadiness = check_steadiness(sim, rng)
    invariance = check_torso_invariance(sim)

    failed = (
        max(worst.values()) >= TOL_DEG
        or settled >= SETTLED_TOL_DEG
        or worst_step >= MAX_STEP_DEG
        or max(invariance.values()) >= INVARIANCE_TOL_DEG
        or max(worst for _, worst in steadiness.values()) >= MAX_NOISE_STEP_DEG
        or max(typical for typical, _ in steadiness.values())
        >= NOISE_JITTER_RATIO * steadiness[BASELINE_POSE][0]
    )
    print("PASS" if not failed else "FAIL")

    print(f"\naccuracy of the closed form, {N_SAMPLES} reachable poses per arm")
    for side in ("left", "right"):
        print(f"  {side:>21}: worst limb direction error {worst[side]:.3e} deg")

    print(f"\ntracking through the live path, settled over {SETTLE_FRAMES} frames")
    print(f"  {'observable':>21}: worst limb direction error {settled:6.2f} deg")
    print(
        f"  {'unobservable':>21}: {settled_unobservable:6.2f} deg over "
        f"{n_unobservable} poses (informational — converges to rest by design)"
    )

    print("\ncontinuity, sweeping through both singularities")
    print(f"  {'worst frame step':>21}: {worst_step:6.2f} deg")

    print("\ntorso invariance, same pose held while the subject moves in frame")
    for name, error in invariance.items():
        print(f"  {name:>21}: worst joint difference {error:.3e} deg")

    baseline = steadiness[BASELINE_POSE][0]
    print(f"\nsteadiness, subject still, landmark noise sigma {NOISE_SIGMA}")
    for name, (typical, worst_frame) in steadiness.items():
        print(
            f"  {name:>21}: frame step p95 {typical:5.2f} deg "
            f"({typical / baseline:.1f}x baseline), worst {worst_frame:5.2f} deg"
        )

    print(
        f"\ntolerances: {TOL_DEG:.0e} deg closed-form error, {SETTLED_TOL_DEG:g} deg "
        f"settled, {MAX_STEP_DEG:g} deg per frame sweeping, {INVARIANCE_TOL_DEG:g} deg "
        f"under torso rotation, {NOISE_JITTER_RATIO:g}x baseline p95 and "
        f"{MAX_NOISE_STEP_DEG:g} deg worst on noise alone"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
