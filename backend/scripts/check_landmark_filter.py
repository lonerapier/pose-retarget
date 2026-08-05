"""What the 1 Euro landmark filter costs and what it buys (NEXT_STEPS.md §6.3).

Run:  just filter-check
      (or: cd backend && PYTHONPATH=. uv run python scripts/check_landmark_filter.py)

The filter has exactly one interesting property and it is a *trade*: smoothing
that removes jitter also delays real motion, and `beta` is the dial between
them. A single number cannot say whether a setting is good, so this measures
both ends of the trade against the same synthetic subject and prints them side
by side, one row per candidate setting.

  jitter  subject holds a pose; only landmark noise moves. The 95th percentile
          frame-to-frame joint step is what a person holding still sees the arm
          do. Measured at a well-conditioned pose *and* at full stretch, since
          near a singularity sew.py's own shrinkage is already suppressing
          noise and a second filter there mostly buys lag.

  lag     subject sweeps an arm through a quarter turn in half a second, noise
          free, well away from both singular bands. What is measured is the
          angle between where the robot's limbs point and where the human's
          actually are, which for a filter reads directly as delay.

Neither is zero at `off`: the input is noisy, and sew.py's trust shrinkage is
itself a low-pass. `off` is the baseline every other row is read against, not a
target.

This is a *tuning* report that also fails: the defaults in smoothing.py have to
earn their place by cutting jitter materially without pushing tracking lag past
what the rest of the pipeline already spends (~30ms of capture-to-solve).
"""

import numpy as np

from app.retarget.mapper import Landmark, PoseFrame
from app.retarget.retargeter import Retargeter
from app.retarget.smoothing import NOMINAL_DT, FilterConfig
from app.sim.simulator import Simulator
from scripts.check_sew_retarget import (
    ArmProbe,
    angle_between,
    limbs_pointing,
    pose_frame,
)

# Per-coordinate landmark noise, in metres — MediaPipe's world landmarks jitter
# by roughly a centimetre on a subject holding still at webcam resolution.
# Applied to the landmarks themselves rather than to the limb directions,
# because the landmarks are what the filter sees.
NOISE_SIGMA = 0.01
NOISE_FRAMES = 300

# Frames held before a measurement starts, so the filter is converged and the
# first-sample pass-through is not counted as tracking.
SETTLE_FRAMES = 30

# The sweep: a quarter turn in half a second is a brisk, deliberate reach — fast
# enough that a badly tuned filter shows up, slow enough to be a real motion
# rather than the teleport the accuracy check uses.
SWEEP_DEGREES = 90.0
SWEEP_SECONDS = 0.5
SWEEP_BEND_DEG = 60.0
SWEEP_TILT = 0.25  # sideways component, see sweep_direction

# Poses the jitter measurement is taken at, as (direction, elbow bend).
JITTER_POSES = {
    "comfortable": (np.array([0.7, 0.3, -0.6]), 60.0),
    "full stretch": (np.array([1.0, 0.0, 0.0]), 5.0),
}
BASELINE_POSE = "comfortable"

# What the shipped defaults have to achieve, measured against the `off` row.
# Both are set below what the defaults measure, with enough margin that ordinary
# drift does not trip them: this is a guard against a future edit quietly
# trading one for the other, not a restatement of today's numbers.
MIN_JITTER_REDUCTION = 0.6
MAX_ADDED_LAG_MS = 25.0

# The frontier, plus the two settings worth showing to be wrong. Every row moves
# along the same trade; what the grid behind these showed is that `min_cutoff`
# buys jitter far more expensively than `beta` does, so the good settings all sit
# at a low cutoff with a large beta rather than in the middle.
CANDIDATES: tuple[tuple[str, FilterConfig], ...] = (
    ("off", FilterConfig(enabled=False)),
    # §6.3's suggested starting point, carried over from the paper's
    # normalized-coordinate setting. Included to show what it does at metre
    # scale: beta*speed never reaches a tenth of a Hz, so it is a fixed 1Hz
    # low-pass with the adaptivity switched off in all but name.
    ("1.0 / 0.007 / 1 (§6.3)", FilterConfig(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)),
    ("0.5 / 2 / 1  smoothest", FilterConfig(min_cutoff=0.5, beta=2.0, d_cutoff=1.0)),
    ("0.5 / 8 / 1", FilterConfig(min_cutoff=0.5, beta=8.0, d_cutoff=1.0)),
    ("0.5 / 8 / 3  DEFAULT", FilterConfig()),
    ("0.5 / 16 / 3 fastest", FilterConfig(min_cutoff=0.5, beta=16.0, d_cutoff=3.0)),
    # Same beta, cutoff raised: pays in jitter without buying back much lag,
    # which is the measurement behind preferring beta as the dial.
    ("1.5 / 8 / 3", FilterConfig(min_cutoff=1.5, beta=8.0, d_cutoff=3.0)),
)


def noisy(frame: PoseFrame, rng, sigma: float) -> PoseFrame:
    """The same frame with independent gaussian noise on every body landmark."""
    fields = {}
    for side in ("left", "right"):
        for part in ("shoulder", "elbow", "wrist", "hip"):
            name = f"{side}_{part}"
            mark = getattr(frame, name)
            fields[name] = Landmark(
                x=mark.x + float(rng.normal(0.0, sigma)),
                y=mark.y + float(rng.normal(0.0, sigma)),
                z=mark.z + float(rng.normal(0.0, sigma)),
            )
    return frame.model_copy(update=fields)


def stage12_of(retargeter: Retargeter, sim: Simulator, solved: np.ndarray, side: str):
    """Joints 1-4 for one arm, out of a full actuator-space command."""
    geometry = retargeter.left if side == "left" else retargeter.right
    index = [
        int(np.where(sim.actuator_qposadr == adr)[0][0])
        for adr in geometry.stage12_qposadr
    ]
    return solved[index]


def measure_jitter(sim: Simulator, config: FilterConfig, rng) -> dict[str, float]:
    """95th-percentile single-frame joint step, in degrees, per held pose."""
    measured = {}
    for name, (direction, bend_deg) in JITTER_POSES.items():
        retargeter = Retargeter(sim, mode="sew", smoothing=config)
        limbs = {
            side: limbs_pointing(direction, bend_deg) for side in ("left", "right")
        }
        clean = pose_frame(limbs)

        commanded = []
        for i in range(SETTLE_FRAMES + NOISE_FRAMES):
            q = retargeter.solve(noisy(clean, rng, NOISE_SIGMA))
            if i >= SETTLE_FRAMES:
                commanded.append(q)

        step = np.degrees(np.abs(np.diff(np.array(commanded), axis=0)))
        measured[name] = float(np.percentile(step, 95))
    return measured


def sweep_direction(t: float) -> np.ndarray:
    """Where the upper arm points, `t` of the way through the sweep.

    Swings forward and up in a plane held slightly off the vertical: straight
    down is parallel to the robot's own +z, where the "across the arm" axis
    `limbs_pointing` builds the forearm from is undefined. Nothing about the
    sweep needs that direction, and the tilt keeps it well away from both
    singular bands too — the upper arm never nears the shoulder axis (+-y) and
    the elbow stays bent, so what this measures is the filter and nothing else.
    """
    angle = np.radians(t * SWEEP_DEGREES)
    v = np.array([np.sin(angle), SWEEP_TILT, -np.cos(angle)])
    return v / np.linalg.norm(v)


def measure_lag(sim: Simulator, config: FilterConfig) -> tuple[float, float, float]:
    """(worst, mean) limb-direction error through a fast sweep, in degrees, and
    the sweep's own angular speed in degrees per second.

    The subject holds still, sweeps, then holds again. Error is accumulated over
    the sweep itself: the first hold exists so the filter starts converged, and
    the trailing frames are not counted so a late-arriving tail is not mistaken
    for a smaller error than it is.
    """
    retargeter = Retargeter(sim, mode="sew", smoothing=config)
    probe = {side: ArmProbe(sim.model, sim.data, side) for side in ("left", "right")}
    frames = max(2, round(SWEEP_SECONDS / NOMINAL_DT))
    errors = []
    swept = 0.0
    previous = None

    for i in range(-SETTLE_FRAMES, frames):
        direction = sweep_direction(max(0.0, i / (frames - 1)))
        limbs = {
            side: limbs_pointing(direction, SWEEP_BEND_DEG)
            for side in ("left", "right")
        }
        solved = retargeter.solve(pose_frame(limbs))
        if i < 0:
            continue

        if previous is not None:
            swept += angle_between(previous, direction)
        previous = direction

        for side, arm in probe.items():
            arm.set_stage12(stage12_of(retargeter, sim, solved, side))
            errors.append(
                max(angle_between(a, b) for a, b in zip(limbs[side], arm.limbs()))
            )

    return float(np.max(errors)), float(np.mean(errors)), swept / SWEEP_SECONDS


def main() -> int:
    sim = Simulator()
    speed_deg_per_s = 0.0

    rows = []
    for name, config in CANDIDATES:
        # One rng per row, seeded the same, so every setting sees an identical
        # noise sequence and the differences between rows are the settings.
        jitter = measure_jitter(sim, config, np.random.default_rng(0))
        worst, mean, speed_deg_per_s = measure_lag(sim, config)
        rows.append((name, jitter, worst, mean))

    baseline = next(row for row in rows if row[0] == "off")
    default = next(row for row in rows if "DEFAULT" in row[0])

    reduction = 1.0 - default[1][BASELINE_POSE] / baseline[1][BASELINE_POSE]
    added_lag_ms = 1e3 * (default[3] - baseline[3]) / speed_deg_per_s
    failed = reduction < MIN_JITTER_REDUCTION or added_lag_ms > MAX_ADDED_LAG_MS
    print("PASS" if not failed else "FAIL")

    poses = list(JITTER_POSES)
    header = (
        f"\n{'min_cutoff / beta / d_cutoff':>28} | "
        + " | ".join(f"{p:>13}" for p in poses)
        + " | sweep worst | sweep lag"
    )
    print(header)
    print("-" * len(header))
    for name, jitter, worst, mean in rows:
        cells = " | ".join(f"{jitter[p]:9.2f} deg" for p in poses)
        lag_ms = 1e3 * mean / speed_deg_per_s
        print(f"{name:>28} | {cells} | {worst:7.2f} deg | {lag_ms:6.1f} ms")

    print(
        f"\njitter: p95 single-frame joint step, subject still, landmark noise "
        f"sigma {NOISE_SIGMA}m\n"
        f"sweep:  limb-direction error through {SWEEP_DEGREES:g} deg in "
        f"{SWEEP_SECONDS:g}s ({speed_deg_per_s:.0f} deg/s), noise free; worst is\n"
        f"        the worst single frame (motion onset), lag is the mean error "
        f"read back as a delay at that speed"
    )
    print(
        f"\ndefaults vs off: jitter {100 * reduction:.0f}% lower at "
        f"'{BASELINE_POSE}' (need {100 * MIN_JITTER_REDUCTION:.0f}%), "
        f"lag +{added_lag_ms:.1f}ms (allow {MAX_ADDED_LAG_MS:g}ms)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
