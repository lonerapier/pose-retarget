"""1 Euro filter on the incoming landmarks (NEXT_STEPS.md §6.3, step 2).

Casiez et al., CHI 2012: https://gery.casiez.net/1euro/

A first-order low-pass whose cutoff rises with the signal's own speed, so a
still subject gets heavy smoothing (jitter dies) and a moving one gets light
smoothing (lag stays small). One filter per coordinate per landmark; there is no
coupling between them.

**Landmark space, not joint space**, and the reason is structural rather than a
preference. `Stage12Solver.solve` picks between two branches by a *discrete*
comparison scored on the unsmoothed solution (sew.py). Filtering joint angles
downstream of that can only smooth a branch flip after it has been decided;
filtering the landmarks changes the decision itself. The same argument applies
to the elbow: q4 is literally the angle between the two limb directions, so
noise on the elbow landmark lands on it undiluted with nothing downstream able
to tell the difference between that and a real bend.

Hips are filtered alongside the arm landmarks because they build the torso frame
(mapper.torso_frame), and noise there rotates *both* arms' inputs at once —
the one landmark whose jitter is not confined to one arm.

Hand landmarks are deliberately not filtered: nothing consumes them yet (stage 3
is unbuilt, §7.4), they come and go frame to frame, and filtering a signal
across its own dropouts needs the gating that step 2 is explicitly not.

The defaults are measured, not inherited — `scripts/check_landmark_filter.py`
grids them against the two ends of the trade and the numbers below are that
table's. Three places the measurement disagreed with the obvious choice:

- **`beta` is in cutoff-Hz per metre-per-second**, because the landmarks are
  MediaPipe *world* landmarks and those are metres (§6.5). §6.3 suggests the
  paper's 0.007, but that number comes from normalized/pixel-space work where
  speeds are two to three orders of magnitude larger. Measured, it lands as a
  fixed 1Hz low-pass with the adaptivity switched off in all but name: same
  jitter as the default, **9x the tracking lag** (134ms against 15ms).
- **A low `min_cutoff` with a high `beta` beats the reverse.** They both trade
  the same two quantities, but not at the same rate: every point on the measured
  frontier has `min_cutoff` at the bottom of its range and pays for
  responsiveness with `beta` instead. Smoothing hard when still and getting out
  of the way completely when moving is strictly better than doing a bit of both.
- **`d_cutoff` is 3Hz, not the paper's 1Hz.** The cutoff can only open once the
  speed estimate has noticed the motion, and at 30fps a 1Hz estimate needs ~5
  frames — a third of a half-second reach, spent lagging. Raising it costs a
  little jitter and takes the worst-case error at motion onset from 12.2 to 9.1
  degrees.

One coupling to revisit: `_trust` in sew.py is already a data-dependent low-pass
in joint space, and `OBSERVABLE_COS_Q2` / `OBSERVABLE_SIN_Q4` are sized for a few
degrees of *raw* landmark noise. Cleaning the input up in front of them makes
those bands conservative, and near a singularity the two filters compound into
lag. Worth retuning now that this exists.

Every parameter is env-overridable so the whole thing can be A/B'd live without
an edit-restart cycle, and `LANDMARK_FILTER=0` turns it off outright.

What this does **not** do, and cannot: reject a bad measurement. 1 Euro is
speed-adaptive by construction, so a landmark that teleports is indistinguishable
from fast motion and is passed through with *less* attenuation than steady noise.
Jitter is the half of the problem it owns; a glitching or absent input needs a
plausibility gate and a defined rest behaviour, which are separate work.
"""

import logging
import math
import os
from dataclasses import dataclass

from app.retarget.mapper import Landmark, PoseFrame

logger = logging.getLogger(__name__)

# The landmarks that reach the solver. Everything here is filtered with one
# shared parameter set: a per-group split (heavier smoothing on the hips, which
# genuinely move slowly) is tempting, but it would give the torso frame's two
# endpoints different lags and so distort the spine direction whenever the
# subject actually moves. One knob until measurement says otherwise.
FILTERED_LANDMARKS = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "right_hip",
)

# Frame interval assumed when the client sends no capture timestamp — the
# synthetic frames the check scripts drive the pipeline with, and any older
# recording made before `t_capture` existed. It has to be the real camera rate
# rather than a wall-clock measurement, or a check that solves 200 frames in a
# millisecond would see a microsecond dt and a filter that never moves.
NOMINAL_DT = 1.0 / 30.0

# Bounds on a measured interval. Below the first, the browser is reporting two
# frames at effectively the same instant and the velocity estimate divides by
# nearly nothing; above the second, whatever the filter is holding is too old to
# blend into the present.
MIN_DT = 1.0 / 240.0
MAX_GAP_SECONDS = 0.5


@dataclass(frozen=True)
class FilterConfig:
    """1 Euro parameters. `beta` is the jitter/lag dial; the rest rarely move.

    At these values, against no filter at all: 77% less jitter on a held pose,
    for 15ms of added tracking lag on a 175 deg/s sweep. Raising `beta` to 16
    halves that lag and gives back ~10 points of the jitter, which is the trade
    to make if the latency budget turns out to be the binding constraint.
    """

    min_cutoff: float = 0.5  # Hz, the cutoff when the subject is still
    beta: float = 8.0  # Hz per m/s, how fast the cutoff opens up with speed
    d_cutoff: float = 3.0  # Hz, smoothing on the speed estimate itself
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "FilterConfig":
        defaults = cls()
        return cls(
            min_cutoff=float(
                os.environ.get("LANDMARK_MIN_CUTOFF", defaults.min_cutoff)
            ),
            beta=float(os.environ.get("LANDMARK_BETA", defaults.beta)),
            d_cutoff=float(os.environ.get("LANDMARK_D_CUTOFF", defaults.d_cutoff)),
            enabled=os.environ.get("LANDMARK_FILTER", "1") not in ("0", "false", "off"),
        )

    def describe(self) -> str:
        if not self.enabled:
            return "landmark filter: off"
        return (
            f"landmark filter: 1 euro, min_cutoff={self.min_cutoff}Hz "
            f"beta={self.beta} d_cutoff={self.d_cutoff}Hz"
        )


def _alpha(cutoff: float, dt: float) -> float:
    """The exponential smoothing weight for a given cutoff frequency and step."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _Axis:
    """One 1 Euro filter, over one coordinate of one landmark."""

    __slots__ = ("config", "value", "speed")

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.value: float | None = None
        self.speed = 0.0

    def update(self, x: float, dt: float) -> float:
        # First sample after a start or a reset: adopt it outright. Smoothing
        # toward an invented starting value would make the arm sweep in from
        # wherever that was.
        if self.value is None:
            self.value = x
            self.speed = 0.0
            return x

        speed = (x - self.value) / dt
        self.speed += _alpha(self.config.d_cutoff, dt) * (speed - self.speed)
        # The whole of the adaptivity: fast motion opens the cutoff up, so the
        # filter gets out of the way rather than lagging behind it.
        cutoff = self.config.min_cutoff + self.config.beta * abs(self.speed)
        self.value += _alpha(cutoff, dt) * (x - self.value)
        return self.value


class LandmarkFilter:
    """Per-landmark 1 Euro filtering of one pose stream.

    Stateful, and the state is per stream: hold one per client rather than one
    globally, or two operators' jitter ends up in each other's filters.
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config if config is not None else FilterConfig.from_env()
        self._axes = {
            name: tuple(_Axis(self.config) for _ in range(3))
            for name in FILTERED_LANDMARKS
        }
        self._t_previous: float | None = None

    def reset(self) -> None:
        """Forget everything: the next frame passes through untouched."""
        self._reset_axes()
        self._t_previous = None

    def _reset_axes(self) -> None:
        for axes in self._axes.values():
            for axis in axes:
                axis.reset()

    def _interval(self, t_capture: float | None) -> float:
        """Seconds since the previous frame, as the filter should treat them."""
        if t_capture is None:
            return NOMINAL_DT

        previous, self._t_previous = self._t_previous, t_capture
        if previous is None:
            return NOMINAL_DT

        dt = t_capture - previous
        if MIN_DT <= dt <= MAX_GAP_SECONDS:
            return dt

        # A backwards, zero or long interval means the stream is not continuous
        # with what the filter is holding: a reloaded page restarts the clock, a
        # backgrounded tab stops requestAnimationFrame entirely. Blending across
        # that gap would drag the arm from a stale pose; start clean instead.
        logger.debug("landmark filter: %.3fs gap, resetting", dt)
        self._reset_axes()
        return NOMINAL_DT

    def apply(self, frame: PoseFrame) -> PoseFrame:
        """A copy of `frame` with its body landmarks smoothed."""
        if not self.config.enabled:
            return frame

        dt = self._interval(frame.t_capture)
        smoothed: dict[str, Landmark] = {}
        for name in FILTERED_LANDMARKS:
            mark = getattr(frame, name)
            axes = self._axes[name]
            # An optional landmark (the hips) that this client does not send, or
            # stopped sending. Drop the history rather than let it bridge the
            # hole when the landmark comes back.
            if mark is None:
                for axis in axes:
                    axis.reset()
                continue
            smoothed[name] = Landmark(
                x=axes[0].update(mark.x, dt),
                y=axes[1].update(mark.y, dt),
                z=axes[2].update(mark.z, dt),
            )
        return frame.model_copy(update=smoothed)
