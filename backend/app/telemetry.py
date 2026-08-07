"""Per-frame telemetry for the pose->joints path.

Everything the old `qpos=[...16 floats...]` line carried, split per arm and
labelled, so a value can be placed at a glance while debugging live. Joint
values stay as plain lists in actuator order (joints 1-7 then the gripper).

`PoseTelemetry` is backend only — it renders log lines and the browser never
sees them. `HudMeter` is the browser-facing counterpart: the same per-frame
state, reduced to the handful of numbers the demo page's HUD draws.
"""

import os
import time

import numpy as np

from app.retarget.mapper import Landmark, PoseFrame
from app.retarget.retargeter import ArmSolve

# Seconds between reports. One a second stays readable live; 0 logs every frame.
LOG_INTERVAL = float(os.environ.get("POSE_LOG_INTERVAL", "1.0"))

# How much of the previous rate an fps sample keeps. Frame arrivals are jittery
# enough that the instantaneous 1/dt flickers by tens of Hz between frames, which
# is unreadable on a HUD; this settles in well under a second.
_FPS_SMOOTHING = 0.85


def _fmt_landmark(mark: Landmark) -> str:
    return f"({mark.x:.3f},{mark.y:.3f},{mark.z:.3f})"


def _fmt_vec(vec: np.ndarray | None) -> str:
    if vec is None:
        return "--"
    return "(" + ",".join(f"{v:.3f}" for v in vec) + ")"


def _fmt_list(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


class PoseTelemetry:
    """Renders one frame's pose->joint state as a few named log lines.

    Every frame is counted but only one per LOG_INTERVAL is rendered, so the
    frame number and fps stay honest even though most frames aren't printed.
    """

    def __init__(self, actuator_names: list[str | None]) -> None:
        # Actuator order is left arm then right arm here, but group by name
        # rather than by slicing so an added actuator can't quietly shift values
        # into the wrong arm's list.
        self.groups: dict[str, list[int]] = {"left": [], "right": [], "other": []}
        for i, name in enumerate(actuator_names):
            name = name or f"act{i}"
            side = next((s for s in ("left", "right") if name.startswith(s)), "other")
            self.groups[side].append(i)

        self.frame_count = 0
        self._since_report = 0
        self._last_report = time.monotonic()

    def record(
        self,
        frame: PoseFrame,
        solves: dict[str, ArmSolve],
        target: np.ndarray,
        actual: np.ndarray,
        solve_seconds: float,
    ) -> list[str]:
        """Count one frame; return log lines, or [] if it isn't time yet."""
        self.frame_count += 1
        self._since_report += 1

        now = time.monotonic()
        elapsed = now - self._last_report
        if elapsed < LOG_INTERVAL:
            return []
        lines = self._render(frame, solves, target, actual, elapsed, solve_seconds)
        self._last_report = now
        self._since_report = 0
        return lines

    def _render(
        self,
        frame: PoseFrame,
        solves: dict[str, ArmSolve],
        target: np.ndarray,
        actual: np.ndarray,
        elapsed: float,
        solve_seconds: float,
    ) -> list[str]:
        fps = self._since_report / elapsed if elapsed > 0 else 0.0
        head = (
            f"frame {self.frame_count} fps={fps:.1f} "
            f"retarget={1e3 * solve_seconds:.2f}ms"
        )
        # t_capture is browser wall clock against server wall clock, so across
        # machines this includes clock skew. On one machine it's the real
        # camera->solve latency.
        if frame.t_capture is not None:
            head += f" capture_lag={1e3 * (time.time() - frame.t_capture):.1f}ms"

        lines = [head]
        for side in ("left", "right"):
            marks = " ".join(
                f"{part}={_fmt_landmark(getattr(frame, f'{side}_{part}'))}"
                for part in ("shoulder", "elbow", "wrist")
            )
            hand = getattr(frame, f"{side}_hand")
            lines.append(f"{side} landmarks: {marks} hand={'yes' if hand else 'no'}")

            solve = solves.get(side)
            if solve is not None:
                held = " HELD (degenerate landmarks)" if solve.held else ""
                # The two unit vectors joints 1-4 are actually solved from, plus
                # the control point they end up putting the hand at.
                lines.append(
                    f"{side} limbs: upper_arm={_fmt_vec(solve.upper_arm)} "
                    f"forearm={_fmt_vec(solve.forearm)} "
                    f"ee_pos={_fmt_vec(solve.ee)}{held}"
                )
                # Which joints the human asked past the arm's limits. Invisible
                # in the joint values themselves — a clamped joint just reads as
                # sitting exactly on its limit.
                if solve.clamped is not None and solve.clamped.any():
                    hit = [str(i + 1) for i, c in enumerate(solve.clamped) if c]
                    lines.append(f"{side} clamped joints: {', '.join(hit)}")

        for side in ("left", "right", "other"):
            idx = self.groups[side]
            if not idx:
                continue
            lines.append(f"{side} target: {_fmt_list(target[idx])}")
            lines.append(f"{side} actual: {_fmt_list(actual[idx])}")
        return lines


class HudMeter:
    """The per-frame numbers the demo page overlays on the video.

    Deliberately a summary and not the whole state: this rides the same
    websocket as the sim frames at pose rate, and the HUD only has room for
    things a viewer can read while the arms are moving.
    """

    def __init__(self) -> None:
        self._last_frame: float | None = None
        self._fps = 0.0
        # Read by the state-streaming task, written by the receive task. Rebound
        # whole rather than mutated, so a reader always gets one frame's numbers
        # and never a half-updated dict.
        self.payload: dict = {}

    def update(
        self,
        frame: PoseFrame,
        solves: dict[str, ArmSolve],
        solve_seconds: float,
    ) -> None:
        now = time.monotonic()
        if self._last_frame is not None:
            delta = now - self._last_frame
            if delta > 0:
                sample = 1 / delta
                self._fps = (
                    sample
                    if self._fps == 0
                    else _FPS_SMOOTHING * self._fps + (1 - _FPS_SMOOTHING) * sample
                )
        self._last_frame = now

        payload: dict = {
            "pose_fps": round(self._fps, 1),
            "retarget_ms": round(1e3 * solve_seconds, 2),
            "arms": {},
        }
        if frame.t_capture is not None:
            # Same caveat as the log line: this is browser wall clock against
            # server wall clock, so it's a true latency only on one machine.
            payload["capture_lag_ms"] = round(1e3 * (time.time() - frame.t_capture), 1)

        for side in ("left", "right"):
            solve = solves.get(side)
            if solve is None:
                continue
            arm: dict = {"held": solve.held}
            if solve.trust is not None:
                # The pose's distance from the nearest singularity, as the solver
                # itself scores it: 1 is fully determined, 0 means at least one
                # shoulder angle has stopped being observable and is relaxing to
                # rest rather than tracking.
                arm["margin"] = round(float(np.min(solve.trust)), 3)
            if solve.clamped is not None:
                arm["clamped"] = [i + 1 for i, hit in enumerate(solve.clamped) if hit]
            payload["arms"][side] = arm

        self.payload = payload
