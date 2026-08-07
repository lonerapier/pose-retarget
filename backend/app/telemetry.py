"""Per-frame telemetry for the pose->joints path.

Everything the old `qpos=[...16 floats...]` line carried, split per arm and
labelled, so a value can be placed at a glance while debugging live. Joint
values stay as plain lists in actuator order (joints 1-7 then the gripper).

One class takes a frame, two methods render it. `PoseTelemetry.record` is the
single place a frame's state arrives; `log_lines` renders it for the backend log
and `hud_payload` reduces it to the numbers the demo page's HUD draws. Splitting
the *ingest* from the two *views* is what keeps the callers from having to hand
the same five values to two different objects, and means a new field is added
once and then rendered wherever it belongs.
"""

import os
import time
from dataclasses import dataclass

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


@dataclass
class Frame:
    """One frame of the pose->joints path, as both renderers need to see it.

    Held whole rather than picked apart on the way in: the log line and the HUD
    want overlapping but different subsets, and which subset each wants is the
    kind of thing that changes.
    """

    pose: PoseFrame
    solves: dict[str, ArmSolve]
    target: np.ndarray  # commanded joint angles, actuator order
    actual: np.ndarray  # where the servos actually are
    solve_seconds: float


class PoseTelemetry:
    """One frame's pose->joint state, rendered for the log or for the browser.

    Every frame is counted and folded into the frame rate, but only one per
    LOG_INTERVAL is rendered into log lines, so the frame number and the rate
    stay honest even though most frames aren't printed. The HUD has no such
    throttle — it reads whatever the last frame was, whenever it asks.
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
        self._last: Frame | None = None
        self._last_arrival: float | None = None
        self._fps = 0.0
        self._last_report = time.monotonic()

    def record(
        self,
        pose: PoseFrame,
        solves: dict[str, ArmSolve],
        target: np.ndarray,
        actual: np.ndarray,
        solve_seconds: float,
    ) -> None:
        """Take one frame. Renders nothing; call `log_lines` or `hud_payload`."""
        self.frame_count += 1
        self._last = Frame(pose, solves, target, actual, solve_seconds)

        now = time.monotonic()
        if self._last_arrival is not None:
            delta = now - self._last_arrival
            if delta > 0:
                sample = 1 / delta
                # Smoothed rather than averaged over the report interval: the HUD
                # asks far more often than the log does, and needs a number that
                # is current without being jittery. One rate for both readers.
                self._fps = (
                    sample
                    if self._fps == 0
                    else _FPS_SMOOTHING * self._fps + (1 - _FPS_SMOOTHING) * sample
                )
        self._last_arrival = now

    def log_lines(self) -> list[str]:
        """This frame as log lines, or [] if nothing new or it isn't time yet."""
        if self._last is None:
            return []
        now = time.monotonic()
        if now - self._last_report < LOG_INTERVAL:
            return []
        self._last_report = now
        return self._render(self._last)

    def hud_payload(self) -> dict:
        """The summary the demo page overlays on the video.

        Deliberately a summary and not the whole state: this rides a websocket at
        pose rate, and the HUD only has room for things a viewer can read while
        the arms are moving.
        """
        if self._last is None:
            return {}
        last = self._last

        payload: dict = {
            "pose_fps": round(self._fps, 1),
            "retarget_ms": round(1e3 * last.solve_seconds, 2),
            "arms": {},
        }
        if last.pose.t_capture is not None:
            # Same caveat as the log line: this is browser wall clock against
            # server wall clock, so it's a true latency only on one machine.
            payload["capture_lag_ms"] = round(
                1e3 * (time.time() - last.pose.t_capture), 1
            )

        for side in ("left", "right"):
            solve = last.solves.get(side)
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

        return payload

    def _render(self, last: Frame) -> list[str]:
        head = (
            f"frame {self.frame_count} fps={self._fps:.1f} "
            f"retarget={1e3 * last.solve_seconds:.2f}ms"
        )
        # t_capture is browser wall clock against server wall clock, so across
        # machines this includes clock skew. On one machine it's the real
        # camera->solve latency.
        if last.pose.t_capture is not None:
            head += f" capture_lag={1e3 * (time.time() - last.pose.t_capture):.1f}ms"

        lines = [head]
        for side in ("left", "right"):
            marks = " ".join(
                f"{part}={_fmt_landmark(getattr(last.pose, f'{side}_{part}'))}"
                for part in ("shoulder", "elbow", "wrist")
            )
            hand = getattr(last.pose, f"{side}_hand")
            lines.append(f"{side} landmarks: {marks} hand={'yes' if hand else 'no'}")

            solve = last.solves.get(side)
            if solve is not None:
                held = " HELD (degenerate landmarks)" if solve.held else ""
                # The two unit vectors joints 1-4 are actually solved from, plus
                # the control point they end up putting the hand at.
                lines.append(
                    f"{side} limbs: upper_arm={_fmt_vec(solve.upper_arm)} "
                    f"forearm={_fmt_vec(solve.forearm)} "
                    f"ee_pos={_fmt_vec(solve.ee)}{held}"
                )
                # How observable the pose made the two shoulder angles. A low
                # margin is why a joint can sit still while the human moves.
                if solve.trust is not None:
                    lines.append(f"{side} margin: {np.min(solve.trust):.3f}")
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
            lines.append(f"{side} target: {_fmt_list(last.target[idx])}")
            lines.append(f"{side} actual: {_fmt_list(last.actual[idx])}")
        return lines
