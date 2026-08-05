"""Per-frame telemetry for the pose->joints path.

Everything the old `qpos=[...16 floats...]` line carried, split per arm and
labelled, so a value can be placed at a glance while debugging live. Joint
values stay as plain lists in actuator order (joints 1-7 then the gripper).

Backend only: the browser gets no extra payload from this.
"""

import os
import time

import numpy as np

from app.retarget.mapper import Landmark, PoseFrame
from app.retarget.retargeter import ArmSolve

# Seconds between reports. One a second stays readable live; 0 logs every frame.
LOG_INTERVAL = float(os.environ.get("POSE_LOG_INTERVAL", "1.0"))


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
