"""Dumps incoming PoseFrames to JSONL so a motion can be replayed later
(NEXT_STEPS §5).

Why this exists at all: you cannot tune jitter by waving at a webcam, because
the input is never the same twice. Recording turns "did that change help?" from
an argument into a measurement.

Recording sits at the websocket boundary, *before* retargeting, and stores the
raw payload dict rather than the parsed PoseFrame. That is deliberate: a clip
recorded today has to stay replayable after PoseFrame grows fields, and storing
the raw dict means an old clip simply lacks the new keys (which are optional)
instead of failing to load. It also means a recording is never contaminated by a
bug in the retargeting code it is supposed to be testing.

File format — JSONL, one clip per file:

    {"kind": "clip",  "version": 1, "name": ..., "recorded_utc": ..., ...}
    {"kind": "frame", "seq": 0, "t_capture": ..., "t_recv": ..., "pose": {...}}
    ...

The header is a line rather than a sidecar file so a clip is a single artifact
you can move around, and `kind` keeps it unambiguous while streaming.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from app.battery import BY_NAME

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Default home for recordings. Kept out of the app package so clips are data,
# not code, and easy to gitignore or share as a tarball.
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "fixtures"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(name: str) -> str:
    """Clip names reach this from an HTTP body, and end up as a path."""
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    if not cleaned:
        raise ValueError("clip name is empty after sanitizing")
    return cleaned


@dataclass
class ClipStatus:
    recording: bool
    name: str | None = None
    path: str | None = None
    frames: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "recording": self.recording,
            "name": self.name,
            "path": self.path,
            "frames": self.frames,
            "seconds": round(self.seconds, 3),
        }


class PoseRecorder:
    """Writes one clip at a time. Not thread-safe; it is only ever touched from
    the event loop, either by the pose websocket or by a /record request."""

    def __init__(self, directory: Path = DEFAULT_DIR) -> None:
        self.directory = directory
        self._file: TextIO | None = None
        self._name: str | None = None
        self._path: Path | None = None
        self._frames = 0
        self._t_first: float | None = None
        self._t_last: float | None = None

    @property
    def recording(self) -> bool:
        return self._file is not None

    def status(self) -> ClipStatus:
        if not self.recording:
            return ClipStatus(recording=False)
        span = 0.0
        if self._t_first is not None and self._t_last is not None:
            span = self._t_last - self._t_first
        return ClipStatus(
            recording=True,
            name=self._name,
            path=str(self._path),
            frames=self._frames,
            seconds=span,
        )

    def start(self, name: str, description: str | None = None) -> ClipStatus:
        """Begin a clip. Restarting while one is open closes it first, so a
        double-click on the record button loses at most the partial take."""
        if self.recording:
            logger.warning("recorder: restarting; discarding open clip %r", self._name)
            self.stop()

        slug = _slug(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{slug}.jsonl"
        spec = BY_NAME.get(slug)

        self._file = path.open("w", encoding="utf-8")
        self._name = slug
        self._path = path
        self._frames = 0
        self._t_first = self._t_last = None
        self._emit({
            "kind": "clip",
            "version": FORMAT_VERSION,
            "name": slug,
            "description": description or (spec.description if spec else None),
            "probes": spec.probes if spec else None,
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "live",
        })
        logger.info("recorder: recording %r -> %s", slug, path)
        return self.status()

    def write(self, pose: dict[str, Any]) -> None:
        """Append one frame. Silently a no-op when not recording, so the
        websocket handler does not need to branch."""
        if self._file is None:
            return
        t_recv = time.time()
        # Fall back to arrival time when the client sends no capture stamp: the
        # clip is still replayable at roughly the right cadence, it just carries
        # the network hop inside its timing. Marked so latency analysis can tell.
        t_capture = pose.get("t_capture")
        stamped = t_capture is not None
        if not stamped:
            t_capture = t_recv
        if self._t_first is None:
            self._t_first = t_capture
        self._t_last = t_capture
        self._emit({
            "kind": "frame",
            "seq": self._frames,
            "t_capture": t_capture,
            "t_recv": t_recv,
            "t_capture_estimated": None if stamped else True,
            "pose": pose,
        })
        self._frames += 1

    def stop(self) -> ClipStatus:
        """Close the clip and return what was captured. Safe to call when idle."""
        if self._file is None:
            return ClipStatus(recording=False)
        final = self.status()
        self._file.close()
        self._file = None
        logger.info(
            "recorder: wrote %d frames (%.1fs) to %s",
            final.frames,
            final.seconds,
            final.path,
        )
        self._name = self._path = None
        self._frames = 0
        self._t_first = self._t_last = None
        # `recording` is False now, but the caller wants the summary of what was
        # just closed, not an empty idle status.
        final.recording = False
        return final

    def _emit(self, record: dict) -> None:
        assert self._file is not None
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Flushed per line so a clip survives Ctrl-C on the backend — which is
        # exactly how a recording session tends to end.
        self._file.flush()


def read_clip(path: Path) -> tuple[dict, list[dict]]:
    """Load a clip as (header, frames). Frames keep the file's order.

    Tolerates a missing header (returns an empty one) so a hand-assembled JSONL
    of bare frame records still replays.
    """
    header: dict = {}
    frames: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: not valid JSON — {err}") from err
            kind = record.get("kind", "frame")
            if kind == "clip":
                header = record
            elif kind == "frame":
                frames.append(record)
            else:
                raise ValueError(f"{path}:{lineno}: unknown record kind {kind!r}")
    if not frames:
        raise ValueError(f"{path}: contains no frames")
    return header, frames
