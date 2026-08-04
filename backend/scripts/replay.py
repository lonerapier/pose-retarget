"""Replay a recorded pose clip through the retargeting stack (NEXT_STEPS §5).

No webcam, no browser, no websocket — just a JSONL clip driving the same
SimState the live server drives, so what you measure here is what the live
server does.

    just replay fixtures/arm_raised_side.jsonl
    uv run python scripts/replay.py fixtures/*.jsonl --summary-json runs/before.json

The default mode is deterministic: physics is advanced in lockstep with the
clip's own capture timestamps rather than against the wall clock, so re-running
a clip reproduces every retargeting and tracking number exactly and an A/B
difference is entirely attributable to the code change. The exception is
`solve_ms`, which is a wall-clock measurement and varies run to run like any
benchmark — read its trend, not its digits. `--realtime` gives up determinism
altogether in exchange for being watchable next to `--viewer`.

The metrics below are the §6.4 set, minus the ones that need code that does not
exist yet (no orientation is commanded, so there is no orientation error to
report). Two are worth knowing about specifically:

  target_delta  L2 change in qpos_target between consecutive pose frames, in
                radians. Measured before physics, so it is a property of the
                retargeter alone — the cleanest jitter number available, and the
                one that should collapse when filtering lands.
  qvel_rms      RMS actuated joint velocity. Jitter as the servos actually see
                it, i.e. including everything the controller adds.

A lock-up (§2.1) shows up as tracking error that rises and then never recovers;
jitter shows up in target_delta's tail, not its mean, which is why p95 and max
are reported alongside it.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import SimState  # noqa: E402
from app.record import read_clip  # noqa: E402
from app.retarget.mapper import (  # noqa: E402
    PoseFrame,
    arm_extension,
    hand_orientation,
)

logger = logging.getLogger("replay")

# A clip can contain a gap — the subject stepped out of frame, or the tracker
# lost them. Stepping physics across the whole gap is faithful but can be
# minutes of simulation for one dropped frame, so cap it and say so.
MAX_GAP_SECONDS = 1.0

# t_recv - t_capture differences two different machines' clocks (the browser's
# and the backend's), which is only a latency if those clocks agree. A skewed
# client, a laptop resuming from sleep, or a clip whose timestamps were authored
# rather than measured all produce numbers that are not transport at all.
# Outside this window the reading is discarded rather than reported.
MAX_PLAUSIBLE_TRANSPORT_S = 2.0


@dataclass
class ArmSample:
    extension: float
    target: np.ndarray
    achieved: np.ndarray
    error: float
    hand_R: np.ndarray | None


def _transport_ms(record: dict) -> float | None:
    """Client-to-server latency for one frame, or None when it isn't credible.

    See MAX_PLAUSIBLE_TRANSPORT_S: this is a difference between two clocks, and
    a wrong number here would quietly poison the latency budget of §6.4.
    """
    if record.get("t_capture_estimated"):
        return None  # the stamp IS the arrival time; the difference is zero by
        # construction and means nothing.
    delta = record["t_recv"] - record["t_capture"]
    if not 0.0 <= delta <= MAX_PLAUSIBLE_TRANSPORT_S:
        return None
    return round(delta * 1e3, 3)


def _arm_sample(state: SimState, side: str, pose: PoseFrame) -> ArmSample:
    """What the retargeter asked for on one arm, and what the sim delivered."""
    arm = getattr(state.retargeter, side)
    landmarks = (
        getattr(pose, f"{side}_shoulder"),
        getattr(pose, f"{side}_elbow"),
        getattr(pose, f"{side}_wrist"),
    )
    extension = arm_extension(*landmarks)
    target = arm.target(extension)
    # site_xpos on the LIVE data: where the end effector actually got to after
    # the servos tracked this target, not where IK thought it would be.
    achieved = state.sim.data.site_xpos[arm.site_id].copy()
    hand = getattr(pose, f"{side}_hand")
    return ArmSample(
        extension=float(np.linalg.norm(extension)),
        target=target,
        achieved=achieved,
        error=float(np.linalg.norm(achieved - target)),
        hand_R=hand_orientation(hand, side) if hand is not None else None,
    )


def replay(
    path: Path, realtime: bool = False, viewer: bool = False, out: Path | None = None
) -> dict:
    header, records = read_clip(path)
    name = header.get("name", path.stem)
    logger.info(
        "%s: %d frames%s",
        name,
        len(records),
        f" — {header['description']}" if header.get("description") else "",
    )

    state = SimState()
    if viewer:
        import mujoco.viewer

        state.viewer = mujoco.viewer.launch_passive(state.sim.model, state.sim.data)

    t0 = records[0]["t_capture"]
    # Frame i's target is held until frame i+1 arrives, which is exactly how the
    # live loop behaves, so the last frame gets the mean inter-frame interval.
    stamps = [r["t_capture"] - t0 for r in records]
    default_dt = (stamps[-1] - stamps[0]) / max(1, len(stamps) - 1)
    deadlines = stamps[1:] + [stamps[-1] + default_dt]

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    out_file = out.open("w", encoding="utf-8") if out else None
    samples: list[dict] = []
    prev_target: np.ndarray | None = None
    wall_start = time.perf_counter()
    gap_warned = False

    try:
        for record, stamp, deadline in zip(records, stamps, deadlines):
            pose = PoseFrame(**record["pose"])

            t_solve = time.perf_counter()
            state.qpos_target = state.retargeter.solve(pose)
            solve_ms = (time.perf_counter() - t_solve) * 1e3

            target_delta = (
                0.0
                if prev_target is None
                else float(np.linalg.norm(state.qpos_target - prev_target))
            )
            prev_target = state.qpos_target.copy()

            # Advance physics to cover the interval this target is held for.
            limit = min(deadline, state.sim.time + MAX_GAP_SECONDS)
            if limit < deadline and not gap_warned:
                logger.warning(
                    "clip has gaps longer than %.1fs; replay skips ahead over them",
                    MAX_GAP_SECONDS,
                )
                gap_warned = True
            while state.sim.time < limit:
                state.step()
            if state.sim.time < deadline:
                state.sim.data.time = deadline  # keep sim clock on the clip clock

            if realtime:
                lag = stamp - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)

            left, right = (_arm_sample(state, s, pose) for s in ("left", "right"))
            qvel = state.sim.actuated_qvel()
            sample = {
                "seq": record["seq"],
                "t": round(stamp, 4),
                "sim_t": round(state.sim.time, 4),
                "solve_ms": round(solve_ms, 4),
                "target_delta": round(target_delta, 6),
                "qvel_rms": round(float(np.sqrt(np.mean(qvel**2))), 6),
                # browser -> backend transport, the only latency stage a
                # recording can still testify to at replay time.
                "transport_ms": _transport_ms(record),
                "qpos_target": [round(q, 5) for q in state.qpos_target],
            }
            for side, arm in (("left", left), ("right", right)):
                sample[side] = {
                    "extension": round(arm.extension, 4),
                    "target": [round(v, 5) for v in arm.target],
                    "achieved": [round(v, 5) for v in arm.achieved],
                    "error": round(arm.error, 5),
                    "hand_R": (
                        None
                        if arm.hand_R is None
                        else [round(v, 5) for v in arm.hand_R.flatten()]
                    ),
                }
            samples.append(sample)
            if out_file:
                out_file.write(json.dumps(sample, separators=(",", ":")) + "\n")
    finally:
        if out_file:
            out_file.close()
        if state.viewer is not None:
            state.viewer.close()

    return summarize(name, header, samples, time.perf_counter() - wall_start)


def _stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 5),
        "p95": round(float(np.percentile(a, 95)), 5),
        "max": round(float(a.max()), 5),
    }


def summarize(name: str, header: dict, samples: list[dict], wall: float) -> dict:
    duration = samples[-1]["t"] - samples[0]["t"]
    intervals = np.diff([s["t"] for s in samples])
    transport = [s["transport_ms"] for s in samples if s["transport_ms"] is not None]
    return {
        "clip": name,
        "probes": header.get("probes"),
        "frames": len(samples),
        "duration_s": round(duration, 3),
        "capture_fps": round(len(samples) / duration, 2) if duration > 0 else None,
        # A single long interval is a dropped-tracking event; the mean hides it.
        "worst_frame_gap_ms": round(float(intervals.max()) * 1e3, 2)
        if len(intervals)
        else None,
        "replay_wall_s": round(wall, 2),
        "solve_ms": _stats([s["solve_ms"] for s in samples]),
        "target_delta_rad": _stats([s["target_delta"] for s in samples]),
        "qvel_rms": _stats([s["qvel_rms"] for s in samples]),
        "transport_ms": _stats(transport) if transport else None,
        "left_error_m": _stats([s["left"]["error"] for s in samples]),
        "right_error_m": _stats([s["right"]["error"] for s in samples]),
        "hand_coverage": {
            side: round(
                sum(s[side]["hand_R"] is not None for s in samples) / len(samples), 3
            )
            for side in ("left", "right")
        },
    }


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['clip']} ===")
    if summary["probes"]:
        print(f"  probes: {summary['probes']}")
    print(
        f"  {summary['frames']} frames over {summary['duration_s']}s "
        f"({summary['capture_fps']} fps capture, "
        f"worst gap {summary['worst_frame_gap_ms']}ms)"
    )
    print(f"  {'':22} {'mean':>10} {'p95':>10} {'max':>10}")
    for label, key in (
        ("IK solve (ms)", "solve_ms"),
        ("target delta (rad)", "target_delta_rad"),
        ("joint vel RMS (rad/s)", "qvel_rms"),
        ("transport (ms)", "transport_ms"),
        ("left ee error (m)", "left_error_m"),
        ("right ee error (m)", "right_error_m"),
    ):
        stats = summary[key]
        if stats is None:
            print(f"  {label:22} {'—':>10} {'—':>10} {'—':>10}")
            continue
        print(
            f"  {label:22} {stats['mean']:>10.4f} "
            f"{stats['p95']:>10.4f} {stats['max']:>10.4f}"
        )
    coverage = summary["hand_coverage"]
    print(
        f"  hand orientation available: "
        f"left {coverage['left']:.0%}, right {coverage['right']:.0%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="+", type=Path, help="recorded .jsonl clip(s)")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace to the clip's original timing (watchable, non-deterministic)",
    )
    parser.add_argument(
        "--viewer", action="store_true", help="open a MuJoCo viewer (needs mjpython)"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="write per-frame metrics JSONL here"
    )
    parser.add_argument(
        "--summary-json", type=Path, help="write all clip summaries here, for A/B diffs"
    )
    parser.add_argument("--quiet", action="store_true", help="only print summaries")
    args = parser.parse_args()

    # app.main calls basicConfig at import, which makes a second call here a
    # no-op — set the level explicitly rather than through it.
    logging.basicConfig(format="%(levelname)-8s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.WARNING if args.quiet else logging.INFO)

    summaries = []
    for clip in args.clips:
        out = args.out_dir / f"{clip.stem}.metrics.jsonl" if args.out_dir else None
        summary = replay(clip, realtime=args.realtime, viewer=args.viewer, out=out)
        summaries.append(summary)
        print_summary(summary)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"\nsummaries -> {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
