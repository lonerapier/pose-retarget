"""The fixed motion battery (NEXT_STEPS §5).

One list, in one place, so the record UI in the browser and the recorder itself
cannot drift apart on clip names. The browser fetches it from
`GET /record/battery`.

Each clip exists to stress one specific failure mode. Record the battery ONCE,
then A/B every retargeting change against the same recordings — the whole point
is that the input stops being different every time you look at it.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClipSpec:
    name: str
    description: str
    # What this clip is meant to expose. Written down because six weeks from
    # now "fast_wave" will not obviously mean "find the filter's group delay".
    probes: str
    seconds: float


BATTERY: tuple[ClipSpec, ...] = (
    ClipSpec(
        name="reach_forward_slow",
        description="Slow reach forward with both arms, then back to rest.",
        probes="baseline tracking accuracy with noise and dynamics both minimal",
        seconds=8.0,
    ),
    ClipSpec(
        name="arm_raised_side",
        description="Raise one arm straight out to the side, hold, lower.",
        probes="the current lock-up failure — cos(q2)->0 singularity of §0.2",
        seconds=8.0,
    ),
    ClipSpec(
        name="cross_midline",
        description="Both hands sweep across the chest past the midline.",
        probes="self-collision (§7.5 phase 4) and left/right axis sign errors",
        seconds=8.0,
    ),
    ClipSpec(
        name="fast_wave",
        description="Fast waving, both arms, as quick as the tracker holds up.",
        probes="'what breaks first as motion gets faster' — filter lag, staircase",
        seconds=6.0,
    ),
    ClipSpec(
        name="near_full_extension",
        description="Reach out until both arms are nearly straight, hold there.",
        probes="the straight-arm singularity the MAX_EXTENSION hack papers over",
        seconds=8.0,
    ),
    ClipSpec(
        name="elbow_occluded",
        description="Turn so the torso hides one elbow from the camera.",
        probes="degraded landmark confidence; depth guessing on a hidden joint",
        seconds=8.0,
    ),
    ClipSpec(
        name="torso_rotate",
        description="Rotate the torso about +/-30 degrees with the arms held still.",
        probes="camera-frame vs torso-frame mapping (§6.2) — arms should NOT move",
        seconds=10.0,
    ),
    ClipSpec(
        name="hand_roll",
        description="Arms held out steady, rotating both wrists through their range.",
        probes="hand orientation only — the SEW stage-3 input, nothing else moving",
        seconds=8.0,
    ),
)

BY_NAME = {clip.name: clip for clip in BATTERY}


def as_dicts() -> list[dict]:
    return [asdict(clip) for clip in BATTERY]
