# Motion battery

Recorded pose clips (NEXT_STEPS.md §5). These are the fixed input that every
retargeting change gets A/B'd against — you cannot tune jitter by waving at a
webcam, because the input is never the same twice.

`*.jsonl` here are committed: they cost a person and a camera to reproduce, and
a comparison between two revisions only means anything if both saw the same
frames.

## Recording the battery

```bash
just dev                    # backend + frontend
```

Open the frontend, pick a clip from the dropdown at the bottom-left, hit
**record**, and perform the motion after the 3-second countdown. The frame
counter is live — if a take shows far fewer frames than `seconds × 30`,
tracking dropped and it is worth redoing. Clips land here, named after the
selection, and re-recording overwrites.

The clip list is defined once in [`app/battery.py`](../app/battery.py) and
served to the page from `GET /record/battery`; add a motion there and it shows
up in the dropdown. Each entry records *what the clip is for* — that field is
the difference between a reusable battery and eight mystery files.

Recording can also be driven without the browser:

```bash
curl -XPOST localhost:8000/record/start -H 'content-type: application/json' -d '{"name":"fast_wave"}'
curl -XPOST localhost:8000/record/stop
```

or with `POSE_RECORD=<name> just backend` to start one clip at boot.

## Using it

```bash
just replay fixtures/fast_wave.jsonl   # one clip, with a summary table
just battery before                    # whole battery -> runs/before.json
#   ... make a retargeting change ...
just battery after
diff backend/runs/before.json backend/runs/after.json
```

Replay is deterministic apart from `solve_ms`, so any other difference between
two runs is caused by the code change and nothing else.

## Format

JSONL, one clip per file, first line a header:

```jsonc
{"kind": "clip",  "version": 1, "name": "fast_wave", "source": "live", ...}
{"kind": "frame", "seq": 0, "t_capture": 1785845464.5, "t_recv": ..., "pose": {...}}
```

`pose` is the raw websocket payload, stored verbatim rather than as a parsed
`PoseFrame`. That is what keeps old clips replayable as the schema grows: new
fields are optional, so an old clip simply lacks them. It also means a recording
can never be corrupted by a bug in the retargeting code it exists to test.

Clips already carry hips and hand landmarks even though nothing consumes them
yet — the torso frame (§6.2) and SEW-Mimic stage 3 (§7.4) need them, and this
battery is too expensive to re-shoot.
