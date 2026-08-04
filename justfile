set shell := ["bash", "-cu"]

# List available commands
default:
    @just --list

# Install backend (uv) and frontend (npm) dependencies
install: install-backend install-frontend

install-backend:
    cd backend && uv sync

install-frontend:
    cd frontend && npm install

# Run the FastAPI backend with autoreload (no viewer window)
backend:
    cd backend && uv run uvicorn app.main:app --reload --port 8000

# Run the FastAPI backend with a live MuJoCo viewer window attached to the
# real, shared sim — what you see is exactly what the websocket handlers are
# driving. macOS needs this launched under mjpython, which in turn needs
# DYLD_LIBRARY_PATH pointed at uv's managed Python's shared lib (without it
# mjpython fails to even start). No --reload: the reloader's subprocess
# model doesn't play well with owning a GUI window.
backend-viewer:
    cd backend && DYLD_LIBRARY_PATH="$(uv run python -c 'import sys; print(sys.base_prefix)')/lib" MUJOCO_VIEWER=1 uv run mjpython -m uvicorn app.main:app --port 8000

# Run the Vite frontend dev server
frontend:
    cd frontend && npm run dev

# Run backend (with viewer) + frontend together; Ctrl-C stops both. Then open
# the printed frontend URL in your own browser (needs real webcam access) —
# moving in frame should move the arm in the MuJoCo window.
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    just backend-viewer &
    (cd frontend && npm run dev) &
    wait

# Sanity-check: load the OpenArm model, command a joint target, step physics
sim-check:
    cd backend && uv run python -c "\
    import logging; logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(name)s: %(message)s'); \
    from app.sim.simulator import Simulator; \
    from app.control.position import PositionController; \
    sim = Simulator(); \
    controller = PositionController(sim); \
    target = sim.actuated_qpos(); target[0] = 0.5; \
    [sim.step(controller.ctrl(target)) for _ in range(1000)]; \
    print('time:', sim.time); \
    print('target[0]:', target[0], 'reached:', round(sim.actuated_qpos()[0], 4)); \
    print('actuated qpos:', sim.actuated_qpos())"

# Verify the retargeting pipeline (landmarks -> IK -> qpos_target) with synthetic input, no camera/GUI needed
retarget-check:
    cd backend && PYTHONPATH=. uv run python scripts/retarget_check.py

# Verify the hand-orientation math (SEW-Mimic stage 3's input) against known rotations
hand-check:
    cd backend && uv run python scripts/check_hand_frame.py

# TODO: maybe can remove this as well.
# Verify the closed-form SEW stage 1/2 derivation against MuJoCo forward kinematics
sew-check:
    cd backend && uv run python scripts/check_sew_closed_form.py

# TODO: remove this
# Replay recorded pose clips through the retargeting stack. Deterministic, no
# camera or browser: this is how retargeting changes get A/B'd.
#   just replay fixtures/fast_wave.jsonl
#   just replay "fixtures/*.jsonl --summary-json runs/after.json"
replay *ARGS:
    cd backend && uv run python scripts/replay.py {{ARGS}}

# Replay the whole battery and write summaries, so two revisions can be diffed.
#   just battery before   # ... make a change ...   just battery after
#   diff <(jq . backend/runs/before.json) <(jq . backend/runs/after.json)
battery LABEL="run":
    #!/usr/bin/env bash
    set -euo pipefail
    cd backend
    shopt -s nullglob
    clips=(fixtures/*.jsonl)
    if [ ${#clips[@]} -eq 0 ]; then
      echo "no clips in backend/fixtures/ — record some first, see fixtures/README.md" >&2
      exit 1
    fi
    uv run python scripts/replay.py "${clips[@]}" --quiet \
      --out-dir "runs/{{LABEL}}" --summary-json "runs/{{LABEL}}.json"

# Open the native MuJoCo viewer on the bimanual model, disconnected from the backend (standalone sanity check)
sim-view:
    cd backend && uv run python -m mujoco.viewer --mjcf=../openarm_mujoco/v2/openarm_bimanual.xml

# Type-check the frontend
typecheck:
    cd frontend && npx tsc --noEmit

# Build the frontend for production
build:
    cd frontend && npm run build

# Pull latest changes for the openarm_mujoco submodule
update-submodule:
    git submodule update --remote --merge openarm_mujoco

# Remove build artifacts and dependency caches
clean:
    rm -rf frontend/node_modules frontend/dist backend/.venv
