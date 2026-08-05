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

# Run the FastAPI backend with a live MuJoCo viewer attached to the shared sim —
# what you see is what the websocket handler is driving. macOS needs mjpython,
# which needs DYLD_LIBRARY_PATH pointed at uv's managed Python's shared lib or it
# won't start at all. No --reload: the reloader's subprocess model doesn't get on
# with owning a GUI window.
backend-viewer:
    cd backend && DYLD_LIBRARY_PATH="$(uv run python -c 'import sys; print(sys.base_prefix)')/lib" MUJOCO_VIEWER=1 uv run mjpython -m uvicorn app.main:app --port 8000

# Run the Vite frontend dev server
frontend:
    cd frontend && npm run dev

# Backend (with viewer) + frontend together; Ctrl-C stops both. Open the printed
# frontend URL in a real browser — it needs webcam access — and moving in frame
# should move the arm in the MuJoCo window.
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

# Check the closed-form SEW stage-1/2 derivation against MuJoCo forward kinematics
sew-check:
    cd backend && uv run python scripts/check_sew_closed_form.py

# Check the wired-up SEW path end-to-end: synthetic landmarks -> qpos_target
sew-retarget-check:
    cd backend && PYTHONPATH=. uv run python scripts/check_sew_retarget.py

# Check the hand orientation frame (SEW stage-3 input) against its own properties
hand-check:
    cd backend && PYTHONPATH=. uv run python scripts/check_hand_frame.py

# Run every check
check: sim-check sew-check sew-retarget-check hand-check typecheck

# Open the native MuJoCo viewer on the bimanual model, standalone (no backend)
sim-view:
    cd backend && uv run python -m mujoco.viewer --mjcf=assets/viewer_scene.xml

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
