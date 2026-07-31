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

# Run the FastAPI backend with autoreload
backend:
    cd backend && uv run uvicorn app.main:app --reload --port 8000

# Run the Vite frontend dev server
frontend:
    cd frontend && npm run dev

# Sanity-check: load the OpenArm model, step physics with PD + gravity comp
sim-check:
    cd backend && uv run python -c "\
    import logging; logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(name)s: %(message)s'); \
    from app.sim.simulator import Simulator; \
    from app.control.pd_gravity import PDGravityController; \
    sim = Simulator(); \
    ctrl = PDGravityController(sim); \
    target = sim.actuated_qpos(); \
    [sim.step(ctrl.torque(target)) for _ in range(200)]; \
    print('time:', sim.time); \
    print('actuated qpos:', sim.actuated_qpos())"

# Connect a tiny example client to the running backend and watch the sim state stream
ws-example:
    cd backend && uv run python scripts/ws_client_example.py

# Run the PD+gravity controller with a live GUI window in the same process
viewer-demo:
    cd backend && PYTHONPATH=. uv run python scripts/viewer_demo.py

# Open the native MuJoCo viewer on the bimanual model
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
