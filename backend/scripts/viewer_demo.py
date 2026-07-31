"""Runs the Simulator + PDGravityController with a live GUI window.

`just sim-view` opens its own model/data with no controller attached, so it
just sits there (or drifts under whatever default ctrl MuJoCo initializes).
`just backend` + `just ws-example` runs the real controller, but in a
*separate* process/MjData with no window at all — the two never share state,
so you can't see one from the other.

This script puts the controller and the GUI in the *same* process sharing
the same MjData, so what you see is what the controller is actually doing:

- `mujoco.viewer.launch()` blocks the main thread and free-runs physics
  itself (mj_step every frame, using whatever `data.ctrl` currently holds).
  It does NOT need `mjpython` (only the non-blocking `launch_passive` does,
  and that's broken in this venv — see the dlopen error from `mjpython`).
- A background thread computes PD+gravity torque from the current qpos
  target and writes it into `data.ctrl`, at 60Hz. It never calls mj_step
  itself, to avoid double-stepping physics against the GUI's own loop.
- Every 3s the target alternates between resting and a lifted pose, so you
  can watch the arm actually track it.

Run:
    just viewer-demo
"""

import logging
import threading
import time

import mujoco.viewer
from app.control.pd_gravity import PDGravityController
from app.sim.simulator import Simulator

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("viewer_demo")


def control_loop(sim: Simulator, controller: PDGravityController) -> None:
    rest = sim.actuated_qpos().copy()
    lifted = rest.copy()
    lifted[0] += 0.6  # left_joint1
    lifted[8] += 0.6  # right_joint1

    waypoint_period_s = 3.0
    start = time.monotonic()
    last_target_name = None

    while True:
        elapsed = time.monotonic() - start
        at_lifted = int(elapsed // waypoint_period_s) % 2 == 1
        target = lifted if at_lifted else rest

        target_name = "lifted" if at_lifted else "rest"
        if target_name != last_target_name:
            logger.info("switching target -> %s", target_name)
            last_target_name = target_name

        sim.data.ctrl[:] = controller.torque(target)
        time.sleep(1 / 60)


def main() -> None:
    sim = Simulator()
    controller = PDGravityController(sim)

    thread = threading.Thread(target=control_loop, args=(sim, controller), daemon=True)
    thread.start()

    logger.info("opening viewer — close the window to exit")
    mujoco.viewer.launch(sim.model, sim.data)


if __name__ == "__main__":
    main()
