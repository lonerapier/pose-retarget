"""Joint position commands for the OpenArm's built-in position servos.

Every actuator in openarm_bimanual.xml is a `<position>` actuator with its own
tuned gains (kp=230/kv=2.7 on the DM8009 shoulder joints, down to kp=30/kv=1.5
on the DM4310 wrist joints). MuJoCo runs that PD loop internally, so `ctrl` is
a *target angle in radians*, not a torque — there is no second controller for
us to write. All this class does is clamp the target into each actuator's
declared ctrlrange, since MuJoCo would silently clamp anyway and a caller that
drifts out of range should be able to see it.

Gravity compensation is likewise not ours to add: the servo's own kp term
absorbs the bias force (at the cost of some steady-state droop). Injecting
qfrc_bias here would be adding a torque to a position setpoint. Real gravity
comp would need `<motor>` actuators declared alongside these in the MJCF.
"""

import logging

import numpy as np

from app.sim.simulator import Simulator

logger = logging.getLogger(__name__)


class PositionController:
    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        ctrlrange = sim.model.actuator_ctrlrange
        self.lo = ctrlrange[:, 0].copy()
        self.hi = ctrlrange[:, 1].copy()
        logger.info("PositionController: %d position actuators", sim.model.nu)

    def ctrl(self, qpos_target: np.ndarray) -> np.ndarray:
        """Map a desired actuated-joint configuration to the actuator ctrl vector."""
        clamped = np.clip(qpos_target, self.lo, self.hi)
        logger.debug("target=%s clamped=%s", qpos_target, clamped)
        return clamped
