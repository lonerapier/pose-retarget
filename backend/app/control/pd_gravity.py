"""PD control toward joint targets, plus gravity compensation via MuJoCo's bias term."""

import logging

import numpy as np

from app.sim.simulator import Simulator

logger = logging.getLogger(__name__)


class PDGravityController:
    def __init__(self, sim: Simulator, kp: float = 50.0, kd: float = 5.0) -> None:
        self.sim = sim
        self.kp = kp
        self.kd = kd
        logger.info("PDGravityController: kp=%.1f kd=%.1f", kp, kd)

    def torque(self, qpos_target: np.ndarray) -> np.ndarray:
        error = qpos_target - self.sim.actuated_qpos()
        pd = self.kp * error - self.kd * self.sim.actuated_qvel()
        gravity = self.sim.gravity_compensation()
        logger.debug("error=%s pd=%s gravity=%s", error, pd, gravity)
        return pd + gravity
