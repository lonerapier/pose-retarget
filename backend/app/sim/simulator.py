"""Loads the OpenArm bimanual MuJoCo model and steps physics."""

import logging
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "openarm_mujoco"
    / "v2"
    / "openarm_bimanual.xml"
)

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        logger.info("loading model from %s", model_path)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        # Actuators don't map 1:1 onto the first N joints (e.g. each gripper has
        # two finger joints but only one actuator), so resolve the real
        # qpos/dof address of the joint each actuator drives.
        joint_ids = self.model.actuator_trnid[:, 0]
        self.actuator_qposadr = self.model.jnt_qposadr[joint_ids]
        self.actuator_dofadr = self.model.jnt_dofadr[joint_ids]

        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        logger.info(
            "model loaded: nq=%d nv=%d nu=%d timestep=%.4f",
            self.model.nq,
            self.model.nv,
            self.model.nu,
            self.model.opt.timestep,
        )
        logger.info("actuators (%d): %s", self.model.nu, actuator_names)

    def actuated_qpos(self) -> np.ndarray:
        return self.data.qpos[self.actuator_qposadr].copy()

    def actuated_qvel(self) -> np.ndarray:
        return self.data.qvel[self.actuator_dofadr].copy()

    def step(self, ctrl: np.ndarray) -> None:
        """Advance one timestep. Every actuator in this model is a `<position>`
        actuator, so `ctrl` is a vector of target joint angles in radians (see
        PositionController), not torques."""
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        logger.debug(
            "t=%.3f ctrl=%s qpos=%s", self.data.time, ctrl, self.actuated_qpos()
        )

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def time(self) -> float:
        return self.data.time
