"""Combines landmark->target mapping and IK: given one PoseFrame, produce a
qpos_target (actuator-space) for the position servos to drive toward.
"""

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from app.retarget.ik import solve_site_ik
from app.retarget.mapper import PoseFrame, arm_extension
from app.sim.simulator import Simulator

logger = logging.getLogger(__name__)

# Number of elbow angles sampled when measuring each arm's reach.
_REACH_SAMPLES = 64

# Cap on the commanded radius, as a fraction of max reach. At full extension the
# arm is straight, which is exactly where the IK Jacobian goes singular and
# damped least-squares starts amplifying landmark noise into large joint swings.
# Staying slightly folded keeps the elbow able to contribute.
MAX_EXTENSION = 0.95

# Direction used when the hand is so close to the shoulder that the
# shoulder->wrist direction is numerically meaningless.
_MIN_DIRECTION = 1e-6


@dataclass
class ArmGeometry:
    """Where one arm's workspace sits, measured off the model at startup.

    The workspace is a spherical *shell*, not a ball: the links have length, so
    the control point can no more fold back onto the shoulder than a person can
    touch their own shoulder with their wrist. On this model that inner radius
    is 0.16m against a 0.44m outer radius — a third of full reach — so a folded
    human arm has no exact robot equivalent and has to be clamped outward.
    """

    site_id: int
    shoulder: np.ndarray  # world position of the first shoulder joint's axis
    min_reach: float  # inner radius of the reachable shell
    max_reach: float  # outer radius, arm straight
    neutral_direction: np.ndarray  # unit vector toward the resting control point

    def target(self, extension: np.ndarray) -> np.ndarray:
        """Map a unitless arm-extension vector into a world-frame IK target.

        Direction is taken from the human pose and preserved exactly; only the
        radius is rescaled into the robot's reachable shell. Clamping the radius
        rather than the whole vector means an out-of-range hand still points the
        robot the right way instead of collapsing toward the shoulder.
        """
        distance = float(np.linalg.norm(extension))
        if distance < _MIN_DIRECTION:
            direction = self.neutral_direction
        else:
            direction = extension / distance
        radius = np.clip(
            distance * self.max_reach, self.min_reach, self.max_reach * MAX_EXTENSION
        )
        return self.shoulder + radius * direction


class Retargeter:
    def __init__(self, sim: Simulator) -> None:
        self.model = sim.model
        self.actuator_qposadr = sim.actuator_qposadr

        # Scratch MjData for the IK solve — never the live sim's MjData,
        # since this mutates qpos directly without stepping physics.
        self.ik_data = mujoco.MjData(self.model)
        self._reset_pose()

        self.left = self._measure_arm("left")
        self.right = self._measure_arm("right")
        for side, arm in (("left", self.left), ("right", self.right)):
            logger.info(
                "%s arm: shoulder=%s reachable shell %.3f-%.3fm",
                side,
                np.round(arm.shoulder, 4),
                arm.min_reach,
                arm.max_reach,
            )

        # _measure_arm left qpos wherever the reach scan ended; put it back at
        # the neutral pose so the first solve warm-starts from something sane.
        self._reset_pose()

    def _reset_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.ik_data)
        mujoco.mj_forward(self.model, self.ik_data)

    def _id(self, objtype: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, objtype, name)
        if obj_id < 0:
            raise ValueError(f"{name!r} not found in model")
        return obj_id

    def _measure_arm(self, side: str) -> ArmGeometry:
        """Locate one arm's shoulder and measure how far it can reach.

        The shoulder is the world anchor of joint1, the first joint in the
        chain — the point the whole arm pivots about, and the natural origin
        for a human shoulder->wrist vector. Note this is NOT the resting
        end-effector position: at qpos0 the arm hangs straight down, a full
        arm-length below the shoulder, so anchoring there would put the neutral
        pose on the edge of the workspace and make every downward hand motion
        unreachable.

        Both radii are measured rather than assumed: sweep the elbow across its
        range with the rest of the arm at qpos0 and record how far the control
        point gets from the shoulder. Straight-armed is the maximum and fully
        folded the minimum, but sweeping keeps this correct if qpos0 or the
        joint ranges change.
        """
        site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
        shoulder_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint1")
        elbow_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint4")

        self._reset_pose()
        shoulder = self.ik_data.xanchor[shoulder_joint].copy()
        rest_offset = self.ik_data.site_xpos[site_id] - shoulder
        neutral_direction = rest_offset / np.linalg.norm(rest_offset)

        elbow_adr = self.model.jnt_qposadr[elbow_joint]
        lo, hi = self.model.jnt_range[elbow_joint]
        radii = []
        for angle in np.linspace(lo, hi, _REACH_SAMPLES):
            self.ik_data.qpos[elbow_adr] = angle
            mujoco.mj_forward(self.model, self.ik_data)
            radii.append(
                float(np.linalg.norm(self.ik_data.site_xpos[site_id] - shoulder))
            )

        return ArmGeometry(
            site_id=site_id,
            shoulder=shoulder,
            min_reach=min(radii),
            max_reach=max(radii),
            neutral_direction=neutral_direction,
        )

    @property
    def left_site(self) -> int:
        return self.left.site_id

    @property
    def right_site(self) -> int:
        return self.right.site_id

    def solve(self, frame: PoseFrame) -> np.ndarray:
        left_target = self.left.target(
            arm_extension(frame.left_shoulder, frame.left_elbow, frame.left_wrist)
        )
        right_target = self.right.target(
            arm_extension(frame.right_shoulder, frame.right_elbow, frame.right_wrist)
        )

        # ik_data.qpos is left over from the previous call, so this warm-starts:
        # consecutive frames converge in ~1 iteration and stay near the previous
        # pose instead of jumping between valid configurations (position-only
        # IK is under-constrained — see the redundancy note from earlier).
        solve_site_ik(self.model, self.ik_data, self.left.site_id, left_target)
        solve_site_ik(self.model, self.ik_data, self.right.site_id, right_target)

        return self.ik_data.qpos[self.actuator_qposadr].copy()
