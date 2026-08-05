"""Combines landmark->target mapping and IK: given one PoseFrame, produce a
qpos_target (actuator-space) for the position servos to drive toward.

Two retargeting paths live here, chosen by `mode`:

- "sew"  (default) closed-form SEW-Mimic stages 1-2, NEXT_STEPS §0.2. Matches
         the human's *limb orientations*; joints 1-4 are solved exactly and
         joints 5-7 hold their neutral posture (stage 3 is not built yet).
- "ik"   the original position-only damped-least-squares solve on the ee site.
         Kept for A/B comparison against the battery — it is the thing §0 is
         arguing against, and the argument is worth being able to demonstrate.

Set RETARGET_MODE=ik to select the old path.
"""

import logging
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from app.retarget.ik import solve_site_ik
from app.retarget.mapper import PoseFrame, arm_extension, limb_directions, torso_frame
from app.retarget.sew import STAGE12_JOINTS, Stage12Solver
from app.retarget.smoothing import FilterConfig, LandmarkFilter
from app.sim.simulator import Simulator

logger = logging.getLogger(__name__)

# Retargeting paths, in the order of preference argued for in NEXT_STEPS §0.
MODES = ("sew", "ik")
DEFAULT_MODE = "sew"

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
class ArmSolve:
    """What one arm's retarget did on one frame, for telemetry.

    Kept as a record of the last solve rather than logged from inside the solve
    itself: the interesting line is landmarks *and* joints together, and only
    the caller has both. All fields are None on a held frame except `held`.
    """

    held: bool  # limb directions were degenerate; the arm kept its last pose
    upper_arm: np.ndarray | None = None  # human u, in this arm's base frame
    forearm: np.ndarray | None = None  # human l, same frame
    q: np.ndarray | None = None  # solved joints 1-4
    clamped: np.ndarray | None = None  # per-joint: solve landed outside limits
    ee: np.ndarray | None = None  # resulting control point, world frame


@dataclass
class ArmGeometry:
    """Where one arm's workspace sits, measured off the model at startup.

    The workspace is a spherical *shell*, not a ball: the links have length, so
    the control point can no more fold back onto the shoulder than a person can
    touch their own shoulder with their wrist. On this model that inner radius
    is 0.16m against a 0.44m outer radius — a third of full reach — so a folded
    human arm has no exact robot equivalent and has to be clamped outward.

    Everything below `stage12_range` is used by the position path only. The SEW path
    consumes just the base rotation (to express the human's limb directions in
    the frame the closed form is derived in) and the stage-1/2 joint addresses:
    it never targets a position, so reach cannot fail and nothing needs to be
    clamped into a shell.
    """

    site_id: int
    base_R: np.ndarray  # arm base frame in world axes, columns are its axes
    stage12_qposadr: np.ndarray  # qpos addresses of joints 1-4, in order
    stage12_range: np.ndarray  # (4, 2) joint limits for those same joints
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
    def __init__(
        self,
        sim: Simulator,
        mode: str | None = None,
        smoothing: FilterConfig | None = None,
    ) -> None:
        self.model = sim.model
        self.actuator_qposadr = sim.actuator_qposadr

        self.mode = (mode or os.environ.get("RETARGET_MODE") or DEFAULT_MODE).lower()
        if self.mode not in MODES:
            raise ValueError(f"unknown retarget mode {self.mode!r}, expected {MODES}")

        # Landmark smoothing sits inside the retargeter rather than at the
        # socket so that every caller gets it: the live path, a replay, and the
        # check scripts all measure the same pipeline. Pass
        # FilterConfig(enabled=False) to A/B against the unfiltered input.
        self.filter = LandmarkFilter(smoothing)

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
        # In "sew" mode this is also what leaves joints 5-7 at neutral, which
        # is where they stay — nothing writes them until stage 3 exists.
        self._reset_pose()

        self.stage12 = {
            side: Stage12Solver(
                side,
                lower=arm.stage12_range[:, 0],
                upper=arm.stage12_range[:, 1],
                initial=self.ik_data.qpos[arm.stage12_qposadr],
            )
            for side, arm in (("left", self.left), ("right", self.right))
        }
        # Overwritten every solve; telemetry reads it instead of the solve
        # paths having to know how anything is formatted.
        self.last_solve = {side: ArmSolve(held=False) for side in ("left", "right")}
        # The landmarks the last solve actually ran on — filtered, so not the
        # ones that arrived on the wire. Telemetry logs these: a value that
        # cannot be reconciled with the joints it produced is the thing worth
        # noticing, and the raw landmark is not that value any more.
        self.last_frame: PoseFrame | None = None
        # A client that sends no hips gives every frame a degenerate torso, and
        # a per-frame warning at 30fps would bury the log. Say it once.
        self._warned_no_torso = False
        logger.info("retargeting mode: %s", self.mode)
        logger.info("%s", self.filter.config.describe())

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
        base_body = self._id(mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_base_link")
        stage12 = [
            self._id(mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{n}")
            for n in STAGE12_JOINTS
        ]

        self._reset_pose()
        # The base link is welded to the torso, so its orientation is fixed and
        # reading it once at startup is enough.
        base_R = self.ik_data.xmat[base_body].reshape(3, 3).copy()
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
            base_R=base_R,
            stage12_qposadr=self.model.jnt_qposadr[stage12].copy(),
            stage12_range=self.model.jnt_range[stage12].copy(),
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
        # Smooth first, so that everything downstream — including the discrete
        # branch choice in Stage12Solver, which no amount of joint-space
        # filtering could undo — sees one consistent, de-jittered subject.
        frame = self.filter.apply(frame)
        self.last_frame = frame
        if self.mode == "sew":
            return self._solve_sew(frame)
        return self._solve_site_ik(frame)

    def _solve_sew(self, frame: PoseFrame) -> np.ndarray:
        """Closed-form SEW-Mimic stages 1-2 (NEXT_STEPS §0.2).

        Per arm: read the human's two limb directions, express them in that
        arm's base frame, and solve joints 1-4 in one shot. Joints 5-7 are left
        wherever they are — at neutral, since nothing writes them — because
        stage 3 needs a wrist orientation target we have measured
        (`mapper.hand_orientation`) but do not yet consume.

        Unlike the position path this does no iterating and keeps no null space
        to drift, so a lost or degenerate landmark can simply be skipped: the
        arm holds its last pose and picks straight back up on the next good
        frame, with no accumulated error to unwind.
        """
        torso = torso_frame(frame)
        if torso is None and not self._warned_no_torso:
            self._warned_no_torso = True
            logger.warning(
                "no usable torso frame (hips missing or collapsed); falling back "
                "to camera-frame limb directions, which rotate with the subject"
            )

        for side, arm in (("left", self.left), ("right", self.right)):
            if side == "left":
                landmarks = (frame.left_shoulder, frame.left_elbow, frame.left_wrist)
            else:
                landmarks = (frame.right_shoulder, frame.right_elbow, frame.right_wrist)

            directions = limb_directions(*landmarks)
            if directions is None:
                logger.debug("%s arm: degenerate limb directions, holding pose", side)
                self.last_solve[side] = ArmSolve(held=True)
                continue

            solver = self.stage12[side]
            # Two frame changes, composed once per arm per frame.
            #
            #   torso.T    camera axes -> the subject's own torso frame. Makes
            #              the input body-centric, so leaning or turning moves
            #              nothing (§6.2, and §7.3's prerequisite).
            #   base_R.T   robot world axes -> this arm's base frame, which is
            #              the frame sew.py's closed form is derived in.
            #
            # The step between them is the retargeting itself and needs no
            # matrix: a direction in the human's torso frame is *taken to be*
            # the same direction in the robot's torso frame. That identification
            # is the whole of what "mimic" means here, and it is why link
            # lengths and the shoulder offset never enter.
            to_arm_base = arm.base_R.T if torso is None else arm.base_R.T @ torso.T
            upper_arm = to_arm_base @ directions[0]
            forearm = to_arm_base @ directions[1]
            q = solver.solve(upper_arm, forearm)
            self.ik_data.qpos[arm.stage12_qposadr] = q
            self.last_solve[side] = ArmSolve(
                held=False,
                upper_arm=upper_arm,
                forearm=forearm,
                q=q,
                clamped=solver.last_clamped.copy(),
            )

        # Not needed to build the command — this writes qpos directly — but it
        # keeps ik_data's derived quantities consistent with its qpos, so the
        # site positions stay meaningful for logging and for metrics (§6.4).
        mujoco.mj_forward(self.model, self.ik_data)
        for side, arm in (("left", self.left), ("right", self.right)):
            self.last_solve[side].ee = self.ik_data.site_xpos[arm.site_id].copy()
        return self.ik_data.qpos[self.actuator_qposadr].copy()

    def _solve_site_ik(self, frame: PoseFrame) -> np.ndarray:
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

        # This path has no limb directions or stage-1/2 branch to report; the ee
        # site is the one field that means the same thing in both modes.
        for side, arm in (("left", self.left), ("right", self.right)):
            self.last_solve[side] = ArmSolve(
                held=False, ee=self.ik_data.site_xpos[arm.site_id].copy()
            )
        return self.ik_data.qpos[self.actuator_qposadr].copy()
