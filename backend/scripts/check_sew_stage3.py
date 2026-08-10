"""Checks SEW-Mimic stage 3 — the wrist — against MuJoCo forward kinematics.

Run:  just sew-stage3-check

Stage 3 is the stage most able to look right and be wrong. Every candidate
answer is a well-formed rotation matrix, and a wrist that is 90 degrees out
about the approach axis still points its gripper in a plausible direction. So
this checks properties, the way check_hand_frame.py does, rather than anything
that could be eyeballed in the viewer.

Two groups. The first four are claims about the *model* — they are what license
`solve_wrist` to be a Z-Y-X Euler decomposition rather than a general
Subproblem-1/2 solve, and if the OpenArm submodule ever moves, this is where
that shows up:

  1. The wrist is functionally spherical: joints 6 and 7 share an anchor, and
     joint 5's offset from it lies along joint 5's own axis, so no wrist joint
     can move the wrist centre. (DESIGN.md §3 overstates this as all three
     passing through one point; only two do.)
  2. The tool frame is the 7th joint's frame, i.e. R^{7,T}_local = I. That plus
     R_align = I (mapper.hand_orientation already emits the robot's ee
     convention) is what collapses H[R_align]^T[R^{7,T}_local]^T to plain H.
  3. This is the paper's *perpendicular wrist*: joint 7 turns about x while the
     gripper approaches along -z.
  4. R^{4,7} factors as Rz(s0*q5) Ry(s1*q6) Rx(s2*q7) with the signs
     `wrist_euler_signs` reads off the model.

The rest are claims about the solver:

  5. Round-trip. A reachable wrist pose, forward-kinematic'd and solved back,
     returns the same joints.
  6. Clamped poses stay inside the joint limits, report which joint hit a stop,
     and degrade continuously wherever the target is reachable. Where it is not,
     the two stops of a wrapped angle are both legal and roughly equally wrong,
     so a switch between them cannot be designed away — what is checked is that
     it stays rare, including under noise, which is what the `previous`
     tie-break is for.
  7. Gimbal lock (the paper's Remark 5) is unreachable, and so is the second
     Euler branch — neither needs handling, and this is why.
  8. Stage 3 leaves stages 1-2 alone: adding a wrist does not move the upper arm
     or the forearm.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retarget.sew import (  # noqa: E402
    Stage12Solver,
    axis_rotation,
    solve_wrist,
    wrist_euler_signs,
)

MODEL_PATH = "../openarm_mujoco/v2/openarm_bimanual.xml"
N_SAMPLES = 300
TOL_DEG = 1e-4
TOL = 1e-9

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking `a` to `b`, in degrees."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class Arm:
    """One arm's ids and constants, read off the model."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str) -> None:
        self.side = side
        self.model, self.data = model, data
        self.joints = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{n}"
            )
            for n in range(1, 8)
        ]
        self.qposadr = model.jnt_qposadr[self.joints]
        self.lower, self.upper = model.jnt_range[self.joints].T
        self.axes = model.jnt_axis[self.joints]
        self.base = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_base_link"
        )
        self.forearm = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link4"
        )
        self.ee_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_ee_base_link"
        )
        self.site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point"
        )
        self.fingers = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_ee_{n}_finger"
            )
            for n in ("inner", "outer")
        ]
        self.signs = wrist_euler_signs(self.axes[4:7])

    def fk(self, q: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qposadr] = q
        mujoco.mj_forward(self.model, self.data)

    def base_R(self) -> np.ndarray:
        return self.data.xmat[self.base].reshape(3, 3)

    def forearm_R(self) -> np.ndarray:
        """R^{0,4}: the frame at the base of the wrist, in the arm base frame."""
        return self.base_R().T @ self.data.xmat[self.forearm].reshape(3, 3)

    def tool_R(self) -> np.ndarray:
        """The achieved tool frame, in the arm base frame."""
        return self.base_R().T @ self.data.site_xmat[self.site].reshape(3, 3)

    def random_q(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper)


def report(label: str, worst: float, tol: float, unit_: str = "deg") -> bool:
    ok = bool(worst < tol)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<48} worst {worst:.3e} {unit_}")
    return ok


def check_model(arm: Arm, results: list[bool]) -> None:
    arm.fk(np.zeros(7))

    # --- 1: the wrist is functionally spherical ----------------------------
    #
    # DESIGN.md §3 says joints 5, 6 and 7 all pass through the ee_base_link
    # origin. Only 6 and 7 do; joint 5 sits 120.5 mm up the forearm. What makes
    # the wrist behave as a spherical one anyway is that the offset lies along
    # joint 5's *own* axis, so turning q5 cannot move the wrist centre. Checked
    # that way round, because that is the property anything downstream uses.
    anchors = arm.data.xanchor[arm.joints[4:7]]
    results.append(
        report(
            "joints 6,7 share one anchor (wrist centre)",
            float(np.abs(anchors[1] - anchors[2]).max()),
            1e-9,
            "m",
        )
    )
    offset = anchors[2] - anchors[0]
    off_axis = float(
        np.linalg.norm(np.cross(unit(offset), arm.data.xaxis[arm.joints[4]]))
    )
    results.append(
        report(
            f"joint 5 offset ({np.linalg.norm(offset) * 1e3:.1f} mm) lies on its axis",
            off_axis,
            1e-9,
            "",
        )
    )

    # --- 2: R^{7,T}_local = I ----------------------------------------------
    ee_R = arm.data.xmat[arm.ee_body].reshape(3, 3)
    local = ee_R.T @ arm.data.site_xmat[arm.site].reshape(3, 3)
    results.append(
        report(
            "R^{7,T}_local == I (tool frame is frame 7)",
            float(np.abs(local - np.eye(3)).max()),
            TOL,
            "",
        )
    )
    offset = arm.data.site_xpos[arm.site] - arm.data.xpos[arm.ee_body]
    results.append(
        report(
            "tool site sits on the joint-7 origin",
            float(np.abs(offset).max()),
            1e-9,
            "m",
        )
    )

    # --- 3: perpendicular wrist --------------------------------------------
    #
    # The gripper's approach direction is read off where the fingers are mounted
    # rather than assumed: both finger bodies hang off the ee frame along it.
    # It is 1.2 degrees off exactly -z (the mount has a 1.4 mm x offset), which
    # is why this asks for "perpendicular to within a couple of degrees" rather
    # than exactly 90 — the point of the check is that joint 7 is nowhere near
    # the approach axis, not that it is machined square to it.
    h7 = arm.axes[6]
    approach = unit(np.mean([arm.model.body_pos[f] for f in arm.fingers], axis=0))
    results.append(
        report(
            "joint-7 axis ~perpendicular to gripper approach",
            abs(90.0 - angle_between(h7, approach)),
            2.0,
        )
    )
    print(
        f"        h7 = {h7.astype(int)}  approach = {np.round(approach, 4)}"
        f"  angle {angle_between(h7, approach):.2f} deg"
    )

    # --- 4: R^{4,7} = Rz Ry Rx with the signs read off the model ------------
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(N_SAMPLES):
        q = arm.random_q(rng)
        arm.fk(q)
        achieved = arm.forearm_R().T @ (
            arm.base_R().T @ arm.data.xmat[arm.ee_body].reshape(3, 3)
        )
        s = arm.signs
        factored = (
            axis_rotation(_Z, s[0] * q[4])
            @ axis_rotation(_Y, s[1] * q[5])
            @ axis_rotation(_X, s[2] * q[6])
        )
        worst = max(worst, float(np.abs(factored - achieved).max()))
    results.append(
        report(f"R^4,7 == Rz Ry Rx, signs {arm.signs.astype(int)}", worst, TOL, "")
    )


def check_solver(arm: Arm, results: list[bool]) -> None:
    rng = np.random.default_rng(1)

    # --- 5: round-trip over reachable wrist poses ---------------------------
    worst_q, worst_R = 0.0, 0.0
    for _ in range(N_SAMPLES):
        q = arm.random_q(rng)
        arm.fk(q)
        desired = arm.forearm_R().T @ arm.tool_R()
        wrist = solve_wrist(desired, arm.signs, arm.lower[4:7], arm.upper[4:7])
        worst_q = max(worst_q, float(np.degrees(np.abs(wrist.q - q[4:7])).max()))
        arm.fk(np.concatenate([q[:4], wrist.q]))
        worst_R = max(worst_R, geodesic_deg(arm.forearm_R().T @ arm.tool_R(), desired))
        if wrist.clamped.any():
            worst_q = np.inf  # a reachable pose must never clamp
    results.append(report("reachable pose: q -> H -> q", worst_q, TOL_DEG))
    results.append(
        report("reachable pose: achieved orientation == H", worst_R, TOL_DEG)
    )

    # --- 6: unreachable poses stay legal and degrade continuously -----------
    from scipy.spatial.transform import Rotation

    rotations = Rotation.random(N_SAMPLES, rng=rng).as_matrix()
    worst_violation, worst_mismatch = 0.0, 0.0
    errors = []
    for desired in rotations:
        wrist = solve_wrist(desired, arm.signs, arm.lower[4:7], arm.upper[4:7])
        worst_violation = max(
            worst_violation,
            float(np.maximum(arm.lower[4:7] - wrist.q, wrist.q - arm.upper[4:7]).max()),
        )
        # `clamped` must agree with the joints actually sitting on a stop.
        on_stop = np.isclose(wrist.q, arm.lower[4:7]) | np.isclose(
            wrist.q, arm.upper[4:7]
        )
        worst_mismatch = max(worst_mismatch, float((wrist.clamped & ~on_stop).sum()))
        arm.fk(np.concatenate([np.zeros(4), wrist.q]))
        errors.append(geodesic_deg(arm.forearm_R().T @ arm.tool_R(), desired))
    results.append(
        report(
            "unreachable pose: joints stay inside limits",
            max(0.0, worst_violation),
            TOL,
            "rad",
        )
    )
    results.append(
        report("clamped flag matches joints on a stop", worst_mismatch, 0.5, "")
    )

    errors = np.array(errors)
    exact = float((errors < TOL_DEG).mean())
    print(
        f"        uniform SO(3): {100 * exact:4.1f}% exact, "
        f"median residual {np.median(errors):5.1f} deg, max {errors.max():5.1f} deg"
    )

    # Continuity. Walk full turns of hand orientation about several axes,
    # straight through the saturated regions.
    #
    # Two different properties, and conflating them hid a real bug in the first
    # draft. Where the target is *reachable* the command must be continuous, full
    # stop. Where it isn't, the two stops of a wrapped angle are both legal and
    # roughly equally wrong, and which one is nearer flips as the target crosses
    # the far side — that switch is a real 180 degree command change and no
    # tie-break makes it vanish, it can only be made rare (see the noise test
    # below) and survivable (the slew limiter, §6.3, still to land). So: assert
    # continuity on the reachable side, and bound the *count* of switches on the
    # other.
    axes = (
        unit(np.array([0.3, 1.0, 0.2])),
        unit(np.array([1.0, 0.0, 0.0])),
        unit(np.array([0.0, 0.0, 1.0])),
        unit(np.array([-0.5, 0.4, 1.0])),
    )
    worst_step, switches = 0.0, 0
    for axis in axes:
        previous, was_reachable = None, False
        for t in np.linspace(0.0, 2 * np.pi, 2881):
            wrist = solve_wrist(
                axis_rotation(axis, t),
                arm.signs,
                arm.lower[4:7],
                arm.upper[4:7],
                previous=previous,
            )
            reachable = not wrist.clamped.any()
            if previous is not None:
                step = float(np.degrees(np.abs(wrist.q - previous)).max())
                if reachable and was_reachable:
                    worst_step = max(worst_step, step)
                elif step > 90.0:
                    switches += 1
            previous, was_reachable = wrist.q, reachable
    results.append(report("reachable sweep: command is continuous", worst_step, 0.5))
    results.append(
        report(
            f"saturated sweep: stop switches over {len(axes)} turns",
            float(switches),
            2.0 * len(axes),
            "",
        )
    )

    # Chatter. Park the target exactly on one of those switch points and shake it
    # at the scale the hand frame actually jitters at (§9.4: the four MCP
    # landmarks are nearly coincident at webcam resolution, so the roll about the
    # hand is the noisiest thing in the pipeline). Without the `previous`
    # tie-break this flips stop to stop on half of all frames; the point of the
    # tolerance is that a switch has to be *worth* making, not merely closer.
    from scipy.spatial.transform import Rotation as _R

    noise = np.random.default_rng(7)
    parked = axis_rotation(axes[0], np.radians(198.0))
    previous, flips = None, 0
    for _ in range(2000):
        jitter = _R.from_rotvec(np.radians(5.0) * noise.normal(size=3)).as_matrix()
        wrist = solve_wrist(
            jitter @ parked,
            arm.signs,
            arm.lower[4:7],
            arm.upper[4:7],
            previous=previous,
        )
        if previous is not None and np.degrees(np.abs(wrist.q - previous)).max() > 90.0:
            flips += 1
        previous = wrist.q
    results.append(
        report(
            "held on a switch point, 5 deg noise: flips", float(flips), 100.0, "/2000"
        )
    )

    # --- 7: gimbal lock and the second Euler branch are unreachable ---------
    q6_limit = float(np.degrees(np.abs(arm.lower[5:6]).max()))
    q6_limit = max(q6_limit, float(np.degrees(abs(arm.upper[5]))))
    # Gimbal lock is at |q6| = 90; the other Euler solution needs |pi - b|,
    # i.e. at least 180 - q6_limit. Both are outside the joint's range, which is
    # why solve_wrist has neither a degeneracy guard nor a branch enumeration.
    print(f"        |q6| <= {q6_limit:.0f} deg")
    results.append(report("gimbal lock (|q6| = 90) unreachable", q6_limit, 90.0))
    results.append(
        report(
            f"second Euler branch (|q6| >= {180 - q6_limit:.0f}) unreachable",
            q6_limit,
            180.0 - q6_limit,
        )
    )


def check_composition(arm: Arm, results: list[bool]) -> None:
    """Stage 3 must not disturb stages 1-2.

    The stages run in sequence and stage 3 writes three more joints, so the
    thing to rule out is that adding a wrist quietly moves the limbs the first
    two stages just aimed. Measured as the *difference the wrist makes* — limb
    directions with q5..q7 solved, against the same q1..q4 with the wrist at
    zero — so any residual left by stages 1-2 cancels instead of being counted
    here, where it would not belong.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(2)
    solver = Stage12Solver(arm.side, lower=arm.lower[:4], upper=arm.upper[:4])
    worst_u, worst_l = 0.0, 0.0

    for desired in Rotation.random(50, rng=rng).as_matrix():
        arm.fk(arm.random_q(rng))
        base = arm.base_R()
        u = unit(base.T @ arm.data.xaxis[arm.joints[2]])
        limb = unit(base.T @ arm.data.xaxis[arm.joints[4]])

        # Settle stages 1-2 on this pose: the solver shrinks toward `previous`,
        # so a single frame is still on its way there. That is smoothing, not
        # error, and it is not what this check is about.
        for _ in range(20):
            q14 = solver.solve(u, limb)

        arm.fk(np.concatenate([q14, np.zeros(3)]))
        base = arm.base_R()
        without = (
            unit(base.T @ arm.data.xaxis[arm.joints[2]]),
            unit(base.T @ arm.data.xaxis[arm.joints[4]]),
        )
        wrist = solve_wrist(
            arm.forearm_R().T @ desired, arm.signs, arm.lower[4:7], arm.upper[4:7]
        )

        arm.fk(np.concatenate([q14, wrist.q]))
        base = arm.base_R()
        worst_u = max(
            worst_u,
            angle_between(unit(base.T @ arm.data.xaxis[arm.joints[2]]), without[0]),
        )
        worst_l = max(
            worst_l,
            angle_between(unit(base.T @ arm.data.xaxis[arm.joints[4]]), without[1]),
        )
    # Exactly zero in exact arithmetic — no wrist joint appears in either limb's
    # forward kinematics at all — so the tolerance is only here to absorb the
    # float noise of reading the axes back out of a longer transform chain.
    results.append(report("stage 3 leaves the upper arm alone", worst_u, 1e-3))
    results.append(report("stage 3 leaves the forearm alone", worst_l, 1e-3))


def main() -> int:
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[1] / MODEL_PATH)
    )
    data = mujoco.MjData(model)
    results: list[bool] = []

    for side in ("left", "right"):
        arm = Arm(model, data, side)
        print(f"\n{side} arm")
        print("  -- model --")
        check_model(arm, results)
        print("  -- solver --")
        check_solver(arm, results)
        print("  -- composition --")
        check_composition(arm, results)

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
