"""Checks the closed-form SEW stage-1/2 derivation against MuJoCo forward
kinematics.

Run:  just sew-check

Two claims, both of which the closed form depends on:

  1. Joint 3's axis is collinear with the upper arm (joint3 anchor -> joint4 anchor)
     and joint 5's axis is collinear with the forearm (joint4 anchor -> wrist centre).
     That's what makes "align joint 3's axis with the human upper arm" the same thing
     as "point the robot's upper arm where the human's points".

  2. Given the two human limb directions in the arm base frame, solve_stage12()
     recovers joints 1-4 exactly.

Claim 1 failing means the OpenArm model changed and the approach needs re-deriving.
Claim 2 failing means the trigonometry below is wrong — cheaper to debug here than in
the live pipeline.
"""

import mujoco
import numpy as np

MODEL_PATH = "../openarm_mujoco/v2/openarm_bimanual.xml"
N_SAMPLES = 300
TOL_DEG = 1e-4


def rot(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def solve_stage12(
    side: str,
    u: np.ndarray,
    l: np.ndarray,  # noqa: E741 — u and l are the derivation's own names
    prev: np.ndarray,
) -> list[np.ndarray]:
    """SEW-Mimic stages 1-2, specialised to OpenArm v2.

    u, l: unit vectors for the human upper arm and forearm, in the ARM BASE frame.
    prev: previous frame's (q1, q2, q3, q4), used to resolve the two singularities.

    Returns both branches; the caller filters by joint limits and picks the one
    nearest `prev`. Keeping the choice out of here keeps the branch policy — a
    control decision, not a geometric one — out of the geometry.
    """
    # Joint 1's axis is (0, +1, 0) on the left arm and (0, -1, 0) on the right; every
    # other axis has the same sign on both. sx carries that one difference.
    sx = -1.0 if side == "left" else 1.0
    h1 = np.array([0.0, -sx, 0.0])
    h2 = np.array([-1.0, 0.0, 0.0])

    # Stage 1: u = (sx*sin(q1)*cos(q2), -sin(q2), -cos(q1)*cos(q2))
    sin_q2 = float(np.clip(-u[1], -1.0, 1.0))
    root = float(np.sqrt(max(0.0, 1.0 - sin_q2**2)))

    solutions = []
    for cos_q2 in (root, -root):
        q2 = float(np.arctan2(sin_q2, cos_q2))
        if abs(cos_q2) < 1e-9:
            # Upper arm along the shoulder axis (arm straight out to the side):
            # q1 is indeterminate. Hold the previous value.
            q1 = float(prev[0])
        else:
            q1 = float(np.arctan2(sx * u[0] / cos_q2, -u[2] / cos_q2))

        # Stage 2: with A = R(h1,q1) R(h2,q2) and v = A^T l,
        #          v = (cos(q3)*sin(q4), -sin(q3)*sin(q4), -cos(q4))
        v = (rot(h1, q1) @ rot(h2, q2)).T @ l
        q4 = float(np.arccos(np.clip(-v[2], -1.0, 1.0)))
        if abs(np.sin(q4)) < 1e-9:
            q3 = float(prev[2])  # straight arm: q3 indeterminate
        else:
            q3 = float(np.arctan2(-v[1], v[0]))

        solutions.append(np.array([q1, q2, q3, q4]))
    return solutions


class ArmProbe:
    """Reads limb directions and joint axes off a live MjData."""

    def __init__(self, model, data, side: str):
        self.model, self.data, self.side = model, data, side
        self.jnt = {
            n: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{n}"
            )
            for n in range(1, 8)
        }
        self.base = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_base_link"
        )
        self.qadr = {n: model.jnt_qposadr[j] for n, j in self.jnt.items()}
        self.range = {n: model.jnt_range[j] for n, j in self.jnt.items()}

    def set_q(self, q: dict[int, float]) -> None:
        self.data.qpos[:] = 0.0
        for n, v in q.items():
            self.data.qpos[self.qadr[n]] = v
        mujoco.mj_forward(self.model, self.data)

    def base_R(self) -> np.ndarray:
        return self.data.xmat[self.base].reshape(3, 3)

    def axis_world(self, n: int) -> np.ndarray:
        j = self.jnt[n]
        return (
            self.data.xmat[self.model.jnt_bodyid[j]].reshape(3, 3)
            @ (self.model.jnt_axis[j])
        )

    def limb_world(self, a: int, b: int) -> np.ndarray:
        v = self.data.xanchor[self.jnt[b]] - self.data.xanchor[self.jnt[a]]
        return v / np.linalg.norm(v)

    def limb_dirs_base(self) -> tuple[np.ndarray, np.ndarray]:
        """(upper arm, forearm) unit vectors in the arm base frame."""
        r = self.base_R().T
        return r @ self.limb_world(3, 4), r @ self.limb_world(4, 7)

    def random_q(self, rng) -> dict[int, float]:
        return {n: float(rng.uniform(*self.range[n])) for n in range(1, 8)}


def main() -> int:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    failed = False

    for side in ("left", "right"):
        arm = ArmProbe(model, data, side)

        # --- claim 1: joint axes are collinear with the limbs they represent ---
        worst_upper = worst_fore = 0.0
        for _ in range(N_SAMPLES):
            arm.set_q(arm.random_q(rng))
            worst_upper = max(
                worst_upper, angle_between(arm.axis_world(3), arm.limb_world(3, 4))
            )
            worst_fore = max(
                worst_fore, angle_between(arm.axis_world(5), arm.limb_world(4, 7))
            )

        # --- claim 2: the closed form round-trips ---
        worst_rt = 0.0
        for _ in range(N_SAMPLES):
            arm.set_q(arm.random_q(rng))
            u, l = arm.limb_dirs_base()  # noqa: E741

            best = np.inf
            for q in solve_stage12(side, u, l, prev=np.zeros(4)):
                arm.set_q(dict(zip((1, 2, 3, 4), q)))
                u2, l2 = arm.limb_dirs_base()
                best = min(best, max(angle_between(u, u2), angle_between(l, l2)))
            worst_rt = max(worst_rt, best)

        ok = max(worst_upper, worst_fore, worst_rt) < TOL_DEG
        failed |= not ok
        print(f"{side:>5} arm  [{'PASS' if ok else 'FAIL'}]")
        print(f"        angle(joint3 axis, upper arm)  max {worst_upper:.3e} deg")
        print(f"        angle(joint5 axis, forearm)    max {worst_fore:.3e} deg")
        print(f"        closed-form round-trip error   max {worst_rt:.3e} deg")

    print(f"\ntolerance {TOL_DEG:.0e} deg over {N_SAMPLES} random configs per arm")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
