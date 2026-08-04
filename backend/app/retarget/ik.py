"""Damped least-squares IK: solve for joint deltas that move a site toward a
target position."""

import mujoco
import numpy as np


def damped_least_squares_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    damping: float = 0.05,
) -> np.ndarray:
    """Return a qpos delta (length model.nv) for one IK iteration toward target_pos."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    error = target_pos - data.site_xpos[site_id]
    jjt = jacp @ jacp.T + damping**2 * np.eye(3)
    dq = jacp.T @ np.linalg.solve(jjt, error)
    return dq


def _clamp_to_joint_limits(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Clip qpos to each joint's declared range in place.

    Without this, an unconstrained least-squares step can walk a joint past
    its physical limit (e.g. an elbow bending the "wrong" way) to satisfy the
    site target. That doesn't just look wrong — it can land the arm in a
    configuration where the Jacobian is close to singular, at which point the
    solver stops making progress at all (see the elbow example this caught).
    """
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        adr = model.jnt_qposadr[joint_id]
        lo, hi = model.jnt_range[joint_id]
        data.qpos[adr] = np.clip(data.qpos[adr], lo, hi)


def solve_site_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    max_iters: int = 20,
    tol: float = 1e-4,
    damping: float = 0.05,
) -> None:
    """Repeatedly apply damped_least_squares_step until the site is within
    `tol` meters of target_pos, or max_iters is reached. A single step is only
    a linear approximation (it uses the Jacobian at the current pose), so it
    needs to be iterated to actually converge for anything but a tiny move.

    Mutates `data.qpos` in place. `data` should be a scratch MjData used only
    for this kinematic solve — never the live simulation's MjData, since this
    writes qpos directly without going through torque/mj_step at all.
    """
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        error = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(error) < tol:
            return
        dq = damped_least_squares_step(model, data, site_id, target_pos, damping)
        data.qpos[:] += dq
        _clamp_to_joint_limits(model, data)
    mujoco.mj_forward(model, data)
