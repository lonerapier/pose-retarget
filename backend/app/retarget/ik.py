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
    """Clip qpos to each joint's declared range, in place.

    Without it an unconstrained least-squares step will walk a joint past its
    physical limit to satisfy the site target. That isn't only ugly: it can land
    the arm near a singular Jacobian.
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
    """Step until the site is within `tol` metres of target_pos, or max_iters
    runs out.

    Mutates `data.qpos` in place, so `data` must be scratch MjData and never the
    live sim's, since this writes qpos without going through mj_step.
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
