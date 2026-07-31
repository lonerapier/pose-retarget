"""Damped least-squares IK: solve for joint deltas that move a site toward a target position."""

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
