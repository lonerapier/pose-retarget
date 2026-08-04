"""Landmark schema + math for retargeting: turns MediaPipe body landmarks
into a unitless arm-extension vector in the robot's axes.

The human side of retargeting ends here, and it deliberately produces no
meters: converting to the robot's workspace needs the robot's own shoulder
position and reach, which live in retargeter.py.
"""

import numpy as np
from pydantic import BaseModel

# Below this (in landmark units) the shoulder/elbow/wrist triple is degenerate
# — landmarks collapsed onto each other, or the subject isn't really detected —
# and the direction of the arm is meaningless.
MIN_ARM_LENGTH = 1e-3


class Landmark(BaseModel):
    """MediaPipe body landmark: x/y normalized to the video frame (x: 0=left
    edge, 1=right edge; y: 0=top, 1=bottom), z roughly the same scale with
    the hips as a rough depth origin (more negative = closer to the camera).
    These are the RAW camera-frame landmarks, not the mirrored image shown on
    screen — the mirror is CSS-only, for display, and isn't reflected here.
    """

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class PoseFrame(BaseModel):
    """One frame of the 6 landmarks relevant to bimanual retargeting."""

    left_shoulder: Landmark
    left_elbow: Landmark
    left_wrist: Landmark
    right_shoulder: Landmark
    right_elbow: Landmark
    right_wrist: Landmark


def to_robot_axes(delta: np.ndarray) -> np.ndarray:
    """Permute a landmark-space displacement into the robot's world axes.

    Axis mapping, first-pass and UNVERIFIED against a real camera — flip signs
    by watching motion in the live viewer:
    - forward (robot +x): wrist closer to the camera than the shoulder
      (z more negative) -> reach further forward.
    - left (robot +y): landmarks are in the raw, unmirrored camera frame —
      like a photo, not a mirror — so the subject's own left hand moving
      further left increases its image-x. No extra flip needed.
    - up (robot +z): image y increases downward -> flip sign.

    This is a pure permutation with sign flips, so it preserves vector norms —
    which is what lets arm_extension() normalize before or after mapping.
    """
    return np.array([-delta[2], delta[0], -delta[1]])


def arm_extension(shoulder: Landmark, elbow: Landmark, wrist: Landmark) -> np.ndarray:
    """Where one wrist sits relative to its shoulder, as a fraction of that
    arm's own length, in robot axes. Norm runs 0 (hand at the shoulder) to
    ~1 (arm straight out); landmark noise can push it slightly past 1.

    Normalizing by the subject's *measured* arm length — upper arm plus
    forearm, from the landmarks themselves — rather than a fixed scale factor
    makes this invariant to body size and to how far the subject stands from
    the camera. Both of those otherwise show up directly as a change in how far
    the robot reaches.

    Only the shoulder->wrist displacement is used, never absolute landmark
    positions, so the target tracks arm motion regardless of where the subject
    stands in frame.

    Deliberately unitless and unclamped: what a given extension means in meters,
    and which extensions the robot can actually achieve, are properties of the
    robot's workspace and belong to ArmGeometry in retargeter.py.
    """
    upper_arm = elbow.as_array() - shoulder.as_array()
    forearm = wrist.as_array() - elbow.as_array()
    arm_length = np.linalg.norm(upper_arm) + np.linalg.norm(forearm)
    if arm_length < MIN_ARM_LENGTH:
        return np.zeros(3)
    return to_robot_axes(wrist.as_array() - shoulder.as_array()) / arm_length
