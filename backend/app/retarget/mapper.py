"""Landmark schema and mapping from MediaPipe Landmarks to magnitude and directions
needed for [`retargeting`](retargeter.py).

Converts landmarks in camera frame to body-centric robot axes (torso frame).
Arms are represented using Upper Arm and Forearm unit vectors. Hands are represented
using finger directions.
"""

from typing import Optional

import numpy as np
from pydantic import BaseModel

# Threshold (in metres) below which the shoulder/elbow/wrist triple has collapsed
# and arm has no direction.
MIN_ARM_LENGTH = 1e-3

# Threshold (in metres) below which the hand landmarks are too collapsed for the
# knuckle line to carry a direction.
MIN_HAND_LENGTH = 1e-4

# Threshold (in metres) below which the shoulder line or the spine has collapsed
# and the torso frame is ill-defined.
MIN_TORSO_LENGTH = 1e-3

# Threshold for how square the shoulder line and spine have to be before the
# chest normal built from them is trusted. This is |sin| of the angle between them,
# so 0.1 is about 6 degrees.
MIN_TORSO_SQUARENESS = 0.1


class Landmark(BaseModel):
    """One MediaPipe **world** landmark: metres, right-handed, origin at the
    midpoint of the hips.

    Axes are the RAW camera frame, not the mirrored image on screen, as that
    mirror is CSS-only for display purposes:

    - `x`: toward the image's right edge
    - `y`: downward
    - `z`: away from the camera (more negative = closer to it)
    """

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def _fmt(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class HandLandmarks(BaseModel):
    """The subset of MediaPipe's 21 hand landmarks that fixes a hand frame.

    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

    Same axes and units as Landmark, but MediaPipe centres the hand model's
    world landmarks on the *hand* while the body's sit on the hips. They share
    no origin, so a vector from a body landmark to a hand landmark is
    meaningless.
    """

    wrist: Landmark  # 0
    index_mcp: Landmark  # 5
    middle_mcp: Landmark  # 9
    pinky_mcp: Landmark  # 17
    thumb_tip: Landmark  # 4
    index_tip: Landmark  # 8


class PoseFrame(BaseModel):
    """One frame of the landmarks relevant to bimanual retargeting.

    Everything past the six arm landmarks is optional so a client that only runs
    the pose model still parses. The hips build the torso frame and are used;
    without them `torso_frame` falls back to the camera frame.
    """

    left_shoulder: Landmark
    left_elbow: Landmark
    left_wrist: Landmark
    right_shoulder: Landmark
    right_elbow: Landmark
    right_wrist: Landmark

    left_hip: Optional[Landmark] = None
    right_hip: Optional[Landmark] = None

    left_hand: Optional[HandLandmarks] = None
    right_hand: Optional[HandLandmarks] = None

    # Unix epoch seconds, stamped in the browser when the frame was handed to MediaPipe.
    t_capture: Optional[float] = None

    @property
    def hips(self) -> Optional[tuple[Landmark, Landmark]]:
        """Both hips, or None if either is missing."""
        if self.left_hip is None or self.right_hip is None:
            return None
        return self.left_hip, self.right_hip


def to_robot_axes(delta: np.ndarray) -> np.ndarray:
    """Permute a landmark-space displacement into robot axes.

    The result is still *camera*-frame: it only matches a body-centric direction
    when the subject is square to the camera and upright. `torso_frame` removes
    that assumption, but this stays the common first step, since the torso frame
    is itself built from vectors that came through here.

    - forward (robot +x): wrist closer to the camera than the shoulder
      (z more negative) -> reach further forward.
    - left (robot +y): landmarks are raw and unmirrored, like a photo rather
      than a mirror, so the subject's left hand moving left increases image-x.
    - up (robot +z): image y increases downward, so flip the sign.
    """
    return np.array([-delta[2], delta[0], -delta[1]])


def _midpoint(a: Landmark, b: Landmark) -> np.ndarray:
    return 0.5 * (a.as_array() + b.as_array())


def torso_frame(frame: PoseFrame) -> np.ndarray | None:
    """The subject's body-centric frame, as a rotation matrix in robot axes.

    Columns are the torso's own axes in robot axes, so `R.T @ v` takes a
    camera-frame vector into the torso frame, which is where SEW-Mimic's inputs
    are defined, otherwise `u` and `l` are camera vectors.

    Returns None when the hips are missing or the torso is too collapsed to
    define a frame. Fall back to the camera frame rather than treating None as
    identity: they mean different things.
    """
    hips = frame.hips
    if hips is None:
        return None
    left_hip, right_hip = hips

    # across the chest, right shoulder -> left (robot +y is left)
    y_t = to_robot_axes(
        frame.left_shoulder.as_array() - frame.right_shoulder.as_array()
    )
    # up the spine, hip midpoint -> shoulder midpoint
    z_t = to_robot_axes(
        _midpoint(frame.left_shoulder, frame.right_shoulder)
        - _midpoint(left_hip, right_hip)
    )
    across, spine = float(np.linalg.norm(y_t)), float(np.linalg.norm(z_t))
    if across < MIN_TORSO_LENGTH or spine < MIN_TORSO_LENGTH:
        return None
    y_t, z_t = y_t / across, z_t / spine

    # out of the chest, `y_t x z_t`
    x_t = np.cross(y_t, z_t)
    # |y x z| = sin(angle between them). Near zero, the spine has folded onto
    # the shoulder line and the chest normal is arbitrary.
    squareness = float(np.linalg.norm(x_t))
    if squareness < MIN_TORSO_SQUARENESS:
        return None
    x_t = x_t / squareness

    return np.column_stack([x_t, y_t, np.cross(x_t, y_t)])


def arm_extension(shoulder: Landmark, elbow: Landmark, wrist: Landmark) -> np.ndarray:
    """Where one wrist sits relative to its shoulder, as a fraction of that
    arm's own length, in robot axes. Norm runs 0 (hand at the shoulder) to ~1
    (arm straight out); noise can push it slightly past 1.

    Normalizing by the subject's *measured* arm length (upper arm and forearm),
    from the landmarks themselves, rather than a fixed scale makes this
    invariant to body size and to how far the subject stands from the camera.
    Both otherwise show up directly as a change in how far the robot reaches.

    Only the shoulder->wrist displacement is used, never absolute positions, so
    the target tracks arm motion wherever the subject stands in frame.
    """
    upper_arm = elbow.as_array() - shoulder.as_array()
    forearm = wrist.as_array() - elbow.as_array()
    arm_length = np.linalg.norm(upper_arm) + np.linalg.norm(forearm)
    if arm_length < MIN_ARM_LENGTH:
        return np.zeros(3)
    return to_robot_axes(wrist.as_array() - shoulder.as_array()) / arm_length


def limb_directions(
    shoulder: Landmark, elbow: Landmark, wrist: Landmark
) -> tuple[np.ndarray, np.ndarray] | None:
    """The subject's two limb *directions*: upper arm and forearm, as unit
    vectors in robot axes.

    Returns None when either limb is too short to carry a direction. Hold the
    previous pose rather than steer the arm from a meaningless vector.
    """
    upper_arm = to_robot_axes(elbow.as_array() - shoulder.as_array())
    forearm = to_robot_axes(wrist.as_array() - elbow.as_array())
    upper_len = float(np.linalg.norm(upper_arm))
    fore_len = float(np.linalg.norm(forearm))
    if upper_len < MIN_ARM_LENGTH or fore_len < MIN_ARM_LENGTH:
        return None
    return upper_arm / upper_len, forearm / fore_len


def hand_orientation(hand: HandLandmarks, side: str) -> np.ndarray | None:
    """The subject's hand orientation as a rotation matrix in robot axes.

    Two directions fix the frame:
    - f = wrist -> middle_mcp. along the hand, toward the fingers
    - k = pinky_mcp -> index_mcp. across the knuckles, thumb-ward

    Only `f` is trusted for magnitude. `k` enters through a cross product only,
    so the knuckle line's length, the noisier of the two, since the four MCPs
    are nearly coincident at low resolution, never scales anything.

    `side` fixes the orientation of the hand-frame.

    Result (in robot axes):
    - `x`: back-of-hand normal (dorsal), for BOTH hands
    - `y`: Z x X, the knuckle line with f projected out
    - `z`: -f, from the fingers back toward the wrist

    `z = -f` rather than `+f` to match the robot's end-effector convention: the
    gripper approaches along its -z.

    Returns None when the hand is too collapsed or too flat to fix a frame,
    rather than a near-arbitrary rotation out of a degenerate cross product.
    """
    f = to_robot_axes(hand.middle_mcp.as_array() - hand.wrist.as_array())
    k = to_robot_axes(hand.index_mcp.as_array() - hand.pinky_mcp.as_array())
    f_len, k_len = float(np.linalg.norm(f)), float(np.linalg.norm(k))
    if f_len < MIN_HAND_LENGTH or k_len < MIN_HAND_LENGTH:
        return None
    f, k = f / f_len, k / k_len

    # Human hands are true mirror images, so `cross(f, k)` points dorsally on one
    # and palmward on the other; flipping by side is what makes both right-handed
    # rotations with the same anatomical meaning, exactly as the robot's two
    # mirrored arms each still carry a right-handed ee frame. The y column therefore
    # runs pinky->index on the right and index->pinky on the left.
    sign = 1.0 if side == "right" else -1.0
    dorsal = sign * np.cross(f, k)
    # |f x k| = sin(angle between them).
    dorsal_len = float(np.linalg.norm(dorsal))
    if dorsal_len < MIN_HAND_LENGTH:
        return None

    x = dorsal / dorsal_len  # already exactly perpendicular to f
    z = -f
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
