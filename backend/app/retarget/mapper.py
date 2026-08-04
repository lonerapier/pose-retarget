"""Landmark schema + math for retargeting: turns MediaPipe body landmarks
into a unitless arm-extension vector in the robot's axes, and MediaPipe hand
landmarks into a hand orientation matrix in the same axes.

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

# Same idea one scale down: below this the hand landmarks are too collapsed
# (hand far from the camera, or a bad detection) for a knuckle line to carry a
# meaningful direction.
MIN_HAND_LENGTH = 1e-4


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


class HandLandmarks(BaseModel):
    """The subset of MediaPipe's 21 hand landmarks that fixes a hand frame.

    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

    Same normalized coordinate convention as Landmark, but note MediaPipe scales
    hand-landmark z by the *hand's* own size rather than the body's, so hand and
    body landmarks are not in a common metric space. Only differences taken
    within one hand are used below, which keeps that from mattering.

    `thumb_tip` and `index_tip` are carried but not yet consumed: finger spread
    is the natural source of a gripper command (NEXT_STEPS §15 Q5), and recorded
    battery clips are expensive enough to re-shoot that it is worth capturing
    them now rather than discovering the gap later.
    """

    wrist: Landmark  # 0
    index_mcp: Landmark  # 5
    middle_mcp: Landmark  # 9
    pinky_mcp: Landmark  # 17
    thumb_tip: Landmark  # 4
    index_tip: Landmark  # 8


class PoseFrame(BaseModel):
    """One frame of the landmarks relevant to bimanual retargeting.

    Everything past the six arm landmarks is optional so that a recording made
    before a field existed — or a client that only runs the pose model — still
    parses. `left_*`/`right_*` are always the *subject's* sides, which map
    directly onto the robot's own left/right arms.

    Not yet consumed by the retargeter, but sent and recorded now:
    - hips, which §6.2 needs to build a torso frame
    - hands, which SEW-Mimic stage 3 (§7.4) needs for wrist orientation

    Both exist so the motion battery does not have to be re-recorded when those
    steps land.
    """

    left_shoulder: Landmark
    left_elbow: Landmark
    left_wrist: Landmark
    right_shoulder: Landmark
    right_elbow: Landmark
    right_wrist: Landmark

    left_hip: Landmark | None = None
    right_hip: Landmark | None = None

    left_hand: HandLandmarks | None = None
    right_hand: HandLandmarks | None = None

    # Capture time as a unix epoch float in seconds, stamped in the browser at
    # the moment the frame was handed to MediaPipe. The one number the latency
    # accounting in §6.4 cannot be reconstructed without.
    t_capture: float | None = None


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

    Its determinant is +1, so it is a rotation rather than a reflection: it
    preserves handedness as well as length. hand_orientation() depends on that
    — a reflection here would silently swap the two hands' chirality.
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


def hand_orientation(hand: HandLandmarks, side: str) -> np.ndarray | None:
    """The subject's hand orientation as a rotation matrix in robot axes.

    This is SEW-Mimic's `H` — the stage-3 input, the one thing MediaPipe Pose
    alone cannot supply and the reason joints 5/6/7 currently have no target
    (NEXT_STEPS §7.4 #2). Nothing consumes it yet; it is measured, logged and
    recorded so stage 3 has real data to land on.

    Two landmark-derived directions fix the frame:

        f = wrist -> middle_mcp      along the hand, toward the fingers
        k = pinky_mcp -> index_mcp   across the knuckles, thumb-ward

    Only `f` is trusted for magnitude; `k` enters only through a cross product,
    so the knuckle line's own length (the noisiest of the two, since the four
    MCPs are nearly coincident at low resolution) never scales anything.

    Columns of the returned matrix, in robot axes:

        x  back-of-hand normal (dorsal), for BOTH hands
        y  z x x — the knuckle line with f projected out
        z  -f, from the fingers back toward the wrist

    `z = -f` rather than `+f` so the frame lines up with the robot's own
    end-effector convention: the gripper approaches along its -z (§3), so the
    human "fingers point this way" axis is the ee frame's -z too. That makes
    stage 3's `R⁰⁷_des = H · ...` a comparison of like with like.

    The `side` sign is not a detail to skip. Human hands are true mirror images
    of each other, so `cross(f, k)` points dorsally on one hand and palmward on
    the other. Flipping it by side is what makes both matrices right-handed
    rotations with the same anatomical meaning — the same way the robot's two
    mirrored arms each still carry a right-handed ee frame. As a consequence the
    y column is pinky->index on the right hand and index->pinky on the left;
    that asymmetry is inherent to the mirroring, not a bug.

    Returns None when the hand is too collapsed or too flat to fix a frame,
    rather than a near-arbitrary rotation from a degenerate cross product.
    Callers should hold the previous orientation rather than treat None as
    identity.
    """
    f = to_robot_axes(hand.middle_mcp.as_array() - hand.wrist.as_array())
    k = to_robot_axes(hand.index_mcp.as_array() - hand.pinky_mcp.as_array())
    f_len, k_len = float(np.linalg.norm(f)), float(np.linalg.norm(k))
    if f_len < MIN_HAND_LENGTH or k_len < MIN_HAND_LENGTH:
        return None
    f, k = f / f_len, k / k_len

    sign = 1.0 if side == "right" else -1.0
    dorsal = sign * np.cross(f, k)
    dorsal_len = float(np.linalg.norm(dorsal))
    # |f x k| = sin(angle between them). Near zero means the knuckle line has
    # collapsed onto the finger direction and the roll about the hand is
    # unobservable — the hand-frame analogue of the straight-arm singularity in
    # §0.2, and handled the same way: refuse to guess.
    if dorsal_len < MIN_HAND_LENGTH:
        return None

    x = dorsal / dorsal_len  # already exactly perpendicular to f
    z = -f
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
