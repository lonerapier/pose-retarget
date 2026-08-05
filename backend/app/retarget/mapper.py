"""Landmark schema + math for retargeting: turns MediaPipe body landmarks
into a unitless arm-extension vector in the robot's axes, and MediaPipe hand
landmarks into a hand orientation matrix in the same axes.

The human side of retargeting ends here, and it deliberately produces no
meters: converting to the robot's workspace needs the robot's own shoulder
position and reach, which live in retargeter.py.
"""

import numpy as np
from pydantic import BaseModel

# Below this (in metres) the shoulder/elbow/wrist triple is degenerate,
# landmarks collapsed onto each other, or the subject isn't really detected
# and the direction of the arm is meaningless.
MIN_ARM_LENGTH = 1e-3

# Same idea one scale down: below this the hand landmarks are too collapsed
# (hand far from the camera, or a bad detection) for a knuckle line to carry a
# meaningful direction.
MIN_HAND_LENGTH = 1e-4

# Below this the shoulder line or the spine has collapsed and the torso frame
# it would define is arbitrary. Same units and the same reasoning as
# MIN_ARM_LENGTH; a real torso is a third of a metre across either way.
MIN_TORSO_LENGTH = 1e-3

# How square the shoulder line and the spine have to be before the chest normal
# built from them is trusted. This is |sin| of the angle between the two, so
# 0.1 is about 6 degrees — far below anything anatomy produces, and reached only
# when a landmark is badly wrong.
MIN_TORSO_SQUARENESS = 0.1


class Landmark(BaseModel):
    """One MediaPipe **world** landmark: metres, right-handed, with the origin
    at the midpoint of the hips.

    Axis directions are the same in both spaces and are the RAW camera frame,
    not the mirrored image shown on screen — the mirror is CSS-only, for
    display, and isn't reflected here:

        x  toward the image's right edge
        y  downward
        z  away from the camera (more negative = closer to it)
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

    Same axes and units as Landmark. These are the hand model's own world
    landmarks, but MediaPipe centres them on the *hand*, while the body's are
    centred on the hips. The two therefore share no origin, and a vector from a
    body landmark to a hand landmark would be meaningless. Only differences
    taken within hand landmarks are useful.

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
    before a field existed, or a client that only runs the pose model, still
    parses. `left_*`/`right_*` are always the *subject's* sides, which map
    directly onto the robot's own left/right arms.

    `left_hip`/`right_hip` build the torso frame and are consumed. They stay optional
    so a frame from a client that only sends the six arm landmarks still parses, and
    `torso_frame` falls back to the camera frame.

    Hands are sent and logged but not yet consumed: SEW-Mimic stage 3 (§7.4)
    needs them for wrist orientation, and joints 5-7 have no target until it
    lands.
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

    @property
    def hips(self) -> tuple[Landmark, Landmark] | None:
        """Both hips, or None if either is missing."""
        if self.left_hip is None or self.right_hip is None:
            return None
        return self.left_hip, self.right_hip


def to_robot_axes(delta: np.ndarray) -> np.ndarray:
    """Permute a landmark-space displacement into robot axes.

    This is a *camera*-frame result: it maps the camera's axes onto the robot's,
    and is only the same thing as a body-centric direction when the subject
    happens to be square to the camera and upright. `torso_frame` is what
    removes that assumption; this stays as the common first step because the
    torso frame is itself built out of vectors that have been through here.

    Axis mapping:
    - forward (robot +x): wrist closer to the camera than the shoulder
      (z more negative) -> reach further forward.
    - left (robot +y): landmarks are in the raw, unmirrored camera frame,
      like a photo, not a mirror, so the subject's own left hand moving
      further left increases its image-x.
    - up (robot +z): image y increases downward, so we flip the sign.
    """
    return np.array([-delta[2], delta[0], -delta[1]])


def _midpoint(a: Landmark, b: Landmark) -> np.ndarray:
    return 0.5 * (a.as_array() + b.as_array())


def torso_frame(frame: PoseFrame) -> np.ndarray | None:
    """The subject's body-centric frame, as a rotation matrix in robot axes.

    Columns are the torso's own axes expressed in robot axes, so `R.T @ v`
    takes a camera-frame vector into the torso frame — which is the frame
    SEW-Mimic's inputs are defined in. Without it, `u` and `l` are camera
    vectors.

    y_t: across the chest, right shoulder -> left     (robot +y is left)
    z_t: up the spine, hip midpoint -> shoulder midpoint
    x_t: out of the chest, y_t x z_t

    Built in that order because the two measured directions are not exactly
    perpendicular — a real spine is not square to a real shoulder line, and the
    landmarks are noisy besides. The cross products resolve that in favour of
    the shoulder line, which is measured across the widest, best-localized pair
    of landmarks on the upper body; the spine only has to fix the roll about it.

    Facing the camera square and upright this returns approximately the identity,
    which is the sanity check to run first if the arms start moving the wrong way.

    Returns None when the hips are absent (a client sending only the six arm
    landmarks), or when the torso is too collapsed to define a frame. Callers
    should fall back to the camera frame rather than treat None as identity
    silently — it is the difference between "no torso correction available" and
    "the torso is upright".
    """
    hips = frame.hips
    if hips is None:
        return None
    left_hip, right_hip = hips

    y_t = to_robot_axes(
        frame.left_shoulder.as_array() - frame.right_shoulder.as_array()
    )
    z_t = to_robot_axes(
        _midpoint(frame.left_shoulder, frame.right_shoulder)
        - _midpoint(left_hip, right_hip)
    )
    across, spine = float(np.linalg.norm(y_t)), float(np.linalg.norm(z_t))
    if across < MIN_TORSO_LENGTH or spine < MIN_TORSO_LENGTH:
        return None
    y_t, z_t = y_t / across, z_t / spine

    x_t = np.cross(y_t, z_t)
    # |y x z| = sin(angle between them); near zero means the spine has folded
    # onto the shoulder line and the chest normal is arbitrary.
    squareness = float(np.linalg.norm(x_t))
    if squareness < MIN_TORSO_SQUARENESS:
        return None
    x_t = x_t / squareness

    return np.column_stack([x_t, y_t, np.cross(x_t, y_t)])


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


def limb_directions(
    shoulder: Landmark, elbow: Landmark, wrist: Landmark
) -> tuple[np.ndarray, np.ndarray] | None:
    """The subject's two limb *directions* — upper arm and forearm — as unit
    vectors in robot axes. SEW-Mimic's stage-1/2 input (NEXT_STEPS §0.2).

    This is the orientation-retargeting counterpart of arm_extension(): same
    three landmarks, but the length of the arm is thrown away instead of being
    the whole signal. That is the point — a direction is what the robot's
    shoulder and elbow joints can actually reproduce exactly, whereas a
    shoulder->wrist position has to be squeezed into a reachable shell first.
    It also demotes monocular depth error from perturbing a target position to
    perturbing a direction.

    Returns None when either limb is too short in landmark units to carry a
    direction — collapsed landmarks, or no real detection. Callers should hold
    the previous pose rather than steer the arm from a meaningless vector.
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
