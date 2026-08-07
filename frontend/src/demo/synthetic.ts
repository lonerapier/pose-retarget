import type { PoseLandmarkerResult } from '@mediapipe/tasks-vision'

/**
 * A moving stick figure, in the same world-landmark space MediaPipe produces.
 *
 * Two uses, both about not needing to be in front of the camera. Framing and
 * recording settings can be checked on a pose that repeats exactly, instead of
 * one that depends on standing still. And because these go down the same
 * websocket as real landmarks, it drives the sim end to end — retarget, solver,
 * physics — with no webcam involved at all, which is what makes the demo page
 * checkable on a machine with no camera permission.
 *
 * Landmark axes, as everywhere: x toward the image's right edge (so the
 * subject's *left* side, the frames being unmirrored), y down, z away from the
 * camera. Metres, origin at the hip midpoint.
 */

type Vec3 = { x: number; y: number; z: number; visibility: number }

const UPPER_ARM = 0.28
const FOREARM = 0.25
const SHOULDER_DROP = -0.45
const SHOULDER_HALF = 0.18
const HIP_HALF = 0.09

function at(x: number, y: number, z: number): Vec3 {
  return { x, y, z, visibility: 1 }
}

function unit(v: [number, number, number]): [number, number, number] {
  const n = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / n, v[1] / n, v[2] / n]
}

function cross(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

/**
 * One arm: shoulder, elbow, wrist.
 *
 * `spread` is the angle away from straight down and `twist` swings that out of
 * the coronal plane, which is what puts motion on the z axis — the axis a single
 * webcam estimates worst, and the one the 3D pane exists to show.
 */
function arm(
  shoulder: Vec3,
  side: 1 | -1,
  spread: number,
  twist: number,
  bend: number,
): [Vec3, Vec3] {
  const upper = unit([
    side * Math.sin(spread) * Math.cos(twist),
    Math.cos(spread),
    -Math.sin(spread) * Math.sin(twist),
  ])
  // Bend the forearm away from the upper arm in a fixed plane, so the elbow
  // behaves like an elbow instead of the forearm folding through the torso.
  const hinge = unit(cross(upper, [side, 0, 0]))
  const forearm = unit([
    upper[0] * Math.cos(bend) + hinge[0] * Math.sin(bend),
    upper[1] * Math.cos(bend) + hinge[1] * Math.sin(bend),
    upper[2] * Math.cos(bend) + hinge[2] * Math.sin(bend),
  ])

  const elbow = at(
    shoulder.x + upper[0] * UPPER_ARM,
    shoulder.y + upper[1] * UPPER_ARM,
    shoulder.z + upper[2] * UPPER_ARM,
  )
  const wrist = at(
    elbow.x + forearm[0] * FOREARM,
    elbow.y + forearm[1] * FOREARM,
    elbow.z + forearm[2] * FOREARM,
  )
  return [elbow, wrist]
}

/**
 * The 33 pose landmarks at time `t` seconds, wrapped so it can go anywhere a
 * real result goes.
 *
 * Only `worldLandmarks` is filled: that is the whole of what the demo page and
 * the websocket read. Anything reaching for the normalized `landmarks` — hand
 * assignment, the 2D overlay — would need a camera anyway, and gets an empty
 * list rather than plausible-looking numbers that aren't.
 */
export function syntheticPose(t: number): PoseLandmarkerResult {
  const wave = (phase: number) => 0.5 + 0.5 * Math.sin(t * 0.8 + phase)

  const leftShoulder = at(SHOULDER_HALF, SHOULDER_DROP, 0)
  const rightShoulder = at(-SHOULDER_HALF, SHOULDER_DROP, 0)
  // The two arms run out of phase so the pose is never symmetric — a symmetric
  // one hides a left/right mix-up completely.
  const [leftElbow, leftWrist] = arm(
    leftShoulder,
    1,
    0.25 + 1.3 * wave(0),
    0.9 * Math.sin(t * 0.53),
    0.35 + 0.9 * wave(1.9),
  )
  const [rightElbow, rightWrist] = arm(
    rightShoulder,
    -1,
    0.25 + 1.3 * wave(2.4),
    0.9 * Math.sin(t * 0.53 + 1.1),
    0.35 + 0.9 * wave(0.6),
  )

  const landmarks: Vec3[] = new Array(33).fill(null).map(() => at(0, 0, 0))
  landmarks[0] = at(0, SHOULDER_DROP - 0.17, -0.05) // nose
  landmarks[11] = leftShoulder
  landmarks[12] = rightShoulder
  landmarks[13] = leftElbow
  landmarks[14] = rightElbow
  landmarks[15] = leftWrist
  landmarks[16] = rightWrist
  landmarks[23] = at(HIP_HALF, 0, 0)
  landmarks[24] = at(-HIP_HALF, 0, 0)
  landmarks[25] = at(HIP_HALF + 0.01, 0.45, 0.01)
  landmarks[26] = at(-HIP_HALF - 0.01, 0.45, 0.01)
  landmarks[27] = at(HIP_HALF + 0.02, 0.9, 0)
  landmarks[28] = at(-HIP_HALF - 0.02, 0.9, 0)
  landmarks[29] = at(HIP_HALF + 0.02, 0.94, 0.04)
  landmarks[30] = at(-HIP_HALF - 0.02, 0.94, 0.04)
  landmarks[31] = at(HIP_HALF + 0.02, 0.96, -0.1)
  landmarks[32] = at(-HIP_HALF - 0.02, 0.96, -0.1)

  return {
    landmarks: [],
    worldLandmarks: [landmarks],
    segmentationMasks: undefined,
    close: () => {},
  } as unknown as PoseLandmarkerResult
}
