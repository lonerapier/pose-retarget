import type { Landmark, PoseLandmarkerResult } from '@mediapipe/tasks-vision'
import { HAND_INDEX, type HandAssignment } from './hands'

// MediaPipe Pose landmark indices for the joints retargeting cares about.
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const LANDMARK_INDEX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const

function toPayload(lm: Landmark) {
  return { x: lm.x, y: lm.y, z: lm.z }
}

/** The six landmarks that fix a hand frame, out of MediaPipe's 21. */
function handPayload(landmarks: Landmark[] | undefined) {
  if (!landmarks) return null
  return {
    wrist: toPayload(landmarks[HAND_INDEX.wrist]),
    index_mcp: toPayload(landmarks[HAND_INDEX.indexMcp]),
    middle_mcp: toPayload(landmarks[HAND_INDEX.middleMcp]),
    pinky_mcp: toPayload(landmarks[HAND_INDEX.pinkyMcp]),
    thumb_tip: toPayload(landmarks[HAND_INDEX.thumbTip]),
    index_tip: toPayload(landmarks[HAND_INDEX.indexTip]),
  }
}

/**
 * Streams the arm-relevant landmarks from a PoseLandmarkerResult to the backend
 * over a WebSocket.
 *
 * Everything sent comes from MediaPipe's **world** landmarks: origin at
 * the hip midpoint, never the normalized image ones. Retargeting consumes limb
 * *directions*, and normalized landmarks can't express one:
 *
 * - `x` is divided by frame width and `y` by frame height, so on this 4:3
 *   capture a vertical displacement is inflated 1.33x against an identical
 *   horizontal one, tilting every direction toward vertical.
 * - `z` is documented only as "roughly the same scale as x", with no metric
 *   meaning. Measured live it put the forearm at 1.8x the upper arm, which is
 *   anatomically impossible, and since the elbow angle is exactly the angle
 *   between the two limb vectors the error lands straight on it: one frame
 *   solved for a 70 degree elbow against a real one near 130.
 *
 * World landmarks are isotropic and in one unit, which removes both. Their z is
 * still a monocular estimate and still the weakest axis, just no longer on an
 * arbitrary scale. Axis directions are the same in both spaces (x right, y down,
 * z negative toward the camera), so `mapper.to_robot_axes` needs no change.
 */
export class PoseSocket {
  private ws: WebSocket
  private open = false

  constructor(url: string) {
    this.ws = new WebSocket(url)
    this.ws.onopen = () => {
      this.open = true
      console.log('pose socket: connected')
    }
    this.ws.onclose = () => {
      this.open = false
      console.log('pose socket: disconnected')
    }
    this.ws.onerror = (err) => console.error('pose socket: error', err)
    this.ws.onmessage = (event) => console.debug('pose socket: ack', event.data)
  }

  /** Sends one frame's landmarks if a person is detected and the socket is
   * ready. Returns whether it sent. */
  send(result: PoseLandmarkerResult, hands: HandAssignment = {}): boolean {
    if (!this.open || this.ws.readyState !== WebSocket.OPEN) return false
    const landmarks = result.worldLandmarks[0]
    if (!landmarks) return false

    this.ws.send(
      JSON.stringify({
        // Unix epoch seconds, to match Python's time.time(). performance.now()
        // alone is monotonic but has an arbitrary origin, so the backend
        // couldn't difference it against its own clock to get a latency.
        t_capture: (performance.timeOrigin + performance.now()) / 1000,
        left_shoulder: toPayload(landmarks[LANDMARK_INDEX.leftShoulder]),
        left_elbow: toPayload(landmarks[LANDMARK_INDEX.leftElbow]),
        left_wrist: toPayload(landmarks[LANDMARK_INDEX.leftWrist]),
        right_shoulder: toPayload(landmarks[LANDMARK_INDEX.rightShoulder]),
        right_elbow: toPayload(landmarks[LANDMARK_INDEX.rightElbow]),
        right_wrist: toPayload(landmarks[LANDMARK_INDEX.rightWrist]),
        // Hips build the torso frame. Their midpoint is the world-landmark
        // origin by construction, so these carry the pelvis *roll* rather than
        // a position — which is what the spine axis needs anyway.
        left_hip: toPayload(landmarks[LANDMARK_INDEX.leftHip]),
        right_hip: toPayload(landmarks[LANDMARK_INDEX.rightHip]),
        // Hand world landmarks are centred on the hand, not the hips, so they
        // share no origin with the six above. Only differences within one hand
        // are used downstream, which is what keeps that from mattering.
        left_hand: handPayload(hands.left?.worldLandmarks),
        right_hand: handPayload(hands.right?.worldLandmarks),
      }),
    )
    return true
  }

  close(): void {
    this.ws.close()
  }
}
