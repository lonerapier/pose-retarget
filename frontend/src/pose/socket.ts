import type { NormalizedLandmark, PoseLandmarkerResult } from '@mediapipe/tasks-vision'
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

function toPayload(lm: NormalizedLandmark) {
  return { x: lm.x, y: lm.y, z: lm.z }
}

/** The six landmarks that fix a hand frame, out of MediaPipe's 21. */
function handPayload(landmarks: NormalizedLandmark[] | undefined) {
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

/** Streams the arm-relevant landmarks from a PoseLandmarkerResult to the backend over a WebSocket. */
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

  /** Sends one frame's landmarks if a person is detected and the socket is ready. Returns whether it sent. */
  send(result: PoseLandmarkerResult, hands: HandAssignment = {}): boolean {
    if (!this.open || this.ws.readyState !== WebSocket.OPEN) return false
    const landmarks = result.landmarks[0]
    if (!landmarks) return false

    this.ws.send(
      JSON.stringify({
        // Unix epoch seconds, to match Python's time.time(). performance.now()
        // alone is monotonic but has an arbitrary origin, so the backend could
        // not difference it against its own clock to get a latency.
        t_capture: (performance.timeOrigin + performance.now()) / 1000,
        left_shoulder: toPayload(landmarks[LANDMARK_INDEX.leftShoulder]),
        left_elbow: toPayload(landmarks[LANDMARK_INDEX.leftElbow]),
        left_wrist: toPayload(landmarks[LANDMARK_INDEX.leftWrist]),
        right_shoulder: toPayload(landmarks[LANDMARK_INDEX.rightShoulder]),
        right_elbow: toPayload(landmarks[LANDMARK_INDEX.rightElbow]),
        right_wrist: toPayload(landmarks[LANDMARK_INDEX.rightWrist]),
        // Hips and hands are not retargeted yet — the torso frame (§6.2) and
        // SEW stage 3 (§7.4) need them. They are sent now so that recorded
        // battery clips do not have to be re-shot when those land.
        left_hip: toPayload(landmarks[LANDMARK_INDEX.leftHip]),
        right_hip: toPayload(landmarks[LANDMARK_INDEX.rightHip]),
        left_hand: handPayload(hands.left),
        right_hand: handPayload(hands.right),
      }),
    )
    return true
  }

  close(): void {
    this.ws.close()
  }
}
