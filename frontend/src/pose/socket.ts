import type { NormalizedLandmark, PoseLandmarkerResult } from '@mediapipe/tasks-vision'

// MediaPipe Pose landmark indices for the joints retargeting cares about.
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const LANDMARK_INDEX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
} as const

function toPayload(lm: NormalizedLandmark) {
  return { x: lm.x, y: lm.y, z: lm.z }
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
  send(result: PoseLandmarkerResult): boolean {
    if (!this.open || this.ws.readyState !== WebSocket.OPEN) return false
    const landmarks = result.landmarks[0]
    if (!landmarks) return false

    this.ws.send(
      JSON.stringify({
        left_shoulder: toPayload(landmarks[LANDMARK_INDEX.leftShoulder]),
        left_elbow: toPayload(landmarks[LANDMARK_INDEX.leftElbow]),
        left_wrist: toPayload(landmarks[LANDMARK_INDEX.leftWrist]),
        right_shoulder: toPayload(landmarks[LANDMARK_INDEX.rightShoulder]),
        right_elbow: toPayload(landmarks[LANDMARK_INDEX.rightElbow]),
        right_wrist: toPayload(landmarks[LANDMARK_INDEX.rightWrist]),
      }),
    )
    return true
  }

  close(): void {
    this.ws.close()
  }
}
