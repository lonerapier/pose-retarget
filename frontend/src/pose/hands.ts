import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type Landmark,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'

// MediaPipe hand landmark indices.
// https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
export const HAND_INDEX = {
  wrist: 0,
  thumbTip: 4,
  indexMcp: 5,
  indexTip: 8,
  middleMcp: 9,
  pinkyMcp: 17,
} as const

/**
 * Detected hand, in both of MediaPipe's coordinate spaces.
 *
 * Both are needed and they aren't interchangeable. `landmarks` are normalized to
 * the image, the only space comparable against the pose model's wrists for side
 * assignment. `worldLandmarks` are metric and isotropic, the only space where
 * *directions* between landmarks mean anything: normalized x and y are divided
 * by frame width and height, so a non-square frame carries an aspect-ratio
 * stretch that bends every direction taken from them.
 */
export type HandFrame = {
  landmarks: NormalizedLandmark[]
  worldLandmarks: Landmark[]
}

/** The detected hands, already assigned to a side of the *subject's* body. */
export type HandAssignment = {
  left?: HandFrame
  right?: HandFrame
}

/**
 * Runs MediaPipe HandLandmarker alongside the pose model.
 *
 * SEW-Mimic's third stage needs a hand orientation, and MediaPipe Pose can't 
 * supply one - its wrist landmark is a point, and a point has no rotation. 
 * Without it the robot's joints 5-7 have no target at all.
 *
 * A second model per frame isn't free, so it's optional: `?hands=0` in the page
 * URL falls back to pose only.
 */
export class HandTracker {
  private landmarker: HandLandmarker | null = null
  private video: HTMLVideoElement
  private lastResult: HandLandmarkerResult | null = null

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  async init(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
    this.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    })
  }

  get ready(): boolean {
    return this.landmarker !== null
  }

  /** Detect on the current video frame. `timestamp` must match the one given to
   * the pose model, and must increase monotonically or MediaPipe throws. */
  detect(timestamp: number): HandLandmarkerResult | null {
    if (!this.landmarker) return null
    this.lastResult = this.landmarker.detectForVideo(this.video, timestamp)
    return this.lastResult
  }

  get result(): HandLandmarkerResult | null {
    return this.lastResult
  }
}

/**
 * Assign detected hands to the subject's left/right by proximity to the pose
 * model's own wrist landmarks.
 *
 * Deliberately not using MediaPipe's `handedness` output. It's documented as
 * assuming a mirrored selfie-view image, and the frames we feed it are
 * unmirrored - the CSS `scaleX(-1)` in index.html is display-only, so its
 * "Left" is our right. Nearest-wrist needs no convention - the pose model
 * already said where each wrist is, in the same frame.
 *
 * `maxDistance` is in normalized image units; a hand further than that from
 * either wrist belongs to a bystander and is dropped rather than forced onto a
 * side.
 */
export function assignHands(
  hands: HandLandmarkerResult | null,
  wrists: { left: NormalizedLandmark; right: NormalizedLandmark },
  maxDistance = 0.15,
): HandAssignment {
  const assignment: HandAssignment = {}
  if (!hands) return assignment

  const best: Record<'left' | 'right', number> = {
    left: maxDistance,
    right: maxDistance,
  }

  for (const [i, landmarks] of hands.landmarks.entries()) {
    const wrist = landmarks[HAND_INDEX.wrist]
    const worldLandmarks = hands.worldLandmarks[i]
    if (!wrist || !worldLandmarks) continue
    for (const side of ['left', 'right'] as const) {
      // Image-plane distance against the normalized wrists. This is the one
      // comparison that has to stay in image space: the pose model's world
      // landmarks are hip-centred and the hand model's are hand-centred, so a
      // distance between them would be meaningless. z is dropped for the same
      // reason.
      const dx = wrist.x - wrists[side].x
      const dy = wrist.y - wrists[side].y
      const distance = Math.hypot(dx, dy)
      if (distance < best[side]) {
        best[side] = distance
        assignment[side] = { landmarks, worldLandmarks }
      }
    }
  }

  // Both wrists claimed the same detection: one hand occluding the other, or a
  // badly mislocated wrist. Keep the closer match rather than sending the same
  // hand as both.
  if (assignment.left && assignment.left.landmarks === assignment.right?.landmarks) {
    delete assignment[best.left <= best.right ? 'right' : 'left']
  }
  return assignment
}
