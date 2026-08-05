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
 * One detected hand, carrying both of MediaPipe's coordinate spaces.
 *
 * Both are needed and they are not interchangeable. `landmarks` are normalized
 * to the image, which is the only space that can be compared against the pose
 * model's wrists for side assignment. `worldLandmarks` are metric and
 * isotropic, which is the only space in which the *directions* between two
 * landmarks are meaningful — normalized x and y are divided by frame width and
 * height respectively, so on a non-square frame they carry an aspect-ratio
 * stretch that bends every direction derived from them.
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
 * This exists for one reason: SEW-Mimic's third stage needs a hand orientation,
 * and MediaPipe Pose cannot supply one — its wrist landmark is a point, and a
 * point has no rotation. Without it the robot's joints 5/6/7 have no target at
 * all (see NEXT_STEPS §7.4 #2, phase B).
 *
 * Running a second model per frame is not free, so it is optional: construct
 * with `enabled: false` (or pass `?hands=0` in the page URL) and the pipeline
 * falls back to pose only.
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

  /** Detect on the current video frame. `timestamp` must match the one handed
   * to the pose model, and must increase monotonically or MediaPipe throws. */
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
 * Deliberately NOT using MediaPipe's `handedness` output. That field is
 * documented as assuming a mirrored (selfie-view) image, and the frames we feed
 * it are unmirrored — the CSS `scaleX(-1)` in index.html is display-only. So its
 * "Left" is our right, which is exactly the kind of double negative that works
 * on one machine and silently inverts on another. Nearest-wrist needs no
 * convention: the pose model already told us where each of the subject's wrists
 * is, in the same coordinate frame.
 *
 * `maxDistance` is in normalized image units; a hand further than that from
 * either wrist is a bystander's, and is dropped rather than forced onto a side.
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
      // Image-plane distance only, and against the *normalized* wrists: this is
      // the one comparison that has to stay in image space, since the pose
      // model's world landmarks are hip-centred and the hand model's are
      // hand-centred — two different origins, so a distance between them would
      // be meaningless. Dropping z for the same reason.
      const dx = wrist.x - wrists[side].x
      const dy = wrist.y - wrists[side].y
      const distance = Math.hypot(dx, dy)
      if (distance < best[side]) {
        best[side] = distance
        assignment[side] = { landmarks, worldLandmarks }
      }
    }
  }

  // Both wrists claimed the same detection — one hand occluding the other, or
  // one wrist badly mislocated. Keep the closer match and drop the other rather
  // than sending the same hand as both.
  if (assignment.left && assignment.left.landmarks === assignment.right?.landmarks) {
    delete assignment[best.left <= best.right ? 'right' : 'left']
  }
  return assignment
}
