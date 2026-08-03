import {
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision'

export type PoseCallback = (result: PoseLandmarkerResult) => void

/** Runs MediaPipe PoseLandmarker in-browser (WASM/GPU) against a live webcam stream. */
export class PoseTracker {
  private landmarker: PoseLandmarker | null = null
  private video: HTMLVideoElement
  private running = false

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  async init(): Promise<void> {
    // Served from public/, copied out of node_modules by `npm run sync-wasm`. The WASM
    // build must match the installed @mediapipe/tasks-vision exactly — pulling it from
    // a CDN's `@latest` drifts and blows up with `ASM_CONSTS[code] is not a function`.
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
    this.landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    })

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    })
    this.video.srcObject = stream
    await this.video.play()
  }

  start(onResult: PoseCallback): void {
    if (!this.landmarker) throw new Error('PoseTracker.init() must resolve first')
    this.running = true
    const loop = () => {
      if (!this.running) return
      const result = this.landmarker!.detectForVideo(this.video, performance.now())
      onResult(result)
      requestAnimationFrame(loop)
    }
    requestAnimationFrame(loop)
  }

  stop(): void {
    this.running = false
  }
}
