import {
  DrawingUtils,
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision'

/** Draws raw MediaPipe landmarks over the webcam feed for visual sanity-checking. */
export class PoseOverlay {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private drawing: DrawingUtils

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.drawing = new DrawingUtils(this.ctx)
  }

  resize(width: number, height: number): void {
    this.canvas.width = width
    this.canvas.height = height
  }

  draw(result: PoseLandmarkerResult, hands: HandLandmarkerResult | null = null): void {
    this.ctx.save()
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    for (const landmarks of result.landmarks) {
      this.drawing.drawLandmarks(landmarks, { radius: 3 })
      this.drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS)
    }
    // Hands in a different colour: overlapping the pose skeleton's own wrist,
    // they are otherwise impossible to tell apart at a glance.
    for (const landmarks of hands?.landmarks ?? []) {
      this.drawing.drawLandmarks(landmarks, { radius: 2, color: '#ff9f43' })
      this.drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: '#ff9f43',
        lineWidth: 2,
      })
    }
    this.ctx.restore()
  }
}
