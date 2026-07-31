import { DrawingUtils, PoseLandmarker, type PoseLandmarkerResult } from '@mediapipe/tasks-vision'

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

  draw(result: PoseLandmarkerResult): void {
    this.ctx.save()
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    for (const landmarks of result.landmarks) {
      this.drawing.drawLandmarks(landmarks, { radius: 3 })
      this.drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS)
    }
    this.ctx.restore()
  }
}
