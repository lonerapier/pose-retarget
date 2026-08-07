/**
 * The demo view: the MuJoCo sim beside the pose that's driving it, composited
 * into one canvas and recordable straight from the page.
 *
 * `main.ts` is the operator's view — webcam with landmarks drawn on top, for
 * checking that detection is tracking. This is the view for showing someone what
 * the system does: the robot on the left, the world landmarks the retargeter
 * actually consumes on the right, and nothing of the webcam except a small inset
 * so it's clear a person is driving it.
 */
import { PoseTracker } from './pose/tracker'
import { HandTracker, assignHands } from './pose/hands'
import { PoseSocket } from './pose/socket'
import { Skeleton3D } from './pose/skeleton3d'
import { SimStream } from './demo/simstream'
import { CanvasRecorder, download } from './demo/recorder'
import { drawCaption, drawHud, drawRecordingBadge } from './demo/hud'
import { syntheticPose } from './demo/synthetic'
import type { PoseLandmarkerResult } from '@mediapipe/tasks-vision'

// Two portrait panes side by side. Fixed, and matched to the backend's render
// size, so the sim pane is drawn 1:1 and the recording comes out a standard
// 720p regardless of the window it was recorded from.
const PANE_WIDTH = 640
const PANE_HEIGHT = 720
const WIDTH = PANE_WIDTH * 2
const HEIGHT = PANE_HEIGHT
const RECORD_FPS = 30

const PIP_WIDTH = 224
const PIP_MARGIN = 18

const video = document.querySelector<HTMLVideoElement>('#video')!
const canvas = document.querySelector<HTMLCanvasElement>('#stage')!
const status = document.querySelector<HTMLSpanElement>('#status')!
const recordButton = document.querySelector<HTMLButtonElement>('#record')!
const pipToggle = document.querySelector<HTMLInputElement>('#pip')!
const frameToggle = document.querySelector<HTMLInputElement>('#torso-frame')!

canvas.width = WIDTH
canvas.height = HEIGHT
const ctx = canvas.getContext('2d')!

const params = new URLSearchParams(location.search)
const handsEnabled = params.get('hands') !== '0'
// `?synthetic=1` drives everything from a generated pose instead of the webcam:
// no camera, no models, but the same landmarks down the same socket, so the
// robot still moves. For checking framing and recording without being in shot.
const synthetic = params.get('synthetic') === '1'

const tracker = new PoseTracker(video)
const hands = new HandTracker(video)
const socket = new PoseSocket('ws://127.0.0.1:8000/ws/pose')
const sim = new SimStream('ws://127.0.0.1:8000/ws/sim')
const skeleton = new Skeleton3D()
const recorder = new CanvasRecorder(canvas, RECORD_FPS)

const LEFT_WRIST = 15
const RIGHT_WRIST = 16

// Latest detection, written by the MediaPipe callback and read by the draw
// loop. The two run at different rates — detection is throttled by the models,
// drawing by the display — so the loop repaints the last known pose rather than
// dropping to the detector's framerate.
let latest: {
  result: PoseLandmarkerResult
  hands: ReturnType<typeof assignHands>
} | null = null

// Sim frames measured here rather than taken from the backend's configured
// framerate, so the HUD shows what actually arrived.
let streamFps = 0
let lastStreamFrames = 0
let lastStreamSample = performance.now()

function sampleStreamFps(): void {
  const now = performance.now()
  const elapsed = now - lastStreamSample
  if (elapsed < 500) return
  streamFps = ((sim.frames - lastStreamFrames) * 1000) / elapsed
  lastStreamFrames = sim.frames
  lastStreamSample = now
}

function drawSimPane(): void {
  const frame = sim.frame
  if (frame) {
    ctx.drawImage(frame, 0, 0, PANE_WIDTH, PANE_HEIGHT)
    return
  }
  ctx.fillStyle = '#12181f'
  ctx.fillRect(0, 0, PANE_WIDTH, PANE_HEIGHT)
  ctx.fillStyle = '#5d7186'
  ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.textAlign = 'center'
  ctx.fillText('waiting for sim video…', PANE_WIDTH / 2, PANE_HEIGHT / 2)
  ctx.textAlign = 'left'
}

/** The webcam inset, straddling the seam between the panes. */
function drawPip(): void {
  if (!pipToggle.checked || !video.videoWidth) return
  const height = (PIP_WIDTH * video.videoHeight) / video.videoWidth
  const x = WIDTH / 2 - PIP_WIDTH / 2
  const y = HEIGHT - height - PIP_MARGIN

  ctx.save()
  ctx.beginPath()
  ctx.roundRect(x, y, PIP_WIDTH, height, 8)
  ctx.clip()
  // Mirrored, like the operator view and like the skeleton pane: the raw frame
  // is unmirrored, and an inset that moves opposite to the figure beside it
  // reads as a bug.
  ctx.translate(x + PIP_WIDTH, y)
  ctx.scale(-1, 1)
  ctx.drawImage(video, 0, 0, PIP_WIDTH, height)
  ctx.restore()

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.roundRect(x, y, PIP_WIDTH, height, 8)
  ctx.stroke()
}

function render(): void {
  sampleStreamFps()
  skeleton.showTorsoFrame = frameToggle.checked

  drawSimPane()
  skeleton.draw(
    ctx,
    PANE_WIDTH,
    0,
    PANE_WIDTH,
    PANE_HEIGHT,
    latest?.result ?? null,
    latest?.hands ?? {},
  )

  // Seam, so the two panes read as two views rather than one wide image.
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)'
  ctx.fillRect(PANE_WIDTH - 1, 0, 2, HEIGHT)

  drawHud(ctx, 16, 16, socket.state?.hud, {
    streamFps,
    connected: socket.isOpen && sim.connected,
  })
  drawCaption(ctx, 16, HEIGHT - 16, 'OpenArm v2 · MuJoCo')
  drawCaption(ctx, PANE_WIDTH + 16, HEIGHT - 16, 'MediaPipe world landmarks')
  drawPip()
  if (recorder.recording) drawRecordingBadge(ctx, WIDTH - 16, 16, recorder.elapsed)

  requestAnimationFrame(render)
}

// Drag anywhere on the skeleton pane to orbit it. Being able to swing round to
// a profile view is the quickest way to see whether the depth estimate is
// sensible, which is the axis a single webcam is weakest on.
let dragging = false
let dragX = 0
let dragY = 0
canvas.addEventListener('pointerdown', (event) => {
  const bounds = canvas.getBoundingClientRect()
  if ((event.clientX - bounds.left) / bounds.width < 0.5) return
  dragging = true
  dragX = event.clientX
  dragY = event.clientY
  canvas.setPointerCapture(event.pointerId)
})
canvas.addEventListener('pointermove', (event) => {
  if (!dragging) return
  skeleton.yaw += (event.clientX - dragX) * 0.008
  skeleton.pitch = Math.max(
    -0.9,
    Math.min(0.9, skeleton.pitch + (event.clientY - dragY) * 0.005),
  )
  dragX = event.clientX
  dragY = event.clientY
})
canvas.addEventListener('pointerup', (event) => {
  // Guarded: a press on the sim pane never captured the pointer, and releasing
  // a capture that was never taken throws.
  if (!dragging) return
  dragging = false
  canvas.releasePointerCapture(event.pointerId)
})

recordButton.addEventListener('click', async () => {
  if (recorder.recording) {
    recordButton.disabled = true
    const clip = await recorder.stop()
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    download(clip, `bim-demo-${stamp}.webm`)
    recordButton.textContent = 'Record'
    recordButton.disabled = false
  } else {
    recorder.start()
    recordButton.textContent = 'Stop & save'
  }
})

function runSynthetic(): void {
  status.textContent = 'synthetic pose (no camera)'
  const started = performance.now()
  // At roughly the rate the real pipeline delivers, so the backend sees the same
  // cadence and the HUD's pose rate stays comparable between the two modes.
  setInterval(() => {
    const result = syntheticPose((performance.now() - started) / 1000)
    latest = { result, hands: {} }
    socket.send(result)
  }, 1000 / 30)
}

async function main() {
  if (synthetic) {
    runSynthetic()
    return
  }
  status.textContent = 'loading models…'
  await tracker.init()
  if (handsEnabled) await hands.init()
  status.textContent = 'running'

  tracker.start((result, timestamp) => {
    const handResult = handsEnabled ? hands.detect(timestamp) : null
    const pose = result.landmarks[0]
    const assigned = pose
      ? assignHands(handResult, { left: pose[LEFT_WRIST], right: pose[RIGHT_WRIST] })
      : {}
    latest = { result, hands: assigned }
    socket.send(result, assigned)
  })
}

// Started before the models are, and outside main()'s error path: a machine
// with no camera should still show the sim pane and the HUD rather than a blank
// page, and that's also the state the page sits in while the models load.
requestAnimationFrame(render)

main().catch((err) => {
  console.error(err)
  status.textContent = `error — ${err.message}`
})
