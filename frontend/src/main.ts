import { PoseTracker } from './pose/tracker'
import { PoseOverlay } from './pose/overlay'
import { PoseSocket } from './pose/socket'

const video = document.querySelector<HTMLVideoElement>('#video')!
const canvas = document.querySelector<HTMLCanvasElement>('#pose-canvas')!
const status = document.querySelector<HTMLDivElement>('#status')!

const overlay = new PoseOverlay(canvas)
const tracker = new PoseTracker(video)
const socket = new PoseSocket('ws://127.0.0.1:8000/ws/pose')

async function main() {
  await tracker.init()
  overlay.resize(video.videoWidth, video.videoHeight)
  status.textContent = 'pose: running'

  tracker.start((result) => {
    overlay.draw(result)
    const sent = socket.send(result)
    status.textContent = `pose: ${result.landmarks.length} person(s) detected${sent ? ' · sending' : ''}`
  })
}

main().catch((err) => {
  console.error(err)
  status.textContent = `pose: error — ${err.message}`
})
