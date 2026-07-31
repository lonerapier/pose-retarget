import { PoseTracker } from './pose/tracker'
import { PoseOverlay } from './pose/overlay'

const video = document.querySelector<HTMLVideoElement>('#video')!
const canvas = document.querySelector<HTMLCanvasElement>('#pose-canvas')!
const status = document.querySelector<HTMLDivElement>('#status')!

const overlay = new PoseOverlay(canvas)
const tracker = new PoseTracker(video)

async function main() {
  await tracker.init()
  overlay.resize(video.videoWidth, video.videoHeight)
  status.textContent = 'pose: running'

  tracker.start((result) => {
    overlay.draw(result)
    status.textContent = `pose: ${result.landmarks.length} person(s) detected`
  })
}

main().catch((err) => {
  console.error(err)
  status.textContent = `pose: error — ${err.message}`
})
