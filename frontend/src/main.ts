import { PoseTracker } from './pose/tracker'
import { HandTracker, assignHands } from './pose/hands'
import { PoseOverlay } from './pose/overlay'
import { PoseSocket } from './pose/socket'

const video = document.querySelector<HTMLVideoElement>('#video')!
const canvas = document.querySelector<HTMLCanvasElement>('#pose-canvas')!
const status = document.querySelector<HTMLDivElement>('#status')!

// Two models per frame costs real time on a laptop GPU. `?hands=0` turns the
// hand model off to isolate the pose pipeline's own framerate.
const handsEnabled = new URLSearchParams(location.search).get('hands') !== '0'

const overlay = new PoseOverlay(canvas)
const tracker = new PoseTracker(video)
const hands = new HandTracker(video)
const socket = new PoseSocket('ws://127.0.0.1:8000/ws/pose')

// Wrist indices from the pose model, used to work out which detected hand
// belongs to which arm (see assignHands).
const LEFT_WRIST = 15
const RIGHT_WRIST = 16

async function main() {
  await tracker.init()
  if (handsEnabled) await hands.init()
  overlay.resize(video.videoWidth, video.videoHeight)
  status.textContent = 'pose: running'

  tracker.start((result, timestamp) => {
    // Both models get the same timestamp
    const handResult = handsEnabled ? hands.detect(timestamp) : null
    const pose = result.landmarks[0]
    const assigned = pose
      ? assignHands(handResult, { left: pose[LEFT_WRIST], right: pose[RIGHT_WRIST] })
      : {}

    overlay.draw(result, handResult)
    const sent = socket.send(result, assigned)

    const handCount = [assigned.left, assigned.right].filter(Boolean).length
    status.textContent =
      `pose: ${result.landmarks.length} person(s)` +
      (handsEnabled ? ` · ${handCount}/2 hands` : ' · hands off') +
      (sent ? ' · sending' : '')
  })
}

main().catch((err) => {
  console.error(err)
  status.textContent = `pose: error — ${err.message}`
})
