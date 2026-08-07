/**
 * Records the composited canvas to a WebM file.
 *
 * The whole reason the demo page composites into one canvas instead of laying
 * two elements out with CSS: `captureStream` gives a real video track of exactly
 * what was drawn, at a fixed size, with no window chrome, no cursor and no
 * dependence on screen-recording tooling or how the operator happened to
 * arrange their windows.
 */

// Best first. VP9 is roughly half the size of VP8 at the same quality, and Safari
// takes neither, so the bare `video/webm` and the empty string are there to let
// `isTypeSupported` pick whatever the browser will actually give us.
const MIME_TYPES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  '',
]

function pickMimeType(): string {
  return MIME_TYPES.find((type) => type === '' || MediaRecorder.isTypeSupported(type)) ?? ''
}

export class CanvasRecorder {
  private canvas: HTMLCanvasElement
  private fps: number
  private recorder: MediaRecorder | null = null
  private chunks: Blob[] = []
  private startedAt = 0

  constructor(canvas: HTMLCanvasElement, fps = 30) {
    this.canvas = canvas
    this.fps = fps
  }

  get recording(): boolean {
    return this.recorder !== null && this.recorder.state === 'recording'
  }

  /** Seconds since recording started, for the on-screen timer. */
  get elapsed(): number {
    return this.recording ? (performance.now() - this.startedAt) / 1000 : 0
  }

  start(): void {
    if (this.recording) return
    const stream = this.canvas.captureStream(this.fps)
    const mimeType = pickMimeType()
    this.chunks = []
    this.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data)
    }
    // A timeslice, so a long take arrives as chunks instead of one buffer the
    // recorder holds until stop().
    this.recorder.start(1000)
    this.startedAt = performance.now()
  }

  /** Stop and hand back the finished clip. Resolves once the recorder flushes. */
  stop(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const recorder = this.recorder
      if (!recorder || recorder.state === 'inactive') {
        reject(new Error('not recording'))
        return
      }
      recorder.onstop = () => {
        this.recorder = null
        resolve(new Blob(this.chunks, { type: recorder.mimeType || 'video/webm' }))
      }
      recorder.stop()
    })
  }
}

/** Save a recorded clip to the user's downloads. */
export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  // Revoking immediately can cancel the download in some browsers; one turn of
  // the event loop is enough for the click to have been taken up.
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
