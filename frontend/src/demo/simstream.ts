/**
 * Receives the backend's offscreen MuJoCo render as a stream of JPEG frames.
 *
 * A websocket rather than an MJPEG `<img src>`: the demo page composites the
 * robot and the skeleton into one canvas and records that canvas, and an image
 * loaded from another origin taints the canvas, which makes `captureStream`
 * throw. Frames arriving as binary blobs and going through `createImageBitmap`
 * never touch the origin rules at all.
 */
export class SimStream {
  private ws: WebSocket
  private bitmap: ImageBitmap | null = null
  private decoding = false
  /** Frames received, whether or not they were decoded. */
  frames = 0
  connected = false

  constructor(url: string) {
    this.ws = new WebSocket(url)
    this.ws.binaryType = 'blob'
    this.ws.onopen = () => {
      this.connected = true
      console.log('sim stream: connected')
    }
    this.ws.onclose = () => {
      this.connected = false
      console.log('sim stream: disconnected')
    }
    this.ws.onerror = (err) => console.error('sim stream: error', err)
    this.ws.onmessage = (event) => this.receive(event.data)
  }

  private async receive(data: unknown): Promise<void> {
    if (!(data instanceof Blob)) return
    this.frames++
    // Decoding is async and the backend keeps sending. Dropping frames that
    // arrive mid-decode keeps the displayed frame the newest one rather than
    // building a queue that falls further behind the longer it runs.
    if (this.decoding) return
    this.decoding = true
    try {
      const bitmap = await createImageBitmap(data)
      // ImageBitmaps hold their pixels outside the JS heap, so an unclosed one
      // per frame at 30fps is a real leak, not something GC gets to in time.
      this.bitmap?.close()
      this.bitmap = bitmap
    } catch (err) {
      console.error('sim stream: decode failed', err)
    } finally {
      this.decoding = false
    }
  }

  /** The most recent decoded frame, or null before the first one arrives. */
  get frame(): ImageBitmap | null {
    return this.bitmap
  }

  close(): void {
    this.ws.close()
    this.bitmap?.close()
    this.bitmap = null
  }
}
