type ClipSpec = {
  name: string
  description: string
  probes: string
  seconds: number
}

type ClipStatus = {
  recording: boolean
  name: string | null
  path: string | null
  frames: number
  seconds: number
}

/**
 * Drives the backend's clip recorder from the page (NEXT_STEPS §5).
 *
 * The recording itself happens on the backend, at the websocket boundary — this
 * is only the button. It lives in the browser because that is where the person
 * performing the motions is looking; recording the battery from a second
 * terminal means turning away from the camera between every take, which is both
 * awkward alone and a good way to record eight clips of yourself reaching for a
 * keyboard.
 *
 * A countdown before recording starts exists for the same reason: it buys the
 * time to get back into frame and into the starting pose.
 */
export class RecordControls {
  private base: string
  private root: HTMLElement
  private select: HTMLSelectElement
  private button: HTMLButtonElement
  private info: HTMLElement
  private battery: ClipSpec[] = []
  private countdown: number | null = null
  private recording = false

  constructor(root: HTMLElement, base = 'http://127.0.0.1:8000') {
    this.base = base
    this.root = root
    this.root.innerHTML = `
      <select id="clip-select"></select>
      <button id="clip-record" disabled>record</button>
      <span id="clip-info">recorder: connecting…</span>
    `
    this.select = root.querySelector('#clip-select')!
    this.button = root.querySelector('#clip-record')!
    this.info = root.querySelector('#clip-info')!
    this.button.addEventListener('click', () => this.toggle())
    this.select.addEventListener('change', () => this.describe())
  }

  async init(): Promise<void> {
    try {
      const response = await fetch(`${this.base}/record/battery`)
      const body = await response.json()
      this.battery = body.battery
      this.select.innerHTML = this.battery
        .map((clip) => `<option value="${clip.name}">${clip.name}</option>`)
        .join('')
      this.button.disabled = false
      // A clip may already be open — the backend outlives a page reload.
      await this.refresh()
      this.describe()
    } catch (err) {
      this.info.textContent = `recorder: unavailable (${(err as Error).message})`
    }
  }

  private get selected(): ClipSpec | undefined {
    return this.battery.find((clip) => clip.name === this.select.value)
  }

  private describe(): void {
    if (this.recording || this.countdown !== null) return
    const clip = this.selected
    this.info.textContent = clip
      ? `${clip.seconds}s — ${clip.description} (${clip.probes})`
      : 'recorder: ready'
  }

  private async refresh(): Promise<void> {
    const status: ClipStatus = await (await fetch(`${this.base}/record`)).json()
    this.applyStatus(status)
  }

  private applyStatus(status: ClipStatus): void {
    this.recording = status.recording
    this.button.textContent = status.recording ? 'stop' : 'record'
    this.root.classList.toggle('recording', status.recording)
    this.select.disabled = status.recording
    if (status.recording) {
      this.info.textContent = `● recording ${status.name} — ${status.frames} frames`
    }
  }

  private async toggle(): Promise<void> {
    if (this.countdown !== null) {
      // Second click during the countdown cancels, rather than queuing a take
      // nobody is ready for.
      window.clearInterval(this.countdown)
      this.countdown = null
      this.button.textContent = 'record'
      this.describe()
      return
    }
    if (this.recording) {
      await this.stop()
    } else {
      this.startCountdown()
    }
  }

  private startCountdown(seconds = 3): void {
    let left = seconds
    this.button.textContent = 'cancel'
    const tick = () => {
      if (left === 0) {
        window.clearInterval(this.countdown!)
        this.countdown = null
        void this.start()
        return
      }
      this.info.textContent = `starting ${this.select.value} in ${left}…`
      left -= 1
    }
    tick()
    this.countdown = window.setInterval(tick, 1000)
  }

  private async start(): Promise<void> {
    const response = await fetch(`${this.base}/record/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: this.select.value }),
    })
    this.applyStatus(await response.json())
    // Poll while recording so the frame counter is live: a take that captured
    // 4 frames because tracking dropped should be obvious before you move on.
    const poll = window.setInterval(async () => {
      if (!this.recording) {
        window.clearInterval(poll)
        return
      }
      await this.refresh()
    }, 500)
  }

  private async stop(): Promise<void> {
    const response = await fetch(`${this.base}/record/stop`, { method: 'POST' })
    const status: ClipStatus = await response.json()
    this.recording = false
    this.applyStatus({ ...status, recording: false })
    this.info.textContent = `saved ${status.name}: ${status.frames} frames, ${status.seconds}s`
  }
}
