import type { Hud } from '../pose/socket'

/**
 * The readout drawn over the sim pane.
 *
 * Two blocks, because they answer two different questions a viewer of the clip
 * will have. The rates say whether this is running live rather than being played
 * back slowly. The margin bars say how hard the current pose is for the solver:
 * they are the solver's own observability weight, which falls to zero exactly at
 * the shoulder singularities where an iterative IK would be thrashing and this
 * closed form instead stops trusting the measurement.
 */

const PANEL = 'rgba(4, 10, 14, 0.72)'
const LABEL = '#8fa8bd'
const VALUE = '#e8f4ff'
const WARN = '#ff9f43'

const PAD = 12
const LINE = 17
const BAR_WIDTH = 116
const BAR_HEIGHT = 9

/** Bar colour by margin: comfortable, getting close, at the singularity. */
function marginColor(margin: number): string {
  if (margin > 0.6) return '#3ddc84'
  if (margin > 0.3) return '#ffcc55'
  return '#ff5f57'
}

function panel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  ctx.fillStyle = PANEL
  ctx.beginPath()
  ctx.roundRect(x, y, w, h, 6)
  ctx.fill()
}

/** Draws both blocks at the top-left of the pane. Returns nothing; pure paint. */
export function drawHud(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  hud: Hud | undefined,
  extra: { streamFps: number; connected: boolean },
): void {
  ctx.save()
  ctx.textBaseline = 'top'

  const rates: [string, string][] = [
    ['pose', hud?.pose_fps !== undefined ? `${hud.pose_fps.toFixed(1)} Hz` : '--'],
    [
      'retarget',
      hud?.retarget_ms !== undefined ? `${hud.retarget_ms.toFixed(2)} ms` : '--',
    ],
    [
      'camera lag',
      hud?.capture_lag_ms !== undefined ? `${hud.capture_lag_ms.toFixed(0)} ms` : '--',
    ],
    ['sim video', `${extra.streamFps.toFixed(0)} fps`],
  ]

  const ratesWidth = 176
  const ratesHeight = PAD * 2 + LINE * (rates.length + 1)
  panel(ctx, x, y, ratesWidth, ratesHeight)

  let ty = y + PAD
  ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.fillStyle = extra.connected ? VALUE : WARN
  ctx.fillText(extra.connected ? 'LIVE — SEW retarget' : 'backend offline', x + PAD, ty)
  ty += LINE + 3

  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace'
  for (const [label, value] of rates) {
    ctx.fillStyle = LABEL
    ctx.fillText(label, x + PAD, ty)
    ctx.fillStyle = VALUE
    ctx.textAlign = 'right'
    ctx.fillText(value, x + ratesWidth - PAD, ty)
    ctx.textAlign = 'left'
    ty += LINE
  }

  const arms = hud?.arms
  const barsY = y + ratesHeight + 8
  const barsWidth = 176
  const barsHeight = PAD * 2 + LINE * 3 + 6
  panel(ctx, x, barsY, barsWidth, barsHeight)

  let by = barsY + PAD
  ctx.fillStyle = LABEL
  ctx.fillText('singularity margin', x + PAD, by)
  by += LINE + 4

  for (const side of ['left', 'right'] as const) {
    const arm = arms?.[side]
    ctx.fillStyle = LABEL
    ctx.fillText(side, x + PAD, by)

    const barX = x + barsWidth - PAD - BAR_WIDTH
    const barY = by + 2
    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)'
    ctx.beginPath()
    ctx.roundRect(barX, barY, BAR_WIDTH, BAR_HEIGHT, 4)
    ctx.fill()

    const margin = arm?.margin
    if (margin !== undefined) {
      ctx.fillStyle = marginColor(margin)
      ctx.beginPath()
      // Floor the drawn width so a margin at zero still shows a sliver of red
      // rather than an empty track that reads as "no data".
      ctx.roundRect(barX, barY, Math.max(3, BAR_WIDTH * margin), BAR_HEIGHT, 4)
      ctx.fill()
    }

    // Held and clamped both mean the arm isn't reproducing the pose, for
    // different reasons, and both are invisible in the joint angles themselves.
    // So is a wrist running on a stale hand: joints 5-7 sitting still looks the
    // same whether the hand was measured and steady or never detected at all.
    const flags: string[] = []
    if (arm?.held) flags.push('HELD')
    if (arm?.wrist_held) flags.push('NO HAND')
    else if (arm?.approach_err !== undefined && arm.approach_err > 10)
      flags.push(`wrist ${arm.approach_err.toFixed(0)}° off`)
    if (arm?.clamped?.length) flags.push(`J${arm.clamped.join(',')} at limit`)
    if (flags.length) {
      ctx.fillStyle = WARN
      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
      ctx.textAlign = 'right'
      ctx.fillText(flags.join(' · '), x + barsWidth - PAD, by + BAR_HEIGHT + 4)
      ctx.textAlign = 'left'
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace'
    }
    by += LINE + 4
  }

  ctx.restore()
}

/** Pane caption, bottom-left of whichever pane it's given. */
export function drawCaption(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
): void {
  ctx.save()
  ctx.font = '600 13px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.textBaseline = 'bottom'
  const width = ctx.measureText(text).width + PAD * 2
  ctx.fillStyle = PANEL
  ctx.beginPath()
  ctx.roundRect(x, y - 26, width, 26, 6)
  ctx.fill()
  ctx.fillStyle = VALUE
  ctx.fillText(text, x + PAD, y - 7)
  ctx.restore()
}

/** The recording dot and timer, top-right. Drawn into the clip on purpose: a
 * take with no indicator is one you can't tell apart from a still frame. */
export function drawRecordingBadge(
  ctx: CanvasRenderingContext2D,
  right: number,
  y: number,
  seconds: number,
): void {
  ctx.save()
  const minutes = Math.floor(seconds / 60)
  const rest = Math.floor(seconds % 60)
  const text = `REC ${minutes}:${rest.toString().padStart(2, '0')}`
  ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace'
  const width = ctx.measureText(text).width + 34
  const x = right - width
  panel(ctx, x, y, width, 26)
  // Blink at 1Hz, the same tell as every camera.
  ctx.globalAlpha = seconds % 1 < 0.5 ? 1 : 0.3
  ctx.fillStyle = '#ff5f57'
  ctx.beginPath()
  ctx.arc(x + 15, y + 13, 5, 0, Math.PI * 2)
  ctx.fill()
  ctx.globalAlpha = 1
  ctx.fillStyle = VALUE
  ctx.textBaseline = 'middle'
  ctx.fillText(text, x + 26, y + 14)
  ctx.restore()
}
