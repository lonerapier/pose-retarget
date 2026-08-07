import { HandLandmarker, type PoseLandmarkerResult } from '@mediapipe/tasks-vision'
import type { HandAssignment } from './hands'

/**
 * A point in landmark space. MediaPipe's own `Landmark` also carries a
 * `visibility`, which nothing here reads, and requiring it would mean every
 * derived point — a grid corner, a translated hand joint, an axis tip — had to
 * invent one.
 */
type Vec3 = { x: number; y: number; z: number }

/**
 * Draws MediaPipe's **world** landmarks as a 3D stick figure.
 *
 * The 2D overlay in `overlay.ts` draws the normalized landmarks on top of the
 * webcam image, which is the right thing for checking that detection tracks the
 * person. This draws a different thing for a different reason: world landmarks
 * are what actually reaches the retargeter, and they carry a depth the image
 * overlay throws away. Rendering them in perspective, with no video behind them,
 * shows the pose the robot is really being driven from — including the z axis
 * that a single webcam is worst at, which is exactly the part worth being able
 * to see.
 *
 * Landmark axes (see `mapper.Landmark`): x toward the image's right edge, y
 * down, z away from the camera. Metres, origin at the hip midpoint.
 */

// Pose landmark indices, grouped by what each group is for. Arms drive the
// robot and are drawn brightest; everything else is context.
const ARM_BONES: [number, number][] = [
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
]
const TORSO_BONES: [number, number][] = [
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
]
const LEG_BONES: [number, number][] = [
  [23, 25],
  [25, 27],
  [27, 29],
  [29, 31],
  [27, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [30, 32],
  [28, 32],
]
const HEAD_BONES: [number, number][] = [
  [0, 11],
  [0, 12],
]

const LEFT_SHOULDER = 11
const RIGHT_SHOULDER = 12
const LEFT_WRIST = 15
const RIGHT_WRIST = 16
const LEFT_HIP = 23
const RIGHT_HIP = 24

const COLOR = {
  arm: '#38e8d0',
  torso: '#8fb4d9',
  leg: '#4a6076',
  head: '#4a6076',
  hand: '#ff9f43',
  joint: '#e8f4ff',
  grid: '#1d3b3a',
  axisX: '#ff5f57',
  axisY: '#3ddc84',
  axisZ: '#4d9fff',
}

/** Metres below the hip midpoint to put the grid.
 *
 * Fixed rather than measured off the ankles: the ankles are frequently out of
 * frame at a desk, and a floor that jumps whenever the legs appear reads far
 * worse than one that sits at roughly hip height for a standing adult. It is a
 * depth cue, not a measurement. */
const FLOOR_Y = -0.95
const GRID_EXTENT = 1.5
const GRID_STEP = 0.3

/** Camera distance in metres, and focal length as a multiple of pane height.
 * Together these frame a person about 1.6m tall with margin at the top. */
const CAMERA_DISTANCE = 3.0
const FOCAL_SCALE = 1.45
/** Where the hip origin sits down the pane, as a fraction of its height. */
const ORIGIN_Y = 0.4

type Point = { x: number; y: number; depth: number }

/** A projected bone, held so everything can be depth-sorted before it's drawn. */
type Segment = { a: Point; b: Point; color: string; width: number }

export type Skeleton3DOptions = {
  /** Yaw about the vertical axis, radians. 0 looks straight at the subject. */
  yaw?: number
  /** Pitch, radians. Positive looks down on the subject. */
  pitch?: number
  /** Draw the torso frame's axes at the hip origin. */
  showTorsoFrame?: boolean
}

export class Skeleton3D {
  /**
   * Mirrors the view, so the figure moves the way the operator sees themselves
   * move. Matches the webcam inset, which the page mirrors for the same reason.
   * The landmarks themselves are never mirrored — only this projection is — so
   * nothing that reaches the backend is affected.
   */
  private static readonly MIRROR = -1

  yaw: number
  pitch: number
  showTorsoFrame: boolean

  constructor(options: Skeleton3DOptions = {}) {
    this.yaw = options.yaw ?? 0.35
    this.pitch = options.pitch ?? 0.12
    this.showTorsoFrame = options.showTorsoFrame ?? true
  }

  /** Draw one frame into the rect (`x`, `y`, `w`, `h`) of `ctx`. */
  draw(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    result: PoseLandmarkerResult | null,
    hands: HandAssignment = {},
  ): void {
    ctx.save()
    ctx.beginPath()
    ctx.rect(x, y, w, h)
    ctx.clip()

    this.background(ctx, x, y, w, h)

    const project = this.projector(x, y, w, h)
    this.grid(ctx, project)

    const landmarks = result?.worldLandmarks?.[0]
    if (!landmarks) {
      ctx.restore()
      return
    }

    const segments: Segment[] = []
    const push = (bones: [number, number][], color: string, width: number) => {
      for (const [i, j] of bones) {
        const a = landmarks[i]
        const b = landmarks[j]
        if (a && b) segments.push({ a: project(a), b: project(b), color, width })
      }
    }
    push(LEG_BONES, COLOR.leg, 2)
    push(HEAD_BONES, COLOR.head, 2)
    push(TORSO_BONES, COLOR.torso, 3)
    push(ARM_BONES, COLOR.arm, 5)

    for (const side of ['left', 'right'] as const) {
      const hand = hands[side]?.worldLandmarks
      const wrist = landmarks[side === 'left' ? LEFT_WRIST : RIGHT_WRIST]
      if (!hand || !wrist) continue
      // Hand world landmarks are centred on the hand, not the hips, so they
      // share no origin with the body's. Only the offsets within one hand carry
      // meaning; pinning its wrist to the pose model's wrist is what puts the
      // hand back on the end of the arm.
      const origin = hand[0]
      const place = (lm: Vec3): Vec3 => ({
        x: wrist.x + (lm.x - origin.x),
        y: wrist.y + (lm.y - origin.y),
        z: wrist.z + (lm.z - origin.z),
      })
      for (const bone of HandLandmarker.HAND_CONNECTIONS) {
        const a = hand[bone.start]
        const b = hand[bone.end]
        if (a && b) {
          segments.push({
            a: project(place(a)),
            b: project(place(b)),
            color: COLOR.hand,
            width: 2,
          })
        }
      }
    }

    // Painter's algorithm. Without it a forearm swung behind the torso draws
    // over it, and the pose reads inside out at exactly the moments depth is the
    // thing you're looking for.
    segments.sort((p, q) => q.a.depth + q.b.depth - (p.a.depth + p.b.depth))
    for (const segment of segments) this.stroke(ctx, segment)

    for (const lm of landmarks) {
      const p = project(lm)
      const radius = Math.max(1.2, 3.2 * this.fade(p.depth))
      ctx.globalAlpha = 0.45 + 0.55 * this.fade(p.depth)
      ctx.fillStyle = COLOR.joint
      ctx.beginPath()
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.globalAlpha = 1

    if (this.showTorsoFrame) this.torsoFrame(ctx, project, landmarks)
    ctx.restore()
  }

  /**
   * Landmark point -> pane pixels, as a closure so the trig runs once per frame
   * rather than once per landmark.
   *
   * Landmark axes go to a conventional view frame first (x right, y up, z toward
   * the viewer), then yaw about the vertical and pitch about the horizontal, then
   * a pinhole divide.
   */
  private projector(x: number, y: number, w: number, h: number) {
    const cosYaw = Math.cos(this.yaw)
    const sinYaw = Math.sin(this.yaw)
    const cosPitch = Math.cos(this.pitch)
    const sinPitch = Math.sin(this.pitch)
    const focal = h * FOCAL_SCALE
    const cx = x + w / 2
    const cy = y + h * ORIGIN_Y

    return (lm: Vec3): Point => {
      const px = Skeleton3D.MIRROR * lm.x
      const py = -lm.y
      const pz = -lm.z

      const rx = px * cosYaw + pz * sinYaw
      const rz = -px * sinYaw + pz * cosYaw
      const uy = py * cosPitch + rz * sinPitch
      const uz = -py * sinPitch + rz * cosPitch

      // Guard the divide rather than clip properly: at this camera distance a
      // landmark can only get behind the lens if detection has produced a wild
      // z, and pinning that frame's point to the near plane is a less alarming
      // failure than a coordinate through infinity.
      const depth = Math.max(0.2, CAMERA_DISTANCE - uz)
      const scale = focal / depth
      return { x: cx + rx * scale, y: cy - uy * scale, depth }
    }
  }

  /** 1 for the nearest plausible depth, falling toward 0 further back. */
  private fade(depth: number): number {
    const t = (depth - (CAMERA_DISTANCE - 0.7)) / 1.4
    return Math.min(1, Math.max(0, 1 - t))
  }

  private stroke(ctx: CanvasRenderingContext2D, segment: Segment): void {
    const near = this.fade((segment.a.depth + segment.b.depth) / 2)
    ctx.globalAlpha = 0.35 + 0.65 * near
    ctx.lineWidth = segment.width * (0.6 + 0.4 * near)
    ctx.strokeStyle = segment.color
    ctx.lineCap = 'round'
    ctx.beginPath()
    ctx.moveTo(segment.a.x, segment.a.y)
    ctx.lineTo(segment.b.x, segment.b.y)
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  private background(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
  ): void {
    const gradient = ctx.createLinearGradient(x, y, x, y + h)
    gradient.addColorStop(0, '#0a1a1c')
    gradient.addColorStop(1, '#04090b')
    ctx.fillStyle = gradient
    ctx.fillRect(x, y, w, h)
  }

  private grid(ctx: CanvasRenderingContext2D, project: (lm: Vec3) => Point): void {
    // The grid lives in landmark axes like everything else, so y is *down* and
    // the floor is a positive y.
    const floor = -FLOOR_Y
    ctx.strokeStyle = COLOR.grid
    ctx.lineWidth = 1
    ctx.globalAlpha = 0.9
    for (let t = -GRID_EXTENT; t <= GRID_EXTENT + 1e-9; t += GRID_STEP) {
      const lines: [Vec3, Vec3][] = [
        [
          { x: t, y: floor, z: -GRID_EXTENT },
          { x: t, y: floor, z: GRID_EXTENT },
        ],
        [
          { x: -GRID_EXTENT, y: floor, z: t },
          { x: GRID_EXTENT, y: floor, z: t },
        ],
      ]
      for (const [from, to] of lines) {
        const a = project(from)
        const b = project(to)
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1
  }

  /**
   * The subject's own frame, drawn at the hip origin.
   *
   * This is the frame `mapper.torso_frame` builds and every limb direction is
   * expressed in, and it's the reason the retarget survives the operator turning
   * or leaning: the axes turn with them, so the directions handed to the robot
   * don't change. That's invisible in the skeleton alone and obvious once the
   * triad is on screen.
   */
  private torsoFrame(
    ctx: CanvasRenderingContext2D,
    project: (lm: Vec3) => Point,
    landmarks: Vec3[],
  ): void {
    const ls = landmarks[LEFT_SHOULDER]
    const rs = landmarks[RIGHT_SHOULDER]
    const lh = landmarks[LEFT_HIP]
    const rh = landmarks[RIGHT_HIP]
    if (!ls || !rs || !lh || !rh) return

    const mid = (a: Vec3, b: Vec3): Vec3 => ({
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      z: (a.z + b.z) / 2,
    })
    const sub = (a: Vec3, b: Vec3): Vec3 => ({
      x: a.x - b.x,
      y: a.y - b.y,
      z: a.z - b.z,
    })
    const norm = (v: Vec3): Vec3 | null => {
      const n = Math.hypot(v.x, v.y, v.z)
      // Same collapse guard as the backend's: with the shoulders or the spine
      // gone the frame is arbitrary, and drawing an arbitrary triad is worse
      // than drawing none.
      if (n < 1e-3) return null
      return { x: v.x / n, y: v.y / n, z: v.z / n }
    }
    const cross = (a: Vec3, b: Vec3): Vec3 => ({
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x,
    })

    const shoulders = mid(ls, rs)
    const hips = mid(lh, rh)
    const across = norm(sub(ls, rs))
    const spine = norm(sub(shoulders, hips))
    if (!across || !spine) return
    const normal = norm(cross(across, spine))
    if (!normal) return

    const length = 0.22
    const axes: [Vec3, string][] = [
      [normal, COLOR.axisX],
      [across, COLOR.axisY],
      [cross(normal, across), COLOR.axisZ],
    ]
    const origin = project(hips)
    for (const [axis, color] of axes) {
      const tip = project({
        x: hips.x + axis.x * length,
        y: hips.y + axis.y * length,
        z: hips.z + axis.z * length,
      })
      ctx.strokeStyle = color
      ctx.lineWidth = 2.5
      ctx.globalAlpha = 0.9
      ctx.beginPath()
      ctx.moveTo(origin.x, origin.y)
      ctx.lineTo(tip.x, tip.y)
      ctx.stroke()
    }
    ctx.globalAlpha = 1
  }
}
