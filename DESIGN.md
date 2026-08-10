# Next Steps: webcam → bimanual OpenArm retargeting

**TL;DR** — the current pipeline works but the *objective is under-determined*. We
solve position-only IK for one site on a 7-DoF arm, which leaves a 4-dimensional
null space with nothing anchoring it. Everything else (jitter, lock-ups, the missing
shoulder DoF) is downstream of that. The fix with the best payoff-to-effort ratio is
to stop retargeting *positions* and start retargeting *limb orientations* —
i.e. SEW-Mimic.

> **Working under a one-day deadline? Go straight to §0.** It supersedes the
> ordering in §13. Sections 6–12 remain the reference material behind it, and §14
> is the reading list for the design document.

---

## 0. The under-a-day plan

**The key finding that makes this tractable: for OpenArm v2, SEW-Mimic stages 1–2
collapse to ~25 lines of closed-form trigonometry.** No Paden–Kahan library, no
external code, no iteration, no solver. See §0.2 — it is derived and numerically
validated below. That removes the "budget for a from-scratch implementation" risk
in §7.6 almost entirely.

Total: ~8 hours with slack. Every step is independently shippable, so if you run out
of time you stop wherever you are and still have something strictly better than today.

### 0.1 The schedule

| #   | Time          | Task                                                    | Detail            |
| --- | ------------- | ------------------------------------------------------- | ----------------- |
| 1   | ~~0:00–0:45~~ | ~~**Record/replay harness + motion battery**~~ **DONE** | §5                |
| 2   | ~~0:45–1:15~~ | ~~**1€ filter on landmarks**~~ **DONE**                 | §6.3              |
| 3   | 1:15–1:45     | **Slew limit + interpolate `qpos_target`**              | §6.3              |
| 4   | ~~1:45–2:30~~ | ~~**Torso frame**~~ + ~~world landmarks~~ **DONE**      | §6.2, §6.5        |
| 5   | ~~2:30–4:15~~ | ~~**Closed-form stages 1–2**~~ **DONE** — the real fix  | §0.2              |
| 6   | ~~4:15–5:00~~ | ~~**Wire joints 5–7**~~ **DONE** (§7.8) · retire `solve_site_ik` still to do | §0.3, §7.8 |
| 7   | 5:00–6:00     | **Intermediate target schema + JSONL metrics**          | §11, §6.4         |
| 8   | 6:00–7:00     | Run the battery, tune `beta`, record the video          |                   |
| 9   | 7:00–8:00     | Design document (3 pages)                               | §14 for citations |

**Step 1 is built** — see §5 for how to drive it. It buys deterministic A/B testing
for the rest of the day; without it you would spend more than forty-five minutes
waving at a webcam trying to decide whether a change helped. Hand tracking landed
alongside it (§7.4 #2 phase B), so clips already carry the stage-3 input.

**Hard time-box on step 5: if the unit test in §0.4 isn't passing by 3:30, abandon it
and fall back to §6.1** (add an elbow task to the existing DLS solver — ~45 min, gets
maybe 70% of the benefit). Do not let the closed form eat the whole day.

Steps 2–4 are worth doing *even if* step 5 gets abandoned; they're independent.

### 0.2 The closed form (steps 5)

Two facts from §3, both **numerically verified against `mj_forward`** over 200 random
full-range configurations per arm (max deviation **2×10⁻⁶ degrees**):

- joint 3's axis is collinear with the upper arm
- joint 5's axis is collinear with the forearm

So aligning those two axes with the human's two limb directions *is* SEW-Mimic
stages 1–2. Because OpenArm's axes are `y`, `−x`, `−z`, `−y` at `qpos=0`, the general
Subproblem-2 machinery specializes to plain trigonometry.

Given human limb unit vectors in the **arm base frame**:

```
u = unit(elbow  − shoulder)     upper arm
l = unit(wrist  − elbow)        forearm
sx = −1 for the left arm, +1 for the right     # joint 1 axis is (0,+1,0) / (0,−1,0)
```

**Stage 1 — joints 1, 2.** Forward relation is `u = (sx·sin q₁·cos q₂, −sin q₂, −cos q₁·cos q₂)`:

```python
sin_q2 = np.clip(-u[1], -1.0, 1.0)
for cos_q2 in (+np.sqrt(1 - sin_q2**2), -np.sqrt(1 - sin_q2**2)):   # two branches
    q2 = np.arctan2(sin_q2, cos_q2)
    if abs(cos_q2) < 1e-9:
        q1 = q1_prev            # SINGULAR — see below
    else:
        q1 = np.arctan2(sx * u[0] / cos_q2, -u[2] / cos_q2)
```

**Stage 2 — joints 3, 4.** With `A = R(h₁, q₁) · R(h₂, q₂)`, `h₁ = (0, −sx, 0)`,
`h₂ = (−1, 0, 0)`, and `v = Aᵀ·l`, the forward relation is
`v = (cos q₃·sin q₄, −sin q₃·sin q₄, −cos q₄)`:

```python
v  = A.T @ l
q4 = np.arccos(np.clip(-v[2], -1.0, 1.0))        # in [0, π]
q3 = q3_prev if abs(np.sin(q4)) < 1e-9 else np.arctan2(-v[1], v[0])
```

That's the whole thing. **Joint 3 — the shoulder roll we've been missing entirely
(§2.3) — falls out of stage 2 for free**, and there is no null space left to drift.

Three details you must handle:

1. **Branch selection.** The `±cos q₂` gives two solutions and *both* can be inside
   joint limits (left `joint2` range is `[−3.316, 0.175]`, so the `cos q₂ < 0` branch
   is legal). Filter by joint limits, then pick the branch closest to the previous
   frame's `q`. Skipping this gives you visible branch-flip pops.
2. **Clamp `q₄`.** `arccos` returns `[0, π]` but joint 4's range is `[0, 2.444]`.
   A human folds their elbow tighter than the robot can — clamp, don't wrap.
3. **The two singularities, and why they're exactly your bug.**
   - `cos q₂ → 0` means `u = (0, ±1, 0)`: **the upper arm pointing straight out to
     the side**, along the shoulder axis. `q₁` becomes indeterminate. This is
     precisely the "raise the arm to the side and the robot doesn't follow" symptom.
     Handle it by holding `q₁` from the previous frame and blending out over a few
     frames as `|cos q₂|` grows — don't let it snap.
   - `sin q₄ → 0` means a straight arm; `q₃` becomes indeterminate. Same treatment.
     This is the principled replacement for the `MAX_EXTENSION = 0.95` hack.

   Note both are *representation* singularities of the alignment, not kinematic
   singularities of the arm, and they're unavoidable — see the stereographic-SEW
   result in §8.2, which proves some singular direction must exist for any
   parameterization.

**Validation result** (round-trip over 300 random configs per arm: take a known `q`,
forward-kinematic it to get `u` and `l`, solve, compare limb directions):

```
left  arm: worst-case limb direction error = 0.0000012 deg
right arm: worst-case limb direction error = 0.0000015 deg
```

Exact to floating point, both arms.

### 0.3 What this deletes (step 6)

The closed form replaces the entire IK path. On the way through, this becomes dead code:

- `solve_site_ik` / `damped_least_squares_step` / `_clamp_to_joint_limits`
  ([`ik.py`](backend/app/retarget/ik.py)) — no iteration, no Jacobian, no warm start,
  and therefore **no cross-frame state to drift** (§2.1) and no clamping pathology (§2.2)
- most of `ArmGeometry` ([`retargeter.py:31-64`](backend/app/retarget/retargeter.py:31))
  — `min_reach`, `max_reach`, `MAX_EXTENSION`, `neutral_direction` and the reach scan
  all exist to map a *position* into a reachable shell. Orientation retargeting never
  targets a position, so reachability cannot fail and none of it is needed.
- the `_measure_arm` reach sweep at startup

Keep `ArmGeometry` only for the base-frame rotation and the shoulder anchor.

**Joints 5–7** — this said "hold them at a fixed neutral posture for today; stage 3
needs a hand orientation we don't have". Both halves are now out of date: the input
landed in §7.7 and the stage in **§7.8**, and stage 3 turned out to be the *cheapest*
of the three because OpenArm is the paper's perpendicular-wrist case. MediaPipe Hands →
HaMeR remains the improvement path, but as a *perception* upgrade decided by §9.4's
measurement, not as missing work.

**Deleting code is a deliverable here.** The rubric asks you to justify each component;
"we removed the reach-clamping machinery because orientation retargeting makes
reachability unfalsifiable" is a strong paragraph.

### 0.4 The risk is already retired

**[`backend/scripts/check_sew_closed_form.py`](backend/scripts/check_sew_closed_form.py)
is written, runs, and passes.** It contains a reference `solve_stage12()` and validates
both claims over 300 random full-range configurations per arm:

```bash
cd backend && uv run python scripts/check_sew_closed_form.py
```

```
 left arm  [PASS]      right arm  [PASS]
   angle(joint3 axis, upper arm)  max 1.7e-06 deg      1.9e-06 deg
   angle(joint5 axis, forearm)    max 1.9e-06 deg      2.3e-06 deg
   closed-form round-trip error   max 1.2e-06 deg      1.2e-06 deg
```

So step 5 is *porting* `solve_stage12()` into `retarget/`, not deriving it. What's
left to build around it: the branch-selection policy (limits + nearest to previous),
the singularity hold-and-blend, and the `q₄` clamp — all listed in §0.2.

**Step 5 has now landed**, in [`retarget/sew.py`](backend/app/retarget/sew.py)
(`solve_branch` + `Stage12Solver`) and `Retargeter._solve_sew`. `mode="sew"` is the
default; **`RETARGET_MODE=ik` still selects the old position-only path**, kept for
A/B comparison against the battery rather than deleted. §0.3's deletions are step 6
and have not been done.

The derivation was right and the port was faithful, and it still shook the arm apart
on the first live run — every real problem was in the *policy* wrapped around the
closed form, not in the trigonometry. What §0.2 calls "three details you must handle"
is closer to five, and the two it does not mention are the ones that bite:

1. **Which full turn of an angle to command.** `arctan2` returns the principal value,
   but joint 1's range is `[-3.491, 1.396]` — wider than half a turn on one side — so
   an angle and its `±2π` twin are not interchangeable: one is reachable, one is not.
   Worse, *interpolating along the shortest arc* between two reachable angles can
   leave the range when the range is that wide. The blend then clamps to a stop, the
   stop becomes the next frame's starting point, and the joint never comes back —
   measured, joint 1 stuck at its limit **59.7° off** a pose it could hold exactly.
   Fix: put the raw measurement on a reachable turn *first* (`_nearest_turn`), then
   interpolate linearly, then clamp. Order matters.
2. **Choose the branch on the unsmoothed solution.** Scoring the two branches after
   shrinkage compares two half-converged poses, which look alike enough that the arm
   settles on whichever it started nearest and stays there — **82°** of settled error
   on a reachable pose. `Stage12Solver.solve` now scores where each branch is
   *heading* (`smoothing=False`) and smooths only the winner.
3. **Score branches by where they point, not by whether every joint is in range.** A
   solution that pins joint 3 at its stop and still points nearly the right way beats
   one that stays inside every limit by throwing the arm to the other branch. Under
   the old per-joint reachability test, joint 3 saturating condemned the good branch
   and triggered a 180° whole-arm flip mid-motion — visible in the frame 11535→11546
   logs from the first live run.
4. **Clamping one angle changes the best value of the next.** With q3 pinned at its
   limit, q4 is re-solved against the *clamped* q3 (`arctan2` form, which reduces
   exactly to `arccos(-v[2])` when q3 is free), so the forearm points as close to the
   human's as the roll limit allows instead of solving for a roll the arm never
   applied.
5. **The singular bands have to be sized for sensor noise, not for exact
   degeneracy.** §0.2's `1e-9` epsilons only catch the exact singularity. What matters
   is the amplification `1/|cos q2|` and `1/|sin q4|`: at an 8°-off-straight arm — which
   is where landmark noise alone puts a "straight" arm — the roll is already amplifying
   noise into ±90° swings. `OBSERVABLE_*` are set at ~3x amplification, and the hard
   threshold is replaced by a shrinkage weight `s²/(s²+floor²)`, since a hard
   threshold puts a discontinuity exactly where the measurement is noisiest.
6. **Shrinking toward the previous frame is not enough — it needs somewhere to fall
   back to.** Shrinkage slows an unobservable angle down but does not anchor it, so
   noise random-walks it: joint 3 wandered from 1° to 90° and parked against its stop
   while the subject held a straight arm. This is §2.1's unresolved null space, one
   joint wide, and it takes the same answer — a restoring term toward the rest posture
   (`REST_RELAXATION`, scaled by `(1-trust)²` so it all but vanishes where the angle
   *is* observable).

**Where it ended up** — `just sew-retarget-check`, four properties that trade against
each other, so a single number would hide which one regressed:

|                                        | result                                                            |
| -------------------------------------- | ----------------------------------------------------------------- |
| closed form, 300 reachable poses/arm   | `1.7e-06 deg` worst limb-direction error                          |
| settled tracking through the live path | `14.0 deg` worst, observable poses                                |
| sweep through both singularities       | `1.6 deg` worst single-frame joint step                           |
| held still, landmark noise σ=0.04      | full stretch p95 `5.9 deg` — **1.1x** the well-conditioned pose's |

That last row is the point of the whole exercise: full stretch is no longer a special
place. Before these fixes the same test gave **113 of 199 frames with a >30° jump and
a worst jump of 175°**; it is now 0 of 199 and 23.5°. The residual is honest tracking
of a noisy input — a well-conditioned pose jitters just as much — which is what steps
2 and 3 (1€ filter, slew limit) are for. **Step 2 has since landed** (§6.3), taking
that row to 5.9° → 2.3° p95 and 23.5° → 18.6° worst; step 3 is still unstarted.

The one deliberate accuracy cost: the branch policy may keep a branch up to
`BRANCH_SWITCH_TOLERANCE` (20°) worse than the best rather than slam the arm across.
Stated as a bound, not as a weight on distance, so the worst it can ever cost is that
number and not a function of how far the arm happened to move.

Re-run this if you ever bump the OpenArm submodule; it's a regression test for the
model, not just for the math. Note it uses `data.xanchor[joint_id]` and
`data.xmat[body_id]`, so no XML edits are needed anywhere in this approach.

### 0.5 What you consciously give up

Worth writing down, because the rubric asks what you considered and what it would
have cost:

| Dropped                           | Why                                                                                          | Cost                                                                                 |
| --------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| ~~Wrist orientation (stage 3)~~   | ~~no hand orientation input yet~~ **built** — input §7.7, stage §7.8                         | ~~joints 5–7 static~~ EE orientation now tracked; 16.3% of SO(3) exactly reachable   |
| Self-collision filter (§7.5 ph.4) | XPBD is a half-day on its own                                                                | arms may interpenetrate at the midline — **test this in the battery and mention it** |
| mink / QP refinement (§8.1)       | closed form makes it unnecessary today                                                       | no exact wrist *position*, no collision constraints                                  |
| Limb-length constraint (§6.6)     | ~half a day                                                                                  | depth noise still enters `u` and `l` — though far less than it entered positions     |
| Perception upgrade (§9.2)         | GPU + setup burden                                                                           | justify MediaPipe per §9.1 instead                                                   |

One genuinely good line for the design document: **switching to orientation-based
retargeting demotes monocular depth from a first-order error source to a second-order
one.** Depth error perturbs the *direction* of `u` and `l` rather than the *position*
of a target, and direction is what the robot consumes. That is the honest answer to
"how is monocular depth ambiguity handled", and it is a design decision rather than a
patch.

---

## 1. Where we are

```
browser (MediaPipe Pose + Hands)        backend (FastAPI + MuJoCo)
  33 pose + 2x21 hand landmarks           PoseFrame
  camera frame                              6 arm landmarks  -> retargeted
  ──── websocket /ws/pose ─────────►        2 hips, 2 hands  -> recorded only
                                            t_capture        -> latency accounting
                                            │
                                            ├─ recorder.write()   (§5, when armed)
                                            ├─ LandmarkFilter.apply()  ← 1€, §6.3
                                            │    per-coordinate, speed-adaptive
                                            ├─ mapper.arm_extension()
                                            │    (wrist − shoulder) / |arm|, robot axes
                                            ├─ ArmGeometry.target()
                                            │    direction preserved, radius clamped
                                            │    into the reachable shell
                                            ├─ solve_site_ik()   ← damped least squares
                                            │    20 iters, position only, warm-started
                                            └─ PositionController.ctrl()
                                                 clamp → MuJoCo <position> servos
                                            physics loop: 8 × 2 ms = 62.5 Hz
```

Files: [`mapper.py`](backend/app/retarget/mapper.py),
[`retargeter.py`](backend/app/retarget/retargeter.py),
[`smoothing.py`](backend/app/retarget/smoothing.py),
[`ik.py`](backend/app/retarget/ik.py),
[`position.py`](backend/app/control/position.py),
[`main.py`](backend/app/main.py).

Harness (§5): [`record.py`](backend/app/record.py),
[`battery.py`](backend/app/battery.py),
[`replay.py`](backend/scripts/replay.py),
[`fixtures/README.md`](backend/fixtures/README.md).

---

## 2. Diagnosis — four independent root causes

### 2.1 The null space is unresolved (causes the lock-ups)

`solve_site_ik` constrains **3 DoF** (site position) on a **7-DoF** chain. The
remaining 4 DoF are whatever damped least squares happens to leave behind.
Because we warm-start from the previous `qpos`
([`retargeter.py:163-168`](backend/app/retarget/retargeter.py:163)), the
null-space component of each `dq` **integrates frame over frame**. It is a random
walk driven by landmark noise, with no restoring term. Over seconds it wanders into
a corner of joint space and stays there.

This is exactly the "arm gets locked into strange positions" symptom.

### 2.2 Joint limits are clamped, not constrained (makes 2.1 worse)

[`ik.py:66`](backend/app/retarget/ik.py:66) clamps `qpos` *inside* the iteration
loop. The next iteration computes a Jacobian that still asks the saturated joint to
move into its stop; the clamp cancels that component, the residual error persists, so
DLS pushes harder and the *other* joints take large compensating deltas. There is no
projection of the limit-active direction out of the Jacobian.

The correct formulation is limits as **inequality constraints in a QP**, not
post-hoc clipping. See §8.1.

### 2.3 Joint 3 (shoulder roll) is geometrically unobservable from our input

[`mapper.py:82-87`](backend/app/retarget/mapper.py:82) computes `wrist − shoulder`
and uses the elbow **only** to measure arm length for normalization.

A shoulder→wrist vector carries exactly **2 DoF of direction + 1 of extension**.
Extension sets the elbow flexion (joint 4); direction sets joints 1–2. The rotation
of the elbow *around* the shoulder→wrist axis — the **swivel angle**, the standard
free parameter of an S-R-S arm — is thrown away. That rotation is precisely joint 3.

We already have the landmark that determines it. We are discarding it.

### 2.4 No filtering, no rate matching, no orientation target

- ~~No filter anywhere between MediaPipe and `ctrl`.~~ **Fixed for jitter** — a 1€
  filter now runs on the landmarks (§6.3). Nothing still rejects an *implausible*
  input, which is a separate failure (§6.7). DLS amplifies noise as the arm
  straightens — `MAX_EXTENSION = 0.95` only partially patches this.
- `qpos_target` is a ~30 Hz step function read by a 62.5 Hz physics loop
  ([`main.py:68`](backend/app/main.py:68)). No interpolation, no slew limit. The
  servos see a staircase.
- Joints 5/6/7 have **no target at all**. The rubric grades end-effector *pose*.
- The mapping is **camera-frame, not torso-frame**
  ([`mapper.py:45-60`](backend/app/retarget/mapper.py:45)). If the operator rotates
  their torso, the whole target frame shears with the camera.

---

## 3. OpenArm v2 kinematics (measured, not assumed)

From `openarm_mujoco/v2/openarm_bimanual.xml`, left arm, lines 178–235. **No `<body>`
carries a `quat`**, so at `qpos=0` every link frame is axis-aligned with the base.

| joint | axis @ qpos0 | origin offset from parent  | range (rad)       | role             |
| ----- | ------------ | -------------------------- | ----------------- | ---------------- |
| 1     | `(0, 1, 0)`  | `(0, 0.0625, 0)` from base | `[-3.491, 1.396]` | shoulder swing   |
| 2     | `(-1, 0, 0)` | `(0, 0.06, 0)`             | `[-3.316, 0.175]` | shoulder swing   |
| **3** | `(0, 0, -1)` | `(0, 0, -0.06625)`         | `[-1.571, 1.571]` | **humeral roll** |
| 4     | `(0, -1, 0)` | `(0, 0, -0.15375)`         | `[0, 2.444]`      | elbow flexion    |
| 5     | `(0, 0, -1)` | `(0, 0, -0.0955)`          | `[-1.571, 1.571]` | forearm roll     |
| 6     | `(0, -1, 0)` | `(0, 0, -0.1205)`          | `[-0.785, 0.785]` | wrist            |
| 7     | `(1, 0, 0)`  | `(0, 0, 0)`                | `[-1.571, 1.571]` | wrist            |

Right arm mirrors: joint 1 axis is `(0, -1, 0)`, joint 6 axis is `(0, 1, 0)`, ranges
negated. Joints 2/3/4/5/7 keep the same axis signs.

**Derived facts that matter:**

- **All consecutive joint axes are mutually perpendicular.** ✅ (SEW-Mimic prerequisite)
- **Upper arm = 153.75 mm**, from joint 3's origin to joint 4's origin, along `(0,0,-1)`
  — **collinear with joint 3's axis**. ✅ Joint 3 is an exact upper-arm-direction proxy.
- **Forearm = 216 mm** (0.0955 + 0.1205), joint 4's origin to the wrist centre, along
  `(0,0,-1)` — **collinear with joint 5's axis**. ✅ Joint 5 is an exact
  forearm-direction proxy.
- **Wrist is *functionally* spherical.** Joints 6 and 7 pass through the `ee_base_link`
  origin; joint **5 does not** — it sits 120.5 mm up the forearm. (An earlier draft of
  this section claimed all three did.) It behaves as a spherical wrist anyway, because
  that offset lies along joint 5's *own* axis, so `q₅` cannot move the wrist centre.
  `just sew-stage3-check` asserts it that way round, since that is the property
  anything downstream actually uses.
- **Shoulder is *not* spherical**: joints 1 and 2 intersect (their 0.06 offset lies
  along joint 1's own axis), but joint 3's axis is offset **66.25 mm** from that
  intersection. This breaks textbook closed-form S-R-S IK — but it is *irrelevant*
  to orientation-only retargeting (§7).
- **`{side}_ee_control_point` sits at the wrist centre**, not the gripper tip. Fingers
  extend to roughly `(-0.0014, ±0.018, -0.068)` from it.
- **Joint 7's axis is `x`, but the gripper approaches along `-z`.** So joint 7 is *not*
  roll-about-approach-axis; joint 5 is. This matters for SEW-Mimic stage 3 (§7.4).

### 3.1 The embodiment mismatch, quantified

|               | upper arm | forearm  | ratio    |
| ------------- | --------- | -------- | -------- |
| OpenArm v2    | 153.75 mm | 216 mm   | 1 : 1.40 |
| typical human | ~0.186 H  | ~0.146 H | 1 : 0.78 |

The robot's forearm is proportionally **~1.8× longer** relative to its upper arm.

**Consequence:** under this mismatch it is *mathematically impossible* to
simultaneously match the human's wrist position and elbow position with any scaling.
Position-based retargeting must trade one against the other. Orientation-based
retargeting matches both limb *directions* exactly and lets positions fall where the
robot's own link lengths put them. This is the single strongest argument for §7.

---

## 4. Rubric mapping

`problem.pdf` grades four things. Where each fix lands:

| Criterion                                                                                                        | Current state                                 | Fixed by                           |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------- |
| **Retargeting accuracy** — "arm configuration ... **including elbow placement**, not just end-effector position" | Elbow entirely unconstrained                  | §6.1 (elbow task), §7 (SEW)        |
| **Tracking accuracy** — Cartesian error, achieved EE pose vs commanded                                           | Not measured; no orientation commanded at all | §6.4 (metrics), §7.4 (wrist stage) |
| **Latency** — end-to-end, camera to command                                                                      | Not measured                                  | §6.4, §10                          |
| **Framerate** — perception + control, sustained                                                                  | 30 Hz / 62.5 Hz, no interpolation             | §6.3                               |

Also explicitly required and currently missing/weak:

- *"expose an explicit intermediate retargeting target ... logged every timestep and
  invariant to the operator's body size"* → §11. We have something size-invariant
  (`arm_extension`) but it is not logged, not in robot units, and too thin (position only).
- *"Lightweight landmark pipelines are usually not sufficient; if you use one,
  justify it against the alternatives"* → §9. We need the written justification even
  if we keep MediaPipe.
- *"what breaks first as motion gets faster"* → §6.4 must actually measure this.

---

## 5. The harness — BUILT

You cannot tune jitter by waving at a webcam — the input is never the same twice.
Every change below gets A/B'd on a fixed battery instead.

### 5.1 What exists

**Record** — [`app/record.py`](backend/app/record.py) dumps every incoming
websocket payload to JSONL at the socket boundary, before retargeting. It stores
the **raw payload dict**, not a parsed `PoseFrame`: old clips stay replayable as the
schema grows (new fields are optional, so an old clip merely lacks them), and a
recording can never be corrupted by a bug in the code it exists to test.

Driven from the page — clip dropdown + record button, bottom-left, with a 3-second
countdown to get back into frame and a live frame counter so a take that lost
tracking is obvious before you move on. Also `POST /record/start|stop` for curl, and
`POSE_RECORD=<name>` to start one clip at boot.

**Replay** — [`scripts/replay.py`](backend/scripts/replay.py) feeds a clip through
the *same* `SimState` the live server drives, no webcam and no browser. Physics
advances in lockstep with the clip's own capture timestamps rather than the wall
clock, so every retargeting and tracking number reproduces exactly and an A/B
difference is attributable to the code change alone. (`solve_ms` is a wall-clock
benchmark and varies; `--realtime` gives up determinism to be watchable next to
`--viewer`.)

**Battery** — defined once in [`app/battery.py`](backend/app/battery.py), served to
the page from `GET /record/battery`, with each clip recording *what it probes*.
Eight clips: the seven originally listed here plus `hand_roll` for the new stage-3
input. **Not yet recorded** — that needs a person and a camera; see
[`fixtures/README.md`](backend/fixtures/README.md).

```bash
just replay fixtures/fast_wave.jsonl   # one clip + summary table
just battery before                    # whole battery -> backend/runs/before.json
#  ... make a change ...
just battery after
diff backend/runs/before.json backend/runs/after.json
```

### 5.2 Metrics it reports

The §6.4 set minus what needs code that doesn't exist yet (nothing commands an
orientation, so there is no orientation error to report). Two worth knowing:

- **`target_delta`** — L2 change in `qpos_target` between consecutive pose frames,
  in radians, measured *before* physics. A property of the retargeter alone, so it
  is the cleanest jitter number available and the one that should collapse when the
  1€ filter lands.
- **`qvel_rms`** — jitter as the servos actually see it, controller included.

A lock-up (§2.1) reads as tracking error that rises and never recovers. Jitter lives
in `target_delta`'s **tail**, not its mean, which is why p95 and max sit next to it.

### 5.3 What to read off the first baseline

Record the battery, run `just battery baseline`, and three sections below get their
number before any of them is built:

- **`torso_rotate` measures §6.2 directly.** Hold the arms rigid relative to the
  torso and rotate. A torso-framed pipeline would emit a *constant* target — δ = 0 —
  so whatever `target_delta` this clip reports **is** the cost of camera-frame
  mapping, in radians. §6.2 should drive it to near zero, which makes this the
  cleanest pass/fail in the battery.
- **`elbow_occluded` vs `reach_forward_slow`** is the same motion with and without a
  hidden elbow. The ratio of their `target_delta` is the case for §6.6's
  limb-length constraint, quantified against a control.
- **`cross_midline` left vs right.** The motion is symmetric, so the two arms' errors
  should be too. If they aren't, that is a lead on §15 Q2's axis-sign question —
  though the arms also differ in joint-limit ranges, which could produce the same
  signature, so confirm before concluding.

Record `reach_forward_slow` first and check it looks sane before shooting the other
seven; it is the easiest motion to perform and the easiest baseline to reason about.

---

## 6. Low-hanging fruit (do these regardless of which big solution wins)

Ordered by payoff ÷ effort. Each is independently shippable.

### 6.1 Add an elbow task — ~half a day, biggest single win

Kills the null-space drift, recovers joint 3, and directly serves the rubric's
"elbow placement" criterion. **No XML edit needed** — the elbow centre is already
available as `data.xanchor[joint4_id]`.

```python
# retargeter.py — sketch
elbow_target = shoulder_world + UPPER_ARM_LEN * unit(human_elbow - human_shoulder)
# stack the two Jacobians and solve once, don't solve them sequentially:
#   J = [ jac_wrist ; jac_elbow ]        (6 × nv)
#   e = [ wrist_err ; W_elbow * elbow_err ]
# W_elbow ≈ 0.3–0.5 so the wrist still wins when they conflict.
```

Two important details:

- Weight the elbow **below** the wrist. Because of the 1:1.40 link ratio (§3.1) the
  two targets are never simultaneously satisfiable; you want a soft preference, not a
  fight.
- Use `UPPER_ARM_LEN = 0.15375` (the robot's), not a scaled human length — this is
  what makes the elbow target reachable by construction.

*Verify:* on the battery, log the elbow position error and the variance of joints 3
and 5 over a static hold. Both should drop hard.

### 6.2 Torso-frame the landmarks — **DONE**

Built as [`mapper.torso_frame`](backend/app/retarget/mapper.py) and composed into the
SEW path in `Retargeter._solve_sew`, which now applies `base_R.T @ torso.T` rather
than `base_R.T` alone. Returns `None` — camera frame, warn once — when the hips are
absent or the torso is too collapsed to define a basis, so a client sending only the
six arm landmarks still works.

The basis, exactly as sketched below:

```python
# mapper.py
y_t = unit(left_shoulder - right_shoulder)          # across the chest
z_t = unit(shoulder_mid - hip_mid)                  # up the spine
x_t = unit(cross(y_t, z_t))                         # out of the chest
z_t = cross(x_t, y_t)                               # re-orthogonalize
R_torso = column_stack([x_t, y_t, z_t])
v_torso = R_torso.T @ v_camera
```

This makes retargeting invariant to where the operator stands and which way they face
— and it is a **hard prerequisite** for SEW-Mimic's optimality guarantee (§7.5).

**Measured**, as the new `torso invariance` block in `just sew-retarget-check`: hold
one pose relative to your own body, rotate the whole subject, and compare the solved
joints. The check is only meaningful because it fails loudly without the frame —
stubbing `torso_frame` to `None` reproduces the old behaviour exactly:

| subject moves | camera frame         | torso frame |
| ------------- | -------------------- | ----------- |
| yaw 35°       | 51.0° of joint error | 1.3e-14°    |
| lean back 25° | 25.0°                | 1.3e-14°    |
| tilt 20°      | 16.3°                | 1.3e-14°    |
| all three     | 16.3°                | 3.8e-14°    |

A lean of 25° producing exactly 25° of joint error is the signature of the bug: the
camera-frame path rotates `u` and `l` bodily with the operator, so the robot's arms
swing while the operator's arms have not moved at all. The `torso_rotate` clip (§5.3)
is still worth shooting to confirm it on real landmarks, where the shoulder and hip
landmarks carry their own noise.

### 6.3 1€ filter — **DONE** · slew limiting + interpolation — still to do

**The filter** is [`retarget/smoothing.py`](backend/app/retarget/smoothing.py)
(`LandmarkFilter`), applied at the top of `Retargeter.solve` so the live path, a
replay and every check script measure the same pipeline. It runs on the six arm
landmarks **and the two hips** — the hips build the torso frame, making them the one
landmark whose jitter is not confined to one arm. Hand landmarks are left alone:
nothing consumes them yet, and they come and go frame to frame, which is a gating
problem rather than a filtering one.

Landmark space rather than joint space, for a stronger reason than "it's upstream":
`Stage12Solver.solve` picks its branch by a **discrete** comparison scored on the
*unsmoothed* solution, so joint-space filtering could only smooth a branch flip after
it had been decided. Filtering the input changes the decision.

**The parameters are measured, not inherited** — `just filter-check` grids them
against both ends of the trade. Three results worth carrying into the design
document, because two of them contradict the obvious choice:

| min_cutoff / beta / d_cutoff         | jitter p95, held pose | tracking lag, 175°/s sweep |
| ------------------------------------ | --------------------- | -------------------------- |
| off                                  | 8.86°                 | 15.0 ms                    |
| 1.0 / 0.007 / 1 — as suggested above | 1.16°                 | **134.4 ms**               |
| 0.5 / 2 / 1                          | 0.79°                 | 68.2 ms                    |
| **0.5 / 8 / 3 — shipped**            | **2.02°**             | **30.1 ms** (+15.1)        |
| 0.5 / 16 / 3                         | 2.90°                 | 23.2 ms                    |

1. **`beta = 0.007` is wrong here by three orders of magnitude.** It comes from
   normalized/pixel-space work; our landmarks are metres (§6.5), so `beta * speed`
   never reaches a tenth of a Hz and the filter degenerates into a fixed 1 Hz
   low-pass — same jitter as the shipped setting, **9× the lag**.
2. **A low `min_cutoff` with a high `beta` beats a middling both.** Every point on the
   measured frontier sits at the bottom of the cutoff range. Smooth hard when still,
   get out of the way entirely when moving.
3. **`d_cutoff` wants 3 Hz, not the paper's 1 Hz.** The cutoff can only open once the
   speed estimate has noticed the motion, and at 30 fps a 1 Hz estimate needs ~5
   frames — a third of a half-second reach, spent lagging. Raising it takes the
   worst-case error at motion onset from 12.2° to 9.1°.

Downstream effect on `just sew-retarget-check`, same test as §0.4's last row: held
still under landmark noise, p95 frame step went **5.9° → 2.3°** at full stretch and
**5.4° → 1.4°** at the well-conditioned pose, worst single frame **23.5° → 18.6°**.

`LANDMARK_FILTER=0` disables it and `LANDMARK_MIN_CUTOFF` / `LANDMARK_BETA` /
`LANDMARK_D_CUTOFF` override the rest, so the whole thing A/Bs live without a rebuild.

**Now due for a retune: `OBSERVABLE_COS_Q2` / `OBSERVABLE_SIN_Q4`.** `_trust` is itself
a data-dependent low-pass in joint space, and those bands are sized for a few degrees
of *raw* landmark noise. With a clean input they are conservative, and near a
singularity the two filters compound into lag.

**Still to do — interpolate `qpos_target`** across substeps in `SimState.step()`
instead of holding a step function ([`main.py:70`](backend/app/main.py:70) reads it
once per tick and holds it for all 8 substeps, so a kp=230 servo eats a position step
every ~33 ms), and clamp per-joint velocity. ~3 rad/s is the sketch, but that is
170 °/s and the sweep above is 175 °/s of *upper arm* — measure before pinning it, and
expect the shoulder joints to want a looser limit than the wrist.

MediaPipe's own `smooth_landmarks` is not a concern: the option existed in the legacy
Solutions API and has no equivalent in `@mediapipe/tasks-vision`'s `PoseLandmarker`,
so there is nothing to double-filter against.

**What the filter cannot do**, and the reason it is only half the answer to "the arm
moves randomly when the input misbehaves": 1€ is speed-adaptive *by construction*, so a
landmark that teleports is indistinguishable from fast motion and is passed through
with **less** attenuation than steady noise. Jitter is the half it owns. A glitching or
absent input needs a plausibility gate and a defined rest behaviour — see §6.7.

### 6.4 Metrics + logging — ~half a day, required by the rubric

Log one JSONL record per control tick containing the §11 intermediate target plus:

- `t_capture`, `t_recv`, `t_solved`, `t_applied` → latency histogram per stage
- upper-arm and forearm **direction error** (degrees) between command and human
- elbow position error (after scale normalization)
- **achieved vs commanded EE pose** — this *is* the rubric's tracking accuracy
- joint velocity / acceleration RMS → the jitter number
- perception fps and control fps, sustained

Then answer *"what breaks first as motion gets faster"* with data instead of a guess.
Candidates to rank: MediaPipe confidence collapse, 1€ filter group delay,
position-servo tracking error at `kp=230`, or the 30→62.5 Hz staircase.

### 6.5 Use MediaPipe world landmarks — **DONE**

`PoseFrame` now carries `worldLandmarks`: **metres, hip-centred, metric**, sent from
[`socket.ts`](frontend/src/pose/socket.ts) for the arm landmarks, the hips and both
hands. It does not fix depth accuracy — BlazePose's depth is still weakly supervised —
but it removes two confounds that were first-order, not cosmetic:

1. **Aspect-ratio anisotropy.** Normalized `x` is divided by frame width and `y` by
   frame height, so on the 4:3 capture a vertical displacement was inflated 1.33x
   against an identical horizontal one. Every direction derived from them was tilted
   toward vertical.
2. **The pseudo-`z`.** Documented only as "roughly the same scale as x", and measured
   live it was far worse than that: on a real hand-toward-face frame the landmarks
   gave a forearm **1.8x the length of the upper arm**, which is anatomically
   impossible.

That second one was the bug behind "the hand never comes near the face". The elbow
angle *is* the angle between `u` and `l` — `q₄ = arccos(-v[2])`, no policy involved —
so a depth scale error lands on it directly:

| that frame's `z`                              | solved `q₄`                                 |
| --------------------------------------------- | ------------------------------------------- |
| as sent (normalized)                          | **70°** — matches the 66.7° in the live log |
| × 0.5                                         | 104°                                        |
| × 0.25                                        | 128°                                        |
| re-derived from a constant limb length (§6.6) | **132°**                                    |

A 70° elbow with the shoulder pointing where the operator's points puts the robot's
hand out in front of the chest. **The solver was reproducing the pose it was given
exactly; the pose was wrong.** Worth remembering as a correction to §0.5's line that
orientation retargeting demotes depth to a second-order error — true of the *target
position*, false of the *included angle between the two limbs*, which is what the
elbow is.

Note this is separate from §6.6, and §6.6 is still worth doing: world landmarks fix
the *scale* of depth, not its accuracy.

### 6.6 Enforce constant limb length — ~half a day

We know the operator's upper-arm and forearm lengths after a brief calibration
(the spec explicitly allows calibration). Enforce them on every incoming frame.

[Seeing Through Occlusion](https://arxiv.org/html/2606.19240v1) does exactly this
geometrically: reconstruct an occluded joint's depth from its reliable 2D position
plus the known limb length (Pythagoras → up to 4 candidates, ranked by spatial
consistency + temporal smoothness + anatomical plausibility). **11 ms** overhead
versus 213 ms for optimization baselines, and it reports >100 cm RMSE reduction under
severe occlusion versus Kalman filtering alone.

This is the cheapest real answer to monocular depth ambiguity — cheaper and more
robust than bolting a depth model on (§9.3).

**Add a T-pose calibration step** at startup: it gives you limb lengths, a
shoulder-width scale, and a torso frame reference, and the rubric asks for a
documented startup procedure anyway.

### 6.7 Plausibility gate + rest behaviour — ~1.5 hours, the other half of §6.3

§6.3 and the slew limit are both *smoothers*: each one passes a sustained bad
measurement through, only more slowly. Neither decides that an input should not be
followed at all, and nothing anywhere defines where the arm goes when there is no
trustworthy input. That is the whole of the "moves randomly when the input
misbehaves" symptom, and it is a detection problem.

What is missing today, concretely:

- **No confidence signal reaches the backend.** `toPayload`
  ([`socket.ts`](frontend/src/pose/socket.ts)) sends `x, y, z` and drops MediaPipe's
  `visibility` / `presence`. §10 flagged this; it is still true.
- **The only "hold" path is a degeneracy that never fires.** `limb_directions` returns
  `None` only when a limb measures under 1 mm ([`mapper.py`](backend/app/retarget/mapper.py)).
  When the subject leaves frame or an arm is occluded, the lite model emits a
  *confident, anatomically plausible, wrong* skeleton, which sails through that test
  and is retargeted faithfully.
- **No liveness timeout.** If frames stop, `qpos_target` keeps its last value forever
  and the physics loop keeps driving to it. `requestAnimationFrame` stops when the tab
  is hidden, so backgrounding the page freezes the arm mid-pose indefinitely.
- **`rest` is not a rest behaviour.** `Stage12Solver.rest` pulls only the *unobservable*
  angles, scaled by `(1-trust)²` — it is for the singular directions, not for "the
  input is not to be trusted".

The three pieces:

1. **A gate producing a per-arm `trust ∈ [0,1]`**, from per-landmark visibility, a
   limb-length constancy test (§6.6's constraint used as a *detector* — no calibration
   needed, since both limbs' lengths should simply be constant), and a per-frame
   direction-velocity test (>25–30° of limb-direction change in one 33 ms frame is
   above human joint speed). A rejected frame contributes *nothing*, unlike a filter,
   which contributes a lagged something.
2. **A tracking state machine with hysteresis.** `TRACKING` → `DEGRADED` (hold pose —
   the `ArmSolve(held=True)` path already is this) → `LOST` (no frame in >250 ms, or N
   consecutive rejects): ramp `qpos_target` toward the neutral posture over ~1 s.
   Require k consecutive good frames to leave `LOST`, or a flickering detection
   flickers the arm.
3. **Order: landmarks → gate → 1€ → solve → slew limit → interpolate.** Gate *before*
   filter, not after: a bad sample admitted into a 1€ filter lives in its state and
   pollutes several following frames. Feed `trust` into the filter's cutoff (§10)
   rather than hard-gating, so a marginal frame is smoothed harder instead of stepped
   over.

The rest ramp needs the slew limiter from §6.3 to exist first — it is the same rate
limit, used for a different reason.

---

## 7. The main event: SEW-Mimic

**[A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot
Teleoperation](https://arxiv.org/abs/2602.01632)** (arXiv 2602.01632).
[HTML version](https://arxiv.org/html/2602.01632).

### 7.1 Why this is the right fit

It reframes retargeting from *"put the end-effector here"* to *"make the robot's
limbs point where the human's limbs point"*. Consequences that map onto our exact
problems:

| Our problem                     | What SEW-Mimic does                                                                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §2.1 null-space drift           | No null space. All 7 joints are determined by the 3 alignment objectives. Closed form, no iteration, **no state carried between frames** → no drift possible.         |
| §2.3 missing joint 3            | Joint 3 *is* the upper-arm alignment variable. Solved for directly, by construction.                                                                                  |
| §3.1 link-ratio mismatch        | Orientation-only, so link lengths never enter. Calibration-free across body sizes.                                                                                    |
| §3 shoulder 66 mm offset        | **Irrelevant** — the algorithm steers axis *directions*, and a positional offset does not change an axis direction. This sidesteps the offset-S-R-S problem entirely. |
| latency budget                  | 0.33 ms per arm, 3 kHz. Two to three orders of magnitude faster than GMR / xr_teleoperate.                                                                            |
| midline crossing self-collision | Built-in XPBD safety filter: ~50% → ~1.3% self-collision, still ~250 Hz bimanual.                                                                                     |

Reported alignment error is near machine precision (~1e-13), with a user study (N=8,
p<0.01) showing improved task success and ~3× higher downstream policy success rate
versus MINK-IK demonstrations.

### 7.2 The objective

```
min_q   μ_c(u, R⁰³(q)·h₃)²  +  μ_c(l, R⁰⁵(q)·h₅)²  +  μ_m(T(q), H)²
s.t.    q within joint limits

u = unit(elbow − shoulder)      human upper arm       (torso frame)
l = unit(wrist − elbow)         human forearm         (torso frame)
H = human hand orientation                            (torso frame)
h₃, h₅ = joint 3 / joint 5 axes in their local frames
μ_c(u,v) = ½ − ½·(u·v)/(‖u‖‖v‖)          ∈ [0,1]   vector cosine error
μ_m(R₁,R₂) = ½‖(R₁ᵀR₂)^½ − I‖_F          ∈ [0,1]   chordal-ish rotation error
```

### 7.3 The algorithm — three sequential stages

Each stage is solved in closed form with Paden–Kahan subproblems.

**Stage 1 — upper arm (solves joints 1, 2).** `AlignAxis(3, q, u)`: rotate joints 1
and 2 so that joint 3's axis points along `u`. This is **Subproblem 2** (intersection
of two cones): up to 2 solution pairs, filter by joint limits, pick the one closest to
the current configuration.

**Stage 2 — forearm (solves joints 3, 4).** `AlignAxis(5, q, l)`: same machinery, one
link down. Joint 3 (the humeral roll we currently ignore) and joint 4 (elbow flexion)
fall out together.

**Stage 3 — wrist (solves joints 5, 6, 7).**
`R⁰⁷_des = H · R_alignᵀ · (R⁷_local)ᵀ` — read right to left as "peel the constants off
the goal": the tool frame must sit where the hand is, and the tool frame is joint 7's
frame plus a fixed axis relabelling (`R_align`) and a fixed mount (`R⁷_local`), so what
the *joints* must deliver is the goal with both stripped off. Then align joint 7's axis
via Subproblem 2 on joints 5–6, and **Subproblem 1** (circle-point) for joint 7's
residual roll. On OpenArm both constants are identity and the two subproblems collapse
into one Euler decomposition — §7.8.

Optimality (Proposition 3) holds when consecutive axes are perpendicular ✅, the
humanoid base-link structure matches ✅, and the input is in a **body-centric frame**
(§6.2 — this is a hard requirement, not a nicety).

### 7.4 OpenArm v2 adaptation notes — read before implementing

Three concrete gotchas, all checked against the XML:

1. **Stage 3 is the paper's *perpendicular wrist* case — DONE, see §7.8.** An earlier
   draft of this note said to keep Algorithm 3's Subproblem-2 structure and swap the
   column index to 0, since joint 7's axis is `x` while the gripper approaches along
   `-z` (§3). That works, but it misses that the paper handles this exact geometry
   separately, in its appendix §C, and prescribes **Euler decomposition instead of
   SP1/SP2**. On OpenArm that collapses to a plain intrinsic Z-Y-X — no subproblem
   library at all, the same way stages 1–2 collapsed to trigonometry.

2. **Hand orientation `H` — DONE.** MediaPipe Pose alone can't give it; a wrist
   landmark is a point, and a point has no rotation. Staged plan:
   - **Phase A** — skip stage 3 entirely; hold joints 5/6/7 at a fixed neutral posture.
     Stages 1–2 alone already fix §2.1, §2.3 and the elbow criterion. Ship this first.
   - **Phase B — DONE on the input side.** See §7.7. `H` is measured, logged and
     recorded on every frame.
   - **Phase B′ — DONE, consuming it.** See §7.8.
   - **Phase C** — [HaMeR](https://github.com/geopavlakos/hamer) or WiLoR for a proper
     MANO wrist rotation, if latency budget allows. §9.4's measurement decides whether
     this is needed; the `hand_roll` clip is what makes that call.

3. **The robot arms hang along `−z` at `qpos=0`, mounted at `(0, ±0.031, 0)`.** Get
   the base-frame convention and the left/right axis sign differences (joints 1 and 6,
   §3) right *once*, in a unit test, before debugging anything visually.

### 7.5 Implementation plan

```
backend/app/retarget/
  sew.py            # all three stages — no subproblem library needed, see §0.2, §7.8
  mapper.py         # torso frame + hand frame (§6.2, §7.7)
  safety.py         # XPBD capsule filter (phase 4, optional, not built)
```

The original plan here listed a `subproblems.py` for Paden–Kahan SP1/SP2 and a separate
`frames.py`. Neither was needed: **all three stages closed-form out to elementary
trigonometry on this arm**, and the frame construction was small enough to live beside
the landmark schema it is built from.

**Phase 0 — validate the geometry.** Write a test that, for a known `q`, does forward
kinematics in MuJoCo and confirms joint 3's axis is collinear with the
link3→link4 vector and joint 5's with link4→wrist. If §3's claims don't hold in
`mj_forward`, stop and re-derive before writing any solver.

**Phase 1 — SP1 and SP2.** Either hand-roll them (~120 lines) or use
[`linearSubproblemSltns`](https://pypi.org/project/linearSubproblemSltns/) (pure
Python, from the Elias & Wen linear-algebra formulation). Unit-test against random
axes with known answers.

**Phase 2 — stages 1 and 2 only**, joints 5–7 held neutral. Run the battery. This
alone should eliminate the lock-ups and fix the raise-arm-to-the-side case.

**Phase 3 — stage 3** once hand orientation exists (7.4 #2). **Done — §7.8.** Phase 1
turned out to be unnecessary for it: the perpendicular-wrist form needs no subproblem
solver either.

**Phase 4 — the XPBD safety filter** for midline crossing, if self-collision shows up
in the battery. Overapproximate each link with a capsule
`{x : ‖x − t·p₁ − (1−t)·p₂‖ ≤ r, t ∈ [0,1]}`, resolve collisions on the SEW keypoints
with Lagrange multipliers, re-project to preserve link lengths, then re-run
SEW-Mimic on the corrected keypoints.

### 7.6 Known risk

**No public code release found** as of this writing — the paper mentions
"open-source, standalone applications" but no repo surfaced in search. Budget for a
from-scratch implementation.

> **Largely mitigated — see §0.2.** For OpenArm v2 specifically, stages 1–2 reduce to
> ~25 lines of closed-form trigonometry with no subproblem library at all, verified
> exact against `mj_forward`. The general Paden–Kahan machinery is only needed if you
> want stage 3 or portability to another arm. Phase 2 is ~2 hours, not 1–2 days.

Recheck for a repo before starting — check the arXiv page, the authors' pages
(Zhenyang Chen, Georgia Tech), and [awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning).

### 7.7 Hand orientation — BUILT (stage 3's input, not stage 3)

[MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
runs in the browser next to the pose model, on the same video frame and the same
timestamp ([`hands.ts`](frontend/src/pose/hands.ts)). Six of its 21 landmarks are
sent: `wrist`, `index_mcp`, `middle_mcp`, `pinky_mcp`, and `thumb_tip` / `index_tip`
for the later gripper question (§15 Q5) — carried now because re-shooting the
battery to add two landmarks would be absurd.

[`mapper.hand_orientation()`](backend/app/retarget/mapper.py) turns those into
SEW-Mimic's `H`, a rotation matrix in robot axes, from two directions:

```
f = wrist -> middle_mcp      along the hand, toward the fingers
k = pinky_mcp -> index_mcp   across the knuckles, thumb-ward

x = ±(f × k), normalized     back-of-hand normal (dorsal), both hands
z = -f                       matches the ee frame, whose gripper approaches along -z (§3)
y = z × x
```

Only `f` is trusted for magnitude — `k` enters solely through a cross product, so
the knuckle line's own length (noisiest of the two; the four MCPs are nearly
coincident at low resolution) never scales anything.

Three decisions worth carrying into the design document:

1. **The `±` is not a detail.** Human hands are true mirror images, so `f × k` points
   dorsally on one and palmward on the other. The side flip is what makes both
   matrices right-handed rotations with the same anatomical meaning — the same way
   the robot's two mirrored arms each still carry a right-handed ee frame.
2. **Handedness comes from the pose model, not MediaPipe's `handedness` field.**
   That field assumes a mirrored selfie-view image; our frames are unmirrored (the
   CSS `scaleX(-1)` is display-only). Each detected hand is assigned to the nearer
   of the two pose wrists instead — no convention to get wrong, in the frame we
   already have.
3. **Degenerate hands return `None`, not identity.** When the knuckle line collapses
   onto the finger direction the roll is genuinely unobservable — the hand-frame
   analogue of the straight-arm singularity in §0.2, and handled the same way:
   refuse to guess, hold the previous value.

[`scripts/check_hand_frame.py`](backend/scripts/check_hand_frame.py) (`just
hand-check`) validates it against properties rather than eyeballs — a sign error
here produces a perfectly well-formed rotation matrix describing the wrong hand,
and no amount of staring at a webcam overlay would show it. Round-trip over 300
random rotations per side recovers `R` to 1.8e-15; mirroring a right hand's
landmarks and reading them as a left hand yields exactly `M·R·M`; all three
degenerate cases return `None`.

**Consumed by stage 3 as of §7.8.** The `hand_roll` battery clip exists to measure
whether it is *good enough*, which is still open (§9.4).

### 7.8 Stage 3 — BUILT

[`sew.py`](backend/app/retarget/sew.py) (`wrist_euler_signs`, `solve_wrist`,
`WristSolver`) and `Retargeter._wrist_target` / `_solve_sew`. `just sew-stage3-check`,
34 properties, both arms.

**The finding that made it small, exactly parallel to §0.2's.** OpenArm is the paper's
*perpendicular wrist*, so its appendix §C applies: Euler decomposition rather than
SP1/SP2. And because no body in the arm chain carries a `quat`, every `R_local` is the
identity and the wrist factors into elementary rotations:

```
R⁴⁷ = Rz(−q₅) · Ry(s·q₆) · Rx(q₇),     s = −1 left, +1 right
```

Verified against `mj_forward` to **7.8e-16** over 200 random full-range configs per
arm. The signs are read off `m.jnt_axis` at startup — the paper's eqs (16)–(18) — and
`wrist_euler_signs` **raises** rather than guessing if the model stops matching.

The first column of `R⁴⁷_des` is where joint 7's axis must point, so `q₅,q₆` aim it and
`q₇` rolls about it. That *is* Algorithm 3's structure (SP2 then SP1), written out.

**Both of the paper's constants are identity here, and both are still in the code.**
`R⁷ᵀ_local = I` because the control point site carries no rotation and sits on joint
7's origin (measured at startup, not assumed). `R_align = I` because
[`hand_orientation`](backend/app/retarget/mapper.py) was written to emit `H` in the
robot's own ee convention — the alignment was paid for in the perception layer. So
`R⁰⁷_des = H R_alignᵀ (R⁷ᵀ_local)ᵀ` reduces to `H`. Named rather than dropped: they
are identity for reasons that a model change or a convention change would break.

**Two structural gifts, both asserted in the check:**

- **No gimbal lock.** The decomposition degenerates at `q₆ = ±90°` — the paper's
  Remark 5, which argues it is *rare*. Joint 6 stops at **±45°**, so it is not rare, it
  is unreachable. Stronger than the paper's own guarantee.
- **No second branch.** The other Euler solution needs `|π − b| ≥ 135°`, also outside
  joint 6's range. So stage 3 is **stateless wherever the target is reachable**.

**The one real problem, which the check caught and the derivation did not predict.**
`q₅` and `q₇` both come from `arctan2`, and an `arctan2` angle *wraps* — while out of
range. At the far side of the wrap it flips between `+π` and `−π`, and the two joint
stops are equally close and equally wrong. Clamping alone picks whichever side it
landed on and **slams the wrist 180°**. `_stops()` therefore returns *both* stops when
an angle doesn't fit, and all candidates are scored with the same bounded-tolerance
tie-break the stage-1/2 branch policy uses (`WRIST_SWITCH_TOLERANCE`, 15°). Measured,
parked on a switch point under 5° of hand-orientation noise:

| | 180° flips / 2000 frames | p95 step |
| --- | --- | --- |
| clamp only | 508 | 180° |
| both stops + tie-break | **16** | **14.0°** |

Worth being straight about the limit: on a *monotone* sweep the tie-break changes
nothing (4 switches either way). It suppresses **chatter**; it cannot prevent a
crossing that genuinely happens. A 180° command change in the unreachable region is
real and remains, which is what makes the slew limiter (§6.3, still unstarted)
load-bearing rather than cosmetic. The check asserts continuity where the target is
reachable and *bounds the count* of switches where it isn't, rather than pretending to
zero.

**Where it ended up:**

| | result |
| --- | --- |
| reachable pose, `q → H → q` round trip | `3.8e-14 deg` |
| reachable pose, achieved orientation vs `H` | `5.1e-06 deg` |
| stage 3 disturbs stages 1–2 by | `1.5e-06 deg` (exactly zero in exact arithmetic) |
| reachable sweep, worst frame step | `0.14 deg` |
| uniform random SO(3) | 16.3% exact, median residual 36.3° |

That last row is the honest one and it is **not** a solver limitation: joint 6 spans
±45° against a human wrist's ~±70° of flexion, so clamping is routine, not
exceptional. `Wrist.clamped` surfaces it per joint so telemetry can tell "the retarget
is wrong" from "the robot doesn't bend that way". Uniform SO(3) is also a pessimistic
prior — real hand orientations correlate strongly with the forearm stage 2 already
matched. Two failure modes worth distinguishing, and the telemetry reports both:
saturating `q₅` costs **roll** and leaves the approach axis exact, while saturating
`q₆` costs the **approach direction** itself, which is the one a viewer notices.

**Policy for a lost hand.** MediaPipe Hands drops out constantly where the pose model
does not, so a wrist that decays on the first miss would sag and snap back with every
flicker. `WristSolver.hold()` keeps the pose through `WRIST_HOLD_FRAMES` (5, ~0.17 s)
and only then eases toward rest at `REST_RELAXATION`. A degenerate hand frame
(`hand_orientation` returning `None`) takes the same path, per-arm and independently.

**Still to do:** the hand landmarks are not filtered at all — note that §6.3's
`smoothing.py` does not exist in this tree, so this is downstream of the 1€ filter
landing rather than an extension of it — and the hand frame's own observability — `|f × k|`,
how well the knuckle line fixes the roll — is measured but thrown away at a hard
threshold rather than fed in as a shrinkage weight. That is §0.4's lesson 5 waiting to
be applied one stage down: `z_H = −f` exactly, so `H` is fully determined by `f` plus
one roll angle about it, and only that angle is poorly conditioned.

---

## 8. Alternative and complementary tracks

### 8.1 mink — differential IK as a QP *(strong fallback / complement)*

[kevinzakka/mink](https://github.com/kevinzakka/mink) · [docs](https://kevinzakka.github.io/mink/)

Near-drop-in replacement for `ik.py`, built directly on MuJoCo, DAQP solver, kHz rates.
Fixes §2.1, §2.2 and §2.4 in one change:

- `FrameTask` — wrist, with **separate position and orientation weights**
- `FrameTask` / position task — elbow (§6.1)
- `PostureTask` — weak pull toward a reference configuration, which is the principled
  way to anchor the leftover null space
- `ConfigurationLimit` — joint limits as **inequality constraints**, not clipping
- `VelocityLimit` — smoothness for free
- `CollisionAvoidanceLimit` — midline crossing

**How this relates to SEW-Mimic:** they're complementary, not competing. SEW-Mimic
gives you a drift-free, closed-form seed that is correct by construction; mink can
refine it against extra objectives (exact wrist position, collision avoidance) with
the seed as its posture reference. If SEW-Mimic implementation stalls, mink alone
still gets you most of §6.1's benefit — and it's ~a day of work with a real library
behind it.

[Pink](https://github.com/stephane-caron/pink) is the Pinocchio equivalent if you
ever want off-MuJoCo.

### 8.2 Stereographic SEW angle — the *position + swivel* alternative

[Elias & Wen, "Redundancy parameterization and inverse kinematics of 7-DOF revolute
manipulators"](https://arxiv.org/abs/2307.13122) ·
[code: rpiRobotics/stereo-sew](https://github.com/rpiRobotics/stereo-sew) ·
[IK-Geo](https://github.com/rpiRobotics/ik-geo)

A generalized SEW angle with a **provably minimal singularity set** (one direction in
the workspace; they prove *some* singularity is unavoidable for any parameterization).
The conventional swivel-angle formulation is singularity-prone — if you go the
"measure the swivel angle from the elbow and feed it to analytic IK" route, use *this*
parameterization, not the textbook one.

**Choose between the two SEW approaches deliberately:**

|                           | SEW-Mimic (§7)                            | Stereographic SEW IK (§8.2)                                                      |
| ------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
| matches                   | limb **orientations**                     | wrist **position** + swivel angle                                                |
| EE position               | wherever robot link lengths put it        | exact                                                                            |
| embodiment mismatch       | immune                                    | must scale workspace, can be unreachable                                         |
| our 66 mm shoulder offset | irrelevant                                | needs the offset-aware variant                                                   |
| rubric fit                | "arm configuration ... elbow placement" ✅ | "Cartesian error" — but that's *achieved vs commanded*, i.e. controller tracking |

Read the rubric carefully: *"Tracking accuracy — Cartesian error between the achieved
end-effector pose and the **commanded target**"* is about controller fidelity, not
retargeting fidelity. SEW-Mimic satisfies it trivially — forward-kinematic the
commanded joint vector to get the commanded pose. So the orientation-first framing
does **not** cost us on that criterion.

### 8.3 RelaxedIK family — explicit smoothness objectives

[RelaxedIK](https://github.com/uwgraphics/relaxed_ik) /
[relaxed_ik_core](https://github.com/uwgraphics/relaxed_ik_core) ·
[analysis paper](https://link.springer.com/article/10.1007/s10514-020-09918-9) ·
CollisionIK · RangedIK.

Weighted-sum nonlinear IK with **joint-velocity and jerk terms in the objective**,
designed explicitly for mimicry-based teleoperation. Rakita's motion-retargeting work
is nearly our problem statement verbatim. **RangedIK** adds tolerance ranges
("wrist roll is free within ±30°") — a principled way to *spend* redundancy rather
than letting it drift.

Worth reading for the design document even if we don't adopt it: it's the best
articulation of "what a better version of the smoothing component looks like",
which the rubric explicitly asks for.

### 8.4 GMR — General Motion Retargeting

[YanjieZe/GMR](https://github.com/YanjieZe/GMR) · [paper](https://arxiv.org/abs/2510.02252) (ICRA 2026)

SMPL/SMPL-X → humanoid, real-time on CPU, the retargeter behind TWIST. Only relevant
if we move to an SMPL-based perception front end (§9.2). SEW-Mimic benchmarks against
it and reports 1–3 orders of magnitude speedup as a drop-in replacement.

### 8.5 Learned swivel priors — the fallback when the elbow is occluded

[HL-IK](https://arxiv.org/pdf/2509.20263) (MLP: EE pose → human-like swivel),
LSTM-based NAPN, [hybrid biomimetic framework](https://www.mdpi.com/2313-7673/11/6/408),
[radial elbow offset IK](https://www.sciencedirect.com/science/article/abs/pii/S1474034622002701).

We can *see* the elbow, so measuring beats predicting. But these are the right answer
for the degraded case — when elbow visibility drops, blend from the measured swivel
toward a learned prior instead of holding the last value. Good "given more time"
material.

---

## 9. Perception track

The rubric warns: *"Lightweight landmark pipelines are usually not sufficient for
precise retargeting; if you use one, justify it against the alternatives."* We need
this written up regardless of what we ship.

### 9.1 The honest case for keeping MediaPipe

- Runs **in the browser**, zero server GPU, ~30 fps on a laptop
- The deployment story the assessment actually asks for (they run it on *their*
  hardware, live)
- With §6.2 (torso frame) + §6.3 (1€ filter) + §6.6 (limb-length constraint), its
  weak axis — depth — is largely constrained away
- **SEW-Mimic only needs limb *directions***, which are far better conditioned than
  absolute positions from a monocular view

That last point is the real justification: switching to orientation-based retargeting
makes the perception stack's weakest output (metric depth) matter much less.

### 9.2 If we upgrade: metric 3D pose

- **[NLF — Neural Localizer Fields](https://github.com/isarandi/nlf-pipeline)**
  (NeurIPS'24, [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/fd23a1f3bc89e042d70960b466dc20e8-Paper-Conference.pdf))
  — absolute 3D joints **in metres**, SOTA on EMDB. Its predecessor MeTRAbs is faster.
- **[SMPLest-X](https://www.researchgate.net/publication/396292526_SMPLest-X_Ultimate_Scaling_for_Expressive_Human_Pose_and_Shape_Estimation)**
  — leads on orientation metrics; heavier.
- **The structural argument**: SMPL-family models output **joint rotations**, not just
  positions. Shoulder roll comes out directly — no swivel inference, no IK for the
  shoulder at all. That's a genuinely different (and cleaner) pipeline shape.
- **Cost**: server-side GPU, +20–40 ms latency, and a much heavier setup story for
  "assume the person running it has not read your code."

### 9.3 Depth models — lower priority than it looks

[Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3),
[UniDepth](https://lpiccinelli-eth.github.io/pub/unidepth/), Metric3D v2.
[AugLift](https://arxiv.org/html/2508.07112v1) shows sampling a depth map at 2D
keypoints measurably improves lifting, ~13 ms with DA-v2-small.

**But**: we don't need *scene* depth, we need *joint* depth. §6.6 (limb-length
geometry) gets most of that benefit for a fraction of the cost and complexity, and
§9.2 gets it properly. Depth models are the *third*-best route here. Deprioritize.

### 9.4 Hands

MediaPipe Hand Landmarker is in and producing a hand frame (§7.7) — free,
in-browser, and the cheap answer first. Whether it is *good enough* is an open
measurement: the four MCP landmarks it derives the knuckle line from are nearly
coincident at webcam resolution, so the roll about the hand is the least
well-conditioned thing in the whole pipeline. Record `hand_roll` and look at the
frame-to-frame variance of `H` during a static hold before trusting it.

If it isn't good enough: [HaMeR](https://github.com/geopavlakos/hamer), WiLoR — MANO
mesh plus a proper **6-DoF wrist orientation** from monocular RGB, at GPU cost.

---

## 10. Filtering and rate matching

Beyond §6.3:

- **[SmoothNet](https://github.com/cure-lab/SmoothNet)** — learned temporal refinement
  over a window of poses; drops in after any estimator.
- **[N-euro predictor](https://jianwang-cmu.github.io/23Neuro/N_euro_predictor.pdf)** —
  a learned 1€ replacement that *predicts ahead*, partially cancelling pipeline
  latency rather than adding to it. Directly relevant to the latency criterion.
- **Where to filter** — landmark space, Cartesian target space, or joint space.
  Landmark space is the right default (§6.3). Document the choice; the rubric asks
  for the filtering rationale explicitly.
- **Confidence gating** — MediaPipe emits per-landmark visibility. Currently ignored.
  Feed it into the filter's cutoff (low confidence → heavier smoothing) rather than
  hard-gating, which would cause visible steps.

---

## 11. The intermediate retargeting target (rubric requirement)

*"Your pipeline must expose an explicit intermediate retargeting target — the desired
robot state, in the robot base frame, in robot units — sitting between perception and
control. It should be logged every timestep and invariant to the operator's body size."*

Current state: `arm_extension()` is size-invariant but is not logged, not in robot
units, and too thin (a single 3-vector per arm — which is exactly the under-determination
of §2.1, surfacing as an API design problem).

**Proposed schema** — one record per control tick, JSONL:

```python
@dataclass
class ArmTarget:
    # size-invariant core — this is what makes the target body-size independent
    upper_arm_dir: np.ndarray   # unit, torso frame
    forearm_dir:   np.ndarray   # unit, torso frame
    hand_R:        np.ndarray | None   # 3x3, torso frame
                                # mapper.hand_orientation() produces this today,
                                # in ROBOT axes; it becomes torso frame for free
                                # once §6.2 lands, since both are one rotation
                                # applied to the same landmark differences.

    # robot units, robot base frame, metres
    shoulder_pos: np.ndarray
    elbow_pos:    np.ndarray
    wrist_pos:    np.ndarray

    # the command itself
    q_cmd:        np.ndarray    # (7,)
    ee_pose_cmd:  tuple         # FK(q_cmd) → (pos, quat) — the "commanded target"
                                # the rubric's tracking metric compares against

    confidence:   np.ndarray    # per-landmark visibility

@dataclass
class RetargetTarget:
    t_capture: float; t_recv: float; t_solved: float   # latency accounting
    frame: str = "robot_base"
    left: ArmTarget
    right: ArmTarget
```

The direction unit-vectors are the size-invariant core; the positions are derived from
them using the *robot's* link lengths, which is what makes the whole thing invariant by
construction rather than by normalization. This schema is also exactly SEW-Mimic's
input and output, so §7 and §11 land together.

---

## 12. Dead ends and corrections

**Freezing the arm and retargeting only the gripper** — backwards relative to the
rubric, which grades "arm configuration ... **including elbow placement**". Freezing
the arm would score *worse*. The salvageable instinct is *decomposition*: let each
observable drive the joints it actually determines (body landmarks → shoulder/elbow,
hand model → wrist), rather than making one 6-DoF IK solve absorb everything. That's
SEW-Mimic's structure.

**[InPose](https://iclrinpose-crypto.github.io/ICLRInPose/)** — worth correcting the
premise: it takes **⟨location, rotation⟩ from 3 body-mounted sensors (head + two
wrists)** — a VR-tracker / IMU setting — and diffusion-priors the full body. We have
neither wrist rotation nor metric location, so it is not a denoiser for webcam
landmarks and won't drop in. The transferable idea (a learned motion prior conditioned
on noisy observations) is real, but in our setting its practical form is *fitting a
body model*, which §9.2 already does with released weights.

**RL policy for smooth retargeting** — lowest priority. Smoothness is available far
more cheaply from §6.3, §8.1's `VelocityLimit`, and §8.3's jerk objectives. Worth one
paragraph in the design doc's "given more time" section; not worth building.
Reference points if asked: [H2O](https://human2humanoid.com/),
[OmniH2O](https://arxiv.org/abs/2406.08858), HumanPlus, TWIST, ExBody — all RGB→humanoid
whole-body teleop via RL, and all far past the scope of an arm-only sim assessment.

---

## 13. Suggested order of work (multi-day version — see §0 if time-boxed to a day)

```
[x] §5   record/replay harness                    ← built
[ ] §5   record the battery                       ← needs a camera; gates every A/B
[x] §7.7 hand tracking → hand orientation H       ← done, and consumed by §7.8
[~] §6.4 metrics + JSONL logging                  ← replay reports most of it;
                                                    the LIVE path still logs nothing
[~] §6.3 1€ filter + slew limit                   ← filter done, slew limit next
[ ] §6.7 plausibility gate + rest behaviour       ← the erratic-input half
[x] §6.5 world landmarks                          ← done; was the elbow-angle bug
[x] §6.2 torso frame                              ← done; invariance in the check
[ ] §6.1 elbow task in the existing DLS solver    ← proves the diagnosis is right
    ─── decision point: measure. is DLS+elbow enough, or go structural? ───
[ ] §7   SEW-Mimic phases 0→2                     ← the real fix
[ ] §11  intermediate target schema               ← lands with §7
[ ] §6.6 limb-length constraint + T-pose calib
[x] §7.4 SEW-Mimic phase 3, consuming H           ← done, §7.8; hand_roll clip still
                                                    needed to measure whether the
                                                    hand frame is good enough (§9.4)
[ ] §8.1 mink, if we want exact wrist position on top of SEW
[ ] §7.5 phase 4: XPBD self-collision filter, if the battery shows collisions
```

Two things the harness needs before it earns its keep:

- **Record the battery.** Nothing above can be A/B'd until `backend/fixtures/` has
  clips in it. It is the one step that cannot be done without a person and a camera,
  so it gates everything else — do it first, in one sitting (§5.3).
- **§6.4 is half-built.** `replay.py` computes the metrics, but only during replay.
  The live websocket path still logs nothing per tick, and the rubric asks for the
  intermediate target logged *every timestep*. Factor the metric computation out of
  the script and call it from both.

§6.1 is deliberately placed *before* §7: it's half a day, and it either confirms the
null-space diagnosis (elbow variance collapses, lock-ups stop) or reveals that
something else is going on — either way you learn it cheaply before committing 1–2
days to a from-scratch solver implementation.

---

## 14. Reading list

**Start here**
- [SEW-Mimic — Closed-Form Geometric Retargeting](https://arxiv.org/abs/2602.01632) ([HTML](https://arxiv.org/html/2602.01632))
- [mink](https://github.com/kevinzakka/mink) · [docs](https://kevinzakka.github.io/mink/)
- [1€ filter](https://gery.casiez.net/1euro/)

**Redundancy / IK**
- [Stereographic SEW angle](https://arxiv.org/abs/2307.13122) · [stereo-sew](https://github.com/rpiRobotics/stereo-sew) · [IK-Geo](https://github.com/rpiRobotics/ik-geo)
- [RelaxedIK](https://github.com/uwgraphics/relaxed_ik) · [analysis](https://link.springer.com/article/10.1007/s10514-020-09918-9)
- [HL-IK](https://arxiv.org/pdf/2509.20263) · [biomimetic redundancy resolution](https://www.mdpi.com/2313-7673/11/6/408) · [radial elbow offset](https://www.sciencedirect.com/science/article/abs/pii/S1474034622002701)
- [IKLink](https://arxiv.org/html/2402.16154v1) — minimizing IK branch reconfigurations
- [Paden–Kahan subproblems](https://en.wikipedia.org/wiki/Paden%E2%80%93Kahan_subproblems)

**Retargeting systems**
- [GMR](https://github.com/YanjieZe/GMR) · [Retargeting Matters](https://arxiv.org/abs/2510.02252)
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) · [Open-TeleVision](https://www.alphaxiv.org/overview/2407.01512)
- [H2O](https://human2humanoid.com/) · [OmniH2O](https://arxiv.org/abs/2406.08858)

**Perception**
- [NLF pipeline](https://github.com/isarandi/nlf-pipeline) · [NLF paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/fd23a1f3bc89e042d70960b466dc20e8-Paper-Conference.pdf)
- [HaMeR](https://github.com/geopavlakos/hamer)
- [Seeing Through Occlusion (AKC)](https://arxiv.org/html/2606.19240v1)
- [AugLift](https://arxiv.org/html/2508.07112v1) · [Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3) · [UniDepth](https://lpiccinelli-eth.github.io/pub/unidepth/)
- [ESFP — Estimating, Smoothing, Filtering, Pose-mapping](https://arxiv.org/pdf/2506.21234)

**Robot**
- [openarm_mujoco](https://github.com/enactic/openarm_mujoco) · [OpenArm docs](https://docs.openarm.dev/simulation/mujoco/)

---

## 15. Open questions

1. Does SEW-Mimic have a public repo yet? Recheck before implementing from scratch (§7.6).
2. Is the raise-arm-to-the-side failure purely null-space drift, or is there also a
   sign error in `to_robot_axes`? **§6.1 should disambiguate this.** If the elbow task
   fixes it, it was drift; if not, check the axis mapping against §3's left/right sign
   differences.
3. What's the actual measured latency split? Unknown until §6.4 exists.
4. Do we ship MediaPipe (browser, easy deployment) or NLF (GPU, better accuracy)? The
   rubric wants precision but also wants *them* to run it live on *their* hardware.
   Leaning MediaPipe + strong geometric constraints, with the tradeoff documented — but
   this should be a measured decision, not a guess.
5. Gripper: the rubric says there's no manipulation task, but a gripper command from
   finger spread is nearly free once MediaPipe Hands is in (§7.4 Phase B). Worth it?
