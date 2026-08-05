# Webcam to Bimanual Arm Human Pose Retargeting

Track human pose using webcam and drive a bimanual OpenArm in MuJoCo in real time.

The browser runs MediaPipe and streams landmarks over a websocket; the backend
holds one shared MuJoCo sim, retargets each frame to joint targets, and steps
physics at 62.5Hz. `problem.pdf` is the original brief, `NEXT_STEPS.md` the
working design doc, `notes.md` a MuJoCo primer and the measured OpenArm joint
axes.

## Solution

Divide the problem into perception, retargeting, IK, filtering stages.

- Perception: Use RGB webcam and Mediapipe Posture + Hands to track Human pose. MediaPipe's **world** landmarks, not the normalized image ones - those are divided by frame width and height separately, so on a 4:3 capture every direction taken from them is tilted toward vertical. See [`socket.ts`](frontend/src/pose/socket.ts) for the measurements behind that.
- Retargeting: Build a torso frame from shoulders and hips, express both limb directions in it, and hand those to the robot as-is. Working body-centric is what makes the result invariant to the operator turning or leaning ([`mapper.py`](backend/app/retarget/mapper.py)).
- IK: Closed-form SEW-Mimic stages 1-2 solve joints 1-4 directly — no Jacobian, no iteration ([`sew.py`](backend/app/retarget/sew.py)). Damped least squares on the end-effector site is kept as the alternative path, `RETARGET_MODE=ik` ([`ik.py`](backend/app/retarget/ik.py)).
- Filtering: The two shoulder rolls stop being observable at the singularities, i.e. arm straight, or raised out to the side - so each measured angle is weighted by how observable it is and unobservable ones relax toward the rest posture rather than tracking noise. Joints 5-7 hold neutral until stage 3 lands.

Why orientation based approach:
- Invariant to teleoperator's and robot's size
- Enables optimal geometric solution
- The embodiment mismatch makes the position approach unwinnable anyway: OpenArm v2's upper arm to forearm ratio is 1:1.40 against a human's 1:0.78, so no scaling matches wrist *and* elbow position at once. Directions match both exactly and let positions fall where the robot's link lengths put them.

## Architecture

```
browser                                  backend (FastAPI + MuJoCo)
  MediaPipe Pose + Hands                   PoseFrame
  world landmarks, 30fps                     6 arm landmarks -> retargeted
  ──── websocket /ws/pose ────────►          2 hips          -> torso frame
                                             2 hands         -> logged only
                                             t_capture       -> latency
                                             │
                                             ├─ torso_frame()
                                             │    camera axes -> body axes
                                             ├─ limb_directions()
                                             │    unit u, l in robot axes
                                             ├─ Stage12Solver.solve()
                                             │    closed form, both branches
                                             │    scored, joints 1-4
                                             └─ PositionController.ctrl()
                                                  clamp -> <position> servos
                                             physics: 8 x 2ms = 62.5Hz
```

One shared sim, not one per connection, so the native viewer window shows
exactly what the websocket handler is driving. Frontend entry point is
[`main.ts`](frontend/src/main.ts), backend is
[`main.py`](backend/app/main.py).

### Running

```bash
just install && just dev
```

`just dev` runs the backend with a MuJoCo viewer plus the Vite dev server; open
the printed URL in a real browser, since it needs webcam access. `just backend`
runs headless, `just --list` shows the rest.

`just check` runs everything verifiable without a camera: the closed form against
MuJoCo forward kinematics, the wired-up path against synthetic landmarks
(accuracy, continuity through both singularities, torso invariance, steadiness
under landmark noise), and the hand frame against its own symmetries.

## Improvements

- Wire SEW-Mimic stage 3 so joints 5-7 track wrist orientation.
  `mapper.hand_orientation()` already produces the input and logs it; nothing
  consumes it yet.
- Slew-limit and interpolate `qpos_target` between the 30fps pose stream and the
  62.5Hz physics loop.
- Drive the gripper from finger spread — `thumb_tip` and `index_tip` are already
  being carried through the wire format for this.

What other type of solvers can be considered?
- Keypoint-based retargeting
- Analytic Geometric IK Solvers

## Further Explorations

- Qin, Yuzhe, et al. "Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system." arXiv preprint arXiv:2307.04577 (2023).
  - [GitHub - dexsuite/dex-retargeting: Various retargeting optimizers to translate human hand motion to robot hand motion. · GitHub](https://github.com/dexsuite/dex-retargeting)
- Chen, Zhenyang, et al. "WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations." arXiv preprint arXiv:2606.29940 (2026).
- [Humanoid Motion Retargeting - cuRobo](https://nvlabs.github.io/curobo/latest/getting-started/humanoid_retargeting.html)
- [GitHub - kevinzakka/mink: Python inverse kinematics based on MuJoCo. · GitHub](https://github.com/kevinzakka/mink)

## References

- [OpenArm](https://github.com/enactic/openarm_mujoco): the arm model, vendored as the `openarm_mujoco` submodule. v2 bimanual, 7 DoF plus a gripper per arm.
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [MuJoCo documentation](https://mujoco.readthedocs.io/): `notes.md` has the terminology primer for the parts used here.
- Kong, Chuizheng, et al. "A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation." arXiv preprint arXiv:2602.01632 (2026). Sew-Mimic solver is used for retargeting pipeline.
