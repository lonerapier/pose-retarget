"""FastAPI app: WebSocket streams sim state to the frontend and accepts pose targets."""

import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.control.pd_gravity import PDGravityController
from app.sim.simulator import Simulator

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Frontend (Vite, e.g. localhost:5173) and backend (localhost:8000) are
# different origins in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class PoseFrame(BaseModel):
    """One frame of MediaPipe body landmarks relevant to bimanual retargeting.

    Coordinates are as MediaPipe reports them: x/y normalized to [0, 1]
    against the video frame, z roughly in the same units with the hip
    midpoint as origin. Not yet mapped into the robot's frame — this
    endpoint only proves the landmarks are arriving; retargeting math
    comes next.
    """

    left_shoulder: Landmark
    left_elbow: Landmark
    left_wrist: Landmark
    right_shoulder: Landmark
    right_elbow: Landmark
    right_wrist: Landmark


def _fmt(lm: Landmark) -> str:
    return f"({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f})"


@app.websocket("/ws/pose")
async def pose_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("pose client connected: %s", websocket.client)
    frame_count = 0
    try:
        while True:
            msg = await websocket.receive_json()
            frame = PoseFrame(**msg)
            frame_count += 1
            logger.debug("pose frame %d: %s", frame_count, frame)
            if frame_count % 30 == 0:  # ~once/sec at 30fps
                logger.info(
                    "pose frame %d: L wrist=%s R wrist=%s",
                    frame_count,
                    _fmt(frame.left_wrist),
                    _fmt(frame.right_wrist),
                )
            await websocket.send_json({"status": "ok", "frame": frame_count})
    except WebSocketDisconnect:
        logger.info("pose client disconnected: %s", websocket.client)


@app.websocket("/ws/sim")
async def sim_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("client connected: %s", websocket.client)
    sim = Simulator()
    controller = PDGravityController(sim)
    qpos_target = sim.actuated_qpos()

    async def receive_targets() -> None:
        nonlocal qpos_target
        while True:
            msg = await websocket.receive_json()
            if "qpos_target" in msg:
                qpos_target = qpos_target.copy()
                for i, v in enumerate(msg["qpos_target"][: sim.model.nu]):
                    qpos_target[i] = v
                logger.info("received qpos_target: %s", qpos_target)

    receiver = asyncio.create_task(receive_targets())
    step_count = 0
    try:
        while True:
            torque = controller.torque(qpos_target)
            sim.step(torque)
            step_count += 1
            if step_count % 60 == 0:  # once a second at 60Hz
                logger.info(
                    "t=%.2fs qpos=%s", sim.time, [round(q, 3) for q in sim.actuated_qpos()]
                )
            await websocket.send_json({
                "time": sim.time,
                "qpos": sim.actuated_qpos().tolist(),
            })
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        logger.info("client disconnected: %s", websocket.client)
    finally:
        receiver.cancel()
