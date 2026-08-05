"""FastAPI app. One shared Simulator drives everything rather than one per
websocket connection, so a single live MuJoCo viewer window (MUJOCO_VIEWER=1,
or `just backend-viewer`) mirrors exactly what the handlers are doing to it.

A background task steps physics at 60Hz whether or not anyone is connected;
the websocket handler just updates `qpos_target` from retargeted landmarks and
streams state back.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.control.position import PositionController
from app.retarget.mapper import PoseFrame
from app.retarget.retargeter import Retargeter
from app.sim.simulator import Simulator
from app.telemetry import PoseTelemetry

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


TARGET_TICK_SECONDS = 1 / 60  # rounded to a whole substep count below
MAX_CATCHUP_TICKS = 3  # how much backlog the physics loop will try to repay


class SimState:
    def __init__(self) -> None:
        self.sim = Simulator()
        self.controller = PositionController(self.sim)
        self.retargeter = Retargeter(self.sim)
        self.qpos_target = self.sim.actuated_qpos()
        self.viewer = None
        self.tick_count = 0

        # The model's 2ms timestep is much shorter than a 60Hz tick, so one
        # mj_step per tick would run the sim at 0.12x realtime and the arm could
        # never catch a live pose. Step enough substeps to cover a whole tick.
        #
        # The tick period is then derived back from the substep count instead of
        # being pinned at 1/60, which isn't a whole number of 2ms steps: sleeping
        # 1/60 while stepping 8x2ms leaves the sim permanently 3.6% slow.
        # Deriving it keeps sim and wall time in step, at 62.5Hz not 60Hz.
        timestep = self.sim.model.opt.timestep
        self.substeps = max(1, round(TARGET_TICK_SECONDS / timestep))
        self.tick_seconds = self.substeps * timestep
        logger.info(
            "physics: %d substeps x %.4fs = %.4fs tick (%.1fHz)",
            self.substeps,
            timestep,
            self.tick_seconds,
            1 / self.tick_seconds,
        )

    def step(self) -> None:
        # Read qpos_target once per tick: the websocket handler can replace it
        # between substeps, and every substep should track one target.
        ctrl = self.controller.ctrl(self.qpos_target)
        for _ in range(self.substeps):
            self.sim.step(ctrl)
        self.tick_count += 1
        if self.viewer is not None:
            self.viewer.sync()
        if self.tick_count % round(1 / self.tick_seconds) == 0:  # ~once a second
            logger.debug(
                "t=%.2fs qpos=%s",
                self.sim.time,
                [round(q, 3) for q in self.sim.actuated_qpos()],
            )


async def _physics_loop(state: SimState) -> None:
    """Steps the sim on a fixed wall-clock cadence.

    Sleeping a flat tick_seconds adds the stepping time and the event loop's
    timer granularity (~1ms per sleep on macOS) to every period, which measured
    at ~0.9x realtime. Tracking an absolute deadline and sleeping only the
    remainder lets a short tick make up for the previous long one.

    Catch-up is capped at MAX_CATCHUP_TICKS, drop the events between the lag.
    """
    loop = asyncio.get_running_loop()
    next_tick = loop.time()
    while True:
        state.step()
        next_tick += state.tick_seconds
        await asyncio.sleep(max(0, next_tick - loop.time()))
        backlog_floor = loop.time() - MAX_CATCHUP_TICKS * state.tick_seconds
        if next_tick < backlog_floor:
            logger.warning(
                "physics loop fell %.2fs behind realtime; dropping the backlog",
                backlog_floor - next_tick,
            )
            next_tick = loop.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = SimState()
    if os.environ.get("MUJOCO_VIEWER"):
        import mujoco.viewer

        logger.info("opening MuJoCo viewer (requires running under mjpython on macOS)")
        state.viewer = mujoco.viewer.launch_passive(state.sim.model, state.sim.data)
    app.state.sim_state = state

    loop_task = asyncio.create_task(_physics_loop(state))
    yield
    loop_task.cancel()
    if state.viewer is not None:
        state.viewer.close()


app = FastAPI(lifespan=lifespan)

# Vite (localhost:5173) and the backend (localhost:8000) are different origins
# in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


async def _stream_state(websocket: WebSocket, state: SimState) -> None:
    while True:
        await websocket.send_json({
            "time": state.sim.time,
            "qpos": state.sim.actuated_qpos().tolist(),
        })
        await asyncio.sleep(1 / 60)


@app.websocket("/ws/pose")
async def pose_socket(websocket: WebSocket) -> None:
    """Drives the shared sim from live webcam landmarks. Each incoming PoseFrame
    is retargeted to a qpos_target for the position servos to track; sim state
    streams back on the same socket.
    """
    await websocket.accept()
    logger.info("pose client connected: %s", websocket.client)
    state: SimState = websocket.app.state.sim_state
    telemetry = PoseTelemetry(state.sim.actuator_names)

    sender = asyncio.create_task(_stream_state(websocket, state))
    try:
        while True:
            msg = await websocket.receive_json()
            frame = PoseFrame(**msg)
            started = time.perf_counter()
            state.qpos_target = state.retargeter.solve(frame)
            for line in telemetry.record(
                frame,
                state.retargeter.last_solve,
                state.qpos_target,
                state.sim.actuated_qpos(),
                time.perf_counter() - started,
            ):
                logger.info("%s", line)
    except WebSocketDisconnect:
        logger.info("pose client disconnected: %s", websocket.client)
    finally:
        sender.cancel()
