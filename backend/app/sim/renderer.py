"""Offscreen rendering of the shared sim, for the browser demo view.

`just backend-viewer` opens a native MuJoCo window, which is the right tool for
poking at the model but the wrong one for recording: the robot ends up in one OS
window and the pose skeleton in another, and lining those up is the operator's
problem every take. This renders the same shared sim into JPEG frames that
`/ws/sim` streams to the demo page, so robot, skeleton and HUD composite inside
one canvas that the browser can record directly.

MuJoCo's GL context belongs to the thread that made it, so the renderer owns a
thread and never hands the `Renderer` out. Callers only ever see the latest
encoded frame.
"""

import io
import logging
import os
import threading
import time

import mujoco
from PIL import Image

from app.sim.simulator import Simulator

logger = logging.getLogger(__name__)

# Portrait, and half of a 1280x720 composite: the demo page puts this pane
# beside an equally sized skeleton pane, and both subjects — a pair of hanging
# arms and a standing person — are far taller than they are wide. Anything
# larger than this also needs `offwidth`/`offheight` raised in viewer_scene.xml,
# which is what bounds MuJoCo's offscreen framebuffer.
WIDTH = int(os.environ.get("SIM_RENDER_WIDTH", "640"))
HEIGHT = int(os.environ.get("SIM_RENDER_HEIGHT", "720"))
FPS = float(os.environ.get("SIM_RENDER_FPS", "30"))
# JPEG rather than PNG: the frames are photographic-ish (gradient skybox, shaded
# meshes) so PNG barely compresses them, and at 30fps the wire cost matters more
# than the compression artefacts do.
QUALITY = int(os.environ.get("SIM_RENDER_QUALITY", "80"))

# Which side the azimuth puts each arm on is the part that matters, not the
# prettiness of the angle: the demo page mirrors its pose pane, so the operator's
# left arm appears on the left there, and this view is turned to put the robot's
# matching arm on the left too. Add 180 and the two panes move in opposite
# directions, which reads as a broken retarget however correct the joints are.
AZIMUTH = float(os.environ.get("SIM_CAM_AZIMUTH", "-30"))
ELEVATION = float(os.environ.get("SIM_CAM_ELEVATION", "-8"))

# Distance is set by the widest pose, not the rest pose. At rest the arms hang in
# a narrow column and almost anything frames well, but an arm raised out to the
# side reaches ~0.8m off centre — and in a portrait pane the horizontal field is
# the narrow one, since MuJoCo's fov is vertical. That leaves vertical slack the
# arms only use at the extremes of their swing, which is the right way round.
DISTANCE = float(os.environ.get("SIM_CAM_DISTANCE", "2.35"))
# Midway between hanging (~-0.75) and fully raised (~+0.3), so the framing is
# centred on the motion rather than on any one pose.
LOOKAT_Z = float(os.environ.get("SIM_CAM_LOOKAT_Z", "-0.22"))


class SimRenderer:
    """Renders the shared sim to JPEG on its own thread, on demand.

    Rendering is pure overhead when nobody is watching, so the thread only runs
    while at least one client is subscribed (`acquire`/`release`). `just
    backend` with no demo page open costs exactly what it did before.
    """

    def __init__(
        self,
        sim: Simulator,
        data_lock: threading.Lock,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: float = FPS,
        quality: int = QUALITY,
    ) -> None:
        self.sim = sim
        # Held by the physics loop while it steps. mjv_updateScene reads the
        # whole of MjData, and reading it mid-`mj_step` yields a frame with some
        # bodies advanced and some not — a visible shear on a fast motion.
        self.data_lock = data_lock
        self.width = width
        self.height = height
        self.period = 1 / fps
        self.quality = quality

        self._frame: bytes | None = None
        self._seq = 0
        self._frame_lock = threading.Lock()

        self._clients = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()

    def acquire(self) -> None:
        """Subscribe; starts the render thread if it isn't already running."""
        with self._lifecycle_lock:
            self._clients += 1
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="sim-renderer", daemon=True
            )
            self._thread.start()
            logger.info(
                "sim renderer started: %dx%d @ %.0ffps q%d",
                self.width,
                self.height,
                1 / self.period,
                self.quality,
            )

    def release(self) -> None:
        """Unsubscribe; stops the render thread once the last client is gone."""
        with self._lifecycle_lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._stop.set()
                self._thread = None
                logger.info("sim renderer stopped (no clients)")

    def shutdown(self) -> None:
        """Stop regardless of the client count, for app shutdown."""
        with self._lifecycle_lock:
            self._clients = 0
            self._stop.set()
            self._thread = None

    def latest(self) -> tuple[int, bytes | None]:
        """The most recent encoded frame and its sequence number.

        The sequence number is what lets a consumer send each frame once without
        having to compare frame bytes or block on the renderer.
        """
        with self._frame_lock:
            return self._seq, self._frame

    def _camera(self) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.sim.model, camera)
        camera.azimuth = AZIMUTH
        camera.elevation = ELEVATION
        camera.distance = DISTANCE
        camera.lookat[:] = [0.0, 0.0, LOOKAT_Z]
        return camera

    def _run(self) -> None:
        # Both the Renderer and its GL context are created here and dropped here
        # — neither is safe to touch from another thread.
        renderer = mujoco.Renderer(self.sim.model, self.height, self.width)
        camera = self._camera()
        buffer = io.BytesIO()
        try:
            next_frame = time.monotonic()
            while not self._stop.is_set():
                with self.data_lock:
                    renderer.update_scene(self.sim.data, camera)
                pixels = renderer.render()

                buffer.seek(0)
                buffer.truncate()
                Image.fromarray(pixels).save(buffer, "JPEG", quality=self.quality)
                with self._frame_lock:
                    self._frame = buffer.getvalue()
                    self._seq += 1

                # Absolute deadline, not a flat sleep: rendering takes a good
                # fraction of the period, and adding it on top would put the
                # stream well under the requested framerate.
                next_frame += self.period
                self._stop.wait(max(0.0, next_frame - time.monotonic()))
                next_frame = max(next_frame, time.monotonic() - self.period)
        except Exception:
            logger.exception("sim renderer thread died")
        finally:
            renderer.close()
