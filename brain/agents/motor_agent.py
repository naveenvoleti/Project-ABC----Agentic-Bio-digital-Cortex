"""
MotorAgent — translates high-level movement commands to hardware calls.
Maps to the Motor Cortex + Cerebellum.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.motor_driver import MotorDriver
from brain.hardware.hw_detector import HardwareCapabilities
from brain.hardware.wheel_driver import WheelDriver
from brain.utils.logger import get_logger

log = get_logger(__name__)


class MotorAgent(BaseAgent):
    name = "motor_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        hw: HardwareCapabilities,
        motor: MotorDriver,
        saccade_smoothing: bool = True,
        saccade_steps: int = 8,
        saccade_duration_ms: int = 200,
        wheel_driver: WheelDriver | None = None,
        gaze_rotate_threshold: float = 45.0,
        gaze_center_threshold: float = 15.0,
    ):
        super().__init__(bus)
        self._hw = hw
        self._motor = motor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_pan: float = 0.0
        self._current_tilt: float = 0.0
        self._saccade_smoothing = saccade_smoothing
        self._saccade_steps = max(2, saccade_steps)
        self._saccade_duration_s = saccade_duration_ms / 1000.0
        # V6.0 — differential drive + gaze hysteresis
        self._wheel_driver = wheel_driver
        self._gaze_rotate_threshold = gaze_rotate_threshold
        self._gaze_center_threshold = gaze_center_threshold
        self._gaze_rotating = False   # hysteresis state flag

    async def start(self) -> None:
        await super().start()
        if not self._hw.motor_available:
            log.info("MotorAgent: no motors detected, passive mode")
        else:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._motor.init
            )
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._motor.center
            )
            log.info(f"MotorAgent started: {self._hw.motor_type}")

        while self._running:
            await asyncio.sleep(1)

    async def handle(self, message: AgentMessage) -> None:
        if not self._hw.motor_available:
            return

        if message.type == MessageType.ACTION_MOVE:
            await self._handle_move(message.data)

        elif message.type == MessageType.WHEELED_ROTATE:
            await self._handle_wheeled_rotate(message.data)

        elif message.type == MessageType.SPATIAL_POINT:
            await self._gaze_snap(message.data)

        elif message.type == MessageType.VLA_CONTROL_TOKEN:
            await self._handle_vla_token(message.data)

        elif message.type == MessageType.EMOTION_CHANGE:
            await self._handle_emotion_motion(message.data.get("motor_action", "still"))

        elif message.type == MessageType.CURIOSITY_TRIGGER:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._motor.scan
            )

    async def _handle_move(self, data: dict) -> None:
        cmd = data.get("command", "")
        loop = asyncio.get_event_loop()

        if cmd == "pan":
            await loop.run_in_executor(self._executor, self._motor.pan, data.get("degrees", 0))
        elif cmd == "tilt":
            await loop.run_in_executor(self._executor, self._motor.tilt, data.get("degrees", 0))
        elif cmd == "look_at":
            await self._smooth_look_at(data.get("pan", 0.0), data.get("tilt", 0.0))
        elif cmd == "scan":
            await loop.run_in_executor(self._executor, self._motor.scan)
        elif cmd == "center":
            await loop.run_in_executor(self._executor, self._motor.center)
            self._current_pan = 0.0
            self._current_tilt = 0.0
        elif cmd == "forward" and self._hw.motor_type == "wheels":
            await loop.run_in_executor(
                self._executor, self._motor.move,
                data.get("speed", 50), "forward", data.get("duration", 1.0)
            )

    async def _handle_wheeled_rotate(self, data: dict) -> None:
        if self._wheel_driver is None:
            return
        direction = data.get("direction", "right")
        degrees = float(data.get("degrees", 90))
        speed = int(data.get("speed", 0)) or None
        loop = asyncio.get_event_loop()
        fn = (self._wheel_driver.rotate_right
              if direction == "right"
              else self._wheel_driver.rotate_left)
        await loop.run_in_executor(self._executor, fn, degrees, speed)
        # Re-center pan servo after body rotation completes
        await loop.run_in_executor(self._executor, self._motor.center)
        self._current_pan = 0.0
        self._gaze_rotating = False

    async def _smooth_look_at(self, target_pan: float, target_tilt: float) -> None:
        """Sigmoid (smoothstep) interpolation to target — mimics biological saccades.
        Falls back to single direct move when saccade_smoothing is disabled."""
        loop = asyncio.get_event_loop()
        if not self._saccade_smoothing:
            await loop.run_in_executor(self._executor, self._motor.look_at, target_pan, target_tilt)
            self._current_pan, self._current_tilt = target_pan, target_tilt
            return
        step_delay = self._saccade_duration_s / self._saccade_steps
        start_pan, start_tilt = self._current_pan, self._current_tilt
        for i in range(1, self._saccade_steps + 1):
            t = i / self._saccade_steps
            progress = t * t * (3.0 - 2.0 * t)  # smoothstep ease-in/ease-out
            pan = start_pan + (target_pan - start_pan) * progress
            tilt = start_tilt + (target_tilt - start_tilt) * progress
            await loop.run_in_executor(self._executor, self._motor.look_at, pan, tilt)
            await asyncio.sleep(step_delay)
        self._current_pan = target_pan
        self._current_tilt = target_tilt
        await self._check_gaze_nav()

    async def _check_gaze_nav(self) -> None:
        """Hysteresis-safe gaze-driven navigation trigger."""
        if self._wheel_driver is None:
            return
        pan = abs(self._current_pan)
        if not self._gaze_rotating and pan > self._gaze_rotate_threshold:
            self._gaze_rotating = True
            direction = "right" if self._current_pan > 0 else "left"
            log.info("Gaze-nav: pan=%.1f° → WHEELED_ROTATE %s", self._current_pan, direction)
            await self.publish(AgentMessage(
                type=MessageType.WHEELED_ROTATE,
                source=self.name,
                data={"direction": direction, "degrees": pan},
                priority=2,
            ))
        elif self._gaze_rotating and pan < self._gaze_center_threshold:
            log.info("Gaze-nav: pan=%.1f° — hysteresis cleared, rotation done", self._current_pan)
            self._gaze_rotating = False

    async def _gaze_snap(self, data: dict) -> None:
        """V7.0 Spatial Grounding — instantly snap gaze to a normalised (x, y) point.

        Bypasses saccade smoothing for real-time visual servoing responsiveness.
        x=0.5, y=0.5 is centre frame; x∈[0,1]→pan∈[-90,90], y∈[0,1]→tilt∈[45,-45].
        """
        x = float(data.get("x", 0.5))
        y = float(data.get("y", 0.5))
        confidence = float(data.get("confidence", 0.0))
        if confidence < 0.1:
            return
        pan = (x - 0.5) * 180.0
        tilt = (y - 0.5) * -90.0
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._motor.look_at, pan, tilt)
        self._current_pan = pan
        self._current_tilt = tilt
        log.debug("gaze_snap: x=%.3f y=%.3f → pan=%.1f° tilt=%.1f°", x, y, pan, tilt)
        await self._check_gaze_nav()

    async def _handle_vla_token(self, data: dict) -> None:
        """V7.0 VLA Signal Mapping — execute a Gemini VLA action token via VLAMapper."""
        from brain.hardware.vla_mapper import VLAMapper
        token = data.get("token", {})
        confidence_threshold = float(data.get("confidence_threshold", 0.3))
        result = VLAMapper.parse(token)
        if result is None or result.get("confidence", 0.0) < confidence_threshold:
            log.debug("vla_token: below confidence threshold or unmapped — skipping")
            return
        action = result.get("action_type", "")
        params = result.get("params", {})
        loop = asyncio.get_event_loop()
        if action == "move_pan":
            await loop.run_in_executor(self._executor, self._motor.pan, params.get("degrees", 0))
            self._current_pan = float(params.get("degrees", self._current_pan))
        elif action == "move_tilt":
            await loop.run_in_executor(self._executor, self._motor.tilt, params.get("degrees", 0))
            self._current_tilt = float(params.get("degrees", self._current_tilt))
        elif action == "rotate_wheels":
            direction = params.get("direction", "right")
            degrees = float(params.get("degrees", 45))
            await self._handle_wheeled_rotate({"direction": direction, "degrees": degrees})
        elif action == "stop":
            if self._wheel_driver:
                await loop.run_in_executor(self._executor, self._wheel_driver.stop)
        log.info("vla_token executed: %s params=%s conf=%.2f",
                 action, params, result.get("confidence", 0.0))

    async def _handle_emotion_motion(self, action: str) -> None:
        loop = asyncio.get_event_loop()
        if action == "nod_up":
            await loop.run_in_executor(self._executor, self._motor.nod)
        elif action == "shake":
            await loop.run_in_executor(self._executor, self._motor.shake)
        elif action == "scan":
            await loop.run_in_executor(self._executor, self._motor.scan)
        elif action == "random_drift":
            await loop.run_in_executor(self._executor, self._motor.random_drift)

    async def stop(self) -> None:
        await super().stop()
        if self._hw.motor_available:
            self._motor.stop()
        self._executor.shutdown(wait=False)
