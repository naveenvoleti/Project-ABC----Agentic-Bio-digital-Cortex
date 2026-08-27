"""
SensoryAgent — GPIO sensors: touch, proximity, temperature, buttons.
Maps to the Somatosensory Cortex.
"""
from __future__ import annotations

import asyncio

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.gpio_driver import GPIODriver
from brain.hardware.hw_detector import HardwareCapabilities
from brain.utils.logger import get_logger

log = get_logger(__name__)

PRIVACY_BUTTON_PIN = 17
TOUCH_SENSOR_PIN = 27
PROXIMITY_SENSOR_TRIG = 23
PROXIMITY_SENSOR_ECHO = 24
TEMP_SENSOR_PIN = 4

PROXIMITY_ALERT_CM = 30.0
POLL_INTERVAL = 0.1


class SensoryAgent(BaseAgent):
    name = "sensory_agent"

    def __init__(self, bus: asyncio.Queue, hw: HardwareCapabilities, gpio: GPIODriver):
        super().__init__(bus)
        self._hw = hw
        self._gpio = gpio
        self._cpu_temp_high = False
        self._privacy_on = False

    async def start(self) -> None:
        await super().start()
        if not self._hw.gpio_available:
            log.info("SensoryAgent: no GPIO, passive mode")
            await self._monitor_cpu_only()
            return

        self._gpio.init()
        self._gpio.setup_input(TOUCH_SENSOR_PIN)
        self._gpio.add_event_detect(
            PRIVACY_BUTTON_PIN,
            callback=self._on_privacy_button,
        )

        log.info("SensoryAgent started")
        while self._running:
            await self._poll_sensors()
            await asyncio.sleep(POLL_INTERVAL)

    async def _monitor_cpu_only(self) -> None:
        while self._running:
            temp = self._gpio.read_cpu_temp()
            if temp > 0:
                await self._handle_temp(temp)
            await asyncio.sleep(5)

    async def _poll_sensors(self) -> None:
        # CPU temperature
        temp = self._gpio.read_cpu_temp()
        if temp > 0:
            await self._handle_temp(temp)

        # Touch sensor
        touch = self._gpio.read(TOUCH_SENSOR_PIN)
        if touch:
            await self.publish(AgentMessage(
                type=MessageType.PERCEPTION_SENSOR,
                source=self.name,
                data={"sensor": "touch", "value": True},
                priority=3,
            ))

        # V10.0 — Ultrasonic distance sensor (HC-SR04 on TRIG/ECHO pins)
        # ProprioceptionAgent receives this to ground odometry estimation.
        dist_cm = self._gpio.read_distance(PROXIMITY_SENSOR_TRIG, PROXIMITY_SENSOR_ECHO)
        if dist_cm is not None and dist_cm > 0:
            await self.publish(AgentMessage(
                type=MessageType.PERCEPTION_SENSOR,
                source=self.name,
                data={"sensor": "distance", "distance_cm": round(dist_cm, 1)},
                priority=6,
            ))
            if dist_cm < PROXIMITY_ALERT_CM:
                await self.publish(AgentMessage(
                    type=MessageType.PERCEPTION_SENSOR,
                    source=self.name,
                    data={"sensor": "proximity_alert", "distance_cm": round(dist_cm, 1)},
                    priority=3,
                ))

    async def _handle_temp(self, temp: float) -> None:
        was_high = self._cpu_temp_high
        self._cpu_temp_high = temp > 75.0
        if self._cpu_temp_high and not was_high:
            log.warning(f"CPU temp high: {temp:.1f}°C — throttle ON")
            await self.publish(AgentMessage(
                type=MessageType.PERCEPTION_SENSOR,
                source=self.name,
                data={"sensor": "cpu_temp", "value": temp, "throttle": True},
                priority=2,
            ))
        elif not self._cpu_temp_high and was_high:
            log.info(f"CPU temp normal: {temp:.1f}°C — throttle OFF")
            await self.publish(AgentMessage(
                type=MessageType.PERCEPTION_SENSOR,
                source=self.name,
                data={"sensor": "cpu_temp", "value": temp, "throttle": False},
                priority=5,
            ))

    def _on_privacy_button(self, channel: int) -> None:
        self._privacy_on = not self._privacy_on
        asyncio.get_event_loop().create_task(
            self.publish(AgentMessage(
                type=MessageType.PRIVACY_MODE,
                source=self.name,
                data={"enabled": self._privacy_on},
                priority=1,
            ))
        )
        log.info(f"Privacy mode: {self._privacy_on}")

    async def handle(self, message: AgentMessage) -> None:
        pass

