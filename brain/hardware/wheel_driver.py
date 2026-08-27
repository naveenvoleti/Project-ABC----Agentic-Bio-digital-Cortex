"""
WheelDriver — L298N H-Bridge driver with linear PWM ramp for differential wheel drive.
"""
from __future__ import annotations

import time

from brain.utils.logger import get_logger

log = get_logger(__name__)


class WheelDriver:
    def __init__(
        self,
        left_pins: list[int] | None = None,
        right_pins: list[int] | None = None,
        speed_default: int = 50,
        rotate_speed: int = 35,
        ramp_ms: int = 50,
        mock: bool = False,
    ):
        self.left_pins = left_pins or []
        self.right_pins = right_pins or []
        self.speed_default = speed_default
        self.rotate_speed = rotate_speed
        self.ramp_ms = ramp_ms
        self.mock = mock

        self._left_motor = None
        self._right_motor = None

    def init(self) -> bool:
        if self.mock:
            log.info("WheelDriver: mock mode")
            return True
        try:
            # Try to import and setup hardware
            from gpiozero import Motor
            # Assuming left_pins = [forward, backward, en] (or just 2 pins)
            if len(self.left_pins) >= 2:
                self._left_motor = Motor(forward=self.left_pins[0], backward=self.left_pins[1])
            if len(self.right_pins) >= 2:
                self._right_motor = Motor(forward=self.right_pins[0], backward=self.right_pins[1])
            log.info("WheelDriver initialized successfully")
            return True
        except Exception as e:
            log.error(f"WheelDriver init error: {e} — falling back to mock")
            self.mock = True
            return True

    def _ramp(self, motor, target_speed: float):
        # target_speed is 0.0 to 1.0
        # In a real implementation this would linearly ramp the PWM
        if self.mock or motor is None:
            return
        
        # Simple mock ramp
        if self.ramp_ms > 0:
            steps = 5
            sleep_time = (self.ramp_ms / 1000.0) / steps
            for i in range(1, steps + 1):
                val = target_speed * (i / steps)
                if val >= 0:
                    motor.forward(val)
                else:
                    motor.backward(-val)
                time.sleep(sleep_time)
        else:
            if target_speed >= 0:
                motor.forward(target_speed)
            else:
                motor.backward(-target_speed)

    def rotate_right(self, degrees: float, speed: int | None = None) -> None:
        spd = speed if speed is not None else self.rotate_speed
        log.info(f"WheelDriver: rotating right {degrees} degrees at speed {spd}")
        if not self.mock:
            # left forward, right backward
            self._ramp(self._left_motor, spd / 100.0)
            self._ramp(self._right_motor, -spd / 100.0)
            # wait proportional to degrees
            time.sleep(abs(degrees) * 0.01) 
            self.stop()

    def rotate_left(self, degrees: float, speed: int | None = None) -> None:
        spd = speed if speed is not None else self.rotate_speed
        log.info(f"WheelDriver: rotating left {degrees} degrees at speed {spd}")
        if not self.mock:
            # left backward, right forward
            self._ramp(self._left_motor, -spd / 100.0)
            self._ramp(self._right_motor, spd / 100.0)
            # wait proportional to degrees
            time.sleep(abs(degrees) * 0.01) 
            self.stop()

    def move(self, speed: int, direction: str = "forward", duration: float = 1.0) -> None:
        log.info(f"WheelDriver: moving {direction} at speed {speed} for {duration}s")
        if not self.mock:
            spd = speed / 100.0
            if direction == "forward":
                self._ramp(self._left_motor, spd)
                self._ramp(self._right_motor, spd)
            elif direction == "backward":
                self._ramp(self._left_motor, -spd)
                self._ramp(self._right_motor, -spd)
            
            time.sleep(duration)
            self.stop()

    def stop(self) -> None:
        log.info("WheelDriver: stop")
        if not self.mock:
            if self._left_motor:
                self._left_motor.stop()
            if self._right_motor:
                self._right_motor.stop()
