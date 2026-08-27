"""Motor driver — pan-tilt servos, differential drive wheels, arm control."""
from __future__ import annotations

import time

from brain.utils.logger import get_logger

log = get_logger(__name__)

PAN_MIN = -90
PAN_MAX = 90
TILT_MIN = -45
TILT_MAX = 45


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MotorDriver:
    def __init__(self, motor_type: str = "pantilt", mock: bool = False):
        self.motor_type = motor_type
        self.mock = mock
        self._pca = None
        self._pan_deg = 0.0
        self._tilt_deg = 0.0

    def init(self) -> bool:
        if self.mock:
            log.info("Motor: mock mode")
            return True
        try:
            from adafruit_pca9685 import PCA9685
            import board, busio
            i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(i2c)
            self._pca.frequency = 50
            log.info(f"Motor driver initialized: {self.motor_type}")
            return True
        except Exception as e:
            log.error(f"Motor init error: {e} — mock fallback")
            self.mock = True
            return True

    def _deg_to_duty(self, deg: float) -> int:
        pulse_min = 0x0666   # ~1ms
        pulse_max = 0x1999   # ~2ms
        frac = (deg + 90) / 180.0
        return int(pulse_min + frac * (pulse_max - pulse_min))

    def pan(self, degrees: float) -> None:
        self._pan_deg = _clamp(degrees, PAN_MIN, PAN_MAX)
        if self.mock:
            log.debug(f"[MOCK] Pan → {self._pan_deg}°")
            return
        if self._pca:
            self._pca.channels[0].duty_cycle = self._deg_to_duty(self._pan_deg)

    def tilt(self, degrees: float) -> None:
        self._tilt_deg = _clamp(degrees, TILT_MIN, TILT_MAX)
        if self.mock:
            log.debug(f"[MOCK] Tilt → {self._tilt_deg}°")
            return
        if self._pca:
            self._pca.channels[1].duty_cycle = self._deg_to_duty(self._tilt_deg)

    def look_at(self, pan: float, tilt: float) -> None:
        self.pan(pan)
        self.tilt(tilt)

    def center(self) -> None:
        self.look_at(0, 0)

    def scan(self, steps: int = 5, delay: float = 0.3) -> None:
        positions = [PAN_MIN, 0, PAN_MAX, 0]
        for pos in positions:
            self.pan(pos)
            time.sleep(delay)

    def nod(self) -> None:
        for deg in [0, 20, 0, 20, 0]:
            self.tilt(deg)
            time.sleep(0.2)

    def shake(self) -> None:
        for deg in [0, -20, 20, -20, 0]:
            self.pan(deg)
            time.sleep(0.15)

    def random_drift(self) -> None:
        import random
        self.pan(random.uniform(-30, 30))
        self.tilt(random.uniform(-15, 15))

    def move(self, speed: int = 50, direction: str = "forward", duration: float = 1.0) -> None:
        if self.motor_type != "wheels":
            return
        log.debug(f"[MOCK] Move {direction} speed={speed} for {duration}s")
        time.sleep(duration)

    def stop(self) -> None:
        if self.mock:
            return
        if self._pca:
            for ch in self._pca.channels:
                ch.duty_cycle = 0
