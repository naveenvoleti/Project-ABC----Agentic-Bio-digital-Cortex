"""Generic GPIO abstraction — sensors, buttons, LEDs."""
from __future__ import annotations

from brain.utils.logger import get_logger

log = get_logger(__name__)


class GPIODriver:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self._gpio = None
        self._setup_pins: set[int] = set()

    def init(self) -> bool:
        if self.mock:
            log.info("GPIO: mock mode")
            return True
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._gpio = GPIO
            log.info("GPIO initialized (BCM mode)")
            return True
        except Exception as e:
            log.warning(f"GPIO unavailable: {e} — mock mode")
            self.mock = True
            return True

    def setup_input(self, pin: int, pull_up: bool = True) -> None:
        if self.mock:
            return
        if self._gpio:
            pud = self._gpio.PUD_UP if pull_up else self._gpio.PUD_DOWN
            self._gpio.setup(pin, self._gpio.IN, pull_up_down=pud)
            self._setup_pins.add(pin)

    def setup_output(self, pin: int) -> None:
        if self.mock:
            return
        if self._gpio:
            self._gpio.setup(pin, self._gpio.OUT)
            self._setup_pins.add(pin)

    def read(self, pin: int) -> bool:
        if self.mock:
            return False
        if self._gpio:
            if pin not in self._setup_pins:
                self.setup_input(pin)
            return bool(self._gpio.input(pin))
        return False

    def write(self, pin: int, value: bool) -> None:
        if self.mock:
            log.debug(f"[MOCK] GPIO pin {pin} → {value}")
            return
        if self._gpio:
            if pin not in self._setup_pins:
                self.setup_output(pin)
            self._gpio.output(pin, self._gpio.HIGH if value else self._gpio.LOW)

    def read_cpu_temp(self) -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return 0.0

    def add_event_detect(self, pin: int, callback, bouncetime: int = 200) -> None:
        if self.mock or not self._gpio:
            return
        self.setup_input(pin)
        self._gpio.add_event_detect(
            pin,
            self._gpio.FALLING,
            callback=callback,
            bouncetime=bouncetime,
        )

    def read_distance(self, trig_pin: int, echo_pin: int) -> float | None:
        """V10.0 — HC-SR04 ultrasonic distance measurement in centimetres.

        Returns None in mock mode, on error, or when the pulse times out.
        SensoryAgent passes this to ProprioceptionAgent to ground dead-reckoning.
        """
        if self.mock or not self._gpio:
            return None
        try:
            import time as _time
            GPIO = self._gpio
            # Ensure pins are configured
            if trig_pin not in self._setup_pins:
                GPIO.setup(trig_pin, GPIO.OUT)
                self._setup_pins.add(trig_pin)
            if echo_pin not in self._setup_pins:
                GPIO.setup(echo_pin, GPIO.IN)
                self._setup_pins.add(echo_pin)
            # Send 10µs trigger pulse
            GPIO.output(trig_pin, False)
            _time.sleep(0.000002)
            GPIO.output(trig_pin, True)
            _time.sleep(0.00001)
            GPIO.output(trig_pin, False)
            # Measure echo pulse width (timeout 30ms = ~5m max range)
            timeout = _time.time() + 0.03
            while GPIO.input(echo_pin) == 0:
                if _time.time() > timeout:
                    return None
            start = _time.time()
            timeout = _time.time() + 0.03
            while GPIO.input(echo_pin) == 1:
                if _time.time() > timeout:
                    return None
            elapsed = _time.time() - start
            distance_cm = (elapsed * 34300.0) / 2.0   # speed of sound
            return round(distance_cm, 1)
        except Exception as e:
            log.debug("GPIO: read_distance error: %s", e)
            return None

    def cleanup(self) -> None:
        if self._gpio:
            self._gpio.cleanup()

