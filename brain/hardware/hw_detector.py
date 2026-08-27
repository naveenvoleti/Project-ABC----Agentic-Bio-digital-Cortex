"""
Hardware Capability Detector — scans available hardware on boot.
Agents receive a HardwareCapabilities instance and never probe hardware themselves.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from brain.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HardwareCapabilities:
    # Camera
    camera_available: bool = False
    camera_source: str = "none"          # "usb" | "csi" | "none"
    camera_device_index: int = 0

    # Microphone
    mic_available: bool = False
    mic_device_name: str = ""

    # Speaker
    speaker_available: bool = False
    speaker_type: str = "none"           # "usb" | "aux" | "bluetooth" | "none"
    bluetooth_speaker_mac: str = ""

    # Display
    display_available: bool = False
    display_type: str = "none"           # "ili9341" | "st7789" | "ssd1306" | "mock" | "none"

    # Motor
    motor_available: bool = False
    motor_type: str = "none"             # "pantilt" | "wheels" | "arm" | "none"

    # V6.0 — Differential wheel drive (L298N)
    has_differential_drive: bool = False
    wheel_left_pins: tuple[int, int] = (0, 0)   # (IN1, IN2) BCM
    wheel_right_pins: tuple[int, int] = (0, 0)  # (IN3, IN4) BCM

    # V8.0 — Arduino / USB serial device
    has_arduino: bool = False
    arduino_port: str = ""

    # GPIO sensors
    gpio_available: bool = False
    gpio_devices: list[str] = field(default_factory=list)

    # Network
    network_available: bool = False
    bluetooth_available: bool = False

    # Platform
    is_raspberry_pi: bool = False
    cpu_temp_available: bool = False
    mock_mode: bool = False


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "raspberry pi" in f.read().lower()
    except Exception:
        return False


def _check_camera() -> tuple[bool, str, int]:
    try:
        import cv2
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.release()
                log.info(f"Camera found: USB device {idx}")
                return True, "usb", idx
        cap.release()
    except ImportError:
        pass
    # Check CSI camera
    if Path("/dev/video0").exists():
        return True, "csi", 0
    return False, "none", 0


def _check_microphone() -> tuple[bool, str]:
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"]
                pa.terminate()
                log.info(f"Microphone found: {name}")
                return True, name
        pa.terminate()
    except Exception:
        pass
    return False, ""


def _check_speaker() -> tuple[bool, str, str]:
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                name = info["name"].lower()
                pa.terminate()
                if "bluetooth" in name or "bt" in name:
                    log.info(f"Bluetooth speaker found: {info['name']}")
                    return True, "bluetooth", ""
                if "usb" in name:
                    return True, "usb", ""
                return True, "aux", ""
        pa.terminate()
    except Exception:
        pass
    # AUX is always assumed available on Pi
    return True, "aux", ""


def _check_display() -> tuple[bool, str]:
    import platform
    if os.getenv("MOCK_HARDWARE", "false").lower() == "true":
        return True, "mock"
    # On Windows/Mac there's no SPI display — use web UI
    if platform.system() != "Linux":
        return True, "web"
    # SPI devices indicate a display is wired (Pi only)
    if Path("/dev/spidev0.0").exists():
        log.info("SPI display detected")
        return True, "ili9341"
    # I2C OLED
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", "1"], capture_output=True, text=True, timeout=3
        )
        if "3c" in result.stdout or "3d" in result.stdout:
            return True, "ssd1306"
    except Exception:
        pass
    return True, "web"   # fall back to web display on Linux too


def _check_arduino() -> tuple[bool, str]:
    """Detect a connected Arduino (or USB serial device) by scanning serial ports."""
    import glob as _glob
    import sys as _sys

    # Linux / Raspberry Pi
    candidates: list[str] = _glob.glob("/dev/ttyUSB*") + _glob.glob("/dev/ttyACM*")

    # Windows
    if _sys.platform == "win32":
        try:
            import serial.tools.list_ports
            candidates += [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            pass

    if candidates:
        log.info("Arduino detected: %s", candidates[0])
        return True, candidates[0]

    # Try arduino-cli if installed
    try:
        r = subprocess.run(
            ["arduino-cli", "board", "list"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "COM" in line or "/dev/tty" in line:
                port = line.split()[0]
                log.info("Arduino detected via arduino-cli: %s", port)
                return True, port
    except Exception:
        pass

    return False, ""


def _check_differential_drive(config: dict) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    """Return (enabled, left_pins, right_pins) from wheel_drive config block."""
    wd = config.get("wheel_drive", {})
    if not wd.get("enabled", False):
        return False, (0, 0), (0, 0)
    left = tuple(wd.get("left_pins", [17, 27]))[:2]
    right = tuple(wd.get("right_pins", [22, 23]))[:2]
    log.info("Differential drive configured: left=%s right=%s", left, right)
    return True, left, right


def _check_motors() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", "1"], capture_output=True, text=True, timeout=3
        )
        if "40" in result.stdout:  # PCA9685 default address
            log.info("PCA9685 motor driver detected at I2C 0x40")
            return True, "pantilt"
    except Exception:
        pass
    return False, "none"


def _check_network() -> bool:
    import socket
    try:
        # TCP connect is more reliable than ICMP ping (works even when ping is blocked)
        sock = socket.create_connection(("8.8.8.8", 53), timeout=3)
        sock.close()
        return True
    except Exception:
        return False


def _check_bluetooth() -> bool:
    import platform
    if platform.system() == "Windows":
        return False   # BT managed by Windows natively; not used for speaker on laptop
    try:
        result = subprocess.run(
            ["bluetoothctl", "show"], capture_output=True, text=True, timeout=3
        )
        return "Powered: yes" in result.stdout
    except Exception:
        return False


def _check_cpu_temp() -> bool:
    return Path("/sys/class/thermal/thermal_zone0/temp").exists()


def scan_hardware(mock_mode: bool = False, config: dict | None = None) -> HardwareCapabilities:
    """Scan all hardware and return capabilities. Called once on boot.

    mock_mode behaviour:
      - Pi-specific hardware (GPIO, SPI, I2C, motors, BT speaker) → always off
      - Real AV hardware (USB mic, USB camera, laptop speakers) → detected normally
      - Falls back to fully-simulated caps only when even PyAudio/cv2 are missing
    """
    log.info("Scanning hardware capabilities...")

    env_mock = os.getenv("MOCK_HARDWARE", "false").lower() == "true"
    use_mock = mock_mode or env_mock

    cfg = config or {}

    if use_mock:
        log.info("Mock mode: detecting real AV hardware, mocking Pi peripherals...")
        # Try to detect real mic, camera, speaker on the laptop
        cam_ok, cam_src, cam_idx = _check_camera()
        mic_ok, mic_name         = _check_microphone()
        spk_ok, spk_type, _      = _check_speaker()
        net_ok                   = _check_network()

        ard_ok, ard_port = _check_arduino()

        caps = HardwareCapabilities(
            camera_available    = cam_ok,
            camera_source       = cam_src,
            camera_device_index = cam_idx,
            mic_available       = mic_ok,
            mic_device_name     = mic_name,
            speaker_available   = spk_ok,
            speaker_type        = spk_type,
            bluetooth_speaker_mac = "",
            display_available   = True,
            display_type        = "web",   # always web display in mock/laptop mode
            motor_available     = False,   # no real motors on laptop
            motor_type          = "none",
            gpio_available      = False,   # no GPIO on laptop
            gpio_devices        = [],
            network_available   = net_ok,
            bluetooth_available = False,   # no BT speaker mgmt on laptop
            is_raspberry_pi     = False,
            cpu_temp_available  = False,
            mock_mode           = True,
            has_differential_drive = False,
            wheel_left_pins     = (0, 0),
            wheel_right_pins    = (0, 0),
            has_arduino         = ard_ok,
            arduino_port        = ard_port,
        )
        log.info(
            f"Mock-mode scan — "
            f"camera={caps.camera_available}({caps.camera_source}) "
            f"mic={caps.mic_available}({caps.mic_device_name}) "
            f"speaker={caps.speaker_type} "
            f"display={caps.display_type} "
            f"motors=off gpio=off"
        )
        return caps

    # ── Full real scan (Pi or unmodified laptop without --mock) ───────────
    is_pi = _is_raspberry_pi()
    cam_ok, cam_src, cam_idx = _check_camera()
    mic_ok, mic_name = _check_microphone()
    spk_ok, spk_type, bt_mac = _check_speaker()
    disp_ok, disp_type = _check_display()
    motor_ok, motor_type = _check_motors() if is_pi else (False, "none")
    net_ok = _check_network()
    bt_ok = _check_bluetooth() if is_pi else False
    temp_ok = _check_cpu_temp()
    wd_ok, wd_left, wd_right = _check_differential_drive(cfg) if is_pi else (False, (0, 0), (0, 0))
    ard_ok, ard_port = _check_arduino()

    caps = HardwareCapabilities(
        camera_available=cam_ok,
        camera_source=cam_src,
        camera_device_index=cam_idx,
        mic_available=mic_ok,
        mic_device_name=mic_name,
        speaker_available=spk_ok,
        speaker_type=spk_type,
        bluetooth_speaker_mac=bt_mac,
        display_available=disp_ok,
        display_type=disp_type,
        motor_available=motor_ok,
        motor_type=motor_type,
        gpio_available=is_pi,
        network_available=net_ok,
        bluetooth_available=bt_ok,
        is_raspberry_pi=is_pi,
        cpu_temp_available=temp_ok,
        mock_mode=False,
        has_differential_drive=wd_ok,
        wheel_left_pins=wd_left,
        wheel_right_pins=wd_right,
        has_arduino=ard_ok,
        arduino_port=ard_port,
    )

    log.info(
        f"Hardware scan complete — "
        f"camera={caps.camera_available}({caps.camera_source}) "
        f"mic={caps.mic_available} "
        f"speaker={caps.speaker_type} "
        f"display={caps.display_type} "
        f"motors={caps.motor_type} "
        f"network={caps.network_available}"
    )
    return caps
