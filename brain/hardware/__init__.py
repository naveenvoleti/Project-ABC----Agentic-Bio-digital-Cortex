"""Hardware abstraction layer."""
from .hw_detector import HardwareCapabilities, scan_hardware

__all__ = ["HardwareCapabilities", "scan_hardware"]
