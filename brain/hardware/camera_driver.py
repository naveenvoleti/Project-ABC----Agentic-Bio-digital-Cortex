"""Camera driver — USB and CSI camera abstraction."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Generator

from brain.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)


class CameraDriver:
    def __init__(self, source: str = "usb", device_index: int = 0, mock: bool = False):
        self.source = source
        self.device_index = device_index
        self.mock = mock
        self._cap = None
        self._lock = threading.Lock()
        self._last_frame = None  # numpy array at runtime

    def open(self) -> bool:
        if self.mock:
            log.info("Camera: mock mode enabled")
            return True
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.device_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 15)
            ok = self._cap.isOpened()
            if ok:
                log.info(f"Camera opened: {self.source} device {self.device_index}")
            else:
                log.warning("Camera failed to open")
            return ok
        except Exception as e:
            log.error(f"Camera open error: {e}")
            return False

    def capture_frame(self):
        if self.mock:
            try:
                import numpy as np
                return np.zeros((480, 640, 3), dtype=np.uint8)
            except ImportError:
                return None
        if self._cap is None or not self._cap.isOpened():
            return None
        with self._lock:
            ok, frame = self._cap.read()
            if ok:
                self._last_frame = frame
                return frame
        return None

    def get_last_frame(self):
        return self._last_frame

    def stream(self) -> Generator:
        while True:
            frame = self.capture_frame()
            if frame is not None:
                yield frame

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
            log.info("Camera closed")
