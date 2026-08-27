"""
Display driver — SPI LCD / OLED / Mock rendering.
Supports ILI9341, ST7789 (240x320), SSD1306 (128x64), and mock (log only).
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Sequence

from brain.utils.logger import get_logger

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    Image = ImageDraw = ImageFont = None
    _PIL_AVAILABLE = False

log = get_logger(__name__)


class DisplayDriver:
    def __init__(
        self,
        display_type: str = "mock",
        width: int = 240,
        height: int = 320,
        mock: bool = False,
    ):
        self.display_type = display_type
        self.width = width
        self.height = height
        self.mock = mock or display_type == "mock"
        self._device = None
        self._font_sm: ImageFont.FreeTypeFont | None = None
        self._font_md: ImageFont.FreeTypeFont | None = None
        self._gif_frames: list[Image.Image] = []
        self._gif_durations: list[int] = []
        self._current_gif_frame = 0
        self._last_frame_time = 0.0

    def init(self) -> bool:
        if self.mock:
            log.info("Display: mock mode enabled")
            self._load_fonts()
            return True
        try:
            if self.display_type in ("ili9341", "st7789"):
                from luma.lcd.device import ili9341, st7789
                from luma.core.interface.serial import spi
                serial = spi(port=0, device=0, gpio_DC=24, gpio_RST=25)
                if self.display_type == "ili9341":
                    self._device = ili9341(serial, width=self.width, height=self.height)
                else:
                    self._device = st7789(serial, width=self.width, height=self.height)
            elif self.display_type == "ssd1306":
                from luma.oled.device import ssd1306
                from luma.core.interface.serial import i2c
                serial = i2c(port=1, address=0x3C)
                self._device = ssd1306(serial)
                self.width = self._device.width
                self.height = self._device.height
            self._load_fonts()
            log.info(f"Display initialized: {self.display_type} {self.width}×{self.height}")
            return True
        except Exception as e:
            log.error(f"Display init error: {e} — falling back to mock")
            self.mock = True
            return True

    def _load_fonts(self) -> None:
        try:
            self._font_sm = ImageFont.truetype("assets/fonts/liberation-mono.ttf", 14)
            self._font_md = ImageFont.truetype("assets/fonts/liberation-mono.ttf", 20)
        except Exception:
            self._font_sm = ImageFont.load_default()
            self._font_md = self._font_sm

    def _render(self, image: Image.Image) -> None:
        if self.mock:
            return
        if self._device:
            self._device.display(image.convert(self._device.mode))

    def show_image(self, image: Image.Image) -> None:
        img = image.resize((self.width, self.height))
        self._render(img)

    def show_text(
        self,
        text: str,
        color: tuple[int, int, int] = (255, 255, 255),
        bg: tuple[int, int, int] = (0, 0, 30),
    ) -> None:
        img = Image.new("RGB", (self.width, self.height), bg)
        draw = ImageDraw.Draw(img)
        font = self._font_md or ImageFont.load_default()

        # Word-wrap
        words = text.split()
        lines, line = [], ""
        for word in words:
            test = f"{line} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] > self.width - 10:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        y = max(10, (self.height - len(lines) * 24) // 2)
        for ln in lines:
            draw.text((10, y), ln, font=font, fill=color)
            y += 24

        self._render(img)

    def load_gif(self, path: str | Path) -> bool:
        try:
            gif = Image.open(path)
            self._gif_frames = []
            self._gif_durations = []
            for frame_idx in range(getattr(gif, "n_frames", 1)):
                gif.seek(frame_idx)
                frame = gif.convert("RGB").resize((self.width, self.height))
                self._gif_frames.append(frame)
                self._gif_durations.append(gif.info.get("duration", 100))
            self._current_gif_frame = 0
            return True
        except Exception as e:
            log.error(f"GIF load error {path}: {e}")
            return False

    def tick_gif(self) -> None:
        if not self._gif_frames:
            return
        now = time.time() * 1000
        duration = self._gif_durations[self._current_gif_frame]
        if now - self._last_frame_time >= duration:
            self._render(self._gif_frames[self._current_gif_frame])
            self._current_gif_frame = (self._current_gif_frame + 1) % len(self._gif_frames)
            self._last_frame_time = now

    def show_gif(self, path: str | Path) -> None:
        if self.load_gif(path):
            log.debug(f"GIF loaded: {path} ({len(self._gif_frames)} frames)")

    def show_overlay(self, layers: Sequence[Image.Image]) -> None:
        base = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        for layer in layers:
            base.paste(layer.resize((self.width, self.height)), (0, 0))
        self._render(base)

    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        img = Image.new("RGB", (self.width, self.height), color)
        self._render(img)
