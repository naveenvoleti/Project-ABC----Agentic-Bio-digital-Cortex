"""
DisplayAgent — SPI screen rendering: eyes GIF, text, HUD overlay.
Maps to Mirror Neurons. Runs at display FPS via asyncio loop.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.display_driver import DisplayDriver
from brain.utils.logger import get_logger

log = get_logger(__name__)

EMOTIONS_DIR = Path("assets/emotions")
TICK_INTERVAL = 1 / 15  # 15 FPS


LOADING_GIF  = Path("assets/loading.gif")    # hourglass / spinner shown during boot
LOADING_MSGS = [                              # cycling status messages on loading screen
    "Loading memory...",
    "Starting agents...",
    "Warming up LLM...",
    "Almost ready...",
]


class DisplayAgent(BaseAgent):
    name = "display_agent"

    def __init__(self, bus: asyncio.Queue, display: DisplayDriver):
        super().__init__(bus)
        self._display = display
        self._current_emotion = "neutral"
        self._current_text = ""
        self._text_expires = 0.0
        self._privacy_mode = False
        self._boot_complete = False   # gate — block eye expressions until system ready

    async def start(self) -> None:
        await super().start()
        self._display.init()

        # ── Boot screen: hourglass GIF + cycling status text ─────────────────
        if LOADING_GIF.exists():
            self._display.show_gif(str(LOADING_GIF))
        else:
            self._display.show_text("Initializing...", color=(100, 200, 255))
        log.info("DisplayAgent started — showing boot screen until system ready")

        _msg_idx = 0
        _msg_interval = 2.0        # rotate label every 2 s
        _last_msg_time = time.time()

        while self._running:
            if self._privacy_mode:
                await asyncio.sleep(0.5)
                continue

            if not self._boot_complete:
                # Rotate loading message every 2 s; keep hourglass GIF ticking
                now = time.time()
                if now - _last_msg_time >= _msg_interval:
                    self._display.show_text(
                        LOADING_MSGS[_msg_idx % len(LOADING_MSGS)],
                        color=(100, 200, 255),
                    )
                    _msg_idx += 1
                    _last_msg_time = now
                self._display.tick_gif()
                await asyncio.sleep(TICK_INTERVAL)
                continue

            # ── Normal operation after boot ───────────────────────────────────
            if self._current_text and time.time() < self._text_expires:
                self._display.show_text(self._current_text)
            else:
                self._current_text = ""
                self._display.tick_gif()

            await asyncio.sleep(TICK_INTERVAL)

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.EMOTION_CHANGE:
            # Remember the emotion but don't render until boot is done
            self._current_emotion = message.data.get("to", "neutral").lower()
            if not self._boot_complete:
                return
            self._load_emotion_gif(self._current_emotion)

        elif message.type == MessageType.ACTION_DISPLAY:
            data = message.data

            # Boot-complete signal from __main__ — switch to normal eye display
            if data.get("boot_complete"):
                self._boot_complete = True
                log.info("DisplayAgent: boot complete — switching to eye expressions")
                self._load_emotion_gif(self._current_emotion)
                return

            if not self._boot_complete:
                return   # suppress all display commands until system is ready

            if "text" in data:
                self._current_text = data["text"]
                self._text_expires = time.time() + data.get("duration", 5.0)
            elif "gif" in data:
                gif_path = Path(data["gif"])
                if gif_path.exists():
                    self._display.show_gif(str(gif_path))

        elif message.type == MessageType.ACTION_SPEAK:
            if not self._boot_complete:
                return
            text = message.data.get("text", "")
            if text:
                self._current_text = text[:80]
                self._text_expires = time.time() + min(len(text) / 15, 8.0)

        elif message.type == MessageType.PRIVACY_MODE:
            self._privacy_mode = message.data.get("enabled", False)
            if self._privacy_mode:
                self._display.show_text("Privacy Mode", color=(100, 100, 100))
            else:
                self._display.clear()

    def _load_emotion_gif(self, emotion: str) -> None:
        gif_path = EMOTIONS_DIR / f"{emotion}.gif"
        if not gif_path.exists():
            gif_path = EMOTIONS_DIR / "neutral.gif"
        if gif_path.exists():
            self._display.show_gif(str(gif_path))
        log.debug(f"Display emotion → {emotion}")
