"""InteroceptionAgent — Insula mapping.

Polls hardware metrics via psutil and maps them to embodied 'feelings'
(overwhelmed, hot, busy, comfortable). Publishes INTERO_STATE so EmotionEngine
can colour the robot's emotional state with hardware-derived sensations.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

from .base_agent import AgentMessage, BaseAgent, MessageType

log = logging.getLogger(__name__)


# ── Thresholds (overridable via config) ─────────────────────────────────────
_CPU_OVERWHELM   = 85.0   # CPU% → "overwhelmed"
_CPU_BUSY        = 60.0   # CPU% → "busy"
_TEMP_HOT        = 70.0   # °C   → "hot"
_RAM_PRESSURE    = 85.0   # RAM% → "memory pressure"
_POLL_INTERVAL   = 10.0   # seconds between polls


def _read_cpu_temp() -> float:
    """Return CPU temperature in °C, or 0.0 if unavailable."""
    if not _PSUTIL_AVAILABLE:
        return 0.0
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return 0.0
        for key in ("cpu_thermal", "cpu-thermal", "coretemp", "k10temp", "acpitz"):
            if key in temps and temps[key]:
                return temps[key][0].current
        # fallback: first available sensor
        for entries in temps.values():
            if entries:
                return entries[0].current
    except (AttributeError, OSError):
        pass
    return 0.0


def _map_to_feeling(cpu: float, ram_pct: float, temp: float) -> dict[str, Any]:
    """Convert raw metrics to a semantic feeling dictionary."""
    fatigue = "low"
    discomfort = "low"
    overwhelm = "low"
    label = "comfortable"

    if cpu >= _CPU_OVERWHELM:
        fatigue = "high"
        overwhelm = "high"
        label = "overwhelmed"
    elif cpu >= _CPU_BUSY:
        fatigue = "medium"
        label = "busy"

    if temp >= _TEMP_HOT:
        discomfort = "high"
        label = "hot" if label == "comfortable" else label

    if ram_pct >= _RAM_PRESSURE:
        overwhelm = "high"
        label = "memory_pressure" if label == "comfortable" else label

    # v5.0 — Response-length modulation: suggest shorter responses under load
    # CognitionAgent reads this to shorten prompts when hardware is strained.
    if label in ("overwhelmed", "memory_pressure"):
        suggested_max_tokens = 60
    elif label in ("hot", "busy"):
        suggested_max_tokens = 120
    else:
        suggested_max_tokens = None   # no limit — comfortable state

    return {
        "cpu_pct":              round(cpu, 1),
        "ram_pct":              round(ram_pct, 1),
        "temp_c":               round(temp, 1),
        "fatigue":              fatigue,
        "discomfort":           discomfort,
        "overwhelm":            overwhelm,
        "label":                label,
        "suggested_max_tokens": suggested_max_tokens,
        "ts":                   time.time(),
    }


class InteroceptionAgent(BaseAgent):
    """Insula — maps hardware vitals to embodied feelings.

    Passive producer: no inbound message handling.
    Publishes INTERO_STATE whenever the hardware state changes.
    ECO mode triggers on CPU/RAM overwhelm OR when temp_c crosses thermal_eco_threshold —
    whichever comes first, independent of label-change gating.
    """

    name = "interoception_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        thermal_eco_threshold: float = 75.0,
        thinking_budget_default: int = 1024,
        thinking_budget_eco: int = 256,
    ):
        super().__init__(bus)
        self._last_label: str = ""
        self._last_eco_mode: bool = False
        self._thermal_eco_threshold = thermal_eco_threshold
        # V7.0 — Dynamic thinking budget for ER reasoning
        self._thinking_budget_default = thinking_budget_default
        self._thinking_budget_eco = thinking_budget_eco
        self._last_budget_emitted: int = thinking_budget_default

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        if not _PSUTIL_AVAILABLE:
            log.warning(
                "InteroceptionAgent: psutil not installed — hardware sensing disabled. "
                "Run: pip install psutil"
            )
            while self._running:
                await asyncio.sleep(60)
            return
        log.info("InteroceptionAgent: started — polling hardware every %.0fs", _POLL_INTERVAL)
        asyncio.create_task(self._poll_loop(), name="interoception-poll")
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        await super().stop()

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            await self._poll_once()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _poll_once(self) -> None:
        try:
            cpu   = psutil.cpu_percent(interval=None)
            ram   = psutil.virtual_memory().percent
            temp  = _read_cpu_temp()
            state = _map_to_feeling(cpu, ram, temp)

            # INTERO_STATE: publish only on label change (avoids bus flooding)
            if state["label"] != self._last_label:
                self._last_label = state["label"]
                await self.publish(AgentMessage(
                    type=MessageType.INTERO_STATE,
                    source=self.name,
                    data=state,
                    priority=6,
                ))
                log.debug("InteroceptionAgent: state → %s (cpu=%.1f%% ram=%.1f%% temp=%.1f°C)",
                          state["label"], cpu, ram, temp)

            # ECO decision is INDEPENDENT of label-change gating.
            # Triggers when CPU/RAM overwhelms OR temperature crosses thermal threshold —
            # whichever comes first, even if the label string hasn't changed yet.
            eco_needed = (
                state["label"] in ("overwhelmed", "memory_pressure")
                or (temp > 0.0 and temp >= self._thermal_eco_threshold)
            )
            if eco_needed != self._last_eco_mode:
                self._last_eco_mode = eco_needed
                trigger = (
                    "thermal" if temp >= self._thermal_eco_threshold
                    else "interoception"
                )
                await self.publish(AgentMessage(
                    type=MessageType.BEHAVIOR_CHANGE,
                    source=self.name,
                    data={"eco": eco_needed, "trigger": trigger, "label": state["label"],
                          "temp_c": state["temp_c"]},
                    priority=5,
                ))
                log.info("InteroceptionAgent: ECO → %s (trigger=%s label=%s temp=%.1f°C)",
                         eco_needed, trigger, state["label"], temp)

            # V7.0 — Dynamic thinking budget: emit THINKING_BUDGET on ECO transition
            # to throttle ER token usage during thermal stress.
            new_budget = (
                self._thinking_budget_eco if eco_needed else self._thinking_budget_default
            )
            if new_budget != self._last_budget_emitted:
                self._last_budget_emitted = new_budget
                await self.publish(AgentMessage(
                    type=MessageType.THINKING_BUDGET,
                    source=self.name,
                    data={
                        "budget": new_budget,
                        "trigger": "thermal" if eco_needed else "recovery",
                        "temp_c": state["temp_c"],
                    },
                    priority=4,
                ))
                log.info("InteroceptionAgent: THINKING_BUDGET → %d tokens (temp=%.1f°C)",
                         new_budget, temp)
        except Exception as exc:
            log.warning("InteroceptionAgent: poll error — %s", exc)

    # ── Inbound (unused — producer only) ─────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        return None
