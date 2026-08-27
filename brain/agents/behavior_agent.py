"""
BehaviorAgent — top-level behavioral state machine: IDLE/ATTENTIVE/FOCUSED/EXPLORING.
Maps to the Cingulate Cortex.
"""
from __future__ import annotations

import asyncio
import collections
import time
from enum import Enum

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)

IDLE_TIMEOUT        = 300   # seconds of silence before going IDLE
CURIOSITY_DELAY     = 30    # seconds after IDLE before first curiosity trigger
CURIOSITY_INTERVAL  = 60    # seconds between repeated curiosity triggers while idle
REFLECT_INTERVAL    = 1800  # self-reflect every 30 minutes of activity
ECO_UPTIME_HOURS    = 8     # minimum uptime before ECO eligibility
ECO_MAX_HOURLY      = 2     # interactions/hour below this → ECO mode
ECO_CHECK_INTERVAL  = 60    # seconds between fatigue checks


class BehaviorState(str, Enum):
    IDLE      = "IDLE"
    ATTENTIVE = "ATTENTIVE"
    FOCUSED   = "FOCUSED"
    EXPLORING = "EXPLORING"
    ECO       = "ECO"


class BehaviorAgent(BaseAgent):
    name = "behavior_agent"

    def __init__(self, bus: asyncio.Queue, idle_timeout: int = IDLE_TIMEOUT):
        super().__init__(bus)
        self._state = BehaviorState.IDLE
        self._last_activity = time.time()
        self._idle_timeout = idle_timeout
        self._last_reflect = time.time()
        self._last_curiosity = 0.0   # epoch of last CURIOSITY_TRIGGER sent

        # Fatigue / ECO tracking
        self._start_time: float = time.time()
        self._eco_mode: bool = False
        self._hour_interactions: int = 0       # interactions in current hour
        self._hourly_buckets: collections.deque = collections.deque(maxlen=8)  # last 8 hours
        self._last_hour_boundary: float = time.time()
        self._last_eco_check: float = time.time()

    async def start(self) -> None:
        await super().start()
        log.info("BehaviorAgent started")
        while self._running:
            await asyncio.sleep(5)
            await self._check_idle()
            await self._check_fatigue()

    async def _check_idle(self) -> None:
        now = time.time()
        elapsed = now - self._last_activity

        # Transition to IDLE after silence
        if elapsed > self._idle_timeout and self._state not in (
            BehaviorState.IDLE, BehaviorState.EXPLORING
        ):
            await self._set_state(BehaviorState.IDLE)

        # Fire curiosity when idle — first trigger after CURIOSITY_DELAY, then every CURIOSITY_INTERVAL
        if self._state in (BehaviorState.IDLE, BehaviorState.EXPLORING):
            since_last = now - self._last_curiosity
            first_trigger = elapsed > (self._idle_timeout + CURIOSITY_DELAY) and self._last_curiosity == 0
            repeat_trigger = self._last_curiosity > 0 and since_last > CURIOSITY_INTERVAL
            if first_trigger or repeat_trigger:
                self._last_curiosity = now
                await self.publish(AgentMessage(
                    type=MessageType.CURIOSITY_TRIGGER,
                    source=self.name,
                    data={"idle_seconds": elapsed},
                    priority=7,
                ))

        # Periodic self-reflection — every 30 minutes of being active
        if (now - self._last_reflect) > REFLECT_INTERVAL and self._state != BehaviorState.IDLE:
            self._last_reflect = now
            log.info("BehaviorAgent: triggering scheduled self-reflection")
            await self.publish(AgentMessage(
                type=MessageType.SELF_REFLECT,
                source=self.name,
                data={"reason": "scheduled", "uptime": elapsed},
                priority=9,
            ))

    async def _check_fatigue(self) -> None:
        """Circadian-inspired ECO mode — enter low-power state after long uptime with
        few interactions. Mirrors how the brain down-regulates during low-stimulus periods."""
        now = time.time()

        # Roll hourly bucket every 3600s
        if (now - self._last_hour_boundary) >= 3600:
            self._hourly_buckets.append(self._hour_interactions)
            self._hour_interactions = 0
            self._last_hour_boundary = now

        # Only check every ECO_CHECK_INTERVAL seconds
        if (now - self._last_eco_check) < ECO_CHECK_INTERVAL:
            return
        self._last_eco_check = now

        uptime_hours = (now - self._start_time) / 3600

        # Need at least ECO_UPTIME_HOURS uptime and 2 hours of interaction data
        if uptime_hours < ECO_UPTIME_HOURS or len(self._hourly_buckets) < 2:
            return

        avg_hourly = sum(self._hourly_buckets) / len(self._hourly_buckets)

        if not self._eco_mode and avg_hourly < ECO_MAX_HOURLY:
            await self._enter_eco()
        elif self._eco_mode and avg_hourly >= ECO_MAX_HOURLY:
            await self._exit_eco()

    async def _enter_eco(self) -> None:
        self._eco_mode = True
        log.info(
            f"BehaviorAgent: entering ECO mode — "
            f"uptime={((time.time()-self._start_time)/3600):.1f}h, "
            f"avg interactions/hour={sum(self._hourly_buckets)/max(1,len(self._hourly_buckets)):.1f}"
        )
        await self.publish(AgentMessage(
            type=MessageType.BEHAVIOR_CHANGE,
            source=self.name,
            data={"state": BehaviorState.ECO.value, "eco": True},
            priority=6,
        ))

    async def _exit_eco(self) -> None:
        self._eco_mode = False
        log.info("BehaviorAgent: exiting ECO mode — activity detected")
        await self.publish(AgentMessage(
            type=MessageType.BEHAVIOR_CHANGE,
            source=self.name,
            data={"state": self._state.value, "eco": False},
            priority=5,
        ))

    async def _set_state(self, new_state: BehaviorState) -> None:
        if new_state == self._state:
            return
        old = self._state.value
        self._state = new_state
        log.info(f"Behavior: {old} → {new_state.value}")
        await self.publish(AgentMessage(
            type=MessageType.BEHAVIOR_CHANGE,
            source=self.name,
            data={"state": new_state.value, "previous": old},
            priority=5,
        ))

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.PERCEPTION_SPEECH:
            # Count interaction for fatigue tracking
            self._hour_interactions += 1
            # Exit ECO immediately on any user speech
            if self._eco_mode:
                await self._exit_eco()
            # Only wake from idle on speech — don't reset the idle timer yet.
            # The timer resets only when the brain actually processes it (COGNITION_INTENT).
            # This prevents ambient/background noise from blocking autonomous behaviour.
            if self._state in (BehaviorState.IDLE, BehaviorState.EXPLORING):
                await self._set_state(BehaviorState.ATTENTIVE)

        elif message.type in (
            MessageType.PERCEPTION_VISION,
            MessageType.PERCEPTION_SENSOR,
        ):
            # Sensor/vision events are real activity — reset timer
            self._last_activity = time.time()
            if self._state in (BehaviorState.IDLE, BehaviorState.EXPLORING):
                await self._set_state(BehaviorState.ATTENTIVE)

        elif message.type == MessageType.COGNITION_INTENT:
            # Real user interaction confirmed — reset idle timer
            self._last_activity = time.time()
            self._last_curiosity = 0.0
            await self._set_state(BehaviorState.FOCUSED)

        elif message.type == MessageType.COGNITION_RESPONSE:
            # Response generated — transition back to ATTENTIVE (ready for next input)
            self._last_activity = time.time()
            await self._set_state(BehaviorState.ATTENTIVE)

        elif message.type == MessageType.CURIOSITY_TRIGGER:
            await self._set_state(BehaviorState.EXPLORING)

        elif message.type == MessageType.ACTION_SPEAK:
            # Any speech (user response or autonomous thought) resets the idle clock
            # so curiosity doesn't fire again immediately after the brain just spoke
            self._last_activity = time.time()
