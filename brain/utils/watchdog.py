"""
Watchdog — monitors asyncio agent tasks and restarts crashed agents.
Each agent registers itself; watchdog checks heartbeats every 30s.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any

from brain.utils.logger import get_logger

log = get_logger(__name__)

HEARTBEAT_TIMEOUT = 120    # seconds — passive agents can go quiet between events
RESTART_MAX = 5
RESTART_BACKOFF_BASE = 5   # seconds


@dataclass
class _AgentRecord:
    name: str
    factory: Callable[[], Coroutine[Any, Any, None]]
    task: asyncio.Task | None = None
    last_heartbeat: float = field(default_factory=time.time)
    restart_count: int = 0
    enabled: bool = True


class Watchdog:
    def __init__(self):
        self._agents: dict[str, _AgentRecord] = {}
        self._running = False

    def register(self, name: str, factory: Callable[[], Coroutine]) -> None:
        self._agents[name] = _AgentRecord(name=name, factory=factory)

    def heartbeat(self, name: str) -> None:
        if name in self._agents:
            self._agents[name].last_heartbeat = time.time()

    def get_task(self, name: str) -> asyncio.Task | None:
        rec = self._agents.get(name)
        return rec.task if rec else None

    def set_task(self, name: str, task: asyncio.Task) -> None:
        if name in self._agents:
            self._agents[name].task = task

    async def start(self) -> None:
        self._running = True
        log.info("Watchdog started")
        while self._running:
            await asyncio.sleep(30)
            await self._check_agents()

    async def _check_agents(self) -> None:
        now = time.time()
        for name, rec in self._agents.items():
            if not rec.enabled:
                continue
            # Check task alive
            task_dead = rec.task is None or rec.task.done()
            heartbeat_stale = (now - rec.last_heartbeat) > HEARTBEAT_TIMEOUT

            if task_dead or heartbeat_stale:
                reason = "task_dead" if task_dead else "heartbeat_timeout"
                log.warning(f"Agent {name} appears dead ({reason}) — restarting")
                await self._restart(rec)

    async def _restart(self, rec: _AgentRecord) -> None:
        if not self._running:
            return   # don't restart during intentional shutdown
        if rec.restart_count >= RESTART_MAX:
            log.error(f"Agent {rec.name} exceeded max restarts ({RESTART_MAX}), disabling")
            rec.enabled = False
            return

        backoff = RESTART_BACKOFF_BASE * (2 ** rec.restart_count)
        log.info(f"Restarting {rec.name} in {backoff}s (attempt {rec.restart_count + 1})")
        await asyncio.sleep(backoff)

        try:
            coro = rec.factory()
            task = asyncio.create_task(coro, name=rec.name)
            rec.task = task
            rec.restart_count += 1
            rec.last_heartbeat = time.time()
            log.info(f"Agent {rec.name} restarted successfully")
        except Exception as e:
            log.error(f"Failed to restart {rec.name}: {e}")

    def stop(self) -> None:
        self._running = False
