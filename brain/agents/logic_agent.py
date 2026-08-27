"""
LogicAgent — rule-based safety checks and command validation.
Maps to the Basal Ganglia. Filters harmful or nonsensical actions.
"""
from __future__ import annotations

import asyncio
import re

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)

BLOCKED_PATTERNS = [
    r"\b(delete|rm -rf|format|wipe|destroy|kill process|shutdown now)\b",
    r"\b(send email|post to|upload to)\b.*\b(without asking|automatically)\b",
    r"\b(hack|exploit|attack|inject)\b",
]

MOTOR_SPEED_MAX = 80


class LogicAgent(BaseAgent):
    name = "logic_agent"

    def __init__(self, bus: asyncio.Queue):
        super().__init__(bus)

    async def start(self) -> None:
        await super().start()
        log.info("LogicAgent started")
        while self._running:
            await asyncio.sleep(1)

    def is_safe(self, text: str) -> tuple[bool, str]:
        text_lower = text.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, text_lower):
                return False, f"Blocked by safety rule: {pattern}"
        return True, ""

    def validate_motor_command(self, data: dict) -> dict:
        speed = data.get("speed", 50)
        data["speed"] = min(abs(speed), MOTOR_SPEED_MAX)
        return data

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.type == MessageType.COGNITION_RESPONSE:
            response = message.data.get("response", "")
            safe, reason = self.is_safe(response)
            if not safe:
                log.warning(f"LogicAgent blocked response: {reason}")
                await self.publish(AgentMessage(
                    type=MessageType.SYSTEM_ERROR,
                    source=self.name,
                    data={"error": f"Safety block: {reason}"},
                    priority=1,
                ))
                return None

        if message.type == MessageType.ACTION_MOVE:
            message.data = self.validate_motor_command(message.data)

        return message
