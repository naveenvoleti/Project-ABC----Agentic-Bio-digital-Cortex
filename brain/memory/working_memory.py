"""
Working Memory (L1) — volatile session cache with TTL.
Fast in-process dict. Flushed to episodic memory on session end.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Entry:
    value: Any
    expires_at: float  # 0 = never expires


class WorkingMemory:
    def __init__(self, default_ttl: int = 3600):
        self._store: dict[str, _Entry] = {}
        self.default_ttl = default_ttl
        # Conversation context window (ordered list of recent messages)
        self.context_window: list[dict] = []
        self.context_window_size = 20

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires = time.time() + ttl if ttl > 0 else 0
        self._store[key] = _Entry(value=value, expires_at=expires)

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return default
        if entry.expires_at and time.time() > entry.expires_at:
            del self._store[key]
            return default
        return entry.value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self.context_window.clear()

    @property
    def turn_count(self) -> int:
        """Number of user turns in the current context window."""
        return sum(1 for m in self.context_window if m["role"] == "user")

    def add_to_context(self, role: str, content: str) -> None:
        self.context_window.append({"role": role, "content": content})
        if len(self.context_window) > self.context_window_size:
            self.context_window.pop(0)

    def replace_with_summary(self, summary: str, keep_last: int = 4) -> None:
        """Compact: replace all but the last `keep_last` turns with a summary entry."""
        # Guard: -0 == 0 in Python so [-0:] returns the full list, not empty.
        recent = self.context_window[-keep_last:] if keep_last > 0 else []
        self.context_window = [{"role": "system", "content": f"## Conversation Summary\n{summary}"}] + recent

    def remove_last_assistant(self) -> None:
        """Replace the most recent assistant turn with an interruption marker.
        Removing it entirely leaves two consecutive user messages which causes
        the LLM to re-answer the old question. The marker tells it to move on."""
        for i in range(len(self.context_window) - 1, -1, -1):
            if self.context_window[i]["role"] == "assistant":
                self.context_window[i]["content"] = (
                    "[My previous response was interrupted mid-sentence. "
                    "I'll focus on the user's latest message instead.]"
                )
                return

    def clear_context(self) -> None:
        """Wipe the conversation context window (keeps key-value store intact)."""
        self.context_window.clear()

    def get_context(self) -> list[dict]:
        return list(self.context_window)

    def purge_expired(self) -> int:
        now = time.time()
        expired = [k for k, v in self._store.items() if v.expires_at and now > v.expires_at]
        for k in expired:
            del self._store[k]
        return len(expired)

    def dump(self) -> dict:
        self.purge_expired()
        return {k: v.value for k, v in self._store.items()}
