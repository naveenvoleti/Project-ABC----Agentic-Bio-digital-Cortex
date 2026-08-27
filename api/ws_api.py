"""
WebSocket API â€” real-time event stream for monitoring the brain.
Connects at ws://brain.local:8080/ws
Events are JSON-serialized AgentMessage objects.
"""
from __future__ import annotations

import asyncio
import json
import math
from collections import deque

from fastapi import WebSocket, WebSocketDisconnect
from brain.utils.logger import get_logger

log = get_logger(__name__)

# â”€â”€ Boot status â€” updated by __main__.py during initialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_boot_status: str = "Starting upâ€¦"
_boot_log: deque = deque(maxlen=6)   # last 6 messages shown in the browser


def set_boot_status(msg: str) -> None:
    """Call from __main__.py at each major boot step."""
    global _boot_status
    _boot_status = msg
    _boot_log.append(msg)


def _safe_default(obj):
    """JSON encoder fallback â€” handles numpy arrays, floats, and other non-serializable objects."""
    # numpy scalars / arrays
    if hasattr(obj, "tolist"):
        return obj.tolist()
    # NaN / Inf floats (not valid JSON)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    # Bytes â†’ base64 string
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"
    # Enums, dataclasses, other objects
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)

_orchestrator = None


def set_orchestrator(orch) -> None:
    global _orchestrator
    _orchestrator = orch


async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info(f"WebSocket client connected: {websocket.client}")

    # â”€â”€ Wait for the orchestrator to be wired in (brain is still initialising) â”€â”€
    # Instead of immediately closing, hold the connection and send "initializing"
    # heartbeats so the browser loading screen stays alive.
    if not _orchestrator:
        import time
        deadline = time.monotonic() + 300  # wait up to 5 minutes
        while not _orchestrator:
            if time.monotonic() > deadline:
                log.warning("WS: orchestrator never became available â€” closing")
                await websocket.close()
                return
            try:
                await websocket.send_text(json.dumps({
                    "type": "initializing",
                    "status": _boot_status,
                    "log": list(_boot_log),
                }))
            except Exception:
                return  # client disconnected while waiting
            await asyncio.sleep(0.25)

    queue = _orchestrator.subscribe_ws()

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=5.0)
                payload = {
                    "type": msg.type.value,
                    "source": msg.source,
                    "data": msg.data,
                }
                await websocket.send_text(json.dumps(payload, default=_safe_default))
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                except Exception:
                    break
    except (WebSocketDisconnect, asyncio.CancelledError):
        # CancelledError fires on clean brain shutdown â€” not an error
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.warning(f"WebSocket error: {e}")
    finally:
        if _orchestrator:
            _orchestrator.unsubscribe_ws(queue)

