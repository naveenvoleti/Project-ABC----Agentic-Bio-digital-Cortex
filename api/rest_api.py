"""
FastAPI REST API — external control and monitoring for Project-ABC.
Accessible at http://localhost:8080
Display UI at http://localhost:8080/display
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from brain.agents.base_agent import AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
EMOTIONS_DIR = Path("assets/emotions")
CONFIG_FILE = Path("config/config.yaml")

app = FastAPI(
    title="Project-ABC Brain API",
    description="Control and monitor the robotic brain",
    version="1.0.0",
)

# Serve emotion GIFs at /emotions/<name>.gif
if EMOTIONS_DIR.exists():
    app.mount("/emotions", StaticFiles(directory=str(EMOTIONS_DIR)), name="emotions")

# Serve static UI files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Injected by __main__.py
_orchestrator = None
_restart_callback = None


def set_orchestrator(orch) -> None:
    global _orchestrator
    _orchestrator = orch


def set_restart_callback(cb) -> None:
    global _restart_callback
    _restart_callback = cb


# ── Request / Response Models ─────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str
    interrupt: bool = False


class DisplayRequest(BaseModel):
    text: str | None = None
    gif: str | None = None
    duration: float = 5.0


class MoveRequest(BaseModel):
    command: str                # pan | tilt | look_at | scan | center | forward
    degrees: float | None = None
    pan: float | None = None
    tilt: float | None = None
    speed: int = 50
    duration: float = 1.0


class AskRequest(BaseModel):
    text: str
    system_override: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/status")
async def get_status() -> dict:
    if not _orchestrator:
        return {"status": "offline"}
    emotion_agent = _orchestrator._agents.get("emotion_engine")
    behavior_agent = _orchestrator._agents.get("behavior_agent")
    return {
        "status": "online",
        "emotion": emotion_agent.current_emotion if emotion_agent else "UNKNOWN",
        "behavior": behavior_agent._state.value if behavior_agent else "UNKNOWN",
        "agents": list(_orchestrator._agents.keys()),
        "hw": {
            "camera": _orchestrator._hw.camera_available,
            "mic": _orchestrator._hw.mic_available,
            "display": _orchestrator._hw.display_available,
            "motors": _orchestrator._hw.motor_available,
        },
    }


@app.get("/memory/episodes")
async def get_episodes(limit: int = 20, session_only: bool = False) -> list:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    return _orchestrator._episodic.get_recent(n=limit, session_only=session_only)


@app.get("/memory/search")
async def search_memory(q: str, limit: int = 10) -> list:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    return _orchestrator._episodic.search(q, limit=limit)


@app.post("/speak")
async def speak(req: SpeakRequest) -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    await _orchestrator.inject(AgentMessage(
        type=MessageType.ACTION_SPEAK,
        source="api",
        data={"text": req.text, "interrupt": req.interrupt},
        priority=2,
    ))
    return {"status": "queued", "text": req.text}


@app.post("/display")
async def display(req: DisplayRequest) -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    data: dict[str, Any] = {"duration": req.duration}
    if req.text:
        data["text"] = req.text
    if req.gif:
        data["gif"] = req.gif
    await _orchestrator.inject(AgentMessage(
        type=MessageType.ACTION_DISPLAY,
        source="api",
        data=data,
        priority=3,
    ))
    return {"status": "sent"}


@app.post("/move")
async def move(req: MoveRequest) -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    await _orchestrator.inject(AgentMessage(
        type=MessageType.ACTION_MOVE,
        source="api",
        data=req.dict(),
        priority=4,
    ))
    return {"status": "sent"}


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    await _orchestrator.inject(AgentMessage(
        type=MessageType.PERCEPTION_SPEECH,
        source="api",
        data={"text": req.text, "is_wake_word": True, "interrupt": True},
        priority=2,
    ))
    return {"status": "processing", "text": req.text}


@app.get("/soul")
async def get_soul() -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    return {"soul": _orchestrator._soul.soul}


@app.put("/soul")
async def update_soul(body: dict) -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    text = body.get("append", "")
    if text:
        _orchestrator._soul.update_soul(text)
    return {"status": "updated"}


@app.post("/dream")
async def trigger_dream() -> dict:
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Brain offline")
    await _orchestrator.inject(AgentMessage(
        type=MessageType.DREAM_START,
        source="api",
        data={},
        priority=9,
    ))
    return {"status": "dream_triggered"}


@app.get("/world")
async def get_world() -> dict:
    """Return current WorldModel snapshot — what the brain perceives right now."""
    if not _orchestrator:
        return {"scene": "", "present_entities": [], "audio": "", "gaze_direct": False}
    wm = getattr(_orchestrator, "_world_model", None)
    if not wm:
        return {"scene": "", "present_entities": [], "audio": "", "gaze_direct": False}
    return await wm.snapshot()


@app.get("/workspace")
async def get_workspace() -> dict:
    """v5.0 GWT — Return the current Spotlight (top-3 most salient messages in the Blackboard)."""
    if not _orchestrator:
        return {"spotlight": [], "spotlight_size": 3, "pulse_hz": 10}
    spotlight = _orchestrator.get_spotlight()
    gwt_cfg = _orchestrator._config.get("cognition_extensions", {}).get("attention", {})
    return {
        "spotlight":      spotlight,
        "spotlight_size": gwt_cfg.get("spotlight_size", 3),
        "salience_threshold": gwt_cfg.get("salience_threshold", 0.4),
        "pulse_hz":       _orchestrator._pulse_hz,
    }


@app.get("/drives")
async def get_drives() -> dict:
    """v5.0 Homeostatic Drives — Return current Curiosity, Social, and Energy drive levels."""
    if not _orchestrator:
        return {"drives": {}, "homeostatic": {}, "top_drive": "unknown"}
    agent = _orchestrator._agents.get("intrinsic_motivation_agent")
    if not agent:
        return {"drives": {}, "homeostatic": {}, "top_drive": "unknown"}
    drives = dict(getattr(agent, "_drives", {}))
    homeo  = dict(getattr(agent, "_homeo", {}))
    top    = max(drives, key=drives.get) if drives else "unknown"
    return {
        "drives":         {k: round(v, 3) for k, v in drives.items()},
        "homeostatic":    {k: round(v, 3) for k, v in homeo.items()},
        "top_drive":      top,
        "person_present": getattr(agent, "_person_present", False),
        "energy_eco_threshold": 0.2,
    }


@app.get("/snapshot")
async def snapshot():
    """Return the latest camera frame as a JPEG image (poll at desired fps from client)."""
    import io
    frame = None
    if _orchestrator:
        va = _orchestrator._agents.get("vision_agent")
        if va:
            frame = getattr(va, "_current_frame", None)

    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    try:
        import cv2
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ret:
            raise ValueError("encode failed")
        jpeg_bytes = buf.tobytes()
    except Exception:
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=65)
            jpeg_bytes = buf.getvalue()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    from fastapi.responses import Response
    return Response(content=jpeg_bytes, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/display")
async def display_ui() -> FileResponse:
    """Serve the brain display UI page."""
    html_path = STATIC_DIR / "display.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Display UI not found")
    return FileResponse(str(html_path), media_type="text/html",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/")
async def root() -> FileResponse:
    """Redirect root to display UI."""
    html_path = STATIC_DIR / "display.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return JSONResponse({"status": "Brain API online", "display": "/display", "docs": "/docs"})


# ── Config endpoints ──────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _drop_none(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _drop_none(v) for k, v in d.items() if v is not None}
    return d


@app.get("/config")
async def get_config() -> dict:
    """Read config.yaml and return as JSON."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="config.yaml not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/restart")
async def restart_brain() -> dict:
    """Signal the brain to stop and restart via the main loop."""
    async def _do_restart():
        await asyncio.sleep(0.3)
        if _restart_callback:
            _restart_callback()
    asyncio.create_task(_do_restart())
    return {"status": "restarting"}


@app.post("/config")
async def update_config(body: dict) -> dict:
    """Deep-merge body into config.yaml and save. Restart brain to apply."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        patch = _drop_none(body)
        _deep_merge(cfg, patch)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=120)
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
