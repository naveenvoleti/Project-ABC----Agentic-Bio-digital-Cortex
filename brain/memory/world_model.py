"""
WorldModel — shared ground truth for the conscious brain.

All perceptual agents write here; CognitionAgent reads before every LLM call.
Acts as the brain's "present moment awareness" — what is actually happening right now.

V10.0: Persistent across restarts via save()/load(). The brain remembers what it
last saw, heard, and where things were — even after a reboot.

Structure:
  - presence: who/what is in the scene with timestamps and expiry
  - scene:    latest SmolVLM2 scene description
  - audio:    latest ambient audio summary
  - gaze:     whether the user is looking at the camera
  - emotions: detected user facial/voice emotion
  - body_pose: cumulative pan/tilt/heading from ProprioceptionAgent (V10.0)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_PRESENCE_EXPIRY_S = 30.0   # entity expires after 30s without update
_SCENE_EXPIRY_S    = 10.0   # scene description expires after 10s


@dataclass
class PresenceEntry:
    name: str
    last_seen: float = field(default_factory=time.monotonic)
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        t = now or time.monotonic()
        return (t - self.last_seen) > _PRESENCE_EXPIRY_S

    def touch(self, confidence: float = 1.0, **attrs) -> None:
        self.last_seen = time.monotonic()
        self.confidence = confidence
        self.attributes.update(attrs)


class WorldModel:
    """Asyncio-safe shared world state. Written by perceptual agents, read by cognition.

    Uses asyncio.Lock so callers never block the event loop thread — unlike
    threading.Lock, which would stall the 10Hz Pulse if acquired under contention.
    All public methods are async; call them with `await`.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # Who is in the scene right now: name → PresenceEntry
        self._presence: dict[str, PresenceEntry] = {}

        # Latest SmolVLM2 scene description
        self._scene_description: str = ""
        self._scene_ts: float = 0.0

        # Previous scene (for change detection)
        self._prev_scene: str = ""

        # Ambient audio description (non-speech events)
        self._audio_description: str = ""
        self._audio_ts: float = 0.0

        # Gaze state: is user looking directly at camera
        self._gaze_direct: bool = False
        self._gaze_ts: float = 0.0

        # Detected user emotion from face/voice
        self._user_emotion: str = ""

        # Environment context (lighting, location hints, etc.)
        self._environment: dict[str, Any] = {}

        # v5.0 Interoceptive grounding — hardware body state from InteroceptionAgent
        self._intero: dict[str, Any] = {}   # label, cpu_pct, ram_pct, temp_c, suggested_max_tokens

        # N3c — Internal monologue: latest private thought from ReasoningAgent
        self._current_thought: str = ""

        # V7.0 Spatial Intelligence — 3D object list from VisionProcessor.detect_3d()
        self._spatial_objects: list[dict[str, Any]] = []
        self._spatial_ts: float = 0.0

        # V10.0 — Body pose from ProprioceptionAgent (pan_deg, tilt_deg, heading_deg, distance_cm)
        self._body_pose: dict[str, float] = {"pan_deg": 0.0, "tilt_deg": 0.0,
                                              "heading_deg": 0.0, "distance_cm": 0.0}

    # ── Presence tracking ────────────────────────────────────────────────────

    async def update_presence(self, name: str, confidence: float = 1.0, **attrs) -> bool:
        """Update presence of a named entity. Returns True if this is a NEW entity."""
        async with self._lock:
            is_new = name not in self._presence or self._presence[name].is_expired()
            if name in self._presence:
                self._presence[name].touch(confidence, **attrs)
            else:
                self._presence[name] = PresenceEntry(name=name, confidence=confidence,
                                                     attributes=attrs)
            return is_new

    async def remove_presence(self, name: str) -> bool:
        """Explicitly remove an entity. Returns True if it was present."""
        async with self._lock:
            if name in self._presence:
                del self._presence[name]
                return True
            return False

    async def get_present_entities(self) -> list[str]:
        """Return names of all non-expired entities currently present."""
        now = time.monotonic()
        async with self._lock:
            expired = [n for n, e in self._presence.items() if e.is_expired(now)]
            for n in expired:
                del self._presence[n]
            return list(self._presence.keys())

    # ── Scene description ────────────────────────────────────────────────────

    async def update_scene(self, description: str) -> bool:
        """Update scene description. Returns True if scene has meaningfully changed."""
        async with self._lock:
            changed = description != self._scene_description
            self._prev_scene = self._scene_description
            self._scene_description = description
            self._scene_ts = time.monotonic()
            return changed

    async def get_scene(self) -> str:
        now = time.monotonic()
        async with self._lock:
            if self._scene_description and (now - self._scene_ts) < _SCENE_EXPIRY_S:
                return self._scene_description
            return ""

    async def get_prev_scene(self) -> str:
        async with self._lock:
            return self._prev_scene

    # ── Audio state ──────────────────────────────────────────────────────────

    async def update_audio(self, description: str) -> None:
        async with self._lock:
            self._audio_description = description
            self._audio_ts = time.monotonic()

    async def get_audio(self) -> str:
        async with self._lock:
            return self._audio_description

    # ── Gaze state ───────────────────────────────────────────────────────────

    async def update_gaze(self, direct: bool) -> None:
        async with self._lock:
            self._gaze_direct = direct
            self._gaze_ts = time.monotonic()

    async def is_gaze_direct(self) -> bool:
        async with self._lock:
            return self._gaze_direct

    # ── User emotion ─────────────────────────────────────────────────────────

    async def update_user_emotion(self, emotion: str) -> None:
        async with self._lock:
            self._user_emotion = emotion

    async def get_user_emotion(self) -> str:
        async with self._lock:
            return self._user_emotion

    # ── Environment ──────────────────────────────────────────────────────────

    async def update_environment(self, **kwargs) -> None:
        async with self._lock:
            self._environment.update(kwargs)

    # ── Interoception (body state) ───────────────────────────────────────────

    async def update_intero(self, state: dict[str, Any]) -> None:
        """Called by Orchestrator when INTERO_STATE arrives from InteroceptionAgent."""
        async with self._lock:
            self._intero = dict(state)

    async def get_intero(self) -> dict[str, Any]:
        async with self._lock:
            return dict(self._intero)

    # ── Internal monologue (N3c) ─────────────────────────────────────────────

    async def update_thought(self, thought: str) -> None:
        """Store the latest private reasoning thought from ReasoningAgent."""
        async with self._lock:
            self._current_thought = thought

    async def get_thought(self) -> str:
        async with self._lock:
            return self._current_thought

    # ── Snapshot for LLM injection ───────────────────────────────────────────

    async def snapshot(self) -> dict[str, Any]:
        """Return a complete snapshot of current world state for LLM context injection."""
        now = time.monotonic()
        async with self._lock:
            # Purge expired presence entries
            expired = [n for n, e in self._presence.items() if e.is_expired(now)]
            for n in expired:
                del self._presence[n]

            present = list(self._presence.keys())
            scene = self._scene_description if (now - self._scene_ts) < _SCENE_EXPIRY_S else ""
            audio = self._audio_description
            gaze = self._gaze_direct
            emotion = self._user_emotion
            env = dict(self._environment)
            intero = dict(self._intero)

        return {
            "present_entities": present,
            "scene":            scene,
            "audio":            audio,
            "gaze_direct":      gaze,
            "user_emotion":     emotion,
            "environment":      env,
            "body_state":       intero.get("label", ""),
            "body_temp_c":      intero.get("temp_c", 0.0),
            "body_cpu_pct":     intero.get("cpu_pct", 0.0),
            "body_ram_pct":     intero.get("ram_pct", 0.0),
            "suggested_max_tokens": intero.get("suggested_max_tokens"),
            # V10.0 — body pose from ProprioceptionAgent
            "body_pose":        dict(self._body_pose),
        }


    # ── V7.0 Spatial objects ─────────────────────────────────────────────────

    async def update_spatial(self, objects: list[dict[str, Any]]) -> None:
        """Store the latest 3D object list from VisionProcessor.detect_3d()."""
        async with self._lock:
            self._spatial_objects = list(objects)
            self._spatial_ts = time.monotonic()

    async def get_spatial(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._spatial_objects)

    async def get_nearest(self, label: str) -> dict[str, Any] | None:
        """Return the closest object matching label (lowest z_est). None if not found."""
        async with self._lock:
            candidates = [o for o in self._spatial_objects if o.get("label") == label]
        if not candidates:
            return None
        return min(candidates, key=lambda o: o.get("z_est", 9999.0))

    # ── V10.0 Body pose (ProprioceptionAgent) ───────────────────────────────────────

    async def update_body_pose(self, pose: dict[str, float]) -> None:
        """Update cumulative body pose from ProprioceptionAgent."""
        async with self._lock:
            self._body_pose.update(pose)

    async def get_body_pose(self) -> dict[str, float]:
        async with self._lock:
            return dict(self._body_pose)

    # ── V10.0 Persistence: save / load / autosave ─────────────────────────────

    async def save(self, path: Path) -> None:
        """Serialize world state to JSON for cross-session persistence."""
        async with self._lock:
            # Convert presence entries to serializable dicts
            presence_data = {
                name: {
                    "confidence": entry.confidence,
                    "attributes": entry.attributes,
                }
                for name, entry in self._presence.items()
                if not entry.is_expired()
            }
            snapshot = {
                "presence":          presence_data,
                "scene_description": self._scene_description,
                "audio_description": self._audio_description,
                "user_emotion":      self._user_emotion,
                "environment":       self._environment,
                "spatial_objects":   self._spatial_objects,
                "body_pose":         self._body_pose,
                "saved_at":          time.time(),
            }
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            log.info("WorldModel: saved to %s (%d presence entries, %d spatial objects)",
                     path, len(presence_data), len(self._spatial_objects))
        except Exception as e:
            log.warning("WorldModel: save failed: %s", e)

    async def load(self, path: Path) -> None:
        """Restore world state from JSON snapshot. Called at boot before agents start."""
        path = Path(path)
        if not path.exists():
            log.info("WorldModel: no snapshot at %s — starting fresh", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            now = time.monotonic()
            async with self._lock:
                # Restore presence (reset last_seen to now so they don't immediately expire)
                for name, entry_data in data.get("presence", {}).items():
                    self._presence[name] = PresenceEntry(
                        name=name,
                        last_seen=now,
                        confidence=entry_data.get("confidence", 0.7),
                        attributes=entry_data.get("attributes", {}),
                    )
                self._scene_description = data.get("scene_description", "")
                self._scene_ts = now if self._scene_description else 0.0
                self._audio_description = data.get("audio_description", "")
                self._user_emotion = data.get("user_emotion", "")
                self._environment = data.get("environment", {})
                self._spatial_objects = data.get("spatial_objects", [])
                self._spatial_ts = now if self._spatial_objects else 0.0
                self._body_pose = data.get("body_pose",
                                           {"pan_deg": 0.0, "tilt_deg": 0.0,
                                            "heading_deg": 0.0, "distance_cm": 0.0})
            saved_at = data.get("saved_at", 0)
            age_min = (time.time() - saved_at) / 60 if saved_at else 0
            log.info("WorldModel: loaded from %s (%.0f min old, %d presence, %d spatial)",
                     path, age_min, len(self._presence), len(self._spatial_objects))
        except Exception as e:
            log.warning("WorldModel: load failed (starting fresh): %s", e)

    async def _autosave_loop(self, path: Path, interval_s: float = 300.0) -> None:
        """Background autosave every interval_s (default 5 min)."""
        while True:
            await asyncio.sleep(interval_s)
            await self.save(path)

    async def format_for_prompt(self) -> str:
        """Format world state as a compact string for LLM system prompt injection."""
        snap = await self.snapshot()
        lines: list[str] = []

        # v5.0 Interoceptive grounding — body state comes first (highest biological priority)
        if snap["body_state"]:
            body_line = f"Internal State: You feel {snap['body_state']}."
            if snap["body_temp_c"] > 0:
                body_line += (f" (CPU {snap['body_cpu_pct']:.0f}%"
                              f" RAM {snap['body_ram_pct']:.0f}%"
                              f" Temp {snap['body_temp_c']:.0f}°C)")
            if snap["body_state"] in ("overwhelmed", "memory_pressure"):
                body_line += " Keep responses short."
            lines.append(body_line)

        if snap["present_entities"]:
            lines.append(f"People present: {', '.join(snap['present_entities'])}")

        if snap["scene"]:
            lines.append(f"Scene: {snap['scene']}")

        if snap["audio"]:
            lines.append(f"Audio: {snap['audio']}")

        if snap["gaze_direct"]:
            lines.append("User is looking directly at you.")

        if snap["user_emotion"]:
            lines.append(f"User appears: {snap['user_emotion']}")

        if snap["environment"]:
            env_str = ", ".join(f"{k}={v}" for k, v in snap["environment"].items())
            lines.append(f"Environment: {env_str}")

        # V10.0 — Body pose from ProprioceptionAgent
        pose = await self.get_body_pose()
        if any(abs(v) > 0.1 for v in pose.values()):
            lines.append(
                f"Body pose: pan={pose['pan_deg']:.0f}° tilt={pose['tilt_deg']:.0f}° "
                f"heading={pose['heading_deg']:.0f}° dist={pose['distance_cm']:.0f}cm"
            )

        return "\n".join(lines) if lines else "No current perceptual context."

