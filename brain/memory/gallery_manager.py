"""
GalleryManager — Visual long-term memory store.

Saves camera frame thumbnails (JPEG) to disk, indexed by perceptual hash (pHash)
for fast duplicate detection. Acts as the visual episodic gallery — letting the
brain say "I've seen this before" and link back to a saved snapshot.

V9.0 feature: CognitionAgent calls save_frame() when user says "this is my X".
VisionAgent calls lookup_phash() / register_phash() to skip CLIP embedding on
frames it has already processed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


class GalleryManager:
    """
    Manages the on-disk visual gallery.

    Directory layout:
        <gallery_dir>/
            <snap_id>.jpg        — saved JPEG thumbnail
            index.json           — {phash: snap_id, ...} for cross-session dedup
            meta.json            — {snap_id: {description, ts, ...}, ...}
    """

    _INDEX_FILE = "index.json"
    _META_FILE  = "meta.json"

    def __init__(self, gallery_dir: str = "data/gallery") -> None:
        self._dir = Path(gallery_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, str] = self._load_json(self._INDEX_FILE)   # phash → snap_id
        self._meta:  dict[str, dict] = self._load_json(self._META_FILE)   # snap_id → metadata
        log.info("GalleryManager: initialized — %d entries in %s",
                 len(self._index), self._dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_frame(self, frame_b64: str, description: str = "") -> tuple[str, str]:
        """
        Save a base64-encoded JPEG frame to the gallery.

        Returns:
            (snap_id, file_path) — unique snapshot ID and absolute path to saved JPEG.
        """
        snap_id = str(uuid.uuid4())[:12]
        try:
            img_bytes = base64.b64decode(frame_b64)
        except Exception as e:
            log.warning("GalleryManager: could not decode frame_b64: %s", e)
            img_bytes = b""

        file_path = self._dir / f"{snap_id}.jpg"
        if img_bytes:
            file_path.write_bytes(img_bytes)
        else:
            file_path.write_bytes(b"")  # empty placeholder

        # Compute simple SHA-256 as content hash (lightweight alternative to pHash)
        content_hash = hashlib.sha256(img_bytes).hexdigest()[:16]

        self._meta[snap_id] = {
            "snap_id":      snap_id,
            "description":  description,
            "file":         str(file_path),
            "content_hash": content_hash,
            "ts":           time.time(),
        }
        self._save_json(self._META_FILE, self._meta)
        log.info("GalleryManager: saved snap=%s desc='%s'", snap_id, description[:60])
        return snap_id, str(file_path)

    def lookup_phash(self, phash: str) -> bool:
        """
        Return True if this perceptual hash has been seen before.
        Used by VisionAgent to skip CLIP embedding on duplicate frames.
        """
        return phash in self._index

    def register_phash(self, phash: str, snap_id: str) -> None:
        """
        Store a new phash → snap_id mapping for future dedup lookups.
        Persisted immediately so it survives restarts.
        """
        if phash not in self._index:
            self._index[phash] = snap_id
            self._save_json(self._INDEX_FILE, self._index)

    def get_meta(self, snap_id: str) -> dict:
        """Return metadata dict for a given snap_id, or {} if not found."""
        return dict(self._meta.get(snap_id, {}))

    def list_recent(self, n: int = 10) -> list[dict]:
        """Return the N most recently saved gallery entries as meta dicts."""
        entries = sorted(self._meta.values(), key=lambda m: m.get("ts", 0), reverse=True)
        return entries[:n]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_json(self, filename: str) -> dict:
        path = self._dir / filename
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("GalleryManager: could not load %s: %s", filename, e)
        return {}

    def _save_json(self, filename: str, data: dict) -> None:
        path = self._dir / filename
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("GalleryManager: could not save %s: %s", filename, e)
