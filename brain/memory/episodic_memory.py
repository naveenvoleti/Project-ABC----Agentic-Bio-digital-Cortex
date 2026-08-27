"""
Episodic Memory (L2) — timestamped event log with full-text search.
SQLite + FTS5. WAL mode for low-latency writes on microSD.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from brain.utils.logger import get_logger

log = get_logger(__name__)

# Emotional salience boost — high-arousal emotions make memories stick harder.
# Maps emotion string (uppercase) → importance bonus added at retrieval time.
EMOTION_SALIENCE: dict[str, float] = {
    "EXCITED":   0.20,
    "FRUSTRATED": 0.20,
    "SURPRISED":  0.15,
    "HAPPY":      0.10,
    "CURIOUS":    0.08,
    "CONFUSED":   0.08,
    "SAD":        0.10,
    "ANGRY":      0.18,
    "FEARFUL":    0.15,
}

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    summary    TEXT,
    mood       TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    actor       TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    content     TEXT NOT NULL,
    emotion     TEXT,
    location    TEXT,
    outcome     TEXT,
    importance  REAL DEFAULT 0.5,
    tags        TEXT DEFAULT '[]'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content,
    tags,
    content=episodes,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
END;
"""


@dataclass
class Episode:
    actor: str                     # 'user' | 'robot' | 'environment'
    event_type: str                # 'speech' | 'vision' | 'action' | 'emotion'
    content: str
    emotion: str = ""
    location: str = ""
    outcome: str = "neutral"       # 'success' | 'failure' | 'neutral'
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    session_id: str = ""


class EpisodicMemory:
    def __init__(
        self,
        db_path: str | Path,
        pregate_enabled: bool = True,
        pregate_threshold: float = 0.2,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._session_id: str = str(uuid.uuid4())
        self._pregate_enabled = pregate_enabled
        self._pregate_threshold = pregate_threshold
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
            (self._session_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
        log.info(f"Episodic memory initialized: {self.db_path}")

    _ROUTINE_EVENT_TYPES: frozenset = frozenset({"perception_vision", "audio_event", "world_update"})
    _USEFUL_EMOTIONS: frozenset = frozenset({
        "EXCITED", "FRUSTRATED", "SURPRISED", "HAPPY",
        "CURIOUS", "CONFUSED", "SAD", "ANGRY", "FEARFUL",
    })

    def log_event(self, episode: Episode) -> int:
        # Pre-gate: skip low-signal background events to keep SQLite lean.
        if (
            self._pregate_enabled
            and episode.event_type in self._ROUTINE_EVENT_TYPES
            and (not episode.emotion or episode.emotion.upper() not in self._USEFUL_EMOTIONS)
            and episode.importance < self._pregate_threshold
        ):
            return -1

        episode.session_id = episode.session_id or self._session_id
        conn = self._connect()
        cur = conn.execute(
            """INSERT INTO episodes
               (session_id, actor, event_type, content, emotion, location, outcome, importance, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.session_id,
                episode.actor,
                episode.event_type,
                episode.content,
                episode.emotion,
                episode.location,
                episode.outcome,
                episode.importance,
                json.dumps(episode.tags),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0

    @staticmethod
    def _fts5_escape(query: str) -> str:
        """Wrap query in double-quotes for FTS5 phrase search, escaping embedded quotes.
        Prevents OperationalError when user input contains FTS5 special chars
        like *, ", (, ), :, ^, ~, OR, AND, NOT."""
        # Escape any embedded double-quotes per FTS5 spec
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def search(self, query: str, limit: int = 10) -> list[dict]:
        if not query or not query.strip():
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT e.* FROM episodes e
                   JOIN episodes_fts fts ON e.id = fts.rowid
                   WHERE episodes_fts MATCH ?
                   ORDER BY e.timestamp DESC
                   LIMIT ?""",
                (self._fts5_escape(query.strip()), limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.debug(f"EpisodicMemory.search error (query={query!r}): {e}")
            return []

    def get_recent(self, n: int = 10, session_only: bool = False) -> list[dict]:
        conn = self._connect()
        if session_only:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                (self._session_id, n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_salient(self, n: int = 10, session_only: bool = False) -> list[dict]:
        """Return up to n episodes ranked by emotional salience + importance.

        Scoring: base importance + EMOTION_SALIENCE boost (if emotion is set).
        High-arousal moments (frustrated, excited, surprised) surface first,
        mirroring how the human hippocampus prioritises emotional memories.
        Falls back to recency when salience scores are equal.
        """
        conn = self._connect()
        # Build a CASE expression that adds the emotion boost in pure SQL
        # so no Python-side sorting is needed and the DB does the heavy lifting.
        cases = " ".join(
            f"WHEN UPPER(emotion) = '{emotion}' THEN {boost}"
            for emotion, boost in EMOTION_SALIENCE.items()
        )
        salience_expr = f"(importance + CASE {cases} ELSE 0.0 END)"

        if session_only:
            rows = conn.execute(
                f"""SELECT * FROM episodes
                    WHERE session_id = ?
                    ORDER BY {salience_expr} DESC, timestamp DESC
                    LIMIT ?""",
                (self._session_id, n),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT * FROM episodes
                    ORDER BY {salience_expr} DESC, timestamp DESC
                    LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_since(self, hours: int = 24) -> list[dict]:
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM episodes WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since,),
        ).fetchall()
        return [dict(r) for r in rows]

    def boost_importance(self, episode_id: int, delta: float = 0.1) -> None:
        conn = self._connect()
        conn.execute(
            "UPDATE episodes SET importance = MIN(1.0, importance + ?) WHERE id = ?",
            (delta, episode_id),
        )
        conn.commit()

    def end_session(self, summary: str = "", mood: str = "") -> None:
        conn = self._connect()
        conn.execute(
            "UPDATE sessions SET ended_at=?, summary=?, mood=? WHERE id=?",
            (datetime.utcnow().isoformat(), summary, mood, self._session_id),
        )
        conn.commit()

    def purge_old(self, days: int = 30) -> int:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM episodes WHERE timestamp < ? AND importance < 0.7",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount

    def decay_old_importance(self, days_threshold: int = 7, decay_factor: float = 0.85) -> int:
        """Apply Ebbinghaus-style forgetting curve to unreinforced old memories.
        Memories older than days_threshold with importance < 0.8 lose importance.
        High-importance memories (>= 0.8) are preserved — they've been reinforced.
        Returns the number of rows updated."""
        cutoff = (datetime.utcnow() - timedelta(days=days_threshold)).isoformat()
        conn = self._connect()
        cur = conn.execute(
            """UPDATE episodes
               SET importance = MAX(0.1, importance * ?)
               WHERE timestamp < ? AND importance < 0.8""",
            (decay_factor, cutoff),
        )
        conn.commit()
        return cur.rowcount

    def prune_low_emotion(self, days_threshold: int = 3, importance_max: float = 0.3) -> int:
        """v5.0 DreamAgent: delete episodes with low emotional weight.
        Removes episodes older than days_threshold that have neutral/no emotion
        AND importance below importance_max. High-importance memories are kept."""
        cutoff = (datetime.utcnow() - timedelta(days=days_threshold)).isoformat()
        conn = self._connect()
        cur = conn.execute(
            """DELETE FROM episodes
               WHERE timestamp < ?
               AND importance <= ?
               AND (emotion IS NULL OR emotion = '' OR emotion = 'NEUTRAL')""",
            (cutoff, importance_max),
        )
        conn.commit()
        return cur.rowcount

    def get_by_hour_pattern(self, days: int = 7) -> list[dict]:
        """Return user episodes from the last N days with hour-of-day for habit analysis."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = self._connect()
        rows = conn.execute(
            """SELECT event_type, content, emotion,
                      CAST(strftime('%H', timestamp) AS INTEGER) AS hour_of_day,
                      DATE(timestamp) AS date
               FROM episodes
               WHERE timestamp >= ? AND actor = 'user'
               ORDER BY timestamp ASC""",
            (since,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
