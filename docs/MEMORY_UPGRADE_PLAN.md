# Memory Upgrade Plan — Project-ABC
**Goal:** Make Brain remember you like a best friend across every restart, session, and mood.
**Last updated:** 2026-04-14

---

## Current State (What Already Works)

| Layer | Storage | Persists | Works Offline |
|---|---|---|---|
| Episodic events (every conversation, emotion, vision) | `data/episodes.db` SQLite FTS5 | ✅ Yes | ✅ Yes |
| Semantic vector embeddings | `data/semantic.db` sqlite-vec | ✅ Yes | ❌ Needs Ollama |
| User facts (name, role, preferences) | `data/USER.md` markdown | ✅ Yes | ✅ Yes |
| Skills taught by speech | `data/SKILLS.json` | ✅ Yes | ✅ Yes |
| Personality + soul | `data/SOUL.md` | ✅ Yes | ✅ Yes |
| Active conversation context | In-process `WorkingMemory` | ❌ Lost on restart | ✅ Yes |
| Context compaction summary | Episodic DB entry | ✅ Yes | ❌ Not loaded on boot |

---

## Gaps Identified

### Gap 1 — Startup Amnesia
On restart, Brain has no memory of the last conversation.
Working memory (last 10–20 exchanges) is in-process only.
Brain has to start from zero every boot even if it talked to you 5 minutes ago.

### Gap 2 — Semantic Search Blind Without Ollama
`SemanticMemory.search_similar()` depends on embeddings from `LLMRouter.embed()`.
If Ollama is offline (e.g. network-only mode), `embed()` returns `[]`.
`_cosine_sim([], [...])` = 0.0 for everything → semantic search silently returns nothing.
Half the memory system is invisible without a running local model.

### Gap 3 — USER.md Is Unstructured Free Text
Facts are appended as raw markdown blocks.
Over time: duplicates accumulate, old/wrong facts never get corrected, LLM has to parse
messy prose to extract signal. No confidence scores, no timestamps per fact.

### Gap 4 — Retrieval Is Not Context-Aware
Current: semantic search queries with the literal user utterance.
Missing: time-of-day, mood, who's speaking, recent conversation topic, relationship history.
A best friend says "this reminds me of what you told me last Tuesday" — Brain can't do that.

### Gap 5 — sqlite-vec Doesn't Scale
For 100k+ episodes, cosine scan on sqlite-vec is O(n).
No HNSW index → gets slow after months of continuous use.

### Gap 6 — No Long-Term Relationship Arc
Brain has no sense of patterns over weeks/months:
"User seems stressed when mentioning work", "User is more talkative on weekends",
"User's communication style has become more casual over the past month."

---

## Implementation Plan (4 Priorities)

---

### Priority 1 — Startup Context Restoration + Embedding Fallback
**Effort:** Small (1–2 hours) | **Impact:** Immediate — Brain stops forgetting on restart

#### 1A. Startup Context Restoration

**What to build:**
On brain boot, before agents start, load the last session summary into working memory.

**Where:**
- File: `brain/__main__.py` or `brain/orchestrator.py` in the `start()` method
- After agents init, before the main loop begins

**How:**
```python
# In Orchestrator.start(), after all agents are started:
async def _restore_last_session(self) -> None:
    # Load last compact summary if one exists
    recent = self._episodic.get_recent(n=30)
    compact = next(
        (e for e in reversed(recent) if e.get("event_type") == "context_compact"),
        None
    )
    if compact:
        self._working.replace_with_summary(
            f"[Previous session] {compact['content']}",
            keep_last=0
        )
        log.info("Restored last session context from compact summary")

    # Also load last 3 user/brain exchanges so Brain knows what was just discussed
    exchanges = [e for e in recent if e.get("event_type") in ("speech", "response")][-6:]
    for e in exchanges:
        role = "user" if e.get("actor") == "user" else "assistant"
        self._working.add_to_context(role, e.get("content", ""))
    log.info(f"Restored {len(exchanges)} recent exchanges into working memory")
```

**Config to add in `config/config.yaml`:**
```yaml
memory:
  restore_on_startup: true
  restore_exchanges: 6      # how many recent turns to restore
```

---

#### 1B. Embedding Fallback (OpenRouter)

**Problem:** `LLMRouter.embed()` only tries Ollama. If Ollama is down, returns `[]`.

**What to build:**
Add OpenRouter embedding fallback using `text-embedding-ada-002` via their API,
OR use a local sentence-transformers model as a pure-Python fallback.

**Recommended: sentence-transformers fallback (no API cost, works offline)**

```python
# In brain/llm/llm_router.py — update embed():
async def embed(self, text: str) -> list[float]:
    # Try Ollama first
    if await self._check_ollama():
        result = await self._ollama.embed(text, model=self.embed_model)
        if result:
            return result

    # Fallback: local sentence-transformers (tiny model, ~90MB)
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, '_st_model'):
            self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._st_model.encode(text).tolist()
    except ImportError:
        log.warning("sentence-transformers not installed — embedding unavailable")
        return []
```

**Install:** Add `sentence-transformers>=2.7.0` to `requirements.txt`
**Model download:** ~90MB, cached locally after first use, works fully offline.

---

### Priority 2 — Replace sqlite-vec with ChromaDB
**Effort:** Medium (3–4 hours) | **Impact:** Scales to millions of memories, HNSW indexed, persistent

#### Why ChromaDB
- Persistent HNSW index (fast even at 1M+ entries)
- Works fully offline (embedded mode, no server)
- Cross-platform (Windows laptop + Raspberry Pi)
- Built-in deduplication via metadata filters
- Python-native, simple API

#### Install
```bash
pip install chromadb>=0.5.0
```

#### 2A. New SemanticMemory Implementation

**File to rewrite:** `brain/memory/semantic_memory.py`

```python
"""
SemanticMemory — ChromaDB-backed vector store.
Replaces sqlite-vec. Persistent HNSW index, works offline.
"""
from __future__ import annotations
import chromadb
from chromadb.config import Settings
from brain.utils.logger import get_logger

log = get_logger(__name__)

class SemanticMemory:
    def __init__(self, db_path: str = "data/chroma"):
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name="brain_memory",
            metadata={"hnsw:space": "cosine"}
        )
        log.info(f"SemanticMemory: ChromaDB loaded ({self._col.count()} entries)")

    def upsert(self, text: str, embedding: list[float], meta: dict) -> None:
        if not embedding:
            return
        import hashlib
        doc_id = hashlib.md5(text.encode()).hexdigest()
        self._col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )

    def search_similar(self, embedding: list[float], k: int = 5,
                       where: dict | None = None) -> list[dict]:
        if not embedding or self._col.count() == 0:
            return []
        try:
            kwargs = {"query_embeddings": [embedding], "n_results": min(k, self._col.count())}
            if where:
                kwargs["where"] = where
            results = self._col.query(**kwargs)
            out = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                out.append({"content": doc, "meta": meta, "score": 1 - dist})
            return out
        except Exception as e:
            log.error(f"SemanticMemory search error: {e}")
            return []

    def count(self) -> int:
        return self._col.count()
```

#### 2B. Context-Aware Search

**Enhancement to `CognitionAgent._process_intent()`:**
Instead of searching with just the raw query, add context filters:

```python
# In CognitionAgent._process_intent(), replace semantic search block:
query_embedding = await self._llm.embed(text)

# Context-aware search: prefer memories matching current emotion + time of day
import datetime
hour = datetime.datetime.now().hour
time_context = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"

# Search with no filter first, then enrich with context
semantic_facts = self._semantic.search_similar(query_embedding, k=5)

# Also search specifically for user-related memories
user_facts = self._semantic.search_similar(
    query_embedding, k=3,
    where={"actor": "user"}   # ChromaDB metadata filter
)
```

#### 2C. Migration Script

**File to create:** `scripts/migrate_semantic_db.py`

```python
"""
One-time migration: sqlite-vec → ChromaDB.
Run once after upgrading: python scripts/migrate_semantic_db.py
"""
# Reads all rows from data/semantic.db, re-embeds if needed, writes to ChromaDB.
# Safe to run multiple times (upsert is idempotent).
```

**Update `config/config.yaml`:**
```yaml
memory:
  semantic_db: "data/chroma"      # was: "data/semantic.db"
  semantic_backend: "chroma"      # new field
```

---

### Priority 3 — Structured User Profile (USER.json)
**Effort:** Medium (2–3 hours) | **Impact:** Brain builds a clean, queryable model of you

#### 3A. USER.json Schema

**Replace:** `data/USER.md` free-form markdown  
**With:** `data/USER.json` structured profile

```json
{
  "identity": {
    "name": "Naveen",
    "confidence": 0.95,
    "last_updated": "2026-04-14T15:30:00"
  },
  "preferences": [
    {"fact": "prefers direct answers over long explanations", "confidence": 0.8, "observed_at": "2026-04-14"},
    {"fact": "dislikes robotic-sounding voice", "confidence": 0.9, "observed_at": "2026-04-14"}
  ],
  "topics_of_interest": ["robotics", "AI", "project-abc"],
  "communication_style": "casual, direct, impatient with verbose responses",
  "context": {
    "location": "home office",
    "timezone": "Asia/Kolkata",
    "device": "Windows laptop during dev, Pi in production"
  },
  "relationship": {
    "first_interaction": "2026-04-14",
    "total_interactions": 47,
    "trust_level": "developing",
    "notes": "User is building Brain from scratch — treat as creator/collaborator"
  }
}
```

#### 3B. SoulManager Updates

**File:** `brain/memory/soul_manager.py`

Add methods:
```python
def load_user_json(self) -> dict:
    """Load USER.json, return empty schema if missing."""

def upsert_user_fact(self, category: str, fact: str, confidence: float) -> None:
    """Add or update a fact. Deduplicates by semantic similarity before writing."""

def get_user_summary(self) -> str:
    """Return a concise LLM-ready string from USER.json for system prompt injection."""
    # e.g. "Name: Naveen. Prefers direct answers. Interested in robotics and AI."
```

#### 3C. CognitionAgent Updates

- `_capture_user_facts()` writes to `USER.json` via `upsert_user_fact()` instead of appending to markdown
- `_extract_user_insights()` writes structured facts to `USER.json`
- System prompt uses `get_user_summary()` instead of raw markdown file content

---

### Priority 4 — Relationship Arc Tracking
**Effort:** Medium (2–3 hours) | **Impact:** Brain develops genuine long-term understanding of you

#### 4A. Weekly Relationship Summary (DreamAgent extension)

**File:** `brain/agents/dream_agent.py`

Add `_build_relationship_arc()`:
```python
async def _build_relationship_arc(self) -> None:
    """Weekly: analyze patterns across all episodes, update relationship arc in USER.json."""
    # Query last 7 days of episodes
    # Prompt: "What patterns do you notice about this user's mood, topics, communication style?"
    # Store structured result in USER.json["relationship"]["arc"]
    # Add to ChromaDB with metadata: {"type": "relationship_arc", "week": "2026-W15"}
```

#### 4B. Relationship Arc in System Prompt

In `SoulManager.get_system_prompt()`, add a new section:
```
## Relationship Context
You've been talking to Naveen for 3 weeks. They built you from scratch.
They prefer direct answers and get frustrated with robotic responses.
Recent pattern: more talkative in the evenings, asks about what you see frequently.
Last week they seemed excited about the voice quality improvement.
```

#### 4C. Proactive Memory References

In `CognitionAgent._process_intent()`, before building the system prompt:
```python
# Check if current query connects to something from the past
past_ref = await self._find_past_connection(text)
if past_ref:
    # Add to system prompt: "This reminds you of [past event] — reference it naturally if relevant"
```

---

## Implementation Order

```
Priority 1A: Startup context restoration          (brain/orchestrator.py)      ~45 min
Priority 1B: sentence-transformers embedding      (brain/llm/llm_router.py)    ~30 min
Priority 2A: ChromaDB SemanticMemory              (brain/memory/semantic_memory.py)  ~2 hrs
Priority 2B: Context-aware search                 (brain/agents/cognition_agent.py)  ~1 hr
Priority 2C: Migration script                     (scripts/migrate_semantic_db.py)   ~1 hr
Priority 3A/B: USER.json + SoulManager            (brain/memory/soul_manager.py)     ~2 hrs
Priority 3C: CognitionAgent uses USER.json        (brain/agents/cognition_agent.py)  ~1 hr
Priority 4A: Weekly relationship arc              (brain/agents/dream_agent.py)      ~1 hr
Priority 4B/C: Arc in prompt + proactive refs     (brain/memory/soul_manager.py)     ~1 hr
```

**Total estimated effort:** ~10–11 hours of implementation

---

## Files to Create / Modify

| File | Action | Priority |
|---|---|---|
| `brain/orchestrator.py` | Add `_restore_last_session()` called on boot | P1A |
| `brain/llm/llm_router.py` | Add sentence-transformers fallback in `embed()` | P1B |
| `requirements.txt` | Add `sentence-transformers>=2.7.0`, `chromadb>=0.5.0` | P1B+P2 |
| `brain/memory/semantic_memory.py` | Full rewrite: sqlite-vec → ChromaDB | P2A |
| `brain/agents/cognition_agent.py` | Context-aware ChromaDB search with metadata filters | P2B |
| `scripts/migrate_semantic_db.py` | One-time migration from sqlite-vec to ChromaDB | P2C |
| `config/config.yaml` | Add `semantic_backend: chroma`, `restore_on_startup: true` | P2+P1 |
| `data/USER.json` | New structured user profile (replaces USER.md) | P3 |
| `brain/memory/soul_manager.py` | Add `load_user_json()`, `upsert_user_fact()`, `get_user_summary()` | P3B |
| `brain/agents/cognition_agent.py` | `_capture_user_facts()` → writes to USER.json | P3C |
| `brain/agents/dream_agent.py` | Add `_build_relationship_arc()` weekly pass | P4A |
| `brain/memory/soul_manager.py` | Inject relationship arc into system prompt | P4B |

---

## What "Best Friend" Memory Looks Like After All 4 Priorities

1. **Restart Brain** → working memory is restored with last session summary + last 6 exchanges.
   Brain knows you were in the middle of a conversation about the voice quality.

2. **"What do you remember about me?"** → Brain queries ChromaDB with HNSW index,
   returns top-5 relevant memories from months of history in milliseconds.
   Reads USER.json summary: name, preferences, relationship arc, trust level.

3. **"What can you see?"** → Semantic search finds you've asked this 47 times.
   Brain answers differently knowing you ask this when testing vision features.

4. **"I'm tired"** → Relationship arc says you're usually tired on weekday evenings.
   Brain acknowledges the pattern: "You seem tired a lot on weekday evenings — long days at work?"

5. **After 3 months** → USER.json has 50+ structured facts. Dream agent has 12 weekly arcs.
   Brain knows your communication style evolved from formal to casual. It responds accordingly.

---

## Dependencies to Add

```
# requirements.txt additions
sentence-transformers>=2.7.0   # offline embedding fallback (~90MB model download)
chromadb>=0.5.0                # persistent HNSW vector DB
```

Note for Raspberry Pi: ChromaDB works on Pi 4 (4GB RAM). Pi 3B+ (1GB RAM) may need
`chromadb` compiled with reduced HNSWLIB settings. Alternative for Pi 3B+: use
`faiss-cpu` with a flat index, or keep sqlite-vec + ensure Ollama always runs.
