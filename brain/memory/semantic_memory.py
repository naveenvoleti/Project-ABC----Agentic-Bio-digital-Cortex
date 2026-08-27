"""
Semantic Memory (L3) — ChromaDB-backed persistent vector store.
HNSW indexed, works fully offline, scales to millions of memories.
Replaces sqlite-vec for long-term "best friend" memory.

Collections:
  brain_memory        — text embeddings (~384-d, all-MiniLM-L6-v2 / nomic-embed-text)
  brain_visual_memory — CLIP visual embeddings (512-d, clip-ViT-B-32)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from brain.utils.logger import get_logger

log = get_logger(__name__)


class SemanticMemory:
    def __init__(self, db_path: str | Path = "data/chroma"):
        self.db_path = str(Path(db_path))
        self._col = None        # text embeddings (~384-d)
        self._visual_col = None # CLIP visual embeddings (512-d)
        self._init_db()

    def _init_db(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self._col = client.get_or_create_collection(
                name="brain_memory",
                metadata={"hnsw:space": "cosine"},
            )
            # Separate collection for CLIP (512-d) vectors — must not mix with text vectors
            self._visual_col = client.get_or_create_collection(
                name="brain_visual_memory",
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                "SemanticMemory: ChromaDB ready — text=%d visual=%d entries at %s",
                self._col.count(), self._visual_col.count(), self.db_path,
            )
        except ImportError:
            log.warning("chromadb not installed — semantic memory disabled. Run: pip install chromadb")
        except Exception as e:
            log.error(f"SemanticMemory init error: {e}")

    def upsert(
        self,
        content: str,
        embedding: list[float] | None = None,
        category: str = "general",
        confidence: float = 0.8,
        source: str = "observation",
        extra: dict | None = None,
    ) -> None:
        if self._col is None or not embedding:
            return
        import hashlib
        doc_id = hashlib.md5(content.encode()).hexdigest()
        meta = {
            "category": category,
            "confidence": confidence,
            "source": source,
            "created_at": datetime.utcnow().isoformat(),
        }
        if extra:
            # ChromaDB metadata values must be str/int/float/bool
            for k, v in extra.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        try:
            self._col.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
            )
        except Exception as e:
            log.debug(f"SemanticMemory upsert error: {e}")

    def search_similar(
        self,
        embedding: list[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        if self._col is None or not embedding:
            return []
        count = self._col.count()
        if count == 0:
            return []
        try:
            kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": min(k, count),
            }
            if where:
                kwargs["where"] = where
            results = self._col.query(**kwargs)
            out = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                out.append({
                    "content": doc,
                    "category": meta.get("category", "general"),
                    "confidence": meta.get("confidence", 0.8),
                    "source": meta.get("source", ""),
                    "score": round(1.0 - dist, 4),
                })
            return out
        except Exception as e:
            log.debug(f"SemanticMemory search error: {e}")
            return []

    def search_text(self, query: str, k: int = 5) -> list[dict]:
        """Keyword-based fallback search via ChromaDB document filter."""
        if self._col is None:
            return []
        count = self._col.count()
        if count == 0:
            return []
        try:
            results = self._col.query(
                query_texts=[query],
                n_results=min(k, count),
            )
            return [
                {"content": doc, **meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        except Exception as e:
            log.debug(f"SemanticMemory text search error: {e}")
            return []

    def get_by_category(self, category: str, limit: int = 20) -> list[dict]:
        if self._col is None or self._col.count() == 0:
            return []
        try:
            results = self._col.get(
                where={"category": category},
                limit=limit,
            )
            return [
                {"content": doc, **meta}
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        except Exception as e:
            log.debug(f"SemanticMemory get_by_category error: {e}")
            return []

    def hybrid_search(
        self,
        text_embedding: list[float],
        k: int = 5,
        visual_boost: float = 0.2,
    ) -> list[dict]:
        """Text-embedding search with a score boost for entries that have image_path metadata."""
        results = self.search_similar(text_embedding, k=k * 2)
        for r in results:
            if r.get("image_path"):
                r["score"] = min(1.0, r["score"] + visual_boost)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_visual_memories(self, limit: int = 10) -> list[dict]:
        """Return stored visual memories (entries with category=visual_memory)."""
        return self.get_by_category("visual_memory", limit=limit)

    # ── Visual RAG — CLIP-based multimodal storage & retrieval ────────────────

    def upsert_visual_memory(
        self,
        content: str,
        visual_embedding: list[float],
        image_path: str,
        metadata: dict,
    ) -> None:
        """Store a frame's CLIP visual features in the dedicated visual collection.

        The visual_embedding (512-d CLIP vector) is the PRIMARY key used for
        retrieval — allowing the robot to find frames by visual similarity
        (pixels-as-vectors) rather than relying on text captions.

        Args:
            content:          Text caption / scene description for the frame.
            visual_embedding: 512-d CLIP vector from VisionProcessor.encode_visual_features().
            image_path:       Absolute path to the saved JPEG on disk (from GalleryManager).
            metadata:         Dict with at minimum 'image_id'. All values must be str/int/float/bool.
        """
        if self._visual_col is None or not visual_embedding:
            return
        import hashlib
        image_id = metadata.get("image_id") or hashlib.sha256(image_path.encode()).hexdigest()[:16]
        meta = {
            "image_path": image_path,
            "image_id": image_id,
            "category": "visual_memory",
            "created_at": datetime.utcnow().isoformat(),
        }
        # Merge caller-supplied metadata (str/int/float/bool values only)
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        try:
            self._visual_col.upsert(
                ids=[image_id],
                embeddings=[visual_embedding],
                documents=[content],
                metadatas=[meta],
            )
            log.debug("SemanticMemory: visual memory stored — id=%s", image_id)
        except Exception as e:
            log.debug("SemanticMemory: upsert_visual_memory error: %s", e)

    def search_visual_memories(
        self,
        query_embedding: list[float],
        k: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieve the k most visually similar stored frames.

        Accepts either a CLIP image embedding or a CLIP text embedding — both
        live in the same 512-d embedding space, enabling cross-modal retrieval.

        Returns a list of dicts with keys: content, image_path, image_id, score.
        """
        if self._visual_col is None or not query_embedding:
            return []
        count = self._visual_col.count()
        if count == 0:
            return []
        try:
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": min(k, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = self._visual_col.query(**kwargs)
            out = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                out.append({
                    "content":       doc,
                    "image_path":    meta.get("image_path", ""),
                    "image_id":      meta.get("image_id", ""),
                    "created_at":    meta.get("created_at", ""),
                    "thumbnail_b64": meta.get("thumbnail_b64", ""),  # inline thumbnail (no disk I/O)
                    "score":         round(1.0 - dist, 4),
                })
            return out
        except Exception as e:
            log.debug("SemanticMemory: search_visual_memories error: %s", e)
            return []

    def visual_memory_count(self) -> int:
        """Return the number of stored visual memory frames."""
        if self._visual_col is None:
            return 0
        return self._visual_col.count()

    def penalize_memory(self, doc_id: str, penalty: float = 0.3) -> None:
        """Reduce the confidence of a specific memory entry (reinforcement correction).

        Fetches the document by ID, applies a negative_bias penalty to its
        confidence metadata, then re-upserts with the same embedding.
        Floor is 0.05 — the memory stays searchable but ranks very low.
        """
        if self._col is None or not doc_id:
            return
        try:
            result = self._col.get(ids=[doc_id], include=["embeddings", "documents", "metadatas"])
            if not result or not result.get("ids") or not result["ids"]:
                log.debug("SemanticMemory.penalize_memory: id not found — %s", doc_id)
                return
            old_meta = result["metadatas"][0] if result.get("metadatas") else {}
            old_conf = float(old_meta.get("confidence", 0.8))
            new_conf = max(0.05, old_conf - penalty)
            old_meta["confidence"] = new_conf
            old_meta["penalized_at"] = datetime.utcnow().isoformat()
            self._col.upsert(
                ids=[doc_id],
                embeddings=result["embeddings"],
                documents=result["documents"],
                metadatas=[old_meta],
            )
            log.info("SemanticMemory: penalized memory id=%s conf %.2f→%.2f", doc_id, old_conf, new_conf)
        except Exception as e:
            log.debug("SemanticMemory.penalize_memory error: %s", e)

    def upsert_correction(
        self,
        original_query: str,
        corrected_fact: str,
        embedding: list[float],
    ) -> None:
        """Store a user-verified correction at confidence=1.0.

        This becomes the authoritative fact — it will surface first in future
        similarity searches because of its high confidence score.
        """
        self.upsert(
            content=corrected_fact,
            embedding=embedding,
            category="correction",
            confidence=1.0,
            source="user_correction",
            extra={
                "original_query": original_query[:120],
                "corrected_at": datetime.utcnow().isoformat(),
            },
        )
        log.info("SemanticMemory: correction upserted — '%s'", corrected_fact[:60])

    def count(self) -> int:
        if self._col is None:
            return 0
        return self._col.count()

    def close(self) -> None:
        # ChromaDB PersistentClient auto-flushes on exit
        pass
