"""
One-time migration: sqlite-vec → ChromaDB.
Run once after upgrading: python scripts/migrate_semantic_db.py

Safe to run multiple times — upsert is idempotent.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate(sqlite_path: str = "data/semantic.db", chroma_path: str = "data/chroma") -> None:
    old_db = Path(sqlite_path)
    if not old_db.exists():
        print(f"No existing sqlite-vec DB found at {sqlite_path} — nothing to migrate.")
        return

    print(f"Connecting to old DB: {sqlite_path}")
    conn = sqlite3.connect(str(old_db))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute("SELECT * FROM fact_meta").fetchall()
    except Exception as e:
        print(f"Could not read fact_meta table: {e}")
        return
    finally:
        conn.close()

    if not rows:
        print("Old DB is empty — nothing to migrate.")
        return

    print(f"Found {len(rows)} facts to migrate...")

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name="brain_memory",
        metadata={"hnsw:space": "cosine"},
    )

    # Re-embed using sentence-transformers since we can't read raw sqlite-vec blobs portably
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers model (first time: ~90MB download)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        return

    import hashlib
    migrated = 0
    for row in rows:
        content = row["content"]
        if not content:
            continue
        doc_id = hashlib.md5(content.encode()).hexdigest()
        embedding = model.encode(content).tolist()
        meta = {
            "category": row["category"] or "general",
            "confidence": float(row["confidence"] or 0.8),
            "source": row["source"] or "migration",
            "created_at": row["created_at"] or "",
        }
        col.upsert(ids=[doc_id], embeddings=[embedding], documents=[content], metadatas=[meta])
        migrated += 1
        if migrated % 10 == 0:
            print(f"  Migrated {migrated}/{len(rows)}...")

    print(f"\nDone. {migrated} facts migrated to ChromaDB at {chroma_path}")
    print(f"ChromaDB now has {col.count()} total entries.")
    print(f"\nYou can now safely delete {sqlite_path} if migration looks correct.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Migrate sqlite-vec semantic DB to ChromaDB")
    p.add_argument("--sqlite", default="data/semantic.db")
    p.add_argument("--chroma", default="data/chroma")
    args = p.parse_args()
    migrate(args.sqlite, args.chroma)
