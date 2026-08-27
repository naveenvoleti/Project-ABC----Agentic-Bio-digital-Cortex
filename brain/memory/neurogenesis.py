"""
Stage 3: Intrinsic Curiosity Engine (Neurogenesis).
Mimics dendritic growth by forming new semantic node weights when surprised by a recurring pattern.
"""
from __future__ import annotations

import collections
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.llm.llm_router import LLMRouter
    from brain.memory.semantic_memory import SemanticMemory

from brain.utils.logger import get_logger

log = get_logger(__name__)

class NeurogenesisEngine:
    """
    Tracks predictive coding errors (Surprises).
    When surprise combined with recurring observation meets a threshold,
    instantiates new node weights in the Semantic Memory module.
    """
    def __init__(self, semantic: "SemanticMemory", llm: "LLMRouter", threshold: int = 3):
        self._semantic = semantic
        self._llm = llm
        self._surprise_history: collections.deque[str] = collections.deque(maxlen=10)
        self._threshold = threshold

    async def process_surprise(self, scene: str) -> str | None:
        """
        Record a surprising observation. If a pattern forms, synthesize a new
        concept (neurogenesis) and store it in SemanticMemory.
        Returns the new insight string if neurogenesis occurred.
        """
        if not self._semantic or not scene:
            return None
            
        self._surprise_history.append(scene)
        
        if len(self._surprise_history) < self._threshold:
            return None
            
        history_text = "\n".join(f"- {s}" for s in self._surprise_history)
        
        prompt = (
            "You are Brain's subconscious neurogenesis engine.\n"
            "Analyze these recent surprising observations from the robot's camera:\n"
            f"{history_text}\n\n"
            "Is there a clear RECURRING pattern, object, or concept that keeps appearing unexpectedly? "
            "If yes, synthesize it into a single factual statement (a new connection/memory). "
            "Keep it short and factual, e.g., 'A cat frequently enters the room'. "
            "If there is NO clear recurring pattern, reply with EXACTLY 'NO_PATTERN'."
        )
        
        try:
            insight = await self._llm.infer(
                user_message=prompt,
                max_tokens=60,
                skip_cache=True
            )
            
            if insight and insight.strip() and "NO_PATTERN" not in insight:
                log.info(f"Neurogenesis occurred: {insight}")
                
                # Check if LLMRouter has the embed method (depends on LLM router version)
                embedding = None
                if hasattr(self._llm, "embed"):
                    embedding = await self._llm.embed(insight)
                
                # Even if embedding fails, we can store it (SemanticMemory might skip it or handle it)
                if embedding:
                    self._semantic.upsert(
                        content=insight,
                        embedding=embedding,
                        category="neurogenesis",
                        confidence=1.0,
                        source="neurogenesis_engine"
                    )
                    self._surprise_history.clear()
                    return insight
        except Exception as e:
            log.error(f"NeurogenesisEngine error: {e}")
            
        return None
