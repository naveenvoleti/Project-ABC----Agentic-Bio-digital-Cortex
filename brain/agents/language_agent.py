"""
LanguageAgent — NLP: intent classification, entity extraction.
Maps to Broca's and Wernicke's areas.
"""
from __future__ import annotations

import asyncio
import re

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)

# Simple intent patterns (extended by LLM in ReasoningAgent for complex cases)
INTENT_PATTERNS = [
    (r"\b(hello|hi|hey|good morning|good evening)\b", "greeting"),
    (r"\b(what (do you|can you) see|look around|describe)\b", "vision_query"),
    (r"\b(what time|what day|what date)\b", "time_query"),
    # Skill teaching — must come before generic memory_store
    (r"\b(whenever|always do|if .+ then|remember to .+ when)\b", "skill_teach"),
    (r"\b(remember|don't forget|note that)\b", "memory_store"),
    (r"\b(recall|do you remember|what did|tell me about)\b", "memory_recall"),
    (r"\b(play|sing|music)\b", "media_request"),
    (r"\b(move|go|forward|back|turn|rotate)\b", "motor_command"),
    (r"\b(execute task|task|plan|do task|execute)\b", "task"),
    (r"\b(how are you|how do you feel|your mood)\b", "status_query"),
    (r"\b(quiet|stop|shut up|pause)\b", "stop_command"),
    (r"\b(forget (our|the) conversation|start fresh|clear (your |my )?memory|reset (your |the )?context|fresh start)\b", "memory_reset"),
    (r"\b(thank|thanks|great job|well done)\b", "positive_feedback"),
    (r"\b(wrong|no|incorrect|not right)\b", "negative_feedback"),
    (r"\b(help|can you|please|could you)\b", "assistance_request"),
    (r"\b(set alarm|remind me|reminder)\b", "reminder_request"),
    (r"\b(weather|temperature|hot|cold)\b", "environment_query"),
]


class LanguageAgent(BaseAgent):
    name = "language_agent"

    def __init__(self, bus: asyncio.Queue):
        super().__init__(bus)
        self._nlp = None

    def _load_spacy(self) -> None:
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            log.info("spaCy model loaded")
        except Exception as e:
            log.warning(f"spaCy not available: {e} — using pattern matching only")

    def classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        for pattern, intent in INTENT_PATTERNS:
            if re.search(pattern, text_lower):
                return intent
        return "general_query"

    def extract_entities(self, text: str) -> dict:
        entities: dict[str, list[str]] = {}
        if self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    def detect_language(self, text: str) -> str:
        # Basic heuristic — extend with langdetect if needed
        return "en"

    async def start(self) -> None:
        await super().start()
        self._load_spacy()
        log.info("LanguageAgent started")
        # Passive agent — only handles messages
        while self._running:
            await asyncio.sleep(1)

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.type != MessageType.PERCEPTION_SPEECH:
            return None

        text = message.data.get("text", "")
        if not text:
            return None

        intent = self.classify_intent(text)
        entities = self.extract_entities(text)
        lang = self.detect_language(text)
        sentiment = self._quick_sentiment(text)

        result = AgentMessage(
            type=MessageType.COGNITION_INTENT,
            source=self.name,
            data={
                "text": text,
                "intent": intent,
                "entities": entities,
                "language": lang,
                "sentiment": sentiment,
                "is_question": text.strip().endswith("?"),
            },
            priority=3,
        )
        await self.publish(result)
        return result

    def _quick_sentiment(self, text: str) -> str:
        positive = {"good", "great", "love", "thanks", "thank", "perfect", "yes", "awesome"}
        negative = {"bad", "wrong", "no", "stop", "terrible", "hate", "awful"}
        words = set(text.lower().split())
        if words & positive:
            return "positive"
        if words & negative:
            return "negative"
        return "neutral"
