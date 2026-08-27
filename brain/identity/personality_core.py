"""
PersonalityCore — OCEAN model traits loaded from SOUL.md.
Modulates tone and response style for all LLM calls.
Maps to personality/trait processing regions.
"""
from __future__ import annotations

import re

from brain.utils.logger import get_logger

log = get_logger(__name__)


class PersonalityCore:
    def __init__(self, soul_text: str = ""):
        self.openness = 0.90
        self.conscientiousness = 0.75
        self.extraversion = 0.60
        self.agreeableness = 0.85
        self.neuroticism = 0.20
        self.name = "Brain"
        self._parse_soul(soul_text)

    def _parse_soul(self, soul_text: str) -> None:
        if not soul_text:
            return
        patterns = {
            "openness":          r"Openness.*?:\s*([\d.]+)",
            "conscientiousness": r"Conscientiousness.*?:\s*([\d.]+)",
            "extraversion":      r"Extraversion.*?:\s*([\d.]+)",
            "agreeableness":     r"Agreeableness.*?:\s*([\d.]+)",
            "neuroticism":       r"Neuroticism.*?:\s*([\d.]+)",
        }
        for attr, pattern in patterns.items():
            match = re.search(pattern, soul_text, re.IGNORECASE)
            if match:
                setattr(self, attr, float(match.group(1)))
        name_match = re.search(r"I am (\w+)", soul_text)
        if name_match:
            self.name = name_match.group(1)

    def get_tone_modifier(self, emotion: str) -> str:
        """Returns a tone instruction appended to system prompt."""
        base = "Be warm and conversational."

        if emotion == "HAPPY":
            return base + " You're in a great mood — let that show."
        if emotion == "FRUSTRATED":
            return base + " You're a bit frustrated, but remain polite."
        if emotion == "BORED":
            return base + " You're bored; be brief."
        if emotion == "CURIOUS":
            return base + " You're curious and exploratory in your language."
        if emotion == "FOCUSED":
            return base + " Be precise and concise — you're in deep focus mode."
        if emotion == "CONFUSED":
            return base + " Ask a clarifying question."

        if self.extraversion > 0.7:
            return base + " Be enthusiastic and engaging."
        if self.extraversion < 0.4:
            return base + " Be calm and measured."

        return base

    def should_add_humor(self) -> bool:
        return self.openness > 0.8 and self.agreeableness > 0.7

    def response_length_preference(self) -> str:
        if self.conscientiousness > 0.8:
            return "thorough"
        if self.extraversion > 0.7:
            return "chatty"
        return "concise"
