"""
name_reflex.py — Identity query fast-path (V10.0)

Fires when user asks Brain's name or what it is.
Zero LLM latency — hardcoded identity response.
"""
import random

trigger_pattern = r'\b(what is your name|what\'s your name|who are you|what are you called|your name|introduce yourself|what do i call you)\b'

_NAMES = [
    "I'm Brain — your robotic companion. Nice to meet you!",
    "I go by Brain. I'm a robotic brain built to see, listen, and think with you.",
    "My name is Brain. I'm an embodied AI with eyes, ears, and curiosity.",
    "I'm Brain — part robot, part mind, all yours.",
]


def result(match) -> str:
    return random.choice(_NAMES)
