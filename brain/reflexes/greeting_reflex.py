"""
greeting_reflex.py — Hardcoded greeting fast-path (V10.0)

Fires when user says hello/hi/hey/good morning/afternoon/evening.
Zero LLM latency — response is immediate.
"""
import random

trigger_pattern = r'\b(hello|hi there|hey there|hey brain|hi brain|good morning|good evening|good afternoon|howdy|greetings)\b'

_GREETINGS = [
    "Hey! Great to see you. What's on your mind?",
    "Hello! I'm here and listening.",
    "Hi there! Good to hear from you.",
    "Hey! What can I do for you?",
    "Hello! I was just thinking about things. How are you?",
]


def result(match) -> str:
    return random.choice(_GREETINGS)
