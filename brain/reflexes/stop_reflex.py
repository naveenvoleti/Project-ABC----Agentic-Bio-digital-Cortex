"""
stop_reflex.py — Emergency stop fast-path (V10.0)

Fires when user says stop/quiet/silence/be quiet.
Response acknowledges immediately without LLM overhead.
"""
import random

trigger_pattern = r'\b(stop|stop it|be quiet|quiet|silence|shut up|stop talking|enough|that\'s enough)\b'

_STOPS = [
    "Understood. Going quiet.",
    "Okay, I'll stop.",
    "Got it — I'll be quiet.",
    "Stopping now.",
]


def result(match) -> str:
    return random.choice(_STOPS)
