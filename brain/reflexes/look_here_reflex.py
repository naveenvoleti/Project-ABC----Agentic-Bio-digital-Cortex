"""
look_here_reflex.py — Gaze redirect fast-path (V10.0)

Fires when user asks Brain to look at them or look here.
Returns a spoken acknowledgement; the actual motor command is handled
by VisionAgent's visual servo which triggers on this phrase separately.
"""
import random

trigger_pattern = r'\b(look at me|look here|look this way|look over here|face me|turn to me|look at the camera)\b'

_LOOKS = [
    "Looking at you now.",
    "Right here — I see you.",
    "Got you in my sight.",
    "I'm looking your way.",
]


def result(match) -> str:
    return random.choice(_LOOKS)
