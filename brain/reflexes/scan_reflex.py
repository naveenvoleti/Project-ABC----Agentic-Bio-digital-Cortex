"""
scan_reflex.py — Visual scan fast-path (V10.0)

Fires when user asks Brain to look around or describe what it sees.
The spoken response is immediate; VisionAgent independently runs
the VLM scan and publishes its description shortly after.
"""
import random

trigger_pattern = r'\b(what do you see|look around|scan the room|what can you see|describe what you see|look at your surroundings|what\'s around you|scan around)\b'

_SCANS = [
    "Let me take a look around...",
    "Scanning my surroundings now.",
    "Looking around — give me a moment.",
    "I'll survey the area.",
    "Taking a good look at everything around me.",
]


def result(match) -> str:
    return random.choice(_SCANS)
