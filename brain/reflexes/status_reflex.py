"""
status_reflex.py — Self-status report fast-path (V10.0)

Fires when user asks how Brain is doing or if it's okay.
Uses only local state — no LLM call needed.
"""
import random
from datetime import datetime

trigger_pattern = r'\b(how are you|how\'s it going|are you okay|are you ok|how do you feel|you doing okay|you alright|how\'s brain doing)\b'

_STATUS = [
    "I'm doing well — sensors active, mind running smoothly.",
    "All good here! I'm listening and ready.",
    "Running fine. How about you?",
    "I'm operational and curious. What's on your mind?",
    "Feeling responsive today! Everything seems to be working.",
]


def result(match) -> str:
    hour = datetime.now().hour
    time_note = (
        "It's morning — I'm feeling alert."
        if 5 <= hour < 12 else
        "Afternoon energy is good."
        if 12 <= hour < 17 else
        "Evening mode — a bit quieter."
        if 17 <= hour < 21 else
        "Late night — running in low-power mode."
    )
    base = random.choice(_STATUS)
    return f"{base} {time_note}"
