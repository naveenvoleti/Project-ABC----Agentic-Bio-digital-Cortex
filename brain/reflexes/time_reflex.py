"""
time_reflex.py — Time query fast-path (V10.0)

Fires when user asks what time it is.
Returns current time from datetime — zero LLM latency.
"""
from datetime import datetime

trigger_pattern = r'\b(what time is it|what\'s the time|tell me the time|current time|what time|do you know the time)\b'


def result(match) -> str:
    now = datetime.now()
    hour = now.hour
    minute = now.strftime("%M")
    period = "AM" if hour < 12 else "PM"
    display_hour = hour if hour <= 12 else hour - 12
    if display_hour == 0:
        display_hour = 12
    return f"It's {display_hour}:{minute} {period}."
