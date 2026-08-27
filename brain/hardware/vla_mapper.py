"""
VLAMapper — translates Gemini Robotics VLA action token dicts to motor primitives (V7.0).

A VLA action token is a dict produced by LLMRouter.vla_act():
  {"action_type": "move_pan", "params": {"degrees": 30}, "confidence": 0.85}

This module is stateless — all methods are class-level so MotorAgent can call
VLAMapper.parse(token) without instantiation.
"""
from __future__ import annotations

import re
from typing import Any

from brain.utils.logger import get_logger

log = get_logger(__name__)

# Canonical action names the rest of the system understands
ACTION_MAP: dict[str, str] = {
    # Gaze / pan-tilt
    "move_pan":       "move_pan",
    "pan":            "move_pan",
    "look_left":      "move_pan",
    "look_right":     "move_pan",
    "move_tilt":      "move_tilt",
    "tilt":           "move_tilt",
    "look_up":        "move_tilt",
    "look_down":      "move_tilt",
    # Wheel drive
    "rotate_wheels":  "rotate_wheels",
    "rotate":         "rotate_wheels",
    "turn":           "rotate_wheels",
    "drive":          "rotate_wheels",
    # Speech
    "speak":          "speak",
    "say":            "speak",
    # Halt
    "stop":           "stop",
    "halt":           "stop",
    "freeze":         "stop",
}

# Degree sign helpers for fallback_parse
_DEG_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)?')
_DIR_RE = re.compile(r'\b(left|right|forward|backward|back)\b', re.IGNORECASE)


class VLAMapper:
    """Stateless token → motor-primitive translator."""

    @classmethod
    def parse(cls, token: dict[str, Any]) -> dict[str, Any] | None:
        """Map a VLA action token dict to a normalised motor primitive dict.

        Returns {"action_type": str, "params": dict, "confidence": float}
        or None if the token cannot be mapped.
        """
        raw_action = str(token.get("action_type", "")).lower().strip()
        canonical = ACTION_MAP.get(raw_action)
        if canonical is None:
            log.debug("VLAMapper: unknown action_type '%s'", raw_action)
            return None

        params = dict(token.get("params", {}))
        confidence = float(token.get("confidence", 0.5))

        # Normalise direction-based actions that VLA may output with plain verbs
        if canonical == "move_pan":
            if raw_action in ("look_left",):
                params.setdefault("degrees", -30.0)
            elif raw_action in ("look_right",):
                params.setdefault("degrees", 30.0)

        if canonical == "move_tilt":
            if raw_action in ("look_up",):
                params.setdefault("degrees", -20.0)
            elif raw_action in ("look_down",):
                params.setdefault("degrees", 20.0)

        return {"action_type": canonical, "params": params, "confidence": confidence}

    @classmethod
    def fallback_parse(cls, command: str) -> dict[str, Any]:
        """Regex-based fallback when LLM is unavailable.

        Parses natural-language motor commands like "pan left 45 degrees".
        Always returns a dict (never None); confidence is low (0.3).
        """
        cmd = command.lower().strip()

        # Pan / tilt detection
        if any(w in cmd for w in ("pan", "look left", "look right", "rotate head")):
            deg_match = _DEG_RE.search(cmd)
            degrees = float(deg_match.group(1)) if deg_match else 30.0
            if "left" in cmd:
                degrees = -abs(degrees)
            return {"action_type": "move_pan", "params": {"degrees": degrees}, "confidence": 0.3}

        if any(w in cmd for w in ("tilt", "look up", "look down", "nod")):
            deg_match = _DEG_RE.search(cmd)
            degrees = float(deg_match.group(1)) if deg_match else 15.0
            if "down" in cmd:
                degrees = abs(degrees)
            else:
                degrees = -abs(degrees)
            return {"action_type": "move_tilt", "params": {"degrees": degrees}, "confidence": 0.3}

        # Wheel rotation
        if any(w in cmd for w in ("turn", "rotate", "drive", "wheel")):
            deg_match = _DEG_RE.search(cmd)
            degrees = float(deg_match.group(1)) if deg_match else 45.0
            dir_match = _DIR_RE.search(cmd)
            direction = dir_match.group(1).lower() if dir_match else "right"
            return {
                "action_type": "rotate_wheels",
                "params": {"direction": direction, "degrees": degrees},
                "confidence": 0.3,
            }

        # Stop
        if any(w in cmd for w in ("stop", "halt", "freeze", "still")):
            return {"action_type": "stop", "params": {}, "confidence": 0.7}

        # Default: speak
        return {"action_type": "speak", "params": {"text": command}, "confidence": 0.2}
