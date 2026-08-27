"""
brain/reflexes/__init__.py — Hardcoded Reflex Loader (V10.0)

Loads every *_reflex.py module in this directory and exposes a compiled
list of (pattern, result_fn) tuples for zero-latency matching in ReasoningAgent.

Each reflex module must define:
  - trigger_pattern: str  — regex string matched against user speech (case-insensitive)
  - result: str | callable — response string or function(match) -> str

Reflexes in this directory are HARDCODED (never pruned/promoted).
They run BEFORE semantic embedding reflexes for maximum speed.
"""
from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# (compiled_regex, result_str_or_fn, module_name)
_HARDCODED_REFLEXES: list[tuple[re.Pattern, str | Callable, str]] = []


def load_reflexes() -> list[tuple[re.Pattern, str | Callable, str]]:
    """Import every *_reflex.py in this directory and compile their patterns."""
    global _HARDCODED_REFLEXES
    if _HARDCODED_REFLEXES:
        return _HARDCODED_REFLEXES

    reflexes_dir = Path(__file__).parent
    loaded = []
    for path in sorted(reflexes_dir.glob("*_reflex.py")):
        module_name = f"brain.reflexes.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
            pattern_str = getattr(mod, "trigger_pattern", None)
            result = getattr(mod, "result", None)
            if pattern_str and result is not None:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                loaded.append((compiled, result, path.stem))
                log.info("Hardcoded reflex loaded: %s → pattern='%s'", path.stem, pattern_str[:60])
            else:
                log.warning("Reflex %s missing trigger_pattern or result — skipped", path.stem)
        except Exception as e:
            log.error("Failed to load reflex %s: %s", path.stem, e)

    _HARDCODED_REFLEXES = loaded
    log.info("brain/reflexes: %d hardcoded reflexes loaded", len(loaded))
    return loaded


def check_hardcoded_reflexes(text: str) -> str | None:
    """Check text against all hardcoded reflexes. Returns first match result or None."""
    reflexes = load_reflexes()
    text_lower = text.strip()
    for pattern, result, name in reflexes:
        m = pattern.search(text_lower)
        if m:
            try:
                response = result(m) if callable(result) else result
                log.info("Hardcoded reflex '%s' fired on: '%s'", name, text_lower[:60])
                return str(response)
            except Exception as e:
                log.warning("Hardcoded reflex '%s' execution error: %s", name, e)
    return None
