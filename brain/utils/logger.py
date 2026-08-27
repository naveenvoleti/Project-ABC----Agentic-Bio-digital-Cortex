"""Structured logger — uses loguru if available, falls back to stdlib logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Force UTF-8 on Windows terminals (cp1252 can't encode → ← etc.) ─────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from loguru import logger as _loguru_logger
    _USE_LOGURU = True
except ImportError:
    _loguru_logger = None
    _USE_LOGURU = False

_configured = False


def configure_logging(log_file: str = "data/brain.log", level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    if _USE_LOGURU:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
            colorize=True,
        )
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            encoding="utf-8",
        )
    else:
        # Use a UTF-8 StreamHandler instead of basicConfig to avoid cp1252 issues
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        ))
        # Ensure handler stream is UTF-8
        if hasattr(handler.stream, "reconfigure"):
            try:
                handler.stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        logging.root.setLevel(getattr(logging, level.upper(), logging.INFO))
        logging.root.addHandler(handler)

    _configured = True


class _StdlibAdapter:
    """Thin wrapper making stdlib Logger look like a loguru logger."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def debug(self, msg, *a, **kw):    self._log.debug(str(msg), *a)
    def info(self, msg, *a, **kw):     self._log.info(str(msg), *a)
    def warning(self, msg, *a, **kw):  self._log.warning(str(msg), *a)
    def error(self, msg, *a, **kw):    self._log.error(str(msg), *a)
    def critical(self, msg, *a, **kw): self._log.critical(str(msg), *a)

    def bind(self, **kw): return self


class _LoguruAdapter:
    """Loguru wrapper that pre-formats %-style args before passing to loguru.

    Loguru uses {}-style formatting and silently ignores %-style positional args,
    so log.info("val=%d", n) would print "val=%d" literally. This adapter
    pre-formats the message with % before handing it to loguru.
    """

    def __init__(self, name: str):
        self._log = _loguru_logger.bind(name=name)

    @staticmethod
    def _fmt(msg: object, args: tuple) -> str:
        s = str(msg)
        if args:
            try:
                return s % args
            except Exception:
                return s
        return s

    def debug(self, msg, *a, **kw):    self._log.debug(self._fmt(msg, a))
    def info(self, msg, *a, **kw):     self._log.info(self._fmt(msg, a))
    def warning(self, msg, *a, **kw):  self._log.warning(self._fmt(msg, a))
    def error(self, msg, *a, **kw):    self._log.error(self._fmt(msg, a))
    def critical(self, msg, *a, **kw): self._log.critical(self._fmt(msg, a))

    def bind(self, **kw): return self


def get_logger(name: str):
    configure_logging()
    if _USE_LOGURU:
        return _LoguruAdapter(name)
    return _StdlibAdapter(name)
