"""
Download Otto robot emoji GIFs from GitHub and map them to emotion states.

Source: https://github.com/txp666/otto-emoji-gif-component/tree/main/gifs

Available GIFs (6):
  staticstate.gif  →  idle / neutral / focused / attentive / curious
  happy.gif        →  happy / excited
  sad.gif          →  bored / confused
  anger.gif        →  frustrated
  scare.gif        →  surprised / scared
  buxue.gif        →  unknown / fallback (shrug / confused face)

Run once during setup:
    python scripts/download_emotions.py
No API keys required.
"""
import sys

# Windows cp1252 terminals reject Unicode arrows — force UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import shutil
from pathlib import Path

import httpx

GITHUB_RAW = "https://raw.githubusercontent.com/txp666/otto-emoji-gif-component/main/gifs"

# All 6 source GIFs in the repo
SOURCE_GIFS = [
    "staticstate.gif",
    "happy.gif",
    "sad.gif",
    "anger.gif",
    "scare.gif",
    "buxue.gif",
]

# Map each emotion state → which source GIF to use
EMOTION_GIF_MAP = {
    "neutral":    "staticstate.gif",   # calm idle robot face
    "attentive":  "staticstate.gif",   # same idle — eyes open, waiting
    "curious":    "buxue.gif",         # inquisitive / tilted look
    "focused":    "staticstate.gif",   # neutral focus state
    "happy":      "happy.gif",         # happy robot face
    "excited":    "happy.gif",         # excited — reuse happy
    "bored":      "sad.gif",           # droopy / low-energy
    "confused":   "buxue.gif",         # "huh?" expression
    "frustrated": "anger.gif",         # angry / frustrated
    "surprised":  "scare.gif",         # startled / scared
}

OUTPUT_DIR = Path("assets/emotions")
CACHE_DIR  = Path("assets/emotions/_cache")   # raw downloads go here


def download_source_gifs() -> dict[str, Path]:
    """Download all 6 source GIFs into the cache dir. Return {name: path}."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached: dict[str, Path] = {}

    print(f"Downloading Otto GIFs from GitHub -> {CACHE_DIR}/\n")
    for gif_name in SOURCE_GIFS:
        dest = CACHE_DIR / gif_name
        if dest.exists():
            print(f"  [skip] {gif_name} already cached")
            cached[gif_name] = dest
            continue
        url = f"{GITHUB_RAW}/{gif_name}"
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            size_kb = dest.stat().st_size // 1024
            print(f"  [ok]   {gif_name} ({size_kb} KB)")
            cached[gif_name] = dest
        except Exception as e:
            print(f"  [err]  {gif_name}: {e}")

    return cached


def map_emotions(cached: dict[str, Path]) -> None:
    """Copy/link cached GIFs to the final emotion names."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nMapping emotions -> {OUTPUT_DIR}/\n")

    for emotion, source_gif in EMOTION_GIF_MAP.items():
        dest = OUTPUT_DIR / f"{emotion}.gif"
        if dest.exists():
            print(f"  [skip] {emotion}.gif already exists")
            continue
        src = cached.get(source_gif)
        if src is None:
            _create_placeholder(dest, emotion)
            continue
        shutil.copy2(src, dest)
        print(f"  [map]  {emotion}.gif  <-  {source_gif}")


def _create_placeholder(dest: Path, name: str) -> None:
    """Minimal 1×1 GIF89a placeholder when a source is unavailable."""
    gif_bytes = (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00"
        b"\xff\xff\xff\x00\x00\x00!\xf9\x04"
        b"\x00\x00\x00\x00\x00,\x00\x00\x00"
        b"\x00\x01\x00\x01\x00\x00\x02\x02"
        b"D\x01\x00;"
    )
    dest.write_bytes(gif_bytes)
    print(f"  [placeholder] {dest.name}  (source GIF unavailable)")


def main() -> None:
    cached = download_source_gifs()
    if not cached:
        print("\nNo GIFs downloaded — check your internet connection.")
        sys.exit(1)

    map_emotions(cached)

    print("\n[OK] All emotion GIFs ready in assets/emotions/")
    print("\nEmotion -> File mapping:")
    for emotion, source in EMOTION_GIF_MAP.items():
        print(f"  {emotion:<12} <- {source}")


if __name__ == "__main__":
    main()
