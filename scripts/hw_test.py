"""
Hardware validation script — run before starting the brain.
python scripts/hw_test.py
"""
import sys
import subprocess
from pathlib import Path

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(label: str, result: bool, warn_only: bool = False) -> bool:
    tag = PASS if result else (WARN if warn_only else FAIL)
    print(f"  {tag} {label}")
    return result


def test_python():
    print("\n── Python ──")
    import sys
    ver = sys.version_info
    check(f"Python {ver.major}.{ver.minor}.{ver.micro}", ver >= (3, 11))


def test_packages():
    print("\n── Python Packages ──")
    required = ["cv2", "vosk", "pyttsx3", "PIL", "fastapi", "anthropic", "spacy"]
    for pkg in required:
        try:
            __import__(pkg)
            check(pkg, True)
        except ImportError:
            check(pkg, False)


def test_camera():
    print("\n── Camera ──")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, _ = cap.read()
        cap.release()
        check("USB Camera (device 0)", ok, warn_only=True)
    except Exception as e:
        check(f"Camera: {e}", False, warn_only=True)


def test_audio():
    print("\n── Audio ──")
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        n = pa.get_device_count()
        check(f"PyAudio — {n} devices found", n > 0)

        has_input = any(
            pa.get_device_info_by_index(i)["maxInputChannels"] > 0
            for i in range(n)
        )
        check("Microphone input device", has_input, warn_only=True)

        has_output = any(
            pa.get_device_info_by_index(i)["maxOutputChannels"] > 0
            for i in range(n)
        )
        check("Speaker output device", has_output, warn_only=True)
        pa.terminate()
    except Exception as e:
        check(f"PyAudio: {e}", False, warn_only=True)


def test_espeak():
    print("\n── TTS (espeak-ng) ──")
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, text=True, timeout=5
        )
        check("espeak-ng installed", result.returncode == 0)
    except FileNotFoundError:
        check("espeak-ng not found", False, warn_only=True)


def test_ollama():
    print("\n── Ollama ──")
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        check("Ollama running", True)
        check("phi3:mini available", any("phi3" in m for m in models), warn_only=True)
        check("llama3.2 available", any("llama3.2" in m for m in models), warn_only=True)
        check("nomic-embed-text available", any("nomic" in m for m in models), warn_only=True)
    except Exception:
        check("Ollama server (localhost:11434)", False, warn_only=True)


def test_files():
    print("\n── Required Files ──")
    required = [
        Path("config/config.yaml"),
        Path("config/.env"),
        Path("data/SOUL.md"),
        Path("data/USER.md"),
        Path("data/SKILLS.json"),
        Path("assets/emotions"),
    ]
    for p in required:
        check(str(p), p.exists())

    gifs = list(Path("assets/emotions").glob("*.gif")) if Path("assets/emotions").exists() else []
    check(f"Emotion GIFs ({len(gifs)} found)", len(gifs) >= 10, warn_only=True)


def test_gpio():
    print("\n── GPIO (Pi only) ──")
    try:
        import RPi.GPIO as GPIO
        check("RPi.GPIO available", True)
    except ImportError:
        check("RPi.GPIO (not running on Pi)", False, warn_only=True)


def main():
    print("=" * 50)
    print("  Project-ABC Hardware Validation")
    print("=" * 50)

    test_python()
    test_packages()
    test_files()
    test_camera()
    test_audio()
    test_espeak()
    test_ollama()
    test_gpio()

    print("\n" + "=" * 50)
    print("  Validation complete.")
    print("  WARN items are optional — brain will work without them.")
    print("  FAIL items must be resolved before starting.")
    print("=" * 50)


if __name__ == "__main__":
    main()
