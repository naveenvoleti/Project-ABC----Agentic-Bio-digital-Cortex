"""Audio driver — mic input and speaker output, cross-platform.
- Windows: Edge TTS (Microsoft neural voices) or pyttsx3 (SAPI) for TTS, PyAudio for mic
- Linux/Pi: espeak-ng for TTS, PyAudio for mic, BlueALSA for BT
"""
from __future__ import annotations

import asyncio
import platform
import subprocess
import tempfile
import threading
from pathlib import Path
from queue import Queue

from brain.utils.logger import get_logger

log = get_logger(__name__)

# Suppress noisy phonemizer mismatch warnings from Kokoro internals
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*words count mismatch.*", category=UserWarning)
_warnings.filterwarnings("ignore", message=".*words count mismatch.*")  # stdlib fallback

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT_INT16 = 8  # pyaudio.paInt16
IS_WINDOWS = platform.system() == "Windows"

# Edge TTS neural voices — warm, natural-sounding
EDGE_VOICES = {
    "female": "en-US-AriaNeural",      # warm, conversational
    "male":   "en-US-GuyNeural",       # natural male
    "jenny":  "en-US-JennyNeural",     # friendly female
}


class AudioDriver:
    def __init__(
        self,
        speaker_type: str = "default",
        bt_mac: str = "",
        tts_engine: str = "auto",   # auto | edge | elevenlabs | espeak
        edge_voice: str = "en-US-JennyNeural",
        elevenlabs_api_key: str = "",
        elevenlabs_voice_name: str = "Anika",
        elevenlabs_voice_id: str = "",
        mock: bool = False,
    ):
        self.speaker_type = speaker_type
        self.bt_mac = bt_mac
        self.mock = mock
        self._pa = None
        self._stream = None
        self._bt_connected = False
        self._running: bool = True
        self._bt_retry_count: int = 0
        self._BT_MAX_RETRIES: int = 10
        self._edge_voice = edge_voice
        self._el_api_key = elevenlabs_api_key
        self._el_voice_name = elevenlabs_voice_name
        self._el_voice_id: str = elevenlabs_voice_id  # pre-set skips API lookup

        # TTS: dedicated thread owns the engine permanently.
        self._tts_q: Queue = Queue(maxsize=10)
        self._tts_thread: threading.Thread | None = None
        self._tts_ready = threading.Event()
        self._tts_stop_flag = threading.Event()   # set by stop_speaking(); cleared before each utterance

        if tts_engine == "auto":
            if elevenlabs_api_key:
                self.tts_engine = "elevenlabs"
            else:
                self.tts_engine = "kokoro"   # local, free, natural — falls back to edge/espeak on failure
        else:
            self.tts_engine = tts_engine
        self._kokoro_model = None   # pre-warmed in background thread during init()

    def init(self) -> bool:
        if self.mock:
            log.info("Audio: mock mode")
            return True
        # Guard: don't start a second TTS thread on watchdog restart
        if self._tts_thread and self._tts_thread.is_alive():
            log.debug("TTS worker already running, skipping re-init")
            return True
        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
            log.info(f"Audio driver initialized — TTS: {self.tts_engine}, platform: {platform.system()}")
            self._tts_thread = threading.Thread(
                target=self._tts_worker,
                daemon=True,
                name="tts-worker",
            )
            self._tts_thread.start()
            self._tts_ready.wait(timeout=3)
            # Pre-warm Kokoro in background so first utterance has no model-load delay
            if self.tts_engine == "kokoro":
                threading.Thread(
                    target=self._prewarm_kokoro,
                    daemon=True,
                    name="kokoro-prewarm",
                ).start()
            if not IS_WINDOWS and self.speaker_type == "bluetooth" and self.bt_mac:
                self._connect_bluetooth()
            return True
        except Exception as e:
            log.error(f"Audio init error: {e}")
            return False

    def _connect_bluetooth(self) -> None:
        if not self._running or self.mock:
            return
        if self._bt_retry_count >= self._BT_MAX_RETRIES:
            log.error(
                f"BT connect: gave up after {self._BT_MAX_RETRIES} attempts "
                f"for {self.bt_mac} — no further retries"
            )
            return
        try:
            result = subprocess.run(
                ["bluetoothctl", "connect", self.bt_mac],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                self._bt_connected = True
                self._bt_retry_count = 0   # reset on success
                log.info(f"Bluetooth speaker connected: {self.bt_mac}")
            else:
                raise RuntimeError(result.stderr.decode().strip() or "non-zero exit")
        except Exception as e:
            self._bt_retry_count += 1
            remaining = self._BT_MAX_RETRIES - self._bt_retry_count
            log.warning(
                f"BT connect failed (attempt {self._bt_retry_count}/{self._BT_MAX_RETRIES}): "
                f"{e} — {'retrying in 30s' if remaining > 0 else 'giving up'}"
            )
            if remaining > 0 and self._running:
                t = threading.Timer(30, self._connect_bluetooth)
                t.daemon = True
                t.start()

    def _prewarm_kokoro(self) -> None:
        """Load Kokoro ONNX model in the background during startup.
        By the time the user speaks, the model is already in memory."""
        model_path = Path("assets/models/kokoro/kokoro-v1.0.onnx")
        voices_path = Path("assets/models/kokoro/voices-v1.0.bin")
        if not model_path.exists() or not voices_path.exists():
            return
        try:
            import time as _t
            from kokoro_onnx import Kokoro
            t0 = _t.monotonic()
            self._kokoro_model = Kokoro(str(model_path), str(voices_path))
            log.info(f"Kokoro pre-warmed in {_t.monotonic()-t0:.1f}s — first utterance will be instant")
        except Exception as e:
            log.warning(f"Kokoro pre-warm failed (non-fatal): {e}")

    def start_recording(self) -> None:
        if self.mock or self._pa is None:
            return
        try:
            import pyaudio
            self._stream = self._pa.open(
                format=FORMAT_INT16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            log.info("Microphone recording started")
        except Exception as e:
            log.error(f"Failed to start recording: {e}")

    def record_chunk(self) -> bytes | None:
        if self.mock:
            return b"\x00" * CHUNK * 2
        if self._stream is None:
            return None
        try:
            return self._stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            log.warning(f"Record chunk error: {e}")
            return None

    def stop_recording(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ── TTS ──────────────────────────────────────────────────────────────────

    def _tts_worker(self) -> None:
        """Permanent TTS worker thread.  Reads from _tts_q and speaks.
        On Windows: Edge TTS (Microsoft neural) → fallback PowerShell SAPI.
        On Linux:   espeak-ng subprocess.
        """
        log.info(f"TTS worker started (engine: {self.tts_engine})")
        self._tts_ready.set()   # signal init() we are running

        # Edge TTS, ElevenLabs, and Kokoro need their own async event loop on this thread
        if self.tts_engine in ("edge", "elevenlabs", "kokoro"):
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
        else:
            loop = None

        while True:
            item = self._tts_q.get()
            if item is None:
                log.debug("TTS worker shutting down")
                if loop:
                    loop.close()
                break
            # Support (text, volume) or (text, volume, speed) tuples
            text, volume = item[0], item[1]
            speed = item[2] if len(item) > 2 else 1.0
            log.debug(f"TTS dequeued: '{text[:60]}' speed={speed:.2f}")
            try:
                if self.tts_engine == "kokoro" and loop:
                    loop.run_until_complete(self._speak_kokoro(text, volume, speed))
                elif self.tts_engine == "elevenlabs" and loop:
                    loop.run_until_complete(self._speak_elevenlabs(text, volume))
                elif self.tts_engine == "edge" and loop:
                    loop.run_until_complete(self._speak_edge(text, volume))
                elif IS_WINDOWS:
                    self._speak_powershell(text, volume)
                else:
                    self._speak_espeak(text, volume)
            except Exception as e:
                log.error(f"TTS speak error: {e}")
                if IS_WINDOWS:
                    try:
                        self._speak_powershell(text, volume)
                    except Exception:
                        pass

    def _speak_powershell(self, text: str, volume: int) -> None:
        """Windows TTS via PowerShell System.Speech — no COM threading issues.
        Uses SSML with a leading 300ms break to prevent audio device clipping."""
        # Escape XML special chars for SSML
        safe = (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("—", " - ")   # em-dash → readable pause
                .replace("'", "\u2019"))  # smart apostrophe safe in SSML
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="en-US">'
            f'<break time="300ms"/>{safe}'
            f'</speak>'
        )
        # Escape the SSML for PowerShell here-string injection
        ps_ssml = ssml.replace('"', '`"')
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 0; $s.Volume = {min(volume, 100)}; "
            f'$s.SpeakSsml("{ps_ssml}");'
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            timeout=30,
            capture_output=True,
        )

    async def _speak_kokoro(self, text: str, volume: int, speed: float = 1.0) -> None:
        """Kokoro ONNX TTS — true streaming: play each chunk the instant it arrives.
        First audio starts within ~200 ms of calling, regardless of text length.
        speed: 0.8 (slow/sad) → 1.0 (neutral) → 1.2 (excited).
        volume: 0–100 mapped to amplitude scale."""
        import sounddevice as sd
        import numpy as np
        import asyncio as _aio
        import time as _time

        if not self._running:
            return

        clean = self._clean_for_tts(text)
        if not clean:
            return

        model_path = Path("assets/models/kokoro/kokoro-v1.0.onnx")
        voices_path = Path("assets/models/kokoro/voices-v1.0.bin")
        if not model_path.exists() or not voices_path.exists():
            log.warning(
                "Kokoro model files not found — falling back to Edge TTS. "
                "Download kokoro-v1.0.onnx and voices-v1.0.bin from "
                "https://github.com/nazdridoy/kokoro-tts/releases/tag/v1.0.0"
            )
            await self._speak_edge(text, volume)
            return

        try:
            from kokoro_onnx import Kokoro

            if self._kokoro_model is None:
                t0 = _time.monotonic()
                self._kokoro_model = Kokoro(str(model_path), str(voices_path))
                log.info(f"Kokoro ONNX model loaded in {_time.monotonic()-t0:.1f}s")

            self._tts_stop_flag.clear()
            speech_speed = max(0.75, min(1.3, speed))
            vol_scale = max(0.4, min(1.4, (volume - 50) / 40.0 + 0.75))

            SR = 24000
            # sounddevice OutputStream — audio plays in real-time as chunks arrive.
            # blocksize=0 → driver picks optimal size; latency="low" targets <50 ms.
            stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", latency="low")
            stream.start()

            chunks_played = 0
            try:
                async for samples, sr in self._kokoro_model.create_stream(
                    clean, voice="af_heart", speed=speech_speed, lang="en-us"
                ):
                    if self._tts_stop_flag.is_set() or not self._running:
                        break
                    chunk = np.clip(samples.astype(np.float32) * vol_scale, -1.0, 1.0)
                    await asyncio.to_thread(stream.write, chunk)
                    chunks_played += 1
            finally:
                # Drain: wait for sounddevice buffer to empty (playback catches up to writes).
                # We poll write_available — when it returns to the full buffer size, all
                # samples have been consumed by the driver.
                if not self._tts_stop_flag.is_set() and chunks_played > 0:
                    try:
                        full_size = stream.write_bufsize if hasattr(stream, 'write_bufsize') else 2048
                        deadline = _time.monotonic() + 4.0
                        while _time.monotonic() < deadline:
                            if self._tts_stop_flag.is_set():
                                break
                            avail = stream.write_available
                            if avail >= full_size - 64:
                                break
                            await asyncio.sleep(0.02)
                    except Exception:
                        await asyncio.sleep(0.3)   # simple fallback drain
                stream.stop()
                stream.close()
                if self._tts_stop_flag.is_set():
                    sd.stop()

            log.info(f"Kokoro spoke: {len(clean)} chars, {chunks_played} chunks, speed={speech_speed:.2f}")

        except _aio.CancelledError:
            sd.stop()
        except RuntimeError as e:
            if "shutdown" in str(e).lower() or "closed" in str(e).lower():
                return  # normal during graceful shutdown — executor already torn down
            log.error(f"Kokoro TTS error: {e} — falling back to Edge TTS")
            self._kokoro_model = None
            if self._running:
                await self._speak_edge(text, volume)
        except Exception as e:
            log.error(f"Kokoro TTS error: {e} — falling back to Edge TTS")
            self._kokoro_model = None
            if self._running:
                await self._speak_edge(text, volume)

    async def _el_lookup_voice(self) -> None:
        """Resolve voice name → voice_id via ElevenLabs API (cached after first call)."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers={"xi-api-key": self._el_api_key},
                )
                voices = resp.json().get("voices", [])
                for v in voices:
                    if v["name"].lower() == self._el_voice_name.lower():
                        self._el_voice_id = v["voice_id"]
                        log.info(f"ElevenLabs: found voice '{v['name']}' → {self._el_voice_id}")
                        return
                # Fallback: first available voice
                if voices:
                    self._el_voice_id = voices[0]["voice_id"]
                    log.warning(f"ElevenLabs: voice '{self._el_voice_name}' not found, using '{voices[0]['name']}'")
        except Exception as e:
            log.error(f"ElevenLabs voice lookup failed: {e}")

    async def _speak_elevenlabs(self, text: str, volume: int) -> None:
        """ElevenLabs TTS — uses official elevenlabs SDK."""
        import asyncio as _asyncio
        import os
        import pygame

        if not self._el_api_key:
            log.warning("ElevenLabs: no API key — falling back to Edge TTS")
            await self._speak_edge(text, volume)
            return

        if not self._el_voice_id:
            await self._el_lookup_voice()
        if not self._el_voice_id:
            await self._speak_edge(text, volume)
            return

        clean = self._clean_for_tts(text)
        if not clean:
            return

        try:
            from elevenlabs import ElevenLabs

            loop = _asyncio.get_event_loop()
            client = ElevenLabs(api_key=self._el_api_key)

            def _generate():
                return client.text_to_speech.convert(
                    text=clean,
                    voice_id=self._el_voice_id,
                    model_id="eleven_v3",
                    output_format="mp3_44100_128",
                    voice_settings={
                        "stability": 0.45,
                        "similarity_boost": 0.80,
                        "style": 0.30,
                        "use_speaker_boost": True,
                    },
                )

            audio_gen = await loop.run_in_executor(None, _generate)
            audio_bytes = b"".join(audio_gen)

            def _write_tmp():
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio_bytes)
                    return f.name
            
            tmp_path = await asyncio.to_thread(_write_tmp)

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._tts_stop_flag.is_set():
                    pygame.mixer.music.stop()
                    break
                await asyncio.sleep(0.02)
            pygame.mixer.music.unload()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            log.info(f"ElevenLabs spoke: {len(clean)} chars")

        except Exception as e:
            log.error(f"ElevenLabs TTS error: {e} — falling back to Edge TTS")
            await self._speak_edge(text, volume)

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """Strip markdown and symbols that make TTS sound robotic."""
        import re
        # Remove markdown bold/italic
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
        # Remove backtick code spans
        text = re.sub(r'`[^`]*`', '', text)
        # Remove hashtag headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Em-dash → natural pause
        text = text.replace('—', ', ').replace('–', ', ')
        # Ellipsis → pause
        text = text.replace('...', '. ')
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove brackets/parens content that's usually metadata
        text = re.sub(r'\[.*?\]', '', text)
        # Collapse multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def _speak_edge(self, text: str, volume: int) -> None:
        """Edge TTS — Microsoft neural voices via edge-tts package."""
        try:
            import edge_tts
            import pygame
            import os

            clean_text = self._clean_for_tts(text)
            if not clean_text:
                return

            communicate = edge_tts.Communicate(
                clean_text,
                self._edge_voice,
                rate="-8%",    # slightly slower = more natural, less rushed
                pitch="-2Hz",  # very slight pitch drop = warmer, less synthetic
                volume=f"+{max(0, volume - 80)}%",
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            await communicate.save(tmp_path)

            # Play via pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            # Wait for playback to finish — check stop flag every 20 ms
            while pygame.mixer.music.get_busy():
                if self._tts_stop_flag.is_set():
                    pygame.mixer.music.stop()
                    break
                await asyncio.sleep(0.02)

            pygame.mixer.music.unload()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        except ImportError:
            log.warning("edge-tts or pygame not installed — falling back to PowerShell SAPI")
            self._speak_powershell(text, volume)
        except Exception as e:
            log.error(f"Edge TTS error: {e} — falling back to PowerShell SAPI")
            self._speak_powershell(text, volume)

    def speak(self, text: str, volume: int = 80, speed: float = 1.0) -> None:
        if self.mock:
            log.info(f"[MOCK SPEAK] {text}")
            return
        # Send to dedicated TTS thread — never blocks the caller
        try:
            self._tts_q.put_nowait((text, volume, speed))
        except Exception:
            log.warning("TTS queue full — dropping utterance")

    def _speak_espeak(self, text: str, volume: int) -> None:
        try:
            subprocess.run(
                ["espeak-ng", "-v", "en+f3", "-s", "165", "-a", str(volume), text],
                timeout=30,
            )
        except FileNotFoundError:
            log.warning("espeak-ng not found")
        except Exception as e:
            log.error(f"espeak TTS error: {e}")

    async def speak_async(self, text: str, volume: int = 80, speed: float = 1.0) -> None:
        """Non-blocking: queues the utterance and returns immediately.
        The TTS worker thread handles it in sequence."""
        self.speak(text, volume, speed)   # already non-blocking for pyttsx3

    def stop_speaking(self) -> None:
        """Interrupt any in-progress TTS playback and clear the queue."""
        # Signal the Kokoro poll loop to exit within 20 ms
        self._tts_stop_flag.set()
        # Clear pending items from TTS queue
        while not self._tts_q.empty():
            try:
                self._tts_q.get_nowait()
            except Exception:
                break
        # Hard-stop sounddevice in case we're mid-chunk
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        # Stop pygame mid-playback (edge/elevenlabs)
        try:
            import pygame
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def close(self) -> None:
        self._running = False   # stops any pending BT retry timers
        self.stop_recording()
        # Send shutdown sentinel to TTS thread
        if self._tts_thread and self._tts_thread.is_alive():
            try:
                self._tts_q.put_nowait(None)
            except Exception:
                pass
        if self._pa:
            self._pa.terminate()
            self._pa = None
