"""
AuditoryAgent — microphone input, VAD, wake-word detection, STT.
Maps to the Auditory Cortex.

Wake-word pipeline:
  1. openwakeword (default) — neural, runs on-device, no API key.
     Uses onnxruntime backend (tflite not required).
  2. Vosk text-match fallback — checking if transcript contains the wake phrase.

STT:
  Vosk (offline, small-en model) → full transcript after wake word fires.

Thread safety:
  _thread_stop (threading.Event) — set in stop() BEFORE audio is stopped,
  so recording threads exit cleanly before KaldiRecognizer is GC'd.
  _vosk_lock (threading.Lock)    — serialises AcceptWaveform / Result calls.
"""
from __future__ import annotations

import asyncio
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typing import TYPE_CHECKING

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.audio_driver import AudioDriver
from brain.hardware.hw_detector import HardwareCapabilities
from brain.utils.config import agent_feature
from brain.utils.logger import get_logger

if TYPE_CHECKING:
    from brain.memory.world_model import WorldModel

log = get_logger(__name__)

VOSK_MODEL_PATH = "assets/models/vosk-model-small-en-us-0.15"
WAKE_PHRASE     = "hey brain"
SAMPLE_RATE     = 16000
OWW_CHUNK_SIZE  = 1280          # ~80 ms @ 16 kHz

# ── Noise filter ──────────────────────────────────────────────────────────────
_NOISE_WORDS: frozenset[str] = frozenset({
    "huh", "uh", "um", "ah", "oh", "mm", "hmm", "hm", "eh",
    "the", "a", "in", "on", "and", "yeah", "yep", "nah",
})
_MIN_TRANSCRIPT_CHARS  = 4
_MIN_CONFIDENCE        = 0.65   # average Vosk word-confidence threshold
_DEBOUNCE_SECONDS      = 2.0


class AuditoryAgent(BaseAgent):
    name = "auditory_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        hw: HardwareCapabilities,
        audio_driver: AudioDriver,
        cfg: dict | None = None,
        world_model: "WorldModel | None" = None,
    ):
        super().__init__(bus)
        self._hw    = hw
        self._audio = audio_driver
        self._cfg   = cfg or {}
        self._world_model = world_model

        self._executor    = ThreadPoolExecutor(max_workers=2)
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._vosk_rec    = None
        self._vosk_lock   = threading.Lock()      # serialise AcceptWaveform calls
        self._thread_stop = threading.Event()     # signals threads to exit cleanly
        self._oww_model   = None
        self._privacy_mode  = False
        self._wake_active   = False
        self._speaking_until: float = 0.0      # full TTS duration — used for barge-in detection
        self._echo_guard_until: float = 0.0    # short hard mute to suppress initial echo burst

        ww_cfg = self._cfg.get("audio", {}).get("wake_word", {})
        # wake_word enabled: config.yaml AND agents.yaml feature flag must both allow it
        self._ww_enabled   = ww_cfg.get("enabled", True) and agent_feature("auditory_agent", "wake_word", True)
        self._ww_engine    = ww_cfg.get("engine", "openwakeword")
        self._ww_model     = ww_cfg.get("model", "hey_jarvis")
        self._ww_threshold = float(ww_cfg.get("threshold", 0.5))

        # Per-feature toggles from agents.yaml
        self._backchannel_enabled    = agent_feature("auditory_agent", "backchannel",        True)
        self._barge_in_enabled       = agent_feature("auditory_agent", "barge_in",           True)
        self._voice_tone_enabled     = agent_feature("auditory_agent", "voice_tone_analysis", True)

        # Voice tone — set by Vosk thread, read by _process_item
        self._last_voice_tone: str = "calm"
        # Boot guard — backchannels queued before boot_complete are silently dropped.
        # _boot_complete_time starts at inf so every queued_at value is "before" boot.
        self._boot_complete: bool = False
        self._boot_complete_time: float = float("inf")

        # V10.0 — Audio surprise detection
        self._rms_ema: float = 0.03          # rolling average RMS baseline
        self._rms_ema_alpha: float = 0.1     # EMA decay coefficient
        self._last_speech_ts: float = 0.0    # timestamp of last speech detection
        self._audio_surprise_cooldown: float = 0.0  # prevent rapid re-firing


    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_vosk(self) -> bool:
        if not Path(VOSK_MODEL_PATH).exists():
            log.warning(f"Vosk model not found at {VOSK_MODEL_PATH}")
            return False
        try:
            from vosk import Model, KaldiRecognizer
            model = Model(VOSK_MODEL_PATH)
            self._vosk_rec = KaldiRecognizer(model, SAMPLE_RATE)
            log.info("Vosk STT model loaded")
            return True
        except ImportError:
            log.warning("vosk not installed — pip install vosk")
            return False

    def _load_openwakeword(self) -> bool:
        """Try onnxruntime first (no tflite needed), then tflite."""
        for framework in ("onnx", "tflite"):
            try:
                from openwakeword.model import Model as OWWModel
                self._oww_model = OWWModel(
                    wakeword_models=[self._ww_model],
                    inference_framework=framework,
                )
                log.info(
                    f"openwakeword loaded: model='{self._ww_model}' "
                    f"framework={framework} threshold={self._ww_threshold}"
                )
                return True
            except ImportError:
                log.warning("openwakeword not installed — pip install openwakeword")
                return False
            except Exception as e:
                log.debug(f"openwakeword {framework} failed: {e}")
                continue
        log.warning("openwakeword: neither onnx nor tflite backend available")
        return False

    # ── Recording threads ─────────────────────────────────────────────────────

    def _oww_recording_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        """openwakeword pipeline — 80 ms chunks, neural wake-word scoring."""
        import numpy as np
        import time
        buf = b""
        while not self._thread_stop.is_set():
            if self._privacy_mode or not self._hw.mic_available:
                time.sleep(0.2)
                continue
            chunk = self._audio.record_chunk()
            if not chunk:
                continue
            buf += chunk
            while len(buf) >= OWW_CHUNK_SIZE * 2:
                frame = buf[: OWW_CHUNK_SIZE * 2]
                buf   = buf[OWW_CHUNK_SIZE * 2 :]
                audio_arr = np.frombuffer(frame, dtype=np.int16)
                try:
                    preds = self._oww_model.predict(audio_arr)
                except Exception:
                    break
                for model_name, score in preds.items():
                    if score >= self._ww_threshold and not self._wake_active:
                        log.info(f"Wake word! [{model_name}] score={score:.2f}")
                        asyncio.run_coroutine_threadsafe(
                            self._result_queue.put({"type": "wake"}), loop
                        )
                        self._wake_active = True

    def _vosk_recording_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        """Vosk STT — continuous with noise filter, debounce, voice tone analysis,
        and back-channel acknowledgement after 5s of continuous speech."""
        import time
        import numpy as np
        last_text = ""
        last_time = 0.0
        utterance_pcm = bytearray()   # accumulates raw PCM for the current utterance
        utterance_start: float = 0.0  # when the current utterance started accumulating
        backchannel_sent: bool = False # one back-channel per utterance

        while not self._thread_stop.is_set():
            if self._privacy_mode or not self._hw.mic_available:
                time.sleep(0.2)
                continue
            if time.monotonic() < self._echo_guard_until:
                # Hard mute — drain buffer to suppress initial TTS echo burst
                self._audio.record_chunk()
                utterance_pcm.clear()
                utterance_start = 0.0
                backchannel_sent = False
                continue
            chunk = self._audio.record_chunk()
            if not chunk:
                continue

            # Track utterance start for back-channel timing
            if not utterance_pcm:
                utterance_start = time.monotonic()
                backchannel_sent = False

            utterance_pcm.extend(chunk)   # accumulate for tone analysis

            # F3 — Back-channel: if user has been speaking for >8s, emit a brief ack.
            # Only fire when wake-word is active (real conversation) — not on ambient noise.
            if (
                self._backchannel_enabled
                and not backchannel_sent
                and utterance_start > 0
                and (time.monotonic() - utterance_start) > 8.0
                and (self._wake_active or not self._ww_enabled)
            ):
                backchannel_sent = True
                asyncio.run_coroutine_threadsafe(
                    self._result_queue.put({"type": "backchannel",
                                            "queued_at": time.monotonic()}), loop
                )

            # --- lock guards the KaldiRecognizer from concurrent/shutdown access ---
            with self._vosk_lock:
                if self._vosk_rec is None:
                    break          # recogniser was cleared during shutdown
                try:
                    accepted = self._vosk_rec.AcceptWaveform(chunk)
                    if not accepted:
                        continue
                    result = json.loads(self._vosk_rec.Result())
                except Exception as e:
                    log.debug(f"Vosk error (likely shutdown): {e}")
                    break

            # Utterance boundary reached — analyse tone + RMS, then reset buffer
            tone = (
                self._analyze_voice_tone(bytes(utterance_pcm), SAMPLE_RATE)
                if self._voice_tone_enabled else "calm"
            )
            # Compute RMS for volume matching (F11)
            try:
                samples = np.frombuffer(bytes(utterance_pcm), dtype=np.int16).astype(np.float32) / 32768.0
                voice_rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.03
            except Exception:
                voice_rms = 0.03
            utterance_pcm.clear()
            utterance_start = 0.0
            backchannel_sent = False
            self._last_voice_tone = tone

            text = result.get("text", "").strip().lower()
            if not text:
                continue
            # Filter by Vosk per-word confidence
            word_results = result.get("result", [])
            if word_results:
                avg_conf = sum(w.get("conf", 1.0) for w in word_results) / len(word_results)
                if avg_conf < _MIN_CONFIDENCE:
                    log.debug(f"STT low-confidence ({avg_conf:.2f}): '{text}'")
                    continue
            if self._is_noise(text):
                log.debug(f"STT noise filtered: '{text}'")
                # Audio event detection: loud non-speech sound (bang, clap, etc.)
                if voice_rms > 0.08:
                    asyncio.run_coroutine_threadsafe(
                        self._result_queue.put({
                            "type": "audio_event",
                            "rms": voice_rms,
                            "description": "loud ambient sound",
                        }),
                        loop,
                    )
                continue
            now = time.monotonic()
            if text == last_text and (now - last_time) < _DEBOUNCE_SECONDS:
                log.debug(f"STT debounced: '{text}'")
                continue
            last_text = text
            last_time = now
            log.debug(f"Voice tone detected: {tone}, RMS: {voice_rms:.4f}")
            asyncio.run_coroutine_threadsafe(
                self._result_queue.put({
                    "type": "transcript",
                    "text": text,
                    "voice_tone": tone,
                    "voice_rms": voice_rms,
                }),
                loop,
            )

    def _is_noise(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) <= _MIN_TRANSCRIPT_CHARS:
            return True
        if stripped in _NOISE_WORDS:
            return True
        return False

    @staticmethod
    def _analyze_voice_tone(pcm_bytes: bytes, sample_rate: int) -> str:
        """Classify vocal tone from raw int16 PCM using numpy signal features.

        Features used (no extra packages — numpy is already present):
          - RMS energy      → overall loudness
          - Zero Crossing Rate (ZCR) → pitch / frequency content proxy
          - Frame energy variance → amplitude fluctuation (excitement marker)

        Returns one of: 'calm' | 'excited' | 'frustrated' | 'sad' | 'happy'
        Thresholds are calibrated for typical conversational speech at 16 kHz.
        """
        try:
            import numpy as np

            if len(pcm_bytes) < 1024:
                return "calm"

            # int16 PCM → float32 in [-1, 1]
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 1. Overall RMS energy (loudness)
            rms = float(np.sqrt(np.mean(samples ** 2)))

            # 2. Zero Crossing Rate — higher = more high-frequency content (sharper, tenser)
            zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2.0)

            # 3. Per-frame energy variance — high variance = dynamic, expressive speech
            frame_size = max(1, sample_rate // 10)   # 100 ms frames
            n_frames = len(samples) // frame_size
            if n_frames > 1:
                frame_rms = np.array([
                    np.sqrt(np.mean(samples[i * frame_size:(i + 1) * frame_size] ** 2))
                    for i in range(n_frames)
                ])
                energy_var = float(np.var(frame_rms))
            else:
                energy_var = 0.0

            # ── Classification rules ──────────────────────────────────────────
            # very quiet → tired / sad
            if rms < 0.015:
                return "sad"
            # loud + high energy variation → excited
            if rms > 0.055 and energy_var > 0.0008:
                return "excited"
            # loud + low ZCR → tense monotone → frustrated
            if rms > 0.04 and zcr < 0.08:
                return "frustrated"
            # moderately loud + melodic pitch variation → happy
            if rms > 0.035 and zcr >= 0.08 and energy_var > 0.0004:
                return "happy"
            return "calm"

        except Exception:
            return "calm"

    # ── Start / Stop ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        self._thread_stop.clear()   # reset in case we were restarted

        if not self._hw.mic_available:
            log.info("AuditoryAgent: no microphone, passive mode")
            # Still beat so watchdog doesn't restart us
            while self._running:
                self._beat()
                await asyncio.sleep(30)
            return

        self._audio.init()
        self._audio.start_recording()
        vosk_ok = self._load_vosk()
        loop    = asyncio.get_running_loop()

        if (self._ww_enabled
                and self._ww_engine == "openwakeword"
                and self._load_openwakeword()):
            threading.Thread(
                target=self._oww_recording_thread, args=(loop,),
                daemon=True, name="auditory-oww",
            ).start()
            if vosk_ok:
                threading.Thread(
                    target=self._vosk_recording_thread, args=(loop,),
                    daemon=True, name="auditory-vosk",
                ).start()
        elif vosk_ok:
            log.info("AuditoryAgent: using Vosk text-match wake-word fallback")
            threading.Thread(
                target=self._vosk_recording_thread, args=(loop,),
                daemon=True, name="auditory-vosk",
            ).start()
        else:
            log.warning("AuditoryAgent: no STT or wake-word engine available")

        log.info(
            f"AuditoryAgent started — engine={self._ww_engine} "
            f"model={self._ww_model} threshold={self._ww_threshold}"
        )

        while self._running:
            try:
                item = await asyncio.wait_for(self._result_queue.get(), timeout=1.0)
                await self._process_item(item)
            except asyncio.TimeoutError:
                self._beat()   # periodic heartbeat for passive periods
                continue

    async def _process_item(self, item: dict) -> None:
        # F3 — Back-channel acknowledgement: user has been speaking >5s
        # Guard: never fire during boot — mic captures ambient noise for ~40s before
        # boot_complete, which falsely triggers the >5s threshold.
        if item.get("type") == "backchannel":
            # Drop if boot hasn't completed OR if this item was queued before boot completed.
            # (Race: boot_complete arrives first → sets _boot_complete_time, then stale
            #  queue item runs; queued_at < _boot_complete_time → silently dropped.)
            queued_at = item.get("queued_at", 0.0)
            if not self._boot_complete or queued_at < self._boot_complete_time:
                return
            phrases = ["Mm-hmm.", "I see.", "Go on.", "Yeah.", "Mhm."]
            await self.publish(AgentMessage(
                type=MessageType.ACTION_SPEAK,
                source=self.name,
                data={"text": random.choice(phrases)},
                priority=6,
            ))
            return

        if item.get("type") == "wake":
            import time
            interrupted = time.monotonic() < self._speaking_until
            if interrupted:
                # Brain was speaking — clear the mute so mic opens immediately
                self._speaking_until = 0.0
                log.info("AuditoryAgent: wake word during TTS — interrupting speech")
            await self.publish(AgentMessage(
                type=MessageType.PERCEPTION_SPEECH,
                source=self.name,
                data={"text": "", "raw": "", "is_wake_word": True,
                      "interrupt": interrupted, "language": "en",
                      "voice_tone": self._last_voice_tone},
                priority=1,
            ))
            return

        if item.get("type") == "audio_event":
            import time as _time
            rms = item.get("rms", 0.0)
            desc = item.get("description", "ambient sound")
            log.info(f"AuditoryAgent: audio event detected (RMS={rms:.3f}) — {desc}")
            if self._world_model:
                await self._world_model.update_audio(desc)
            await self.publish(AgentMessage(
                type=MessageType.AUDIO_EVENT,
                source=self.name,
                data={"description": desc, "rms": rms},
                priority=4,
            ))
            # V10.0 — Audio surprise: check if this RMS spike is surprising
            await self._check_audio_surprise(rms, "audio_spike", desc)
            return


        if item.get("type") == "transcript":
            import time as _time
            text       = item.get("text", "")
            voice_tone = item.get("voice_tone", self._last_voice_tone)
            voice_rms  = item.get("voice_rms", 0.03)
            is_wake = WAKE_PHRASE in text
            if is_wake and not self._wake_active:
                self._wake_active = True

            # Barge-in: user spoke while brain was still speaking
            brain_speaking = self._barge_in_enabled and _time.monotonic() < self._speaking_until
            if brain_speaking:
                self._speaking_until = 0.0
                self._echo_guard_until = 0.0
                log.info(f"AuditoryAgent: barge-in detected — '{text}' [tone: {voice_tone}]")
                await self.publish(AgentMessage(
                    type=MessageType.PERCEPTION_SPEECH,
                    source=self.name,
                    data={"text": text, "raw": text, "is_wake_word": False,
                          "interrupt": True, "language": "en",
                          "voice_tone": voice_tone, "voice_rms": voice_rms},
                    priority=1,
                ))
                return

            clean = text.replace(WAKE_PHRASE, "").strip()
            # Only log at INFO when in an active conversation; ambient transcriptions are debug noise
            _log = log.info if (self._wake_active or not self._ww_enabled) else log.debug
            _log(f"STT: '{text}' [tone: {voice_tone}] (wake={is_wake}, active={self._wake_active})")
            if self._wake_active or not self._ww_enabled:
                # Update WorldModel with speech context
                if self._world_model and (clean or text):
                    await self._world_model.update_audio(f"User said: {clean or text}")
                    if voice_tone and voice_tone != "calm":
                        await self._world_model.update_user_emotion(voice_tone.upper())
                await self.publish(AgentMessage(
                    type=MessageType.PERCEPTION_SPEECH,
                    source=self.name,
                    data={
                        "text": clean or text,
                        "raw": text,
                        "is_wake_word": is_wake,
                        "language": "en",
                        "voice_tone": voice_tone,
                        "voice_rms": voice_rms,
                    },
                    priority=2,
                ))
                if not is_wake and self._wake_active:
                    self._wake_active = False

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.ACTION_DISPLAY:
            if message.data.get("boot_complete"):
                import time as _t
                self._boot_complete = True
                self._boot_complete_time = _t.monotonic()
                log.debug("AuditoryAgent: boot complete — backchannels enabled")

        elif message.type == MessageType.PRIVACY_MODE:
            self._privacy_mode = message.data.get("enabled", False)
            if self._privacy_mode:
                self._audio.stop_recording()
            else:
                self._audio.start_recording()

        elif message.type == MessageType.ACTION_SPEAK:
            import time
            text = message.data.get("text", "")
            word_count = len(text.split()) if text else 0
            # Full TTS duration estimate (~2.5 words/sec + 0.5s buffer)
            full_secs = max(1.5, word_count / 2.5 + 0.5)
            now = time.monotonic()
            self._speaking_until = now + full_secs
            # Echo guard: only hard-mute the first 1.2s regardless of text length
            self._echo_guard_until = now + 1.2
            log.debug(f"AuditoryAgent: echo guard 1.2s, barge-in window {full_secs:.1f}s")

    async def _check_audio_surprise(self, rms: float, surprise_type: str, detail: str) -> None:
        """V10.0 — Detect surprising audio events and fire WORLD_SURPRISE.

        Maintains an exponential moving average (EMA) of RMS energy.
        Fires if:
          - Sudden spike: current RMS > 1.8× EMA baseline (unexpected loud sound)
          - Unexpected silence: 30s of silence after a period of speech activity
        """
        import time as _t
        now = _t.time()

        # Cooldown: don't fire more than once per 5s
        if now < self._audio_surprise_cooldown:
            # Still update EMA even during cooldown
            self._rms_ema = (1 - self._rms_ema_alpha) * self._rms_ema + self._rms_ema_alpha * rms
            return

        # Update EMA baseline
        prev_ema = self._rms_ema
        self._rms_ema = (1 - self._rms_ema_alpha) * self._rms_ema + self._rms_ema_alpha * rms

        # Check for spike surprise
        surprise_triggered = False
        if prev_ema > 0.005 and rms > prev_ema * 1.8:
            log.info("AuditoryAgent: AUDIO SURPRISE — spike %.3f vs baseline %.3f (%s)",
                     rms, prev_ema, surprise_type)
            surprise_triggered = True
            await self.publish(AgentMessage(
                type=MessageType.WORLD_SURPRISE,
                source=self.name,
                data={
                    "source":   "auditory",
                    "type":     "audio_spike",
                    "detail":   detail,
                    "rms":      rms,
                    "baseline": prev_ema,
                    "delta":    min(1.0, (rms - prev_ema) / prev_ema),
                },
                priority=2,
            ))

        # Check for unexpected silence (30s after last speech with active conversation)
        if (self._last_speech_ts > 0
                and (now - self._last_speech_ts) > 30.0
                and not surprise_triggered):
            log.info("AuditoryAgent: AUDIO SURPRISE — unexpected silence (%.0fs)",
                     now - self._last_speech_ts)
            surprise_triggered = True
            await self.publish(AgentMessage(
                type=MessageType.WORLD_SURPRISE,
                source=self.name,
                data={
                    "source":  "auditory",
                    "type":    "unexpected_silence",
                    "silence_s": round(now - self._last_speech_ts, 1),
                    "delta":   0.5,
                },
                priority=4,
            ))
            self._last_speech_ts = 0.0   # reset to prevent repeated silence alerts

        if surprise_triggered:
            self._audio_surprise_cooldown = now + 5.0

    async def stop(self) -> None:
        await super().stop()
        # 1. Signal threads to stop FIRST — before touching the recogniser
        self._thread_stop.set()
        # 2. Stop audio stream so record_chunk() stops blocking
        self._audio.stop_recording()
        # 3. Nullify recogniser under lock so any in-flight AcceptWaveform sees None
        with self._vosk_lock:
            self._vosk_rec = None
        self._executor.shutdown(wait=False)

