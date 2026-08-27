"""
SmolVLM2Processor — continuous video understanding for the conscious brain.

Architecture:
  - RingBuffer: thread-safe 60-frame (~2s at 30fps) circular buffer
  - Three backends in priority order:
      1. _LlamaCppBackend  — llama.cpp server (GGUF quantized, ~0.3–0.8s) ← FASTEST
      2. _OpenVINOBackend  — Intel OpenVINO IR (~1.5s on Iris Xe)
      3. _HuggingFaceBackend — plain PyTorch CPU (~60s, emergency fallback)
  - SmolVLM2Processor: async wrapper, samples latest frame every scan_interval seconds

Fastest path (matches ngxson/smolvlm-realtime-webcam):
  1. Start llama-server:
       llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --port 8090 -ngl 99
       # -ngl 99 offloads all layers to Intel Iris Xe GPU
  2. Set config: vision.smolvlm2.backend: "auto"  (llamacpp is tried first)
  3. Set config: vision.smolvlm2.scan_interval: 0.5

OpenVINO path (no llama.cpp, ~1.5s):
  pip install optimum[openvino] && pip install transformers --upgrade
  Set config: backend: "openvino", scan_interval: 1.5
"""
from __future__ import annotations

import asyncio
import collections
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SCAN_INTERVAL   = 0.5   # default for llamacpp; overridden by config
RING_SIZE       = 60    # frames in ring buffer (~2s @ 30fps)
SAMPLE_FRAMES   = 6     # frames for HuggingFace/OpenVINO video understanding
MAX_NEW_TOKENS  = 100   # max tokens per VLM response
MAX_FRAME_DIM   = 512   # resize frames before inference

# Per-backend safe minimum intervals (seconds) — prevents OOM on slow backends
_BACKEND_MIN_INTERVALS = {
    "LlamaCpp":    0.3,
    "OpenVINO":    1.0,
    "HuggingFace": 5.0,
    "Mock":        0.5,
}


@dataclass
class SceneResult:
    description: str
    gaze_direct: bool = False
    user_emotion: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0
    backend: str = ""
    timestamp: float = field(default_factory=time.monotonic)


class RingBuffer:
    """Thread-safe fixed-size circular frame buffer."""

    def __init__(self, maxsize: int = RING_SIZE) -> None:
        self._buf: collections.deque = collections.deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame)

    def latest(self) -> np.ndarray | None:
        """Return the most recent frame."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def sample(self, n: int = SAMPLE_FRAMES) -> list[np.ndarray]:
        """Return n evenly-spaced frames from the buffer."""
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return []
        if len(buf) <= n:
            return buf
        indices = [int(i * (len(buf) - 1) / (n - 1)) for i in range(n)]
        return [buf[i] for i in indices]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_frame(frame: np.ndarray, max_dim: int = MAX_FRAME_DIM) -> np.ndarray:
    """Resize frame so the longest edge is max_dim (preserves aspect ratio)."""
    from PIL import Image
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)


def _frame_to_jpeg_b64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode numpy frame as base64 JPEG string (for llama.cpp API)."""
    import base64
    import io
    from PIL import Image
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Backend implementations ──────────────────────────────────────────────────

class _LlamaCppBackend:
    """llama.cpp server backend — fastest real-time path (~0.3–0.8s on Intel Iris Xe).

    Replicates the approach from https://github.com/ngxson/smolvlm-realtime-webcam:
      - Sends a single JPEG frame as base64 to the OpenAI-compatible API
      - llama.cpp runs a GGUF-quantized model with GPU layers offloaded via -ngl 99
      - No Python model loading — inference is pure C++

    Start the server before launching Project-ABC:
      llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --port 8090 -ngl 99

    For Intel Iris Xe GPU acceleration add: --device GPU (requires llama.cpp with SYCL/OpenCL)
    CPU-only (still fast with GGUF quantization):
      llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --port 8090
    """

    def __init__(self, server_url: str = "http://localhost:8090") -> None:
        import urllib.request
        self._base_url = server_url.rstrip("/")
        self._completion_url = self._base_url + "/completion"

        # Verify server is reachable — try /health, fallback to /v1/models
        reachable = False
        for path in ("/health", "/v1/models", "/"):
            try:
                urllib.request.urlopen(self._base_url + path, timeout=3)
                reachable = True
                break
            except Exception:
                continue

        if not reachable:
            raise ConnectionError(
                f"llama.cpp server not reachable at {server_url}. "
                "Start it with: llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --port 8090 -ngl 99"
            )
        log.info(f"SmolVLM2: llama.cpp backend ready at {server_url}")

    def infer(self, frames: list[np.ndarray], prompt: str) -> str:
        import json
        import urllib.request

        # Use most recent frame — single frame for maximum speed
        frame = frames[-1] if frames else frames[0]
        frame = _resize_frame(frame, max_dim=MAX_FRAME_DIM)
        b64 = _frame_to_jpeg_b64(frame, quality=80)

        # Use /completion + image_data — works with all llama.cpp versions including
        # those with --jinja enabled by default (PR #17524). The /v1/chat/completions
        # + image_url path breaks on newer builds with SmolVLM2.
        full_prompt = (
            "<|im_start|>user\n"
            "[img-1]\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        payload = {
            "prompt": full_prompt,
            "image_data": [{"data": b64, "id": 1}],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.1,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._completion_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())

        return result["content"].strip()


class _OpenVINOBackend:
    """Intel OpenVINO optimized backend. ~1.5s on i5-H + Iris Xe.

    Requires: pip install optimum[openvino] && pip install transformers --upgrade
    On first run the model is exported to IR format and saved to data/models/ so
    subsequent starts load instantly from cache.
    """

    def __init__(self, model_id: str, device: str = "CPU") -> None:
        from pathlib import Path
        from transformers import AutoProcessor

        OVClass = None
        for cls_name in ("OVModelForImageTextToText", "OVModelForVisualCausalLM"):
            try:
                import importlib
                mod = importlib.import_module("optimum.intel")
                OVClass = getattr(mod, cls_name)
                log.info(f"SmolVLM2: using OpenVINO class {cls_name}")
                break
            except (ImportError, AttributeError):
                continue
        if OVClass is None:
            raise ImportError("Neither OVModelForImageTextToText nor OVModelForVisualCausalLM found in optimum.intel")

        safe_name = model_id.replace("/", "--")
        cache_dir = Path("data/models") / f"{safe_name}-openvino"
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"SmolVLM2: loading OpenVINO backend ({model_id}, device={device})")
        self._processor = AutoProcessor.from_pretrained(model_id)

        ov_xml = cache_dir / "openvino_model.xml"
        if ov_xml.exists():
            log.info(f"SmolVLM2: loading cached OpenVINO IR from {cache_dir}")
            self._model = OVClass.from_pretrained(str(cache_dir), device=device)
        else:
            log.info("SmolVLM2: exporting to OpenVINO IR (first run — may take a few minutes)…")
            self._model = OVClass.from_pretrained(
                model_id,
                export=True,
                task="image-text-to-text",
                device=device,
            )
            self._model.save_pretrained(str(cache_dir))
            self._processor.save_pretrained(str(cache_dir))
            log.info(f"SmolVLM2: OpenVINO IR saved to {cache_dir}")

        log.info("SmolVLM2: OpenVINO backend ready")

    def infer(self, frames: list[np.ndarray], prompt: str) -> str:
        from PIL import Image
        resized = [_resize_frame(f) for f in frames]
        pil_frames = [Image.fromarray(f) for f in resized]
        content = [{"type": "image"} for _ in pil_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=pil_frames, return_tensors="pt", padding=True)
        generated = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded = self._processor.decode(generated[0], skip_special_tokens=True)
        if "assistant" in decoded.lower():
            idx = decoded.lower().rfind("assistant")
            return decoded[idx + len("assistant"):].strip(": \n")
        return decoded.strip()


class _HuggingFaceBackend:
    """Plain HuggingFace transformers CPU backend. ~60s — emergency fallback only."""

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        log.warning("SmolVLM2: HuggingFace CPU backend loaded — expect ~60s inference. "
                    "Start llama-server for real-time performance.")
        self._processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, _attn_implementation="eager"
        )
        self._model.eval()
        log.info(f"SmolVLM2: HuggingFace backend ready (dtype={dtype})")

    def infer(self, frames: list[np.ndarray], prompt: str) -> str:
        import torch
        from PIL import Image
        resized = [_resize_frame(f) for f in frames]
        pil_frames = [Image.fromarray(f) for f in resized]
        content = [{"type": "image"} for _ in pil_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=pil_frames, return_tensors="pt", padding=True)
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded = self._processor.decode(generated[0], skip_special_tokens=True)
        if "assistant" in decoded.lower():
            idx = decoded.lower().rfind("assistant")
            return decoded[idx + len("assistant"):].strip(": \n")
        return decoded.strip()


class _MockBackend:
    """Mock backend for testing without any model."""

    def infer(self, frames: list[np.ndarray], prompt: str) -> str:
        return "[mock] A person is sitting at a desk, looking at a screen. They appear focused."


# ── Main processor ────────────────────────────────────────────────────────────

class SmolVLM2Processor:
    """
    Async wrapper around SmolVLM2 backend.

    Usage:
        proc = SmolVLM2Processor(config)
        await proc.start()
        proc.push_frame(frame_np)           # called from camera thread
        result = await proc.scan_async()    # on-demand scan
    """

    # Concise prompts — kept under 100 tokens of output
    SCENE_PROMPT = (
        "Briefly describe who is in the scene, what they are doing, "
        "and their emotional state. One or two sentences."
    )

    # Video-aware prompt used when multiple frames are available — asks about
    # actions and motion across time rather than just the static state.
    VIDEO_SCENE_PROMPT = (
        "You are watching a short video clip. Describe: (1) who is present, "
        "(2) what action or movement is happening across the frames (e.g. waving, "
        "nodding, typing, standing up), and (3) their emotional state. Two sentences max."
    )

    GAZE_PROMPT = "Is the person looking directly at the camera? Answer yes or no."

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._ring = RingBuffer(maxsize=RING_SIZE)
        self._backend: Any = None
        self._backend_name: str = ""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smolvlm2")
        self._scan_interval: float = float(config.get("scan_interval", SCAN_INTERVAL))
        self._running = False
        self._server_proc: subprocess.Popen | None = None  # auto-started llama-server

    def push_frame(self, frame: np.ndarray) -> None:
        """Called from camera capture thread. Thread-safe."""
        self._ring.push(frame)

    async def start(self) -> None:
        self._running = True
        await asyncio.get_event_loop().run_in_executor(self._executor, self._load_backend)

    def _try_autostart_server(self, server_url: str) -> bool:
        """Spawn llama-server as a subprocess and wait until it responds.

        Returns True if the server became ready within the startup timeout.
        The subprocess is stored in self._server_proc and terminated on stop().
        """
        import urllib.request

        cmd = self._cfg.get(
            "llamacpp_cmd",
            "llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --port 8090 -ngl 99",
        )
        startup_timeout = int(self._cfg.get("llamacpp_startup_timeout", 60))

        log.info("SmolVLM2: auto-starting llama-server: %s", cmd)
        try:
            self._server_proc = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.warning("SmolVLM2: llama-server not found in PATH — install llama.cpp")
            return False
        except Exception as e:
            log.warning("SmolVLM2: failed to launch llama-server: %s", e)
            return False

        deadline = time.monotonic() + startup_timeout
        poll_interval = 2.0
        log.info("SmolVLM2: waiting up to %ds for llama-server to become ready…", startup_timeout)
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            if self._server_proc.poll() is not None:
                log.warning("SmolVLM2: llama-server process exited prematurely")
                return False
            for path in ("/health", "/v1/models", "/"):
                try:
                    urllib.request.urlopen(server_url.rstrip("/") + path, timeout=3)
                    log.info("SmolVLM2: llama-server ready at %s", server_url)
                    return True
                except Exception:
                    continue
        log.warning("SmolVLM2: llama-server did not become ready within %ds", startup_timeout)
        return False

    def _load_backend(self) -> None:
        """Load backend in thread pool (priority: llamacpp → openvino → huggingface → mock)."""
        cfg = self._cfg

        if cfg.get("mock", False):
            log.info("SmolVLM2: using mock backend")
            self._backend = _MockBackend()
            self._backend_name = "Mock"
            return

        backend_pref = cfg.get("backend", "auto")

        # 1. Try llama.cpp (fastest — replicates ngxson/smolvlm-realtime-webcam)
        if backend_pref in ("llamacpp", "auto"):
            server_url = cfg.get("llamacpp_url", "http://localhost:8090")
            try:
                self._backend = _LlamaCppBackend(server_url)
                self._backend_name = "LlamaCpp"
                # Auto-set safe scan interval for llamacpp if config value is too slow
                if self._scan_interval > 2.0:
                    self._scan_interval = 0.5
                    log.info(f"SmolVLM2: llama.cpp loaded — scan_interval set to {self._scan_interval}s")
                return
            except ConnectionError:
                # Server not running — try to auto-start it
                if cfg.get("auto_start", False):
                    if self._try_autostart_server(server_url):
                        try:
                            self._backend = _LlamaCppBackend(server_url)
                            self._backend_name = "LlamaCpp"
                            if self._scan_interval > 2.0:
                                self._scan_interval = 0.5
                            log.info("SmolVLM2: llama.cpp auto-start succeeded")
                            return
                        except Exception as e:
                            log.warning("SmolVLM2: llama.cpp connect after auto-start failed: %s", e)
                if backend_pref == "llamacpp":
                    log.error("SmolVLM2: llamacpp backend required but server not running")
                    self._backend = _MockBackend()
                    self._backend_name = "Mock"
                    return
                log.info("SmolVLM2: llama.cpp not running, trying OpenVINO…")
            except Exception as e:
                if backend_pref == "llamacpp":
                    log.error(f"SmolVLM2: llamacpp backend required but unavailable: {e}")
                    self._backend = _MockBackend()
                    self._backend_name = "Mock"
                    return
                log.info(f"SmolVLM2: llama.cpp not available ({e}), trying OpenVINO…")

        model_id = cfg.get("model_id", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

        # 2. Try OpenVINO (~1.5s)
        if backend_pref in ("openvino", "auto"):
            try:
                device = cfg.get("openvino_device", "CPU")
                self._backend = _OpenVINOBackend(model_id, device=device)
                self._backend_name = "OpenVINO"
                # Ensure interval is safe for OpenVINO
                self._scan_interval = max(self._scan_interval, _BACKEND_MIN_INTERVALS["OpenVINO"])
                return
            except Exception as e:
                log.warning(f"SmolVLM2: OpenVINO failed ({e}), falling back to HuggingFace")

        # 3. HuggingFace CPU (~60s — last resort)
        if backend_pref not in ("llamacpp", "openvino"):
            try:
                self._backend = _HuggingFaceBackend(model_id)
                self._backend_name = "HuggingFace"
                self._scan_interval = max(self._scan_interval, _BACKEND_MIN_INTERVALS["HuggingFace"])
                return
            except Exception as e:
                log.error(f"SmolVLM2: HuggingFace backend failed ({e}), using mock")

        self._backend = _MockBackend()
        self._backend_name = "Mock"

    async def scan_async(self, prompt: str | None = None) -> SceneResult | None:
        """Run one inference pass. Returns None if no frames available."""
        if not self._backend:
            return None

        # llamacpp only needs the latest frame; others use 6-frame video sample
        # for native temporal understanding (action detection, not just state).
        if self._backend_name == "LlamaCpp":
            frame = self._ring.latest()
            frames = [frame] if frame is not None else []
        else:
            frames = self._ring.sample(SAMPLE_FRAMES)

        if not frames:
            return None

        # Use video-aware prompt when we have multiple frames to reason over
        if prompt is None:
            prompt = self.VIDEO_SCENE_PROMPT if len(frames) > 1 else self.SCENE_PROMPT
        t0 = time.monotonic()

        loop = asyncio.get_event_loop()
        try:
            raw = await loop.run_in_executor(
                self._executor,
                self._backend.infer, frames, prompt,
            )
        except Exception as e:
            log.error(f"SmolVLM2 inference error: {e}")
            return None

        latency_ms = (time.monotonic() - t0) * 1000
        log.debug(f"SmolVLM2 [{self._backend_name}] {latency_ms:.0f}ms: {raw[:80]}")

        lower = raw.lower()
        gaze = any(p in lower for p in [
            "looking directly", "eye contact", "looking at the camera",
            "facing the camera", "staring at the camera",
        ])
        emotion = ""
        for e_word in ["happy", "sad", "angry", "confused", "excited", "neutral", "frustrated"]:
            if e_word in lower:
                emotion = e_word.upper()
                break

        return SceneResult(
            description=raw,
            gaze_direct=gaze,
            user_emotion=emotion,
            raw_response=raw,
            latency_ms=latency_ms,
            backend=self._backend_name,
            timestamp=time.monotonic(),
        )

    async def run_loop(self, callback) -> None:
        """Background loop: scans every scan_interval seconds and calls callback(SceneResult)."""
        log.info(f"SmolVLM2: run_loop started (backend={self._backend_name}, interval={self._scan_interval}s)")
        while self._running:
            await asyncio.sleep(self._scan_interval)
            if not self._ring:
                continue
            result = await self.scan_async()
            if result:
                try:
                    await callback(result)
                except Exception as e:
                    log.error(f"SmolVLM2 callback error: {e}")

    async def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
        if self._server_proc and self._server_proc.poll() is None:
            log.info("SmolVLM2: terminating auto-started llama-server (pid=%d)", self._server_proc.pid)
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
