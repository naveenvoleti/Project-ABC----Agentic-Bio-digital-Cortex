"""Moondream VLM wrapper — semantic vision pre-processor.

Sits between the raw camera feed and the LLM:
  camera frame → Moondream → rich natural-language scene description → LLM context

Capabilities used:
  - caption()  : natural scene description every N seconds
  - query()    : answer specific visual questions on-demand
  - detect()   : locate objects with bounding boxes

Mode selection (auto):
  1. Cloud API  — if MOONDREAM_API_KEY is set  (fastest, needs network)
  2. Local 2B   — if 'moondream' package installed (good for dev/laptop)
  3. Local 0.5B — lighter local model
  4. Mock       — logs fake output, no model needed
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from brain.agents.base_agent import AgentMessage, MessageType
from brain.utils.logger import get_logger

log = get_logger(__name__)

# Visual question keywords — trigger on-demand Moondream query
VISUAL_KEYWORDS = (
    "see", "look", "who", "what's in", "what is in", "show", "scene",
    "room", "face", "person", "people", "around you", "front of you",
    "describe", "notice", "observe", "detect", "find", "spot",
)


class VisionProcessor:
    def __init__(
        self,
        mode: str = "auto",        # auto | cloud | local | mock
        api_key: str = "",         # Moondream cloud API key
        model_size: str = "2b",    # 2b | 0_5b  (local only)
        caption_interval_s: float = 3.0,
        bus: asyncio.Queue | None = None,
    ):
        self._mode = mode
        self._api_key = api_key
        self._model_size = model_size
        self._caption_interval = caption_interval_s
        self._bus = bus
        self._model: Any = None
        self._available = False

        self._last_caption_time: float = 0.0
        self._last_caption: str = ""
        self._last_frame: Any = None  # most recent camera frame (for on-demand queries)

        # Visual RAG — CLIP encoder (lazy-loaded on first call)
        self._clip_encoder: Any = None
        self._clip_available: bool = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def init(self) -> bool:
        if self._mode == "mock":
            self._available = True
            log.info("VisionProcessor: mock mode")
            return True

        # 1. Try cloud API
        if self._mode in ("auto", "cloud") and self._api_key:
            if self._try_cloud():
                return True

        # 2. Try local model
        if self._mode in ("auto", "local"):
            if self._try_local():
                return True

        log.warning(
            "VisionProcessor: Moondream unavailable — "
            "install 'moondream' package or set MOONDREAM_API_KEY"
        )
        return False

    def _try_cloud(self) -> bool:
        try:
            import moondream as md
            self._model = md.vl(api_key=self._api_key)
            self._mode = "cloud"
            self._available = True
            log.info("VisionProcessor: Moondream cloud API ready")
            return True
        except ImportError:
            log.warning("VisionProcessor: 'moondream' package not installed — pip install moondream")
        except Exception as e:
            log.warning(f"VisionProcessor: cloud init failed: {e}")
        return False

    def _try_local(self) -> bool:
        """Try to load moondream locally via transformers (HuggingFace).
        The 'moondream' pip package routes to cloud even when given a model name,
        so we use transformers directly for true local inference."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            from PIL import Image

            repo = "vikhyat/moondream2"
            log.info(f"VisionProcessor: loading {repo} from HuggingFace (first run downloads ~2 GB)…")

            tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                trust_remote_code=True,
                torch_dtype=torch.float32,   # float32 for CPU (no GPU on Pi/laptop CPU)
            )
            model.eval()

            # Verify with a tiny dummy image
            dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
            enc = model.encode_image(dummy)
            model.query(enc, "test")

            # Store as a thin wrapper with the same caption/query/detect interface
            self._model = _TransformersBackend(model, tokenizer)
            self._mode = "local"
            self._available = True
            log.info("VisionProcessor: moondream2 loaded locally via transformers")
            return True
        except ImportError:
            log.warning("VisionProcessor: transformers/torch not installed — pip install transformers torch")
        except Exception as e:
            log.warning(f"VisionProcessor: local init failed: {e}")
        return False

    def _init_encoder(self) -> None:
        """Lazy-load CLIP (clip-ViT-B-32) via sentence-transformers for visual RAG.

        Uses the same CLIP image/text embedding space so cross-modal retrieval
        (text query → similar historical frames) works correctly.
        Requires: pip install sentence-transformers
        """
        if self._clip_available or self._mode == "mock":
            return
        try:
            from sentence_transformers import SentenceTransformer
            import os
            # Try offline-cached model first to avoid network calls at runtime
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self._clip_encoder = SentenceTransformer(
                    "clip-ViT-B-32",
                    local_files_only=True,
                )
            except Exception:
                os.environ.pop("HF_HUB_OFFLINE", None)
                self._clip_encoder = SentenceTransformer("clip-ViT-B-32")
                os.environ["HF_HUB_OFFLINE"] = "1"
            self._clip_available = True
            log.info("VisionProcessor: CLIP encoder (clip-ViT-B-32) ready for Visual RAG")
        except ImportError:
            log.warning(
                "VisionProcessor: sentence-transformers not installed — "
                "visual RAG disabled. Run: pip install sentence-transformers"
            )
        except Exception as e:
            log.warning(f"VisionProcessor: CLIP encoder init failed: {e}")

    # ── Core API ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def update_frame(self, frame: Any) -> None:
        """Store latest camera frame so on-demand queries can use it."""
        self._last_frame = frame

    def caption(self, frame: Any | None = None) -> str:
        """Return natural-language scene description.
        Rate-limited to caption_interval_s — returns cached value in between."""
        now = time.monotonic()
        if now - self._last_caption_time < self._caption_interval:
            return self._last_caption

        target = frame if frame is not None else self._last_frame
        if target is None or not self._available:
            return self._last_caption

        if self._mode == "mock":
            self._last_caption = "A room with ambient lighting and no notable activity."
            self._last_caption_time = now
            return self._last_caption

        # Always update rate-limit timestamp first so failures don't spam
        self._last_caption_time = now

        try:
            img = self._to_pil(target)
            encoded = self._model.encode_image(img)
            result = self._model.caption(encoded)
            text = result.get("caption", "") if isinstance(result, dict) else str(result)
            if self._last_caption and self._is_surprising(text, self._last_caption):
                log.info(f"VisionProcessor: scene surprise detected — '{text[:80]}'")
                if self._bus is not None:
                    try:
                        self._bus.put_nowait(AgentMessage(
                            type=MessageType.WORLD_SURPRISE,
                            source="vision_processor",
                            data={"scene": text, "old_scene": self._last_caption},
                            priority=1,
                        ))
                    except asyncio.QueueFull:
                        pass
            self._last_caption = text
            log.debug(f"Moondream caption: {text[:100]}")
            return text
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err or "403" in err:
                log.warning("VisionProcessor: auth error — disabling Moondream (set MOONDREAM_API_KEY in .env)")
                self._available = False   # stop all future calls
            else:
                log.warning(f"Moondream caption error: {e}")
            return self._last_caption

    def query(self, question: str, frame: Any | None = None) -> str:
        """Answer a visual question about the current frame on-demand."""
        target = frame if frame is not None else self._last_frame
        if target is None or not self._available:
            return ""

        if self._mode == "mock":
            return f"[mock vision] I see a typical indoor scene. ({question})"

        try:
            img = self._to_pil(target)
            encoded = self._model.encode_image(img)
            result = self._model.query(encoded, question)
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            log.debug(f"Moondream query '{question[:60]}': {answer[:80]}")
            return answer
        except Exception as e:
            log.warning(f"Moondream query error: {e}")
            return ""

    def detect(self, object_class: str, frame: Any | None = None) -> list[dict]:
        """Detect objects — returns [{x_min, y_min, x_max, y_max}] (normalised 0–1)."""
        target = frame if frame is not None else self._last_frame
        if target is None or not self._available:
            return []

        if self._mode == "mock":
            return []

        try:
            img = self._to_pil(target)
            encoded = self._model.encode_image(img)
            result = self._model.detect(encoded, object_class)
            return result.get("objects", []) if isinstance(result, dict) else []
        except Exception as e:
            log.warning(f"Moondream detect error: {e}")
            return []

    def scene_for_llm(self, frame: Any | None = None) -> str:
        """Return a ready-to-inject LLM context string with scene description."""
        desc = self.caption(frame)
        return f"[Visual scene] {desc}" if desc else ""

    def spatial_point(self, query: str, frame: Any | None = None) -> dict:
        """V7.0 Spatial Intelligence — return normalised (x, y) centre of queried object.

        Uses detect() bbox; returns {"x": float, "y": float, "confidence": float, "query": str}.
        x, y are in [0, 1] where (0.5, 0.5) is frame centre.
        Returns confidence=0.0 if object not found.
        """
        objects = self.detect(query, frame)
        if not objects:
            if self._mode == "mock":
                return {"x": 0.5, "y": 0.5, "confidence": 0.5, "query": query}
            return {"x": 0.5, "y": 0.5, "confidence": 0.0, "query": query}

        # Use the first (highest-confidence) detection
        obj = objects[0]
        x = (obj.get("x_min", 0.4) + obj.get("x_max", 0.6)) / 2.0
        y = (obj.get("y_min", 0.4) + obj.get("y_max", 0.6)) / 2.0
        # Confidence proxy: larger bbox area = higher confidence
        area = max(0.0, (obj.get("x_max", 0.6) - obj.get("x_min", 0.4)) *
                        (obj.get("y_max", 0.6) - obj.get("y_min", 0.4)))
        confidence = min(1.0, area * 4.0)
        log.debug("spatial_point '%s': x=%.3f y=%.3f conf=%.2f", query, x, y, confidence)
        return {"x": x, "y": y, "confidence": confidence, "query": query}

    def detect_3d(self, frame: Any | None = None) -> list[dict]:
        """V7.0 Spatial Intelligence — estimate 3D bounding boxes for salient objects.

        Returns list of {"label", "x", "y", "z_est", "w", "h", "d_est"} where:
          x, y — normalised frame centre [0–1]
          z_est — depth estimate (metres) inferred from bbox area (monocular heuristic)
          w, h  — normalised bbox dimensions
          d_est — diameter estimate (metres) from w * z_est * FOV_factor
        """
        if self._mode == "mock":
            return [{"label": "person", "x": 0.5, "y": 0.5, "z_est": 1.5,
                     "w": 0.2, "h": 0.4, "d_est": 0.6}]

        # Query a set of common salient object classes
        SALIENT_CLASSES = ["person", "cup", "bottle", "chair", "table", "phone", "hand"]
        results: list[dict] = []
        for label in SALIENT_CLASSES:
            for obj in self.detect(label, frame):
                w = max(0.01, obj.get("x_max", 0) - obj.get("x_min", 0))
                h = max(0.01, obj.get("y_max", 0) - obj.get("y_min", 0))
                x = obj.get("x_min", 0) + w / 2.0
                y = obj.get("y_min", 0) + h / 2.0
                # Monocular depth heuristic: z ≈ k / sqrt(area), k=0.5 calibrated for 60° FOV
                area = w * h
                z_est = round(0.5 / max(0.01, area ** 0.5), 2)
                d_est = round(w * z_est * 1.1, 2)
                results.append({
                    "label": label, "x": round(x, 3), "y": round(y, 3),
                    "z_est": z_est, "w": round(w, 3), "h": round(h, 3), "d_est": d_est,
                })
        log.debug("detect_3d: %d objects found", len(results))
        return results

    # ── Visual RAG Encoding ───────────────────────────────────────────────────

    def encode_visual_features(self, frame: Any) -> list[float]:
        """Encode a camera frame into a CLIP embedding vector (512-d).

        This is the core of Visual RAG — the resulting vector represents the
        'visual meaning' of the scene and can be stored in ChromaDB for
        similarity search later, allowing the robot to find frames that
        *look* like the current scene without needing a text caption.

        Returns a 512-d float list, or a zero-vector in mock/error cases.
        """
        if self._mode == "mock":
            return [0.0] * 512
        if not self._clip_available:
            self._init_encoder()
        if not self._clip_available or self._clip_encoder is None:
            return []
        try:
            img = self._to_pil(frame)
            vec = self._clip_encoder.encode(img)
            return vec.tolist()
        except Exception as e:
            log.warning(f"VisionProcessor: encode_visual_features failed: {e}")
            return []

    def encode_text_features(self, query: str) -> list[float]:
        """Encode a text query in CLIP embedding space (512-d).

        Uses the same CLIP model as encode_visual_features() so text and image
        vectors live in the same embedding space. This enables cross-modal
        retrieval: a text query like 'coffee mug on the counter' can find the
        most visually similar stored frames.

        Returns a 512-d float list, or [] on failure.
        """
        if self._mode == "mock":
            return [0.0] * 512
        if not self._clip_available:
            self._init_encoder()
        if not self._clip_available or self._clip_encoder is None:
            return []
        try:
            vec = self._clip_encoder.encode(query)
            return vec.tolist()
        except Exception as e:
            log.warning(f"VisionProcessor: encode_text_features failed: {e}")
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(frame: Any):
        """Convert OpenCV BGR frame (numpy array) → PIL Image."""
        from PIL import Image
        import cv2
        import numpy as np
        if isinstance(frame, Image.Image):
            return frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _is_surprising(new: str, old: str) -> bool:
        """True when the new caption introduces more than 3 words absent from the old one."""
        return len(set(new.split()) - set(old.split())) > 3

    @staticmethod
    def is_visual_question(text: str) -> bool:
        """Heuristic: does this user utterance ask about what the camera sees?"""
        low = text.lower()
        return any(kw in low for kw in VISUAL_KEYWORDS)


class _TransformersBackend:
    """Thin wrapper around the raw HuggingFace moondream2 model
    so it exposes the same encode_image / caption / query / detect interface
    as the cloud client."""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def encode_image(self, pil_image):
        return self._model.encode_image(pil_image)

    def caption(self, encoded) -> dict:
        result = self._model.caption(encoded, length="normal")
        # moondream2 returns a string directly
        text = result if isinstance(result, str) else result.get("caption", "")
        return {"caption": text}

    def query(self, encoded, question: str) -> dict:
        result = self._model.query(encoded, question)
        answer = result if isinstance(result, str) else result.get("answer", "")
        return {"answer": answer}

    def detect(self, encoded, object_class: str) -> dict:
        try:
            result = self._model.detect(encoded, object_class)
            objects = result if isinstance(result, list) else result.get("objects", [])
            return {"objects": objects}
        except Exception:
            return {"objects": []}
