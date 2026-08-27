"""
VisionAgent — camera capture, object detection, scene understanding.
Maps to the Visual Cortex. Publishes perception events every 500ms.

SmolVLM2 integration:
  - Every captured frame is pushed to the SmolVLM2Processor ring buffer
  - SmolVLM2 scans every 2.5s → writes WorldModel → publishes events
  - Moondream kept as on-demand VLM for direct user visual questions
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.hardware.camera_driver import CameraDriver
from brain.hardware.face_recognizer import FaceRecognizer
from brain.hardware.hw_detector import HardwareCapabilities
from brain.hardware.vision_processor import VisionProcessor
from brain.utils.logger import get_logger

if TYPE_CHECKING:
    from brain.hardware.smolvlm2_processor import SmolVLM2Processor, SceneResult
    from brain.memory.world_model import WorldModel
    from brain.memory.semantic_memory import SemanticMemory
    from brain.memory.gallery_manager import GalleryManager
    from brain.llm.llm_router import LLMRouter

log = get_logger(__name__)

CAPTURE_INTERVAL = 0.5  # seconds
FACE_REC_INTERVAL = 1.0  # run face recognition every N seconds

# Generic visual terms Moondream uses for people — replaced with actual names
_PERSON_PATTERNS = [
    "a man", "a woman", "a person", "an individual", "a human",
    "the man", "the woman", "the person", "the individual",
    "someone", "a figure", "the figure",
]


def _ground_scene(caption: str, known_names: list[str]) -> str:
    """Replace generic person references in a VLM caption with actual names.

    Moondream describes faces as "a man with dark hair" — once face recognition
    identifies the person, we substitute the name so the LLM context says
    "Naveen (a man with dark hair)" instead of discarding the identity.
    """
    if not caption or not known_names:
        return caption

    label = " and ".join(known_names)
    result = caption

    # First match: replace the first generic reference with "Name (original phrase)"
    # so physical context is preserved but identity takes priority.
    lower = caption.lower()
    for pattern in _PERSON_PATTERNS:
        idx = lower.find(pattern)
        if idx != -1:
            orig_phrase = caption[idx: idx + len(pattern)]
            result = caption[:idx] + f"{label} ({orig_phrase})" + caption[idx + len(pattern):]
            return result

    # No pattern matched — prepend name as context
    return f"{label} is present. {caption}"


def _sanitize_person_count(caption: str, face_count: int) -> str:
    """Remove VLM hallucinations about extra people when face recognition
    contradicts them. Only applied when face_count is reliable (> 0 means
    face rec ran successfully).

    Examples:
      face_count=1, caption mentions "two people" → strip that phrase
      face_count=1, caption mentions "friends" or "group" → add correction note
    """
    if not caption or face_count <= 0:
        return caption

    import re
    lower = caption.lower()

    # Patterns that suggest VLM hallucinated multiple people when only 1 is present
    if face_count == 1:
        multi_patterns = [
            r'\b(two|three|four|five|several|multiple|many|a few|a group of|some)\s+(people|persons?|individuals?|others?|friends?|figures?|humans?|men|women|faces?)\b',
            r'\bother\s+(people|persons?|individuals?|figures?)\b',
            r'\bbeside\s+(them|him|her)\b',
            r'\b(his|her|their)\s+friends?\b',
            r'\bgroup\b',
        ]
        for pat in multi_patterns:
            if re.search(pat, lower):
                # Append a correction rather than mangling the sentence
                caption = caption.rstrip(". ") + " [Note: only 1 person detected by face recognition]"
                break

    return caption


class VisionAgent(BaseAgent):
    name = "vision_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        hw: HardwareCapabilities,
        vision_proc: VisionProcessor | None = None,
        face_rec: FaceRecognizer | None = None,
        smolvlm2: "SmolVLM2Processor | None" = None,
        world_model: "WorldModel | None" = None,
        llm: "LLMRouter | None" = None,
        scan_interval_s: float = 3.0,
        semantic: "SemanticMemory | None" = None,
        gallery: "GalleryManager | None" = None,
        soul: "SoulManager | None" = None,
    ):
        super().__init__(bus)
        self._hw = hw
        self._camera: CameraDriver | None = None
        self._vision_proc = vision_proc   # Moondream VLM — may be None (on-demand queries)
        self._face_rec = face_rec         # Face recognizer — may be None
        self._smolvlm2 = smolvlm2        # SmolVLM2 continuous video understanding
        self._world_model = world_model  # Shared world state ground truth
        self._semantic = semantic        # Visual RAG — CLIP vector store
        self._gallery = gallery          # Visual RAG — JPEG snapshot store
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_scene: str = ""
        self._privacy_mode = False
        self._thread_stop = threading.Event()
        self._last_publish_time: float = 0.0
        self._last_person_count: int = 0         # stabilised person count
        self._person_count_votes: list[int] = [] # rolling window for smoothing
        self._last_face_rec_time: float = 0.0    # throttle face recognition
        self._current_frame = None               # latest frame for on-demand use
        # Salience filter state — only publish when something meaningful changed
        self._last_vlm_caption: str = ""         # last caption sent to bus
        self._last_publish_person_count: int = 0 # person count at last publish
        self._motion_was_active: bool = False    # motion state at last publish
        # ECO mode — extended capture interval, face rec paused
        self._capture_interval: float = CAPTURE_INTERVAL
        self._face_rec_paused: bool = False
        # F8 — Gaze tracking: face bbox updated by _detect_objects for motor control
        self._last_face_bboxes: list = []        # list of (x, y, w, h) OpenCV face rects
        self._last_gaze_x: float = 0.0           # last published gaze x offset (-1..1)
        self._last_gaze_y: float = 0.0           # last published gaze y offset (-1..1)
        self._last_gaze_time: float = 0.0        # throttle gaze updates
        self._last_recognized: frozenset = frozenset()  # debounce face-recognition log
        # SmolVLM2 state tracking for event detection
        from brain.memory.lif_network import LIFNetwork
        from brain.memory.anchor_module import AnchorModule
        self._lif_network = LIFNetwork(decay_rate=0.05, threshold=1.0, spike_boost=2.0, max_voltage=5.0)
        self._anchor_module = AnchorModule(soul) if soul else None
        self._last_satiated_entities: frozenset = frozenset()
        self._last_gaze_direct: bool = False
        # v5.0 Predictive Processing — SurpriseHeuristic
        self._predicted_scene: str = ""    # last prediction from ImaginationAgent
        self._surprise_threshold: float = 0.30   # delta > 30% → WORLD_SURPRISE
        # V8.0 — Temporal change monitoring for task outcome verification
        self._monitoring_target: str = ""
        self._monitoring_task: asyncio.Task | None = None
        # Cloud vision scanner — uses LLMRouter with live camera frame for scene descriptions
        self._scan_llm = llm
        self._scan_interval_s = scan_interval_s
        # Visual Hashing — O(1) duplicate detection before CLIP encode
        # OrderedDict acts as a bounded LRU cache (max 2000 frames)
        import collections as _col
        self._phash_cache: _col.OrderedDict = _col.OrderedDict()
        self._PHASH_CACHE_MAX = 2000

    def _setup(self) -> bool:
        if not self._hw.camera_available:
            log.info("VisionAgent: no camera detected, running in passive mode")
            return False
        self._camera = CameraDriver(
            source=self._hw.camera_source,
            device_index=self._hw.camera_device_index,
            mock=False,   # always try real camera if hw reports it available
        )
        return self._camera.open()

    def _detect_objects(self, frame) -> list[str]:
        if frame is None:
            return []
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            variance = float(gray.var())
            mean_brightness = float(gray.mean())

            objects = []

            # Brightness
            if mean_brightness < 40:
                objects.append("dark environment")
            elif mean_brightness > 200:
                objects.append("bright environment")

            # Motion / activity
            if variance > 1500:
                objects.append("high activity scene")
            elif variance > 500:
                objects.append("some motion detected")
            else:
                objects.append("still scene")

            # Face detection with count stabilisation (3-frame rolling majority)
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                raw_count = len(faces)
                # Store bboxes for gaze tracking (F8)
                self._last_face_bboxes = [tuple(int(v) for v in f) for f in faces] if raw_count > 0 else []
                # Keep last 3 readings, use majority vote
                self._person_count_votes.append(raw_count)
                if len(self._person_count_votes) > 3:
                    self._person_count_votes.pop(0)
                stable_count = round(sum(self._person_count_votes) / len(self._person_count_votes))
                self._last_person_count = stable_count
                if stable_count > 0:
                    objects.append(f"{stable_count} person{'s' if stable_count > 1 else ''} in view")
            except Exception:
                self._last_face_bboxes = []

            return objects
        except Exception:
            return []

    def _describe_scene(self, objects: list[str], frame=None) -> str:
        """Natural-language scene description.
        Prefers Moondream VLM caption; falls back to OpenCV summary."""
        if self._vision_proc and self._vision_proc.available and frame is not None:
            vlm_caption = self._vision_proc.caption(frame)
            if vlm_caption:
                # Append quick OpenCV facts (person count, motion) for structured fields
                return vlm_caption
        # Fallback: OpenCV-only
        if not objects:
            return "camera active, nothing notable detected"
        return "; ".join(objects)

    def _frame_to_base64(self, frame) -> str:
        try:
            import cv2
            from PIL import Image
            # OpenCV uses BGR; PIL's fromarray() expects RGB — convert first
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return ""

    async def _on_smolvlm2_result(self, result: "SceneResult") -> None:
        """Called by SmolVLM2Processor.run_loop every scan cycle."""
        if self._privacy_mode:
            return

        wm = self._world_model
        desc = result.description

        # ── Face recognition: authoritative source for WHO is present ─────────
        # Face rec is far more reliable than VLM text for person identity/count.
        # The VLM excels at describing actions/context but frequently hallucinates
        # extra people. We use face rec counts to sanitize VLM person claims.
        face_count = 0       # total faces detected by OpenCV/face_rec
        known: list[str] = []
        if self._face_rec and self._face_rec.available and self._current_frame is not None:
            try:
                faces = await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._face_rec.recognize, self._current_frame
                )
                face_count = len(faces)
                known = [f["name"] for f in faces if f["name"] != "unknown"]
                if known:
                    known_set = frozenset(known)
                    if known_set != self._last_recognized:
                        self._last_recognized = known_set
                        log.info(f"VisionAgent: recognized {known}")
                    desc = _ground_scene(desc, known)

            except Exception:
                pass
        else:
            # No face rec available — use last OpenCV person count as fallback
            face_count = self._last_person_count

        # ── Sanitize VLM hallucinations ───────────────────────────────────────
        # If face rec found only 1 person, strip phrases about multiple people.
        desc = _sanitize_person_count(desc, face_count)

        # ── Build authoritative entity set from face recognition ─────────────
        # Never derive entity presence from VLM text — only from face rec.
        entity_hints: set[str] = set()
        if known:
            for name in known:
                if self._anchor_module:
                    anchored_name = self._anchor_module.get_anchored_identity(name)
                else:
                    anchored_name = name
                entity_hints.add(anchored_name)
                if wm:
                    await wm.update_presence(anchored_name)
        elif face_count > 0:
            # Face detected but not recognized
            entity_hints = {"unknown_person"}
            if wm:
                await wm.update_presence("unknown_person")
        # face_count == 0 → no entities, entity_hints stays empty

        # ── Update WorldModel ─────────────────────────────────────────────────
        if wm:
            await wm.update_scene(desc)
            if result.gaze_direct:
                await wm.update_gaze(True)
            elif self._last_gaze_direct:
                await wm.update_gaze(False)
            if result.user_emotion:
                await wm.update_user_emotion(result.user_emotion)

        # Stage 1: LIF Network processing for habituation
        fired_entities, satiated_entities = self._lif_network.step(entity_hints)

        # PERSON_ENTER: newly fired
        for entity in fired_entities:
            log.info(f"VisionAgent: person entered — {entity}")
            await self.publish(AgentMessage(
                type=MessageType.PERSON_ENTER,
                source=self.name,
                data={"entity": entity, "scene": desc},
                priority=3,
            ))

        # PERSON_LEAVE: entities that were satiated but dropped below threshold
        left_entities = self._last_satiated_entities - frozenset(satiated_entities)
        for entity in left_entities:
            log.info(f"VisionAgent: person left — {entity}")
            await self.publish(AgentMessage(
                type=MessageType.PERSON_LEAVE,
                source=self.name,
                data={"entity": entity},
                priority=3,
            ))
            if wm:
                await wm.remove_presence(entity)
            
        self._last_satiated_entities = frozenset(satiated_entities)

        # GAZE_DIRECT: user just started looking at camera
        if result.gaze_direct and not self._last_gaze_direct:
            log.info("VisionAgent: direct gaze detected")
            await self.publish(AgentMessage(
                type=MessageType.GAZE_DIRECT,
                source=self.name,
                data={"scene": desc},
                priority=2,
            ))
        self._last_gaze_direct = result.gaze_direct

        # SCENE_CHANGE: significant content shift
        current_scene = await wm.get_scene() if wm else ""
        if wm and current_scene and self._caption_diff_pct(desc, self._last_vlm_caption) > 0.35:
            log.info(f"VisionAgent: scene change ({result.latency_ms:.0f}ms via {result.backend})")
            self._last_vlm_caption = desc
            await self.publish(AgentMessage(
                type=MessageType.SCENE_CHANGE,
                source=self.name,
                data={"scene": desc, "prev_scene": await wm.get_prev_scene()},
                priority=4,
            ))
            # ── Visual RAG: encode & store the salient frame ─────────────────
            # Only frames that triggered a scene change are vectorised (Salience Filter).
            # CLIP encoding is offloaded to a thread to keep the async loop unblocked.
            if self._gallery and self._semantic and self._vision_proc and self._current_frame is not None:
                asyncio.create_task(
                    self._store_visual_memory(desc),
                    name="visual-rag-store",
                )

        # v5.0 SurpriseHeuristic — compare to ImaginationAgent prediction
        if self._predicted_scene and desc:
            delta = self._caption_diff_pct(desc, self._predicted_scene)
            if delta > self._surprise_threshold:
                log.info(
                    "VisionAgent: WORLD_SURPRISE (delta=%.0f%% vs prediction)",
                    delta * 100,
                )
                await self.publish(AgentMessage(
                    type=MessageType.WORLD_SURPRISE,
                    source=self.name,
                    data={
                        "scene":          desc,
                        "predicted_scene": self._predicted_scene,
                        "delta":          round(delta, 3),
                        "ts":             time.time(),
                    },
                    priority=1,   # highest — bypasses standard attention gating
                ))

        # WORLD_UPDATE: broadcast updated world snapshot for other agents
        if wm:
            await self.publish(AgentMessage(
                type=MessageType.WORLD_UPDATE,
                source=self.name,
                data=await wm.snapshot(),
                priority=8,
            ))

        # Also update legacy PERCEPTION_VISION for backward compat with other agents
        self._last_scene = desc

    async def _vision_scan_loop(self) -> None:
        """Background scene scanner using LLMRouter with camera frame — cloud vision via Gemma/ER.
        Runs every scan_interval_s seconds. Feeds into the same _on_smolvlm2_result pipeline
        so SCENE_CHANGE, WORLD_UPDATE, PERSON_ENTER/LEAVE events all work.
        """
        import types as _types
        log.info("VisionAgent: cloud vision scan loop started (interval=%.1fs)", self._scan_interval_s)
        while self._running:
            await asyncio.sleep(self._scan_interval_s)
            if self._privacy_mode or self._current_frame is None or not self._scan_llm:
                continue
            frame_b64 = self._frame_to_base64(self._current_frame)
            if not frame_b64:
                continue
            t0 = time.monotonic()
            try:
                desc = await self._scan_llm.infer(
                    user_message=(
                        "Describe what you see in this image concisely. "
                        "Mention people, objects, actions, and setting. "
                        "Two sentences maximum."
                    ),
                    max_tokens=80,
                    frame_b64=frame_b64,
                    skip_cache=True,
                )
            except Exception as e:
                log.warning("VisionAgent: cloud vision scan error: %s", e)
                continue
            if not desc:
                continue
            latency_ms = (time.monotonic() - t0) * 1000
            log.info("VisionAgent: cloud scan %.0fms — %s", latency_ms, desc[:60])
            result = _types.SimpleNamespace(
                description=desc,
                gaze_direct=False,
                user_emotion="",
                raw_response=desc,
                latency_ms=latency_ms,
                backend="cloud-vision",
                timestamp=time.monotonic(),
            )
            await self._on_smolvlm2_result(result)

    async def _monitor_for_change(self, target_description: str, timeout_s: float = 15.0) -> None:
        """V8.0 — Watch for a temporal state change for task outcome verification.

        Unlike _visual_servo() which tracks spatial position, this method watches for
        a STATE CHANGE (e.g. LED starts blinking) by comparing consecutive SmolVLM2 scans.
        Emits OUTCOME_ANALYSIS when change detected or on timeout.
        """
        if not self._smolvlm2:
            await self.publish(AgentMessage(
                type=MessageType.OUTCOME_ANALYSIS,
                source=self.name,
                data={"intent": target_description, "observed_state": "no vision system available",
                      "history": []},
                priority=7,
            ))
            return

        prompt = (
            f"Describe the current state of: {target_description}. "
            "Is it active, on, blinking, or moving? One sentence only."
        )
        descriptions: list[str] = []
        deadline = time.monotonic() + timeout_s
        interval = 1.5

        while time.monotonic() < deadline and self._running:
            try:
                result = await self._smolvlm2.scan_async(prompt)
                desc = result.description if result else ""
                if desc:
                    descriptions.append(desc)
                    if len(descriptions) >= 2 and descriptions[-1] != descriptions[-2]:
                        log.info("VisionAgent: state change detected for '%s'", target_description)
                        break
            except Exception as e:
                log.debug("_monitor_for_change: scan failed: %s", e)
            await asyncio.sleep(interval)

        observed = descriptions[-1] if descriptions else "no observation"
        await self.publish(AgentMessage(
            type=MessageType.OUTCOME_ANALYSIS,
            source=self.name,
            data={"intent": target_description, "observed_state": observed,
                  "history": descriptions[-4:]},
            priority=7,
        ))

    async def _visual_servo(self, target_label: str, timeout_s: float = 5.0) -> bool:
        """V7.0 Visual Servoing — emit SPATIAL_POINT until target is centred or timeout.

        Loops at ~5Hz for up to timeout_s seconds. Each cycle:
          1. Query VisionProcessor.spatial_point(target_label) for current (x, y)
          2. Emit SPATIAL_POINT so MotorAgent snaps gaze
          3. Stop when |x - 0.5| < 0.05 (target within 5% of frame centre)

        Returns True if target was centred, False if timeout or no detection.
        """
        if not self._vision_proc:
            return False

        deadline = time.monotonic() + timeout_s
        log.info("visual_servo: tracking '%s' (timeout=%.1fs)", target_label, timeout_s)

        while time.monotonic() < deadline and self._running:
            point = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._vision_proc.spatial_point,
                target_label,
                self._current_frame,
            )
            x = point.get("x", 0.5)
            confidence = point.get("confidence", 0.0)

            if confidence > 0.1:
                await self.publish(AgentMessage(
                    type=MessageType.SPATIAL_POINT,
                    source=self.name,
                    data=point,
                    priority=2,
                ))
                # Update WorldModel with 3D objects if available
                if self._world_model and self._current_frame is not None:
                    objects_3d = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self._vision_proc.detect_3d, self._current_frame
                    )
                    if objects_3d:
                        await self._world_model.update_spatial(objects_3d)
                        await self.publish(AgentMessage(
                            type=MessageType.SPATIAL_3D,
                            source=self.name,
                            data={"objects": objects_3d},
                            priority=7,
                        ))

                if abs(x - 0.5) < 0.05:
                    log.info("visual_servo: '%s' centred (x=%.3f)", target_label, x)
                    return True
            else:
                log.debug("visual_servo: '%s' not detected (conf=%.2f)", target_label, confidence)

            await asyncio.sleep(0.2)

        log.info("visual_servo: timeout — '%s' not centred", target_label)
        return False

    async def start(self) -> None:
        await super().start()
        self._thread_stop.clear()   # reset on restart
        cam_ok = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._setup
        )
        log.info(f"VisionAgent started (camera={cam_ok})")

        # Start SmolVLM2 continuous video loop if available and not mocked
        if self._smolvlm2:
            await self._smolvlm2.start()
            if getattr(self._smolvlm2, '_backend_name', 'Mock') != 'Mock':
                asyncio.create_task(
                    self._smolvlm2.run_loop(self._on_smolvlm2_result),
                    name="smolvlm2-loop",
                )
                log.info("VisionAgent: SmolVLM2 continuous scan started")
            else:
                log.info("VisionAgent: SmolVLM2 in mock mode — skipping scan loop")

        # Start cloud vision scanner if LLM is available (replaces SmolVLM2 for scene descriptions)
        if self._scan_llm:
            asyncio.create_task(self._vision_scan_loop(), name="vision-scan-loop")

        while self._running:
            if self._privacy_mode or not self._camera:
                self._beat()   # heartbeat even when idle
                await asyncio.sleep(1.0)
                continue

            frame = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._camera.capture_frame
            )

            if frame is not None:
                self._current_frame = frame

                # Push to SmolVLM2 ring buffer for continuous scene understanding
                if self._smolvlm2:
                    self._smolvlm2.push_frame(frame)

                # Let VisionProcessor keep the latest frame for on-demand queries
                if self._vision_proc:
                    self._vision_proc.update_frame(frame)

                objects = await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._detect_objects, frame
                )
                # VLM caption runs inside _describe_scene (rate-limited internally)
                scene = await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._describe_scene, objects, frame
                )

                # Face recognition — throttled to FACE_REC_INTERVAL, paused in ECO
                faces: list[dict] = []
                now = time.monotonic()
                if (
                    self._face_rec
                    and self._face_rec.available
                    and not self._face_rec_paused
                    and (now - self._last_face_rec_time) >= FACE_REC_INTERVAL
                ):
                    self._last_face_rec_time = now
                    faces = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self._face_rec.recognize, frame
                    )
                    if faces:
                        names = [f["name"] for f in faces]
                        known = [n for n in names if n != "unknown"]
                        if known:
                            known_set = frozenset(known)
                            if known_set != self._last_recognized:
                                self._last_recognized = known_set
                                log.info(f"VisionAgent: recognized {known}")
                            # Ground the VLM caption: replace generic "a person / a man /
                            # a woman / someone" references with the actual name so the
                            # LLM context correctly identifies who is in frame rather than
                            # showing a physical description.
                            scene = _ground_scene(scene, known)

                # Derive current motion state from objects list
                motion_active = any("motion" in o or "activity" in o for o in objects)
                person_count  = self._last_person_count   # set by _detect_objects

                # Always update cached scene; publish only when salient + debounced.
                # _is_salient() checks person count change, motion flip, caption drift —
                # much stricter than plain string inequality. Cuts LLM noise by ~70%.
                self._last_scene = scene
                debounce_ok = (now - self._last_publish_time) >= 4.0

                if self._is_salient(scene, person_count, motion_active) and debounce_ok:
                    self._last_publish_time = now
                    # Update salience tracking state so next cycle compares correctly
                    self._last_vlm_caption = scene
                    self._last_publish_person_count = person_count
                    self._motion_was_active = motion_active
                    await self.publish(AgentMessage(
                        type=MessageType.PERCEPTION_VISION,
                        source=self.name,
                        data={
                            "objects": objects,
                            "scene": scene,
                            "has_person": any("person" in o for o in objects),
                            "motion": motion_active,
                            "vlm": self._vision_proc.available if self._vision_proc else False,
                            "faces": faces,
                        },
                        priority=6,
                    ))

                # F8 — Gaze tracking: steer motors toward the primary detected face.
                # Throttled to 1 update per second; only fires when offset is significant.
                if (
                    self._last_face_bboxes
                    and frame is not None
                    and (now - self._last_gaze_time) >= 1.0
                ):
                    h_frame, w_frame = frame.shape[:2]
                    if w_frame > 0 and h_frame > 0:
                        x, y, w, h = self._last_face_bboxes[0]
                        face_cx = x + w // 2
                        face_cy = y + h // 2
                        x_norm = (face_cx - w_frame // 2) / (w_frame // 2)
                        y_norm = (face_cy - h_frame // 2) / (h_frame // 2)
                        # Only publish when face has moved meaningfully (>12% frame)
                        dx = abs(x_norm - self._last_gaze_x)
                        dy = abs(y_norm - self._last_gaze_y)
                        if dx > 0.12 or dy > 0.12:
                            self._last_gaze_x = x_norm
                            self._last_gaze_y = y_norm
                            self._last_gaze_time = now
                            await self.publish(AgentMessage(
                                type=MessageType.ACTION_MOVE,
                                source=self.name,
                                data={
                                    "command": "look_at",
                                    "x_offset": round(x_norm, 2),
                                    "y_offset": round(y_norm, 2),
                                },
                                priority=9,
                            ))

            await asyncio.sleep(self._capture_interval)

    async def handle(self, message: AgentMessage) -> None:
        # V8.0 — Start temporal change monitoring when a task needs visual verification
        if message.type == MessageType.TASK_EXECUTE:
            verify_target = message.data.get("verify_target", "")
            if verify_target:
                if self._monitoring_task and not self._monitoring_task.done():
                    self._monitoring_task.cancel()
                self._monitoring_target = verify_target
                timeout_s = float(message.data.get("timeout_s", 15.0))
                self._monitoring_task = asyncio.create_task(
                    self._monitor_for_change(verify_target, timeout_s),
                    name=f"monitor-{verify_target[:20]}",
                )
            return

        if message.type == MessageType.IMAGINATION_SIMULATE:
            # Store the predicted next world state for SurpriseHeuristic comparison
            predicted = message.data.get("predicted_scene") or message.data.get("simulation", "")
            if predicted:
                self._predicted_scene = predicted
            return

        if message.type == MessageType.BEHAVIOR_CHANGE:
            eco = message.data.get("eco")
            if eco is True:
                self._capture_interval = 30.0   # 30s instead of 0.5s
                self._face_rec_paused = True
                log.info("VisionAgent: ECO mode — capture interval 30s, face rec paused")
            elif eco is False:
                self._capture_interval = CAPTURE_INTERVAL
                self._face_rec_paused = False
                log.info("VisionAgent: full mode restored")

        elif message.type == MessageType.PRIVACY_MODE:
            self._privacy_mode = message.data.get("enabled", False)

        elif message.type == MessageType.VLM_SCAN_NOW:
            # Force an immediate SmolVLM2 scan (e.g., user asked "what do you see?")
            if self._smolvlm2 and not self._privacy_mode:
                prompt = message.data.get("prompt")
                result = await self._smolvlm2.scan_async(prompt)
                if result:
                    await self._on_smolvlm2_result(result)

        elif message.type == MessageType.PERCEPTION_SPEECH:
            text = message.data.get("text", "").strip()
            text_lower = text.lower()

            # ── Face recognition voice commands ───────────────────────────────
            if self._face_rec and self._face_rec.available and self._current_frame is not None:
                # "remember my face as [name]" / "learn my face" / "remember my face"
                if any(kw in text_lower for kw in ("remember my face", "learn my face", "save my face")):
                    # Try to extract a name from the command
                    name = message.data.get("user_name") or self._extract_name(text_lower)
                    if not name:
                        name = "user"
                    ok = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self._face_rec.learn_face, self._current_frame, name
                    )
                    reply = f"Got it! I've saved your face as '{name}'." if ok else "I couldn't detect a clear face. Please try again closer to the camera."
                    await self.publish(AgentMessage(
                        type=MessageType.ACTION_SPEAK,
                        source=self.name,
                        data={"text": reply},
                        priority=2,
                    ))
                    return

                # "forget [name]'s face" / "forget my face"
                elif "forget" in text_lower and "face" in text_lower:
                    name = self._extract_name(text_lower) or "user"
                    ok = self._face_rec.forget_face(name)
                    reply = f"Done, I've forgotten '{name}'." if ok else f"I don't have '{name}' in my memory."
                    await self.publish(AgentMessage(
                        type=MessageType.ACTION_SPEAK,
                        source=self.name,
                        data={"text": reply},
                        priority=2,
                    ))
                    return

                # "do you recognize me" / "who am i" / "do you know me"
                elif any(kw in text_lower for kw in ("do you recognize me", "who am i", "do you know me", "recognize me")):
                    faces = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self._face_rec.recognize, self._current_frame
                    )
                    known = [f for f in faces if f["name"] != "unknown"]
                    if known:
                        names_str = ", ".join(f["name"] for f in known)
                        reply = f"Yes! I recognize you — {names_str}."
                    elif faces:
                        reply = "I can see a face but I don't recognize you yet. You can say 'remember my face' to introduce yourself."
                    else:
                        reply = "I don't see a face clearly right now."
                    await self.publish(AgentMessage(
                        type=MessageType.ACTION_SPEAK,
                        source=self.name,
                        data={"text": reply},
                        priority=2,
                    ))
                    return

                # "who are the known faces" / "who do you know"
                elif any(kw in text_lower for kw in ("who do you know", "known faces", "faces you know")):
                    names = self._face_rec.known_names
                    if names:
                        reply = f"I know: {', '.join(names)}."
                    else:
                        reply = "I don't have any faces saved yet."
                    await self.publish(AgentMessage(
                        type=MessageType.ACTION_SPEAK,
                        source=self.name,
                        data={"text": reply},
                        priority=2,
                    ))
                    return

            # ── Visual servo — "look at X" / "find X" / "track X" ───────────
            import re as _re
            servo_match = _re.search(
                r'\b(?:look at|find|track|locate|focus on|point at)\s+(?:the\s+)?([a-z][a-z\s]{1,30}?)(?:\s+please|\s*$|\s*\.)',
                text_lower,
            )
            if servo_match and self._vision_proc and self._vision_proc.available and not self._privacy_mode:
                target = servo_match.group(1).strip()
                asyncio.create_task(
                    self._visual_servo(target),
                    name=f"visual-servo-{target[:20]}",
                )
                return

            # ── Visual question — Moondream VLM ───────────────────────────────
            if (
                self._vision_proc
                and self._vision_proc.available
                and not self._privacy_mode
            ):
                if VisionProcessor.is_visual_question(text):
                    answer = await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        self._vision_proc.query,
                        text,
                    )
                    if answer:
                        log.info(f"VisionAgent VLM answer: {answer[:80]}")
                        await self.publish(AgentMessage(
                            type=MessageType.PERCEPTION_VISION,
                            source=self.name,
                            data={
                                "scene": answer,
                                "objects": [],
                                "has_person": "person" in answer.lower() or "human" in answer.lower(),
                                "motion": False,
                                "vlm": True,
                                "on_demand": True,
                            },
                            priority=3,   # high priority — user asked directly
                        ))

    @staticmethod
    def _caption_diff_pct(a: str, b: str) -> float:
        """Jaccard word-set distance between two captions — no imports needed.
        Returns 0.0 (identical) to 1.0 (completely different)."""
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa and not wb:
            return 0.0
        intersection = len(wa & wb)
        union = len(wa | wb)
        return 1.0 - (intersection / union) if union else 0.0

    def _is_salient(
        self,
        scene: str,
        person_count: int,
        motion_active: bool,
    ) -> bool:
        """Return True only when the scene contains a meaningful change.

        Rules (any one is sufficient):
          1. Person count changed since last publish (someone arrived or left)
          2. Motion state flipped (started or stopped)
          3. VLM caption differs by >30% word content (Jaccard distance)

        This replaces the raw `scene != last_scene` string comparison which
        fires on every minor caption variation, flooding the LLM with noise.
        """
        if person_count != self._last_publish_person_count:
            return True
        if motion_active != self._motion_was_active:
            return True
        if self._caption_diff_pct(scene, self._last_vlm_caption) > 0.30:
            return True
        return False

    def _extract_name(self, text_lower: str) -> str:
        """Extract a person name from a face command phrase.
        e.g. 'remember my face as naveen' → 'naveen'
             'forget naveen face'         → 'naveen'
        """
        import re
        # "as <name>"
        m = re.search(r"\bas\s+([a-z][a-z\s]{0,20}?)(?:\s+(?:my|their|the|face|please)|\s*$)", text_lower)
        if m:
            return m.group(1).strip().title()
        # "forget <name>'s face" / "forget <name> face"
        m = re.search(r"\bforget\s+([a-z][a-z\s]{1,20}?)(?:'s)?\s+face", text_lower)
        if m:
            return m.group(1).strip().title()
        return ""

    @staticmethod
    def _compute_phash(frame) -> int | None:
        """Compute a 64-bit dHash (difference hash) for duplicate detection.

        Uses 8x8 dHash — fast (~1ms), robust to minor JPEG compression artifacts,
        and tolerant of small perspective shifts. Returns None if imagehash is not
        installed (dedup is silently skipped, no crash).
        """
        try:
            import imagehash
            import cv2
            from PIL import Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            return int(str(imagehash.dhash(img, hash_size=8)), 16)
        except ImportError:
            return None
        except Exception:
            return None

    @staticmethod
    def _make_thumbnail_b64(frame, size: int = 128) -> str:
        """Generate a compact JPEG thumbnail (size×size) from an OpenCV frame.

        The thumbnail is stored directly in ChromaDB metadata (base64 string,
        ~3-6 KB) instead of writing a full JPEG to disk. This keeps the gallery
        clean — disk writes only happen for explicit user-teaching commands.
        """
        try:
            import cv2, io, base64
            from PIL import Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=55)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return ""

    async def _store_visual_memory(self, scene_description: str) -> None:
        """Visual RAG write path — encode salient frame with CLIP and persist.

        Called as a background asyncio.Task on every SCENE_CHANGE so the main
        scan loop is never blocked by the CLIP inference (~20-80ms on CPU).

        No disk writes happen here. A compact 128×128 thumbnail is stored as
        base64 directly inside ChromaDB metadata. The gallery on disk is only
        written when the user explicitly teaches a visual label ("this is my X").

        Pipeline:
            current_frame → CLIP encoder (thread) → SemanticMemory
                          → thumbnail_b64 (in-memory) → ChromaDB metadata
        """
        try:
            frame = self._current_frame
            if frame is None or not self._semantic or not self._vision_proc:
                return

            # 0. Visual Hash dedup — O(1) check before expensive CLIP encode
            #    If we've seen this exact frame before, skip immediately (< 1ms)
            frame_phash = self._compute_phash(frame)
            if frame_phash is not None:
                if frame_phash in self._phash_cache:
                    log.debug("VisionAgent: phash cache hit — skipping CLIP encode (duplicate frame)")
                    return
                # Also check persistent gallery index (survives restarts)
                if self._gallery and self._gallery.lookup_phash(frame_phash):
                    log.debug("VisionAgent: phash gallery hit — skipping CLIP encode (seen before)")
                    return

            # 1. Generate 512-d CLIP visual embedding (CPU-bound — run in thread)
            loop = asyncio.get_event_loop()
            visual_vec = await loop.run_in_executor(
                self._executor,
                self._vision_proc.encode_visual_features,
                frame,
            )
            if not visual_vec:
                log.debug("VisionAgent: CLIP encoder unavailable — skipping visual memory write")
                return

            # 2. Create a compact 128×128 thumbnail stored inside ChromaDB (no disk write)
            thumbnail_b64 = await loop.run_in_executor(
                self._executor,
                self._make_thumbnail_b64,
                frame,
            )

            # 3. Generate a stable content-addressed ID for deduplication
            import hashlib
            import time
            snap_id = hashlib.sha256(scene_description.encode()).hexdigest()[:16]

            # 4. Persist CLIP vector + thumbnail in brain_visual_memory collection
            self._semantic.upsert_visual_memory(
                content=scene_description,
                visual_embedding=visual_vec,
                image_path="",   # no disk path — thumbnail lives in metadata
                metadata={
                    "image_id":      snap_id,
                    "source":        "scene_change",
                    "timestamp":     str(time.time()),
                    "frame_desc":    scene_description[:120],
                    "thumbnail_b64": thumbnail_b64,   # inline thumbnail for LLM grounding
                },
            )

            # 5. Register phash so future identical frames are deduped in < 1ms
            if frame_phash is not None:
                # In-memory cache (LRU eviction)
                self._phash_cache[frame_phash] = snap_id
                if len(self._phash_cache) > self._PHASH_CACHE_MAX:
                    self._phash_cache.popitem(last=False)  # remove oldest
                # Persistent gallery index (survives restarts)
                if self._gallery:
                    self._gallery.register_phash(frame_phash, snap_id)

            log.info(
                "VisionAgent: visual memory stored (in-memory) — id=%s desc='%s'",
                snap_id, scene_description[:60],
            )
        except Exception as e:
            log.warning("VisionAgent: _store_visual_memory error: %s", e)

    async def stop(self) -> None:
        await super().stop()
        self._thread_stop.set()
        if self._smolvlm2:
            try:
                await asyncio.wait_for(self._smolvlm2.stop(), timeout=2.0)
            except (asyncio.TimeoutError, Exception) as e:
                log.warning("VisionAgent: smolvlm2 stop timed out: %s", e)
        if self._camera:
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(self._executor, self._camera.close),
                    timeout=2.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                log.warning("VisionAgent: camera close timed out: %s", e)
            finally:
                self._camera = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        log.info("VisionAgent stopped — camera released")
