"""Face Recognition Driver — identify known people from camera frames.

Backend priority:
  1. face_recognition (dlib) — most accurate, pip install face-recognition
  2. OpenCV LBPH             — lightweight fallback, no extra install needed
  3. mock                    — dev mode, returns empty results

Known faces stored in data/faces/encodings.pkl as {name: [encoding, ...]}
Up to MAX_SAMPLES encodings per person to handle lighting/angle variation.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from brain.utils.logger import get_logger

log = get_logger(__name__)

FACES_DIR        = Path("data/faces")
ENCODINGS_FILE   = FACES_DIR / "encodings.pkl"
UNKNOWN_LABEL    = "unknown"
MAX_SAMPLES      = 10      # max encodings stored per person
DIST_THRESHOLD   = 0.55    # face_recognition distance — lower = stricter match
LBPH_THRESHOLD   = 70.0    # OpenCV LBPH confidence — lower = stricter


class FaceRecognizer:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self._backend: str = "none"
        self._available: bool = False
        self._known: dict[str, list] = {}   # name → list of encodings
        # OpenCV LBPH state
        self._lbph = None
        self._label_to_name: dict[int, str] = {}
        self._lbph_trained: bool = False

    # ── Init ─────────────────────────────────────────────────────────────────

    def init(self) -> bool:
        if self.mock:
            self._available = True
            log.info("FaceRecognizer: mock mode")
            return True

        FACES_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

        try:
            import face_recognition  # noqa: F401
            self._backend = "face_recognition"
            self._available = True
            log.info(f"FaceRecognizer: face_recognition backend — {len(self._known)} known face(s)")
            return True
        except ImportError:
            pass

        try:
            import cv2
            self._lbph = cv2.face.LBPHFaceRecognizer_create()
            self._backend = "opencv_lbph"
            self._available = True
            self._rebuild_lbph()
            log.info(f"FaceRecognizer: OpenCV LBPH backend — {len(self._known)} known face(s)")
            return True
        except (ImportError, AttributeError):
            pass

        log.warning(
            "FaceRecognizer: no backend — install with: pip install face-recognition"
        )
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def known_names(self) -> list[str]:
        return list(self._known.keys())

    def recognize(self, frame) -> list[dict]:
        """Identify faces in frame.
        Returns list of {name, confidence, bbox: {x,y,w,h}}"""
        if not self._available or frame is None or self.mock:
            return []
        if self._backend == "face_recognition":
            return self._recognize_fr(frame)
        if self._backend == "opencv_lbph":
            return self._recognize_lbph(frame)
        return []

    def learn_face(self, frame, name: str) -> bool:
        """Capture and store the largest face in frame under the given name."""
        if not self._available or frame is None:
            return False
        if self._backend == "face_recognition":
            return self._learn_fr(frame, name)
        if self._backend == "opencv_lbph":
            return self._learn_lbph(frame, name)
        return False

    def forget_face(self, name: str) -> bool:
        """Remove all stored encodings for name."""
        if name in self._known:
            del self._known[name]
            self._save()
            if self._backend == "opencv_lbph":
                self._rebuild_lbph()
            log.info(f"FaceRecognizer: forgot '{name}'")
            return True
        return False

    # ── face_recognition backend ──────────────────────────────────────────────

    def _recognize_fr(self, frame) -> list[dict]:
        try:
            import face_recognition
            import cv2
            import numpy as np

            # Downscale 4× for speed; scale results back
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb, model="hog")
            if not locs:
                return []
            encs = face_recognition.face_encodings(rgb, locs)

            # Flatten known encodings into two parallel arrays
            all_names: list[str] = []
            all_encs:  list      = []
            for n, enc_list in self._known.items():
                for e in enc_list:
                    all_names.append(n)
                    all_encs.append(e)

            results = []
            for enc, (top, right, bottom, left) in zip(encs, locs):
                name       = UNKNOWN_LABEL
                confidence = 0.0

                if all_encs:
                    dists    = face_recognition.face_distance(all_encs, enc)
                    best_idx = int(np.argmin(dists))
                    best_d   = float(dists[best_idx])
                    if best_d < DIST_THRESHOLD:
                        name       = all_names[best_idx]
                        confidence = round(1.0 - best_d, 2)

                results.append({
                    "name":       name,
                    "confidence": confidence,
                    "bbox":       {"x": left*4, "y": top*4,
                                   "w": (right-left)*4, "h": (bottom-top)*4},
                })
            return results
        except Exception as e:
            log.warning(f"FaceRecognizer recognize error: {e}")
            return []

    def _learn_fr(self, frame, name: str) -> bool:
        try:
            import face_recognition
            import cv2

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            if not locs:
                log.warning("FaceRecognizer: no face detected — try again closer to camera")
                return False
            encs = face_recognition.face_encodings(rgb, locs)
            if not encs:
                return False

            # Pick largest face (closest to camera)
            areas  = [(b-t)*(r-l) for (t,r,b,l) in locs]
            best   = int(__import__("numpy").argmax(areas))

            bucket = self._known.setdefault(name, [])
            bucket.append(encs[best])
            if len(bucket) > MAX_SAMPLES:
                self._known[name] = bucket[-MAX_SAMPLES:]

            self._save()
            log.info(f"FaceRecognizer: learned '{name}' ({len(self._known[name])} samples)")
            return True
        except Exception as e:
            log.error(f"FaceRecognizer learn error: {e}")
            return False

    # ── OpenCV LBPH backend ──────────────────────────────────────────────────

    def _get_faces_gray(self, frame):
        import cv2
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        return gray, faces

    def _recognize_lbph(self, frame) -> list[dict]:
        if not self._lbph_trained:
            return []
        try:
            import cv2
            gray, faces = self._get_faces_gray(frame)
            results = []
            for (x, y, w, h) in faces:
                roi  = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                label, conf = self._lbph.predict(roi)
                if conf < LBPH_THRESHOLD:
                    name = self._label_to_name.get(label, UNKNOWN_LABEL)
                    confidence = round(1.0 - conf / LBPH_THRESHOLD, 2)
                else:
                    name, confidence = UNKNOWN_LABEL, 0.0
                results.append({"name": name, "confidence": confidence,
                                 "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}})
            return results
        except Exception as e:
            log.warning(f"FaceRecognizer LBPH recognize error: {e}")
            return []

    def _learn_lbph(self, frame, name: str) -> bool:
        try:
            import cv2
            import numpy as np
            gray, faces = self._get_faces_gray(frame)
            if len(faces) == 0:
                log.warning("FaceRecognizer: no face detected")
                return False
            # Largest face
            areas = [w*h for (x,y,w,h) in faces]
            x,y,w,h = faces[int(np.argmax(areas))]
            roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            bucket = self._known.setdefault(name, [])
            bucket.append(roi.tolist())
            if len(bucket) > MAX_SAMPLES:
                self._known[name] = bucket[-MAX_SAMPLES:]
            self._save()
            self._rebuild_lbph()
            log.info(f"FaceRecognizer LBPH: learned '{name}' ({len(self._known[name])} samples)")
            return True
        except Exception as e:
            log.error(f"FaceRecognizer LBPH learn error: {e}")
            return False

    def _rebuild_lbph(self) -> None:
        if not self._known or self._lbph is None:
            self._lbph_trained = False
            return
        try:
            import cv2
            import numpy as np
            images, labels = [], []
            self._label_to_name = {}
            for idx, (name, samples) in enumerate(self._known.items()):
                self._label_to_name[idx] = name
                for s in samples:
                    images.append(np.array(s, dtype=np.uint8))
                    labels.append(idx)
            self._lbph.train(images, np.array(labels))
            self._lbph_trained = True
        except Exception as e:
            log.warning(f"FaceRecognizer: LBPH retrain failed: {e}")
            self._lbph_trained = False

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if ENCODINGS_FILE.exists():
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    self._known = pickle.load(f)
                log.info(f"FaceRecognizer: loaded {len(self._known)} face(s): {list(self._known.keys())}")
            except Exception as e:
                log.warning(f"FaceRecognizer: could not load encodings: {e}")
                self._known = {}

    def _save(self) -> None:
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(self._known, f)
