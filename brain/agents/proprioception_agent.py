"""
ProprioceptionAgent — Somatosensory Cortex + Cerebellum mapping. (V10.0)

Maintains the robot's body schema: cumulative physical pose in space.
Tracks pan/tilt angles, wheel odometry heading, and distance traveled
based on motor commands received — the "efference copy" of movement.

This gives the brain an answer to "where am I pointing?" and "how far
have I moved?" without needing to look at external sensors.

Published: PROPRIOCEPTION_STATE → WorldModel + CognitionAgent + PlannerAgent

Subscribes to:
  - ACTION_MOVE → update pose based on pan/tilt/wheel commands
  - TASK_OUTCOME with motor_reset=True → reset odometry to zero
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent, MessageType

if TYPE_CHECKING:
    from brain.memory.world_model import WorldModel

log = logging.getLogger(__name__)

# Physical limits
_PAN_LIMIT_DEG  = 90.0   # ±90° pan from center
_TILT_LIMIT_DEG = 45.0   # ±45° tilt from center
_PUBLISH_INTERVAL_S = 5.0  # publish pose every 5s
_SIGNIFICANT_CHANGE_DEG = 5.0  # publish immediately if change > 5°


class ProprioceptionAgent(BaseAgent):
    """Somatosensory Cortex — body schema, proprioception, efference copy.

    Tracks cumulative physical state of the robot body so the brain always
    knows its own pose without needing external sensors.
    """

    name = "proprioception_agent"

    def __init__(self, bus: asyncio.Queue, world_model: "WorldModel | None" = None):
        super().__init__(bus)
        self._world_model = world_model

        # Current pose
        self._pan_deg:      float = 0.0   # current pan from center (+ = right)
        self._tilt_deg:     float = 0.0   # current tilt from center (+ = up)
        self._heading_deg:  float = 0.0   # cumulative wheel heading from start
        self._distance_cm:  float = 0.0   # cumulative wheel distance traveled

        # Pose history: deque of (timestamp, pose_dict)
        self._pose_history: collections.deque = collections.deque(maxlen=20)

        # Track last published pose to detect significant changes
        self._last_published_pan:  float = 0.0
        self._last_published_tilt: float = 0.0
        self._last_publish_ts:     float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        log.info("ProprioceptionAgent: started — tracking body pose")
        asyncio.create_task(self._publish_loop(), name="proprio-publish")
        while self._running:
            await asyncio.sleep(1)

    # ── Message handling ──────────────────────────────────────────────────────

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        mtype = message.type

        if mtype == MessageType.ACTION_MOVE:
            await self._update_pose_from_move(message.data)

        elif mtype == MessageType.TASK_OUTCOME:
            if message.data.get("motor_reset"):
                self._reset_odometry()

        elif mtype == MessageType.SPATIAL_POINT:
            # Gaze snap: normalize (0-1) → degrees and update pan/tilt
            x = message.data.get("x", 0.5)
            y = message.data.get("y", 0.5)
            # (0.5, 0.5) = center = 0°, (0, 0) = top-left = (-45°, +22°)
            target_pan  = (x - 0.5) * 2.0 * _PAN_LIMIT_DEG
            target_tilt = -(y - 0.5) * 2.0 * _TILT_LIMIT_DEG  # invert: up = positive tilt
            self._pan_deg  = max(-_PAN_LIMIT_DEG,  min(_PAN_LIMIT_DEG,  target_pan))
            self._tilt_deg = max(-_TILT_LIMIT_DEG, min(_TILT_LIMIT_DEG, target_tilt))
            await self._publish_if_significant()

        elif mtype == MessageType.PERCEPTION_SENSOR:
            # V10.0 — Ground odometry from ultrasonic distance measurement
            if message.data.get("sensor") == "distance":
                dist_cm = float(message.data.get("distance_cm", 0.0))
                if dist_cm > 0:
                    self._distance_cm = dist_cm  # absolute external measurement wins
                    log.debug("Proprioception: distance corrected by ultrasonic → %.1fcm", dist_cm)
                    await self._update_world_model()

        return None


    # ── Internal ──────────────────────────────────────────────────────────────

    async def _update_pose_from_move(self, data: dict) -> None:
        """Update pose based on motor command data."""
        import math

        # Pan/tilt (absolute target from gaze snap or incremental)
        if "pan" in data:
            self._pan_deg = max(-_PAN_LIMIT_DEG, min(_PAN_LIMIT_DEG, float(data["pan"])))
        if "tilt" in data:
            self._tilt_deg = max(-_TILT_LIMIT_DEG, min(_TILT_LIMIT_DEG, float(data["tilt"])))

        # Pan/tilt incremental delta
        if "pan_delta" in data:
            self._pan_deg = max(-_PAN_LIMIT_DEG,
                                min(_PAN_LIMIT_DEG, self._pan_deg + float(data["pan_delta"])))
        if "tilt_delta" in data:
            self._tilt_deg = max(-_TILT_LIMIT_DEG,
                                 min(_TILT_LIMIT_DEG, self._tilt_deg + float(data["tilt_delta"])))

        # Wheel drive — update odometry estimate
        # speed_left/speed_right: -1.0 to 1.0 normalized
        speed_l = float(data.get("speed_left",  data.get("speed", 0.0)))
        speed_r = float(data.get("speed_right", data.get("speed", 0.0)))
        duration_s = float(data.get("duration_s", 0.1))
        # Approximate: assume max wheel speed = 30cm/s, track width = 20cm
        MAX_SPEED_CM_S = 30.0
        TRACK_WIDTH_CM = 20.0
        v_left  = speed_l * MAX_SPEED_CM_S
        v_right = speed_r * MAX_SPEED_CM_S
        v_avg   = (v_left + v_right) / 2.0
        omega   = (v_right - v_left) / TRACK_WIDTH_CM  # rad/s

        self._distance_cm  += abs(v_avg * duration_s)
        self._heading_deg  += math.degrees(omega * duration_s)
        self._heading_deg  %= 360.0  # normalize to 0-360°

        # Rotation command (degrees, e.g., from WHEELED_ROTATE)
        if "rotate_deg" in data:
            self._heading_deg = (self._heading_deg + float(data["rotate_deg"])) % 360.0

        self._record_pose()
        await self._publish_if_significant()

    def _reset_odometry(self) -> None:
        """Reset wheel odometry (not pan/tilt — those are absolute)."""
        self._heading_deg = 0.0
        self._distance_cm = 0.0
        log.info("ProprioceptionAgent: odometry reset")

    def _record_pose(self) -> None:
        """Record current pose in history."""
        self._pose_history.append({
            "ts":          time.time(),
            "pan_deg":     round(self._pan_deg, 1),
            "tilt_deg":    round(self._tilt_deg, 1),
            "heading_deg": round(self._heading_deg, 1),
            "distance_cm": round(self._distance_cm, 1),
        })

    async def _publish_if_significant(self) -> None:
        """Publish immediately if pan or tilt changed significantly."""
        pan_delta  = abs(self._pan_deg  - self._last_published_pan)
        tilt_delta = abs(self._tilt_deg - self._last_published_tilt)
        if pan_delta >= _SIGNIFICANT_CHANGE_DEG or tilt_delta >= _SIGNIFICANT_CHANGE_DEG:
            await self._publish_pose()

    async def _publish_loop(self) -> None:
        """Publish pose every PUBLISH_INTERVAL_S seconds regardless of changes."""
        while self._running:
            await asyncio.sleep(_PUBLISH_INTERVAL_S)
            await self._publish_pose()

    async def _publish_pose(self) -> None:
        """Publish current body pose and update WorldModel."""
        pose = {
            "pan_deg":     round(self._pan_deg, 1),
            "tilt_deg":    round(self._tilt_deg, 1),
            "heading_deg": round(self._heading_deg, 1),
            "distance_cm": round(self._distance_cm, 1),
        }

        # Update WorldModel directly (fast path — no bus round-trip needed)
        if self._world_model:
            try:
                await self._world_model.update_body_pose(pose)
            except Exception as e:
                log.debug("ProprioceptionAgent: world model update failed: %s", e)

        # Publish to bus for CognitionAgent, PlannerAgent
        await self.publish(AgentMessage(
            type=MessageType.PROPRIOCEPTION_STATE,
            source=self.name,
            data={
                **pose,
                "pose_history": list(self._pose_history)[-5:],
                "ts": time.time(),
            },
            priority=8,  # background — not urgent
        ))

        self._last_published_pan  = self._pan_deg
        self._last_published_tilt = self._tilt_deg
        self._last_publish_ts     = time.time()
        log.debug("ProprioceptionAgent: pose published pan=%.1f° tilt=%.1f° head=%.1f°",
                  self._pan_deg, self._tilt_deg, self._heading_deg)
