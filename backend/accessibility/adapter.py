"""Thread-safe in-memory adapter and registry for pedestrian accessibility obstacles.

Enables verified pedestrian pathway obstacles to influence the routing engine,
penalizing blocked or inaccessible road segments in traffic-and-accessibility-aware
route calculations.

DISCLAIMER & LIMITATIONS:
1. Obstacle observations apply only to specific directed OpenStreetMap edges (osm_u, osm_v, key).
2. Expired, unverified, or low-confidence observations are excluded from route calculations.
3. Unmonitored road segments have unknown accessibility status and are NOT assumed accessible.
4. In-Memory Registry Scope: Thread-safe within a single process. Multi-worker deployments require an external datastore.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import threading
from typing import Sequence

from backend.accessibility.contracts import AccessibilityObstacleObservation
from backend.traffic.loader import TrafficEdgeKey


class AccessibilityAdapter:
    """Thread-safe in-memory registry for pedestrian accessibility obstacle observations."""

    def __init__(
        self,
        default_ttl_seconds: float = 86400.0,
        max_future_skew_seconds: float = 5.0,
        min_confidence: float = 0.0,
    ) -> None:
        if default_ttl_seconds < 60.0:
            raise ValueError("default_ttl_seconds must be at least 60.0 seconds.")
        if max_future_skew_seconds < 0.0:
            raise ValueError("max_future_skew_seconds must be non-negative.")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0.")

        self._default_ttl_seconds = float(default_ttl_seconds)
        self._max_future_skew_seconds = float(max_future_skew_seconds)
        self._min_confidence = float(min_confidence)
        self._lock = threading.RLock()
        self._registry: dict[TrafficEdgeKey, AccessibilityObstacleObservation] = {}

    def _normalize_edge_key(self, edge_key: TrafficEdgeKey | Sequence[int]) -> TrafficEdgeKey:
        if isinstance(edge_key, TrafficEdgeKey):
            return edge_key
        if isinstance(edge_key, (list, tuple)):
            if len(edge_key) == 2:
                return TrafficEdgeKey(osm_u=int(edge_key[0]), osm_v=int(edge_key[1]), key=0)
            if len(edge_key) == 3:
                return TrafficEdgeKey(
                    osm_u=int(edge_key[0]), osm_v=int(edge_key[1]), key=int(edge_key[2])
                )
        raise TypeError(
            f"Expected TrafficEdgeKey or 2/3-tuple (osm_u, osm_v[, key]), got {type(edge_key).__name__}."
        )

    def register_obstacle(self, observation: AccessibilityObstacleObservation) -> TrafficEdgeKey:
        """Register a validated obstacle observation for a directed edge."""
        if not isinstance(observation, AccessibilityObstacleObservation):
            raise TypeError(
                f"Expected AccessibilityObstacleObservation, got {type(observation).__name__}."
            )

        edge_key = TrafficEdgeKey(
            osm_u=observation.osm_u,
            osm_v=observation.osm_v,
            key=observation.osm_key,
        )

        with self._lock:
            self._registry[edge_key] = observation
            return edge_key

    def get_valid_obstacle(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        current_time: datetime | None = None,
    ) -> AccessibilityObstacleObservation | None:
        """Retrieve obstacle for edge only if verified and not expired."""
        key = self._normalize_edge_key(edge_key)
        eval_time = current_time or datetime.now(timezone.utc)
        if eval_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware.")

        with self._lock:
            obs = self._registry.get(key)
            if obs is None or not obs.is_verified:
                return None
            if obs.confidence is not None and obs.confidence < self._min_confidence:
                return None

            age = (eval_time - obs.reported_at_utc).total_seconds()
            if age < -self._max_future_skew_seconds:
                return None  # Future dated
            if age > obs.time_to_live_seconds:
                return None  # Stale

            return obs

    def get_obstacle_penalty(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        current_time: datetime | None = None,
    ) -> float | None:
        """Return penalty multiplier if a valid obstacle is present on edge."""
        obs = self.get_valid_obstacle(edge_key, current_time)
        return obs.penalty_multiplier if obs is not None else None

    def has_obstacle(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        current_time: datetime | None = None,
    ) -> bool:
        """Check whether edge has an active verified obstacle."""
        return self.get_valid_obstacle(edge_key, current_time) is not None

    def get_active_obstacles(
        self,
        current_time: datetime | None = None,
    ) -> list[AccessibilityObstacleObservation]:
        """Return all currently active and unexpired obstacles."""
        eval_time = current_time or datetime.now(timezone.utc)
        with self._lock:
            active = []
            for obs in self._registry.values():
                if not obs.is_verified:
                    continue
                if obs.confidence is not None and obs.confidence < self._min_confidence:
                    continue
                age = (eval_time - obs.reported_at_utc).total_seconds()
                if -self._max_future_skew_seconds <= age <= obs.time_to_live_seconds:
                    active.append(obs)
            return active

    def clear(self) -> None:
        """Clear all registered obstacles."""
        with self._lock:
            self._registry.clear()


_DEFAULT_ACCESSIBILITY_ADAPTER: AccessibilityAdapter | None = None
_ADAPTER_INIT_LOCK = threading.Lock()


def get_default_accessibility_adapter() -> AccessibilityAdapter:
    """Retrieve shared default AccessibilityAdapter instance."""
    global _DEFAULT_ACCESSIBILITY_ADAPTER
    if _DEFAULT_ACCESSIBILITY_ADAPTER is None:
        with _ADAPTER_INIT_LOCK:
            if _DEFAULT_ACCESSIBILITY_ADAPTER is None:
                _DEFAULT_ACCESSIBILITY_ADAPTER = AccessibilityAdapter()
    return _DEFAULT_ACCESSIBILITY_ADAPTER
