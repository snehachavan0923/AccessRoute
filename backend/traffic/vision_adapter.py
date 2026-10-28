"""Thread-safe in-memory adapter and registry for computer-vision traffic observations.

Bridges validated VisionEdgeObservation payloads with the AccessRoute traffic and
routing subsystems using existing TrafficEdgeKey abstractions.

DISCLAIMER AND LIMITATIONS:
1. Spatial Calibration: An observation must originate from a certified, spatially
   calibrated camera (homography matrix H verified) to be returned as usable traffic data.
2. Temporal Validity: Traffic is dynamic. Observations older than their Time-To-Live
   (TTL, default 15 minutes) are expired and must not influence routing.
3. No Uncalibrated Congestion Formulation: The adapter does NOT infer or compute
   congestion from raw vehicle detection counts or tracker IDs. If an observation
   lacks an explicit, valid experimental congestion indicator, no vision congestion
   value is returned.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
import threading
from typing import Any, Sequence

from backend.traffic.loader import TrafficEdgeKey
from backend.vision.schemas import VisionEdgeObservation


class ObservationValidationStatus(str, Enum):
    """Detailed validation status explaining why an observation is accepted or rejected."""

    VALID = "valid"
    NOT_FOUND = "not_found"
    UNCALIBRATED = "uncalibrated"
    LOW_QUALITY = "low_quality"
    EXPIRED = "expired"
    FUTURE_SKEW = "future_skew"
    NO_CONGESTION_INDICATOR = "no_congestion_indicator"


class VisionTrafficAdapter:
    """Thread-safe in-memory registry for computer-vision traffic edge observations.

    Stores validated VisionEdgeObservation payloads keyed by directed TrafficEdgeKey
    (osm_u, osm_v, key) and enforces strict validation filters against spatial
    uncalibration, temporal staleness, future clock skew, and low quality.
    """

    def __init__(
        self,
        default_ttl_seconds: float = 900.0,
        min_quality_score: float = 0.70,
        require_spatial_calibration: bool = True,
        max_future_skew_seconds: float = 5.0,
    ) -> None:
        """Initialize the VisionTrafficAdapter with configurable safeguard thresholds.

        Args:
            default_ttl_seconds: Fallback time-to-live in seconds if an observation
                does not specify its own TTL (default: 900.0s / 15 minutes). Must be >= 60.0.
            min_quality_score: Minimum observation quality threshold in [0.0, 1.0]
                (default: 0.70). Observations below this score are rejected.
            require_spatial_calibration: If True, observations must have certified
                spatial calibration (camera_metadata.is_spatially_calibrated is True)
                to be returned as usable traffic data (default: True).
            max_future_skew_seconds: Maximum tolerable clock skew in seconds for
                observations timestamped in the future (default: 5.0s).

        Raises:
            ValueError: If thresholds are outside valid numeric bounds.
        """
        if default_ttl_seconds < 60.0:
            raise ValueError(
                f"default_ttl_seconds must be at least 60.0 seconds, got {default_ttl_seconds}."
            )
        if not (0.0 <= min_quality_score <= 1.0):
            raise ValueError(
                f"min_quality_score must be between 0.0 and 1.0, got {min_quality_score}."
            )
        if max_future_skew_seconds < 0.0:
            raise ValueError(
                f"max_future_skew_seconds must be non-negative, got {max_future_skew_seconds}."
            )

        self._default_ttl_seconds = float(default_ttl_seconds)
        self._min_quality_score = float(min_quality_score)
        self._require_spatial_calibration = bool(require_spatial_calibration)
        self._max_future_skew_seconds = float(max_future_skew_seconds)

        self._lock = threading.RLock()
        self._registry: dict[TrafficEdgeKey, VisionEdgeObservation] = {}

    @property
    def default_ttl_seconds(self) -> float:
        """Configured default time-to-live in seconds."""
        return self._default_ttl_seconds

    @property
    def min_quality_score(self) -> float:
        """Configured minimum quality score threshold."""
        return self._min_quality_score

    @property
    def require_spatial_calibration(self) -> bool:
        """Whether spatial calibration is required for usable observations."""
        return self._require_spatial_calibration

    @property
    def max_future_skew_seconds(self) -> float:
        """Maximum tolerable future clock skew in seconds."""
        return self._max_future_skew_seconds

    def set_observation(
        self, observation: VisionEdgeObservation
    ) -> TrafficEdgeKey:
        """Register or update an observation for a directed road edge.

        Extracts the directed TrafficEdgeKey(osm_u, osm_v, key) directly from the
        observation and stores it in the thread-safe registry.

        Args:
            observation: Validated VisionEdgeObservation instance.

        Returns:
            TrafficEdgeKey under which the observation was registered.

        Raises:
            TypeError: If observation is not a VisionEdgeObservation instance.
        """
        if not isinstance(observation, VisionEdgeObservation):
            raise TypeError(
                f"Expected VisionEdgeObservation, got {type(observation).__name__}."
            )

        edge_key = TrafficEdgeKey(
            osm_u=observation.osm_u,
            osm_v=observation.osm_v,
            key=observation.osm_key,
        )

        with self._lock:
            self._registry[edge_key] = observation

        return edge_key

    def ingest(self, observation: VisionEdgeObservation) -> TrafficEdgeKey:
        """Convenience alias for set_observation."""
        return self.set_observation(observation)

    @staticmethod
    def _normalize_edge_key(edge_key: TrafficEdgeKey | Sequence[int]) -> TrafficEdgeKey:
        """Normalize a TrafficEdgeKey or integer sequence (u, v) / (u, v, key) to TrafficEdgeKey."""
        if isinstance(edge_key, TrafficEdgeKey):
            return edge_key
        if isinstance(edge_key, (tuple, list)):
            return TrafficEdgeKey.from_tuple(edge_key)
        raise TypeError(
            f"Expected TrafficEdgeKey or tuple of ints, got {type(edge_key).__name__}."
        )

    def get_raw_observation(
        self, edge_key: TrafficEdgeKey | Sequence[int]
    ) -> VisionEdgeObservation | None:
        """Retrieve the raw stored observation for an edge without applying filtering.

        Args:
            edge_key: Directed TrafficEdgeKey or tuple (osm_u, osm_v) / (osm_u, osm_v, key).

        Returns:
            The raw stored VisionEdgeObservation if present, or None.
        """
        key = self._normalize_edge_key(edge_key)
        with self._lock:
            return self._registry.get(key)

    def _evaluate_observation_under_lock(
        self,
        edge_key: TrafficEdgeKey,
        eval_time: datetime,
        require_congestion_index: bool = False,
    ) -> ObservationValidationStatus:
        """Internal helper to evaluate observation status while caller holds self._lock."""
        obs = self._registry.get(edge_key)
        if obs is None:
            return ObservationValidationStatus.NOT_FOUND

        # 1. Spatial calibration check
        if self._require_spatial_calibration:
            meta = obs.camera_metadata
            if meta is None or not meta.is_spatially_calibrated:
                return ObservationValidationStatus.UNCALIBRATED

        # 2. Quality score threshold check
        if obs.observation_quality_score < self._min_quality_score:
            return ObservationValidationStatus.LOW_QUALITY

        # 3. Temporal freshness / TTL check
        # Measure age from observation end_time (the latest instant of the sampling window)
        age_seconds = (eval_time - obs.timing.end_time).total_seconds()

        if age_seconds < -self._max_future_skew_seconds:
            return ObservationValidationStatus.FUTURE_SKEW

        effective_ttl = obs.timing.time_to_live_seconds or self._default_ttl_seconds
        if age_seconds > effective_ttl:
            return ObservationValidationStatus.EXPIRED

        # 4. Optional check for presence of a valid congestion indicator
        if require_congestion_index:
            raw_index = obs.experimental_congestion_index
            if raw_index is None or isinstance(raw_index, bool):
                return ObservationValidationStatus.NO_CONGESTION_INDICATOR
            if not (isinstance(raw_index, (int, float)) and math.isfinite(raw_index) and 0.0 <= float(raw_index) <= 1.0):
                return ObservationValidationStatus.NO_CONGESTION_INDICATOR

        return ObservationValidationStatus.VALID

    def evaluate_observation_status(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        current_time: datetime | None = None,
        require_congestion_index: bool = False,
    ) -> ObservationValidationStatus:
        """Evaluate the detailed validation status of an edge observation.

        Args:
            edge_key: Directed TrafficEdgeKey or tuple (osm_u, osm_v) / (osm_u, osm_v, key).
            current_time: Optional reference datetime for expiration evaluation.
                Must be timezone-aware if provided. Defaults to datetime.now(timezone.utc).
            require_congestion_index: If True, also checks that experimental_congestion_index
                is present and within [0.0, 1.0]. Returns NO_CONGESTION_INDICATOR if missing.

        Returns:
            ObservationValidationStatus indicating whether the observation is valid
            or the specific reason it was rejected.

        Raises:
            ValueError: If current_time is provided but is timezone-naive.
        """
        key = self._normalize_edge_key(edge_key)
        eval_time = self._resolve_eval_time(current_time)

        with self._lock:
            return self._evaluate_observation_under_lock(
                key, eval_time, require_congestion_index=require_congestion_index
            )

    def get_valid_observation(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        current_time: datetime | None = None,
    ) -> VisionEdgeObservation | None:
        """Retrieve a stored observation only if it passes all safety and freshness filters.

        Filters applied:
        1. Spatial Calibration: Requires certified camera homography when enabled.
        2. Observation Quality: observation_quality_score >= min_quality_score.
        3. Freshness / TTL: Elapsed age from observation end_time <= effective TTL.
        4. Clock Skew: Rejects observations timestamped too far in the future.

        Retrieval is atomic: evaluation and access occur within a single lock acquisition.

        Args:
            edge_key: Directed TrafficEdgeKey or tuple (osm_u, osm_v) / (osm_u, osm_v, key).
            current_time: Optional reference datetime (must be timezone-aware).
                Defaults to datetime.now(timezone.utc).

        Returns:
            VisionEdgeObservation if all validation checks pass, otherwise None.

        Raises:
            ValueError: If current_time is timezone-naive.
        """
        key = self._normalize_edge_key(edge_key)
        eval_time = self._resolve_eval_time(current_time)
        with self._lock:
            status = self._evaluate_observation_under_lock(key, eval_time)
            if status == ObservationValidationStatus.VALID:
                return self._registry.get(key)
            return None

    def get_congestion_index(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        current_time: datetime | None = None,
    ) -> float | None:
        """Query the usable vision congestion index for a directed edge.

        CRITICAL SAFEGUARD:
        Does NOT invent an uncalibrated congestion formula from raw vehicle counts
        or tracker IDs. If the observation is expired, uncalibrated, low quality,
        or lacks an explicit experimental_congestion_index in [0.0, 1.0], this
        method returns None, allowing the caller to fall back safely to baseline.

        Args:
            edge_key: Directed TrafficEdgeKey or tuple (osm_u, osm_v) / (osm_u, osm_v, key).
            current_time: Optional reference datetime (must be timezone-aware).

        Returns:
            Float congestion index in [0.0, 1.0] if valid and present, otherwise None.
        """
        obs = self.get_valid_observation(edge_key, current_time=current_time)
        if obs is None:
            return None

        raw_index = obs.experimental_congestion_index
        if raw_index is None or isinstance(raw_index, bool):
            return None

        if isinstance(raw_index, (int, float)) and math.isfinite(raw_index):
            val = float(raw_index)
            if 0.0 <= val <= 1.0:
                return val

        return None

    def has_edge(self, edge_key: TrafficEdgeKey | Sequence[int]) -> bool:
        """Check if an observation is registered for the specified edge (regardless of freshness)."""
        key = self._normalize_edge_key(edge_key)
        with self._lock:
            return key in self._registry

    def remove_observation(self, edge_key: TrafficEdgeKey | Sequence[int]) -> bool:
        """Remove an observation from the registry.

        Returns:
            True if an observation was removed, False if not found.
        """
        key = self._normalize_edge_key(edge_key)
        with self._lock:
            return self._registry.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all observations from the registry."""
        with self._lock:
            self._registry.clear()

    def purge_expired(self, current_time: datetime | None = None) -> int:
        """Remove all expired observations from the registry.

        Args:
            current_time: Optional reference datetime for expiration evaluation.

        Returns:
            Count of observations purged.
        """
        eval_time = self._resolve_eval_time(current_time)
        purged_count = 0

        with self._lock:
            expired_keys: list[TrafficEdgeKey] = []
            for edge_key, obs in self._registry.items():
                age_seconds = (eval_time - obs.timing.end_time).total_seconds()
                effective_ttl = obs.timing.time_to_live_seconds or self._default_ttl_seconds
                if age_seconds > effective_ttl:
                    expired_keys.append(edge_key)

            for key in expired_keys:
                del self._registry[key]
                purged_count += 1

        return purged_count

    def get_all_edge_keys(self) -> list[TrafficEdgeKey]:
        """Return a snapshot list of all currently registered edge keys."""
        with self._lock:
            return list(self._registry.keys())

    def __len__(self) -> int:
        """Return total count of stored observations."""
        with self._lock:
            return len(self._registry)

    @staticmethod
    def _resolve_eval_time(eval_time: datetime | None) -> datetime:
        """Resolve and validate reference datetime, enforcing timezone awareness."""
        if eval_time is None:
            return datetime.now(timezone.utc)
        if eval_time.tzinfo is None or eval_time.tzinfo.utcoffset(eval_time) is None:
            raise ValueError(
                "current_time must be timezone-aware (e.g., UTC). "
                "Naive datetime objects are rejected to prevent ambiguous comparisons."
            )
        return eval_time
