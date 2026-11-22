"""Thread-safe adapter for TomTom Traffic Flow Segment Data API.

Provides real-time motor-vehicle traffic flow observations, conservative OSM
road-network geometric matching, in-memory caching with TTL, and strict daily API
request budget protection for AccessRoute.

DISCLAIMER & INTEGRATION SAFEGUARDS:
1. Motor-Vehicle vs Pedestrian Traffic: TomTom Traffic Flow measures motor-vehicle
   speeds on vehicular roadways. It does NOT measure pedestrian sidewalk crowding,
   footpath accessibility, or pedestrian walking speeds. Pure pedestrian paths
   (footways, steps, cycleways) are never queried or matched.
2. Conservative Road Matching: Geometric proximity and road direction/bearing must
   align with the candidate OSMnx edge. If the returned segment is displaced or
   directionally mismatched, the observation is rejected and logged.
3. Transparent Fallback: If TomTom is unconfigured, disabled, timed out, rate-limited,
   low-confidence, or out of daily budget, the system silently cascades to the
   existing simulated Electronic City dataset or physical distance baseline.
4. Road Closures: A validated road closure is preserved as a distinct closure condition
   and not silently mapped to normal speed-based congestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging
import math
import os
import threading
import time
from typing import Any, Sequence

import requests
from requests import RequestException, Timeout

logger = logging.getLogger(__name__)

# Standard motor-vehicle OSM highway classifications suitable for traffic matching
MOTOR_VEHICLE_HIGHWAYS: frozenset[str] = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
        "service",
    }
)

# Non-motorized pedestrian infrastructure that must NEVER be queried as motor roads
EXCLUDED_PEDESTRIAN_HIGHWAYS: frozenset[str] = frozenset(
    {
        "footway",
        "steps",
        "pedestrian",
        "path",
        "cycleway",
        "bridleway",
        "track",
        "corridor",
        "elevator",
        "escalator",
        "platform",
    }
)

DEFAULT_TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


@dataclass(frozen=True)
class TomTomObservation:
    """Validated point-in-time traffic observation for a matched road segment."""

    current_speed_kmh: float
    free_flow_speed_kmh: float
    current_travel_time: float
    free_flow_travel_time: float
    confidence: float
    road_closure: bool
    frc: str
    congestion_index: float  # Normalized in [0.0, 1.0]
    coordinates: tuple[tuple[float, float], ...]  # ((lat, lon), ...)
    fetched_at: float  # Monotonic seconds or unix timestamp


@dataclass(frozen=True)
class TomTomMatchResult:
    """Outcome of attempting to match TomTom traffic data to an OSM edge."""

    matched: bool
    observation: TomTomObservation | None = None
    rejection_reason: str | None = None
    distance_meters: float | None = None
    bearing_diff_degrees: float | None = None


def haversine_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two coordinates in meters."""
    radius_earth_meters = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return radius_earth_meters * c


def distance_point_to_polyline_meters(
    point: tuple[float, float], polyline: Sequence[tuple[float, float]]
) -> float:
    """Calculate the minimum perpendicular distance in meters from a point to a polyline."""
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return haversine_distance_meters(
            point[0], point[1], polyline[0][0], polyline[0][1]
        )

    p_lat, p_lon = point
    min_dist = float("inf")
    meters_per_deg_lat = 111320.0

    for i in range(len(polyline) - 1):
        a_lat, a_lon = polyline[i]
        b_lat, b_lon = polyline[i + 1]

        cos_lat = math.cos(math.radians((a_lat + b_lat) / 2.0))
        meters_per_deg_lon = meters_per_deg_lat * cos_lat

        dx = (b_lon - a_lon) * meters_per_deg_lon
        dy = (b_lat - a_lat) * meters_per_deg_lat
        px = (p_lon - a_lon) * meters_per_deg_lon
        py = (p_lat - a_lat) * meters_per_deg_lat

        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-9:
            dist = math.hypot(px, py)
        else:
            t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
            proj_x = t * dx
            proj_y = t * dy
            dist = math.hypot(px - proj_x, py - proj_y)

        if dist < min_dist:
            min_dist = dist

    return min_dist


def calculate_bearing_degrees(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the forward initial bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
        phi2
    ) * math.cos(delta_lambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing


def bearing_difference_degrees(b1: float, b2: float) -> float:
    """Calculate acute angular difference between two bearings in degrees [0, 180]."""
    diff = abs(b1 - b2) % 360.0
    return min(diff, 360.0 - diff)


def is_suitable_motor_road(edge_data: dict[str, Any]) -> bool:
    """Return True if an OSM edge represents a vehicular road suitable for traffic monitoring."""
    if edge_data.get("is_steps") or str(edge_data.get("stairs", "")).lower() == "yes":
        return False

    raw_highway = edge_data.get("highway")
    if raw_highway is None:
        return False

    highways = raw_highway if isinstance(raw_highway, list) else [raw_highway]
    for h in highways:
        h_str = str(h).strip().lower()
        if h_str in EXCLUDED_PEDESTRIAN_HIGHWAYS:
            return False
        if "footway" in h_str or "pedestrian" in h_str or "path" in h_str:
            return False

    return any(str(h).strip().lower() in MOTOR_VEHICLE_HIGHWAYS for h in highways)


def calculate_normalized_congestion(
    current_speed: float, free_flow_speed: float
) -> float:
    """Calculate normalized traffic congestion index in range [0.0, 1.0].

    Formula: clamp(1.0 - (current_speed / free_flow_speed), 0.0, 1.0)
    Handles edge cases: non-finite values, zero/negative free_flow_speed, current_speed >= free_flow.
    """
    if not (math.isfinite(current_speed) and math.isfinite(free_flow_speed)):
        return 0.0
    if free_flow_speed <= 0.0 or current_speed < 0.0:
        return 0.0
    raw_congestion = 1.0 - (current_speed / free_flow_speed)
    return max(0.0, min(1.0, raw_congestion))


class TomTomTrafficAdapter:
    """Thread-safe adapter for querying and caching TomTom Flow Segment Data API.

    Features:
    - In-memory cache with configurable TTL (default: 900s / 15 min).
    - Daily request counter with configurable hard limit (default: 200).
    - Request timeout (default: 5.0s).
    - Conservative OSM edge matching: proximity and bearing validation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        enabled: bool | None = None,
        ttl_seconds: float | None = None,
        max_daily_requests: int | None = None,
        min_confidence: float | None = None,
        timeout_seconds: float | None = None,
        match_max_distance_meters: float | None = None,
        match_max_bearing_diff_degrees: float | None = None,
        base_url: str = DEFAULT_TOMTOM_BASE_URL,
    ) -> None:
        raw_key = api_key if api_key is not None else os.environ.get("TOMTOM_API_KEY", "")
        self._api_key = str(raw_key).strip()

        # Check if key is a template placeholder (e.g. paste_your_actual_tomtom_key_here)
        is_placeholder = (
            not self._api_key
            or "paste_your" in self._api_key.lower()
            or "your_tomtom" in self._api_key.lower()
        )

        if enabled is not None:
            self._enabled = bool(enabled) and bool(self._api_key)
        else:
            raw_enabled = os.environ.get("TOMTOM_TRAFFIC_ENABLED", "true").strip().lower()
            self._enabled = raw_enabled in ("true", "1", "yes") and not is_placeholder

        self._ttl_seconds = float(
            ttl_seconds
            if ttl_seconds is not None
            else os.environ.get("TOMTOM_TRAFFIC_TTL_SECONDS", "900")
        )
        self._max_daily_requests = int(
            max_daily_requests
            if max_daily_requests is not None
            else os.environ.get("TOMTOM_MAX_DAILY_REQUESTS", "200")
        )
        self._min_confidence = float(
            min_confidence
            if min_confidence is not None
            else os.environ.get("TOMTOM_MIN_CONFIDENCE", "0.70")
        )
        self._timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get("TOMTOM_REQUEST_TIMEOUT_SECONDS", "5")
        )
        self._match_max_distance_meters = float(
            match_max_distance_meters
            if match_max_distance_meters is not None
            else os.environ.get("TOMTOM_MATCH_MAX_DISTANCE_METERS", "35.0")
        )
        self._match_max_bearing_diff_degrees = float(
            match_max_bearing_diff_degrees
            if match_max_bearing_diff_degrees is not None
            else os.environ.get("TOMTOM_MATCH_MAX_BEARING_DIFF_DEGREES", "50.0")
        )
        self._base_url = base_url

        self._lock = threading.RLock()
        # Cache mapping quantized coordinate (round(lat, 4), round(lon, 4)) -> (TomTomObservation, monotonic_time)
        self._cache: dict[tuple[float, float], tuple[TomTomObservation, float]] = {}
        # Daily request tracker
        self._daily_requests_date: date = datetime.now(timezone.utc).date()
        self._daily_requests_count: int = 0
        self._total_requests_count: int = 0
        self._cache_hits: int = 0

    @property
    def is_available(self) -> bool:
        """Return True if TomTom integration is enabled and configured with an API key."""
        return self._enabled and bool(self._api_key)

    @property
    def daily_requests_count(self) -> int:
        """Number of TomTom API requests made today in UTC."""
        with self._lock:
            self._check_and_reset_daily_counter()
            return self._daily_requests_count

    @property
    def max_daily_requests(self) -> int:
        """Internal daily request budget ceiling."""
        return self._max_daily_requests

    @property
    def cache_size(self) -> int:
        """Number of active entries currently held in memory cache."""
        with self._lock:
            return len(self._cache)

    @property
    def cache_hits(self) -> int:
        """Number of cache hits during the adapter's lifecycle."""
        with self._lock:
            return self._cache_hits

    def clear_cache(self) -> None:
        """Evict all entries from the in-memory cache."""
        with self._lock:
            self._cache.clear()

    def reset_daily_counter(self) -> None:
        """Reset the daily request counter (primarily for testing)."""
        with self._lock:
            self._daily_requests_count = 0
            self._daily_requests_date = datetime.now(timezone.utc).date()

    def _check_and_reset_daily_counter(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._daily_requests_date:
            self._daily_requests_date = today
            self._daily_requests_count = 0

    def get_edge_observation(
        self,
        edge_data: dict[str, Any],
        source_coord: tuple[float, float],
        target_coord: tuple[float, float],
    ) -> TomTomMatchResult:
        """Attempt to match a TomTom traffic flow observation to a candidate OSM edge.

        Args:
            edge_data: Dictionary of OSM edge attributes (length, highway, oneway, geometry).
            source_coord: (latitude, longitude) of edge start node u.
            target_coord: (latitude, longitude) of edge end node v.

        Returns:
            TomTomMatchResult with match outcome, observation, and rejection reason if rejected.
        """
        # Step 1: Check suitable road class
        if not is_suitable_motor_road(edge_data):
            return TomTomMatchResult(
                matched=False,
                rejection_reason="non_motor_road_class",
            )

        # Step 2: Check provider availability
        if not self.is_available:
            return TomTomMatchResult(
                matched=False,
                rejection_reason="provider_disabled_or_no_key",
            )

        # Step 3: Compute OSM edge midpoint and bearing
        src_lat, src_lon = source_coord
        tgt_lat, tgt_lon = target_coord
        mid_lat = (src_lat + tgt_lat) / 2.0
        mid_lon = (src_lon + tgt_lon) / 2.0

        # If detailed geometry exists, use its midpoint
        geometry = edge_data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords") and len(geometry.coords) > 1:
            coords = list(geometry.coords)
            # GeoJSON coordinates are (lon, lat)
            mid_idx = len(coords) // 2
            mid_lon, mid_lat = coords[mid_idx]

        osm_bearing = calculate_bearing_degrees(src_lat, src_lon, tgt_lat, tgt_lon)

        # Step 4: Check in-memory cache using quantized coordinates
        cache_key = (round(mid_lat, 4), round(mid_lon, 4))
        obs = self._get_from_cache(cache_key)

        if obs is None:
            # Query TomTom API
            obs, query_rejection = self._query_api(mid_lat, mid_lon)
            if obs is None:
                return TomTomMatchResult(
                    matched=False,
                    rejection_reason=query_rejection or "query_failed",
                )
            self._put_in_cache(cache_key, obs)

        # Step 5: Geometric proximity matching against returned TomTom polyline
        dist = distance_point_to_polyline_meters((mid_lat, mid_lon), obs.coordinates)
        if dist > self._match_max_distance_meters:
            return TomTomMatchResult(
                matched=False,
                rejection_reason="distance_exceeded",
                distance_meters=round(dist, 1),
            )

        # Step 6: Road direction and bearing comparison
        if len(obs.coordinates) >= 2:
            tt_p1 = obs.coordinates[0]
            tt_p2 = obs.coordinates[-1]
            tt_bearing = calculate_bearing_degrees(
                tt_p1[0], tt_p1[1], tt_p2[0], tt_p2[1]
            )
            bearing_diff = bearing_difference_degrees(osm_bearing, tt_bearing)

            is_oneway = bool(edge_data.get("oneway") in (True, "yes", "1", 1))
            if is_oneway:
                if bearing_diff > self._match_max_bearing_diff_degrees:
                    return TomTomMatchResult(
                        matched=False,
                        rejection_reason="direction_mismatch_oneway",
                        distance_meters=round(dist, 1),
                        bearing_diff_degrees=round(bearing_diff, 1),
                    )
                effective_bearing_diff = bearing_diff
            else:
                # On two-way roads, traffic flows bidirectionally: evaluate axial alignment
                axis_diff = min(bearing_diff, 180.0 - bearing_diff)
                if axis_diff > self._match_max_bearing_diff_degrees:
                    return TomTomMatchResult(
                        matched=False,
                        rejection_reason="bearing_mismatch",
                        distance_meters=round(dist, 1),
                        bearing_diff_degrees=round(axis_diff, 1),
                    )
                effective_bearing_diff = axis_diff
        else:
            effective_bearing_diff = None

        return TomTomMatchResult(
            matched=True,
            observation=obs,
            distance_meters=round(dist, 1),
            bearing_diff_degrees=round(effective_bearing_diff, 1) if effective_bearing_diff is not None else None,
        )

    def _get_from_cache(self, key: tuple[float, float]) -> TomTomObservation | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            obs, cached_time = entry
            if time.monotonic() - cached_time <= self._ttl_seconds:
                self._cache_hits += 1
                return obs
            # Expired
            self._cache.pop(key, None)
            return None

    def _put_in_cache(
        self, key: tuple[float, float], obs: TomTomObservation
    ) -> None:
        with self._lock:
            # Bound cache size to prevent unbounded memory growth
            if len(self._cache) > 2000:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)
            self._cache[key] = (obs, time.monotonic())

    def _query_api(
        self, latitude: float, longitude: float
    ) -> tuple[TomTomObservation | None, str | None]:
        with self._lock:
            self._check_and_reset_daily_counter()
            if self._daily_requests_count >= self._max_daily_requests:
                logger.warning(
                    "TomTom daily request limit reached (%d/%d). Falling back.",
                    self._daily_requests_count,
                    self._max_daily_requests,
                )
                return None, "daily_budget_reached"

            self._daily_requests_count += 1
            self._total_requests_count += 1

        url = self._base_url
        params = {
            "key": self._api_key,
            "point": f"{latitude},{longitude}",
            "unit": "kmph",
        }

        try:
            response = requests.get(url, params=params, timeout=self._timeout_seconds)
        except Timeout:
            logger.warning("TomTom Traffic API request timed out for (%f, %f)", latitude, longitude)
            return None, "timeout"
        except RequestException as exc:
            logger.warning("TomTom Traffic API network error: %s", exc)
            return None, "network_error"

        if response.status_code == 429:
            logger.warning("TomTom Traffic API HTTP 429 Too Many Requests.")
            return None, "http_429_rate_limited"
        if response.status_code != 200:
            logger.warning(
                "TomTom Traffic API returned status %d: %s",
                response.status_code,
                response.text[:120],
            )
            return None, f"api_error_{response.status_code}"

        try:
            payload = response.json()
        except ValueError:
            return None, "malformed_json_response"

        flow_data = payload.get("flowSegmentData")
        if not isinstance(flow_data, dict):
            return None, "missing_flow_segment_data"

        # Check required fields
        required_fields = ["currentSpeed", "freeFlowSpeed", "confidence", "coordinates"]
        missing = [f for f in required_fields if f not in flow_data or flow_data[f] is None]
        if missing:
            return None, f"missing_fields_{'_'.join(missing)}"

        # Validate numeric metrics
        try:
            curr_spd = float(flow_data.get("currentSpeed", 0.0))
            free_spd = float(flow_data.get("freeFlowSpeed", 0.0))
            curr_time = float(flow_data.get("currentTravelTime", 0.0))
            free_time = float(flow_data.get("freeFlowTravelTime", 0.0))
            conf = float(flow_data.get("confidence", 0.0))
            road_closed = bool(flow_data.get("roadClosure", False))
            frc_class = str(flow_data.get("frc", "UNKNOWN"))
        except (ValueError, TypeError):
            return None, "invalid_numeric_values"

        if not (math.isfinite(curr_spd) and math.isfinite(free_spd) and math.isfinite(conf)):
            return None, "non_finite_values"

        if conf < self._min_confidence:
            return None, "low_confidence"

        # Extract returned segment coordinates
        coords_dict = flow_data.get("coordinates", {})
        raw_coords = coords_dict.get("coordinate", []) if isinstance(coords_dict, dict) else []
        parsed_coords: list[tuple[float, float]] = []
        for item in raw_coords:
            if isinstance(item, dict) and "latitude" in item and "longitude" in item:
                try:
                    c_lat = float(item["latitude"])
                    c_lon = float(item["longitude"])
                    if math.isfinite(c_lat) and math.isfinite(c_lon):
                        parsed_coords.append((c_lat, c_lon))
                except (ValueError, TypeError):
                    continue

        if not parsed_coords:
            return None, "missing_road_geometry"

        congestion = calculate_normalized_congestion(curr_spd, free_spd)

        obs = TomTomObservation(
            current_speed_kmh=curr_spd,
            free_flow_speed_kmh=free_spd,
            current_travel_time=curr_time,
            free_flow_travel_time=free_time,
            confidence=conf,
            road_closure=road_closed,
            frc=frc_class,
            congestion_index=congestion,
            coordinates=tuple(parsed_coords),
            fetched_at=time.time(),
        )
        return obs, None


# Shared application-wide default TomTom adapter instance
_DEFAULT_TOMTOM_ADAPTER: TomTomTrafficAdapter | None = None
_TOMTOM_INIT_LOCK = threading.Lock()


def get_default_tomtom_adapter() -> TomTomTrafficAdapter:
    """Retrieve the shared application-wide TomTomTrafficAdapter instance."""
    global _DEFAULT_TOMTOM_ADAPTER
    if _DEFAULT_TOMTOM_ADAPTER is None:
        with _TOMTOM_INIT_LOCK:
            if _DEFAULT_TOMTOM_ADAPTER is None:
                _DEFAULT_TOMTOM_ADAPTER = TomTomTrafficAdapter()
    return _DEFAULT_TOMTOM_ADAPTER


def set_default_tomtom_adapter(adapter: TomTomTrafficAdapter) -> None:
    """Override the default TomTom adapter instance (primarily for testing)."""
    global _DEFAULT_TOMTOM_ADAPTER
    with _TOMTOM_INIT_LOCK:
        _DEFAULT_TOMTOM_ADAPTER = adapter
