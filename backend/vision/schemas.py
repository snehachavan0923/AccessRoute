"""Data models and schemas for video-based vehicle and object detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixels or normalized coordinates."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(ge=0.0, description="Left coordinate")
    y_min: float = Field(ge=0.0, description="Top coordinate")
    x_max: float = Field(ge=0.0, description="Right coordinate")
    y_max: float = Field(ge=0.0, description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        """Require strictly positive width and height."""
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be strictly greater than x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("y_max must be strictly greater than y_min.")

        return self


class DetectedObject(BaseModel):
    """A single object detected in a video frame."""

    model_config = ConfigDict(frozen=True)

    class_name: str = Field(
        min_length=1,
        max_length=50,
        description="Detected category, such as car, bus, truck, or motorcycle",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Detection confidence between 0 and 1",
    )

    bounding_box: BoundingBox

    track_id: int | None = Field(
        default=None,
        ge=0,
        description="Optional tracking identifier across frames",
    )


class FrameDetectionResult(BaseModel):
    """Detection results for one processed video frame."""

    model_config = ConfigDict(frozen=True)

    frame_number: int = Field(
        ge=0,
        description="Zero-based frame index",
    )

    timestamp_seconds: float = Field(
        ge=0.0,
        description="Time offset from the beginning of the video",
    )

    detections: list[DetectedObject] = Field(
        default_factory=list,
        description="Objects detected in this frame",
    )

    detection_count: int = Field(
        ge=0,
        description="Number of detections in this frame",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_or_validate_detection_count(cls, data: Any) -> Any:
        """Populate the count when omitted and reject inconsistent values."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        detections = data.get("detections")

        if detections is None:
            detections = []
            data["detections"] = detections

        if not isinstance(detections, (list, tuple)):
            # Let Pydantic report the appropriate list validation error.
            return data

        actual_count = len(detections)
        provided_count = data.get("detection_count")

        if provided_count is None:
            data["detection_count"] = actual_count
        elif provided_count != actual_count:
            raise ValueError(
                f"detection_count ({provided_count}) does not match "
                f"the number of detections ({actual_count})."
            )

        return data

    @classmethod
    def from_detections(
        cls,
        frame_number: int,
        timestamp_seconds: float,
        detections: list[DetectedObject],
    ) -> FrameDetectionResult:
        """Create a frame result and calculate its detection count."""
        return cls(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            detections=detections,
            detection_count=len(detections),
        )


class VideoAnalysisSummary(BaseModel):
    """Summary of object detections across a video.

    class_counts and total_detections count per-frame detection events,
    not unique physical vehicles. This summary does not verify traffic
    congestion or identify accessibility barriers.
    """

    model_config = ConfigDict(frozen=True)

    video_name: str = Field(
        min_length=1,
        description="Video filename or identifier",
    )

    total_frames: int = Field(
        ge=0,
        description="Total frames in the source video",
    )

    processed_frames: int = Field(
        ge=0,
        description="Number of frames processed",
    )

    duration_seconds: float = Field(
        ge=0.0,
        description="Video duration in seconds",
    )

    fps: float = Field(
        ge=0.0,
        description="Source video frame rate",
    )

    class_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-frame detection events by object class; "
            "these are not unique vehicle counts."
        ),
    )

    total_detections: int = Field(
        ge=0,
        description="Total detection events across processed frames",
    )

    tracked_id_counts: dict[str, int] | None = Field(
        default=None,
        description=(
            "Optional count of distinct tracking IDs observed per class. "
            "These are tracker IDs, not verified unique physical vehicles."
        ),
    )

    total_tracked_ids: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total distinct tracking IDs observed across all target classes. "
            "Subject to ID switches, occlusions, and multi-class assignments; "
            "does not represent verified ground-truth vehicles."
        ),
    )

    disclaimer: str = Field(
        default=(
            "These are preliminary per-frame detection events, "
            "not verified unique vehicle counts. They do not establish "
            "traffic congestion levels or detect accessibility barriers."
        ),
        description="Limitations of the analysis",
    )

    @field_validator("tracked_id_counts")
    @classmethod
    def validate_tracked_id_counts(
        cls,
        counts: dict[str, int] | None,
    ) -> dict[str, int] | None:
        """Require non-empty class names and non-negative integer counts when provided."""
        if counts is None:
            return None
        for class_name, count in counts.items():
            if not class_name.strip():
                raise ValueError("Class names must not be empty.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"Tracked count for class '{class_name}' must be an integer."
                )
            if count < 0:
                raise ValueError(
                    f"Tracked count for class '{class_name}' must be non-negative."
                )
        return counts

    @field_validator("class_counts")
    @classmethod
    def validate_class_counts(
        cls,
        counts: dict[str, int],
    ) -> dict[str, int]:
        """Require non-empty class names and non-negative integer counts."""
        for class_name, count in counts.items():
            if not class_name.strip():
                raise ValueError("Class names must not be empty.")

            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"Count for class '{class_name}' must be an integer."
                )

            if count < 0:
                raise ValueError(
                    f"Count for class '{class_name}' must be non-negative."
                )

        return counts

    @model_validator(mode="after")
    def validate_summary_consistency(self) -> VideoAnalysisSummary:
        """Ensure frame totals and detection counts are consistent."""
        if self.processed_frames > self.total_frames:
            raise ValueError(
                "processed_frames cannot exceed total_frames."
            )

        summed_counts = sum(self.class_counts.values())

        if self.total_detections != summed_counts:
            raise ValueError(
                f"total_detections ({self.total_detections}) does not match the sum of class_counts ({summed_counts})."
            )

        return self


class CameraMetadata(BaseModel):
    """Spatial and hardware calibration metadata for an observational traffic camera.

    Encapsulates geographic mount coordinates, orientation angles, and spatial
    calibration status. Does not provide default coordinates or assume a verified
    camera location to prevent fabricating unverified sensor locations.
    """

    model_config = ConfigDict(frozen=True)

    camera_id: str = Field(
        min_length=1,
        max_length=100,
        description="Unique identifier for the observational camera",
    )

    latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Camera mounting latitude in WGS-84 degrees [-90.0, 90.0]",
    )

    longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Camera mounting longitude in WGS-84 degrees [-180.0, 180.0]",
    )

    elevation_meters: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional camera mounting height above ground level in meters",
    )

    heading_degrees: float = Field(
        ge=0.0,
        lt=360.0,
        description="Compass orientation angle in degrees [0.0, 360.0) (0=North, 90=East)",
    )

    tilt_degrees: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Optional camera pitch/tilt angle in degrees [-90.0, 90.0] (-90=downwards/nadir, 0=level)",
    )

    target_osm_u: int | None = Field(
        default=None,
        description="Optional OpenStreetMap source node ID of the targeted roadway segment",
    )

    target_osm_v: int | None = Field(
        default=None,
        description="Optional OpenStreetMap target node ID of the targeted roadway segment",
    )

    target_osm_key: int = Field(
        default=0,
        ge=0,
        description="OpenStreetMap edge key distinguishing parallel ways (default: 0)",
    )

    is_spatially_calibrated: bool = Field(
        default=False,
        description=(
            "Flag indicating whether camera-to-road homography transformation matrix "
            "and geographic projection have been empirically certified."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def handle_camera_aliases(cls, data: Any) -> Any:
        """Support target_u and target_v aliases for OSM node identifiers."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "target_u" in data and "target_osm_u" not in data:
            data["target_osm_u"] = data.pop("target_u")
        if "target_v" in data and "target_osm_v" not in data:
            data["target_osm_v"] = data.pop("target_v")
        if "target_key" in data and "target_osm_key" not in data:
            data["target_osm_key"] = data.pop("target_key")
        return data

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id_not_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only camera identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("camera_id must not be empty or whitespace-only.")
        return cleaned

    @field_validator(
        "latitude",
        "longitude",
        "heading_degrees",
        "elevation_meters",
        "tilt_degrees",
        mode="before",
    )
    @classmethod
    def reject_boolean_for_float_fields(cls, value: Any) -> Any:
        """Reject boolean values for float fields before coercion."""
        if isinstance(value, bool):
            raise ValueError("Geographic coordinate and orientation values must be numeric, not boolean.")
        return value

    @field_validator("latitude", "longitude", "heading_degrees")
    @classmethod
    def validate_finite_coordinates(cls, value: float) -> float:
        """Reject non-finite numeric coordinates (NaN, Inf)."""
        if not math.isfinite(value):
            raise ValueError("Geographic coordinate and heading values must be finite numbers.")
        return value

    @field_validator("elevation_meters", "tilt_degrees")
    @classmethod
    def validate_optional_finite(cls, value: float | None) -> float | None:
        """Reject non-finite optional orientation values when provided."""
        if value is not None and not math.isfinite(value):
            raise ValueError("Elevation and tilt values must be finite numbers.")
        return value

    @field_validator("target_osm_u", "target_osm_v", "target_osm_key", mode="before")
    @classmethod
    def validate_integer_osm_identifiers(cls, value: Any) -> Any:
        """Reject boolean values for OSM node identifiers before coercion."""
        if isinstance(value, bool):
            raise ValueError("OSM node and key identifiers must be integers, not boolean.")
        return value

    @model_validator(mode="after")
    def validate_spatial_calibration(self) -> CameraMetadata:
        """Ensure spatial calibration is not falsely claimed without target road segment."""
        if self.is_spatially_calibrated:
            if self.target_osm_u is None or self.target_osm_v is None:
                raise ValueError(
                    "is_spatially_calibrated cannot be True without identifying the targeted road edge "
                    "(both target_osm_u and target_osm_v are required)."
                )
        if (self.target_osm_u is None) != (self.target_osm_v is None):
            raise ValueError(
                "target_osm_u and target_osm_v must both be specified together to form a valid road edge."
            )
        if self.target_osm_u is not None and self.target_osm_v is not None:
            if self.target_osm_u == self.target_osm_v:
                raise ValueError(
                    "target_osm_u and target_osm_v must be distinct node IDs (self-loops are not allowed)."
                )
        return self


class ObservationTimeEnvelope(BaseModel):
    """Temporal validity window for computer vision video analysis.

    Enforces timezone-aware UTC timestamps, rejects end times earlier than or equal
    to start times, and ensures observation duration matches the elapsed timestamp difference.
    """

    model_config = ConfigDict(frozen=True)

    start_time: datetime = Field(
        description="Observation window start timestamp. Must be timezone-aware (e.g. UTC)."
    )

    end_time: datetime = Field(
        description="Observation window end timestamp. Must be timezone-aware (e.g. UTC)."
    )

    duration_seconds: float = Field(
        gt=0.0,
        description="Observation sampling window length in seconds, strictly greater than zero.",
    )

    time_to_live_seconds: float = Field(
        default=900.0,
        ge=60.0,
        description="Time-to-live in seconds before the observation is considered stale (default: 900s / 15 mins).",
    )

    @property
    def captured_at_utc(self) -> datetime:
        """Alias property matching the integration plan's captured_at_utc convention."""
        return self.start_time

    @model_validator(mode="before")
    @classmethod
    def handle_timing_inputs_and_aliases(cls, data: Any) -> Any:
        """Support 'captured_at_utc' alias and auto-compute duration or end_time when one is omitted."""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # Support captured_at_utc alias
        if "captured_at_utc" in data and "start_time" not in data:
            data["start_time"] = data.pop("captured_at_utc")

        start = data.get("start_time")
        end = data.get("end_time")
        duration = data.get("duration_seconds")

        def _to_dt(val: Any) -> datetime | None:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    return None
            return None

        start_dt = _to_dt(start)
        end_dt = _to_dt(end)

        # 1. Auto-compute duration if omitted and both start & end datetimes are provided
        if duration is None and start_dt is not None and end_dt is not None:
            if start_dt.tzinfo is not None and end_dt.tzinfo is not None:
                elapsed = (end_dt - start_dt).total_seconds()
                if elapsed > 0.0:
                    data["duration_seconds"] = elapsed

        # 2. Auto-compute end_time if omitted and start & duration_seconds are provided
        elif end is None and start_dt is not None and duration is not None:
            try:
                dur_float = float(duration)
                if dur_float > 0.0 and start_dt.tzinfo is not None:
                    data["end_time"] = start_dt + timedelta(seconds=dur_float)
            except (ValueError, TypeError):
                pass

        return data

    @field_validator("start_time", "end_time")
    @classmethod
    def require_timezone_aware(cls, dt: datetime) -> datetime:
        """Require explicit timezone awareness to prevent ambiguous cross-server comparisons."""
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError(
                "Timestamps must be timezone-aware (e.g., UTC with tzinfo=timezone.utc). "
                "Naive timestamps without timezone information are rejected to prevent ambiguity."
            )
        return dt

    @field_validator("duration_seconds", "time_to_live_seconds", mode="before")
    @classmethod
    def reject_boolean_timing_numbers(cls, value: Any) -> Any:
        """Reject boolean values for duration and TTL before coercion."""
        if isinstance(value, bool):
            raise ValueError("Duration and TTL must be numbers, not boolean.")
        return value

    @field_validator("duration_seconds", "time_to_live_seconds")
    @classmethod
    def validate_finite_timing_numbers(cls, value: float) -> float:
        """Reject non-finite numbers for duration and TTL."""
        if not math.isfinite(value):
            raise ValueError("Duration and TTL must be finite numbers, not NaN/Inf.")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> ObservationTimeEnvelope:
        """Require end_time strictly greater than start_time and verify duration consistency."""
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be strictly after start_time.")

        elapsed = (self.end_time - self.start_time).total_seconds()
        if elapsed <= 0.0:
            raise ValueError("Calculated elapsed time between start_time and end_time must be positive.")

        # Validate that declared duration_seconds is consistent with timestamp difference (within 1.0s tolerance)
        if abs(self.duration_seconds - elapsed) > 1.0:
            raise ValueError(
                f"duration_seconds ({self.duration_seconds}) does not match the elapsed time "
                f"between start_time and end_time ({elapsed}s)."
            )

        return self


class VisionEdgeObservation(BaseModel):
    """Standardized payload linking computer-vision vehicle observations to a road edge.

    Bridges video analysis results with an OpenStreetMap directed edge (osm_u, osm_v, osm_key).
    Enforces non-negative vehicle counts, valid quality scores, and explicit disclaimers
    marking any derived traffic indicator as unvalidated research prototypes.
    """

    model_config = ConfigDict(frozen=True)

    camera_id: str = Field(
        min_length=1,
        max_length=100,
        description="Unique identifier for the reporting camera",
    )

    camera_metadata: CameraMetadata | None = Field(
        default=None,
        description="Optional full camera calibration and spatial metadata",
    )

    osm_u: int = Field(
        description="Directed OpenStreetMap source node ID (u)",
    )

    osm_v: int = Field(
        description="Directed OpenStreetMap destination node ID (v)",
    )

    osm_key: int = Field(
        default=0,
        ge=0,
        description="OpenStreetMap parallel edge key (default: 0)",
    )

    timing: ObservationTimeEnvelope = Field(
        description="Temporal validity and duration envelope for this observation",
    )

    unique_vehicles_observed: int = Field(
        ge=0,
        description="Screened, deduplicated count of unique physical vehicles observed",
    )

    vehicles_by_class: dict[str, int] = Field(
        default_factory=dict,
        description="Observed vehicle counts broken down by category (e.g. car, bus, motorcycle, truck)",
    )

    total_detection_events: int | None = Field(
        default=None,
        ge=0,
        description="Optional raw sum of per-frame detection bounding boxes across all frames",
    )

    average_speed_kmh: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional estimated average vehicle speed in km/h if camera homography is calibrated",
    )

    observation_quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Observation quality and confidence score between 0.0 (unusable) and 1.0 (ideal)",
    )

    experimental_congestion_index: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "EXPERIMENTAL AND UNVALIDATED: Proposed dimensionless congestion index in [0.0, 1.0]. "
            "This metric is uncalibrated against real-world road capacity and MUST NOT be used "
            "to modify pedestrian route costs without empirical verification."
        ),
    )

    disclaimer: str = Field(
        default=(
            "This vision edge observation is an experimental research prototype. It does not "
            "establish ground-truth traffic congestion or certify pedestrian route accessibility."
        ),
        description="Limitations and non-guarantee disclaimer",
    )

    @property
    def camera(self) -> CameraMetadata | None:
        """Convenience property for accessing camera_metadata matching research plan convention."""
        return self.camera_metadata

    @property
    def u(self) -> int:
        """OpenStreetMap source node identifier (osm_u) alias."""
        return self.osm_u

    @property
    def v(self) -> int:
        """OpenStreetMap destination node identifier (osm_v) alias."""
        return self.osm_v

    @property
    def key(self) -> int:
        """OpenStreetMap parallel edge key (osm_key) alias."""
        return self.osm_key

    def edge_tuple(self) -> tuple[int, int, int]:
        """Return (osm_u, osm_v, osm_key) matching AccessRoute graph edge representation."""
        return (self.osm_u, self.osm_v, self.osm_key)

    @model_validator(mode="before")
    @classmethod
    def populate_camera_fields(cls, data: Any) -> Any:
        """Allow passing 'camera', 'u', 'v', 'key', and auto-populating edge from camera metadata."""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # Support 'camera' key containing CameraMetadata, dict, or string ID
        camera = data.get("camera")
        if camera is not None:
            if isinstance(camera, CameraMetadata):
                data.setdefault("camera_id", camera.camera_id)
                data.setdefault("camera_metadata", camera)
                if camera.target_osm_u is not None and "osm_u" not in data and "u" not in data:
                    data["osm_u"] = camera.target_osm_u
                if camera.target_osm_v is not None and "osm_v" not in data and "v" not in data:
                    data["osm_v"] = camera.target_osm_v
                if "osm_key" not in data and "key" not in data:
                    data["osm_key"] = camera.target_osm_key
            elif isinstance(camera, dict) and "camera_id" in camera:
                data.setdefault("camera_id", camera["camera_id"])
            elif isinstance(camera, str) and camera.strip():
                data.setdefault("camera_id", camera.strip())

        # Support u, v, key aliases
        if "u" in data and "osm_u" not in data:
            data["osm_u"] = data.pop("u")
        if "v" in data and "osm_v" not in data:
            data["osm_v"] = data.pop("v")
        if "key" in data and "osm_key" not in data:
            data["osm_key"] = data.pop("key")

        return data

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id_non_empty(cls, value: str) -> str:
        """Require non-blank camera identifier."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("camera_id must not be empty or whitespace-only.")
        return cleaned

    @field_validator(
        "osm_u",
        "osm_v",
        "osm_key",
        "unique_vehicles_observed",
        "total_detection_events",
        mode="before",
    )
    @classmethod
    def reject_boolean_integers(cls, value: Any) -> Any:
        """Reject boolean values for integer fields before coercion."""
        if isinstance(value, bool):
            raise ValueError("Node identifiers and vehicle counts must be integers, not boolean.")
        return value

    @field_validator(
        "average_speed_kmh",
        "observation_quality_score",
        "experimental_congestion_index",
        mode="before",
    )
    @classmethod
    def reject_boolean_floats(cls, value: Any) -> Any:
        """Reject boolean values for float fields before coercion."""
        if isinstance(value, bool):
            raise ValueError("Numeric metrics and quality scores must be numbers, not boolean.")
        return value

    @field_validator("disclaimer")
    @classmethod
    def validate_disclaimer_not_empty(cls, value: str) -> str:
        """Require non-empty disclaimer declaring experimental prototype status."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("disclaimer must not be empty or whitespace-only.")
        return cleaned

    @field_validator("vehicles_by_class")
    @classmethod
    def validate_vehicles_by_class(cls, counts: dict[str, int]) -> dict[str, int]:
        """Require non-empty class names and non-negative integer counts."""
        for class_name, count in counts.items():
            if not class_name.strip():
                raise ValueError("Class names in vehicles_by_class must not be empty.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"Count for vehicle class '{class_name}' must be an integer."
                )
            if count < 0:
                raise ValueError(
                    f"Count for vehicle class '{class_name}' must be non-negative."
                )
        return counts

    @field_validator("average_speed_kmh", "observation_quality_score", "experimental_congestion_index")
    @classmethod
    def validate_finite_floats(cls, value: float | None) -> float | None:
        """Ensure numeric metrics are finite floats."""
        if value is not None and not math.isfinite(value):
            raise ValueError("Numeric metric values must be finite numbers.")
        return value

    @model_validator(mode="after")
    def validate_observation_consistency(self) -> VisionEdgeObservation:
        """Enforce geographic and statistical consistency."""
        # 1. Distinct OSM edge endpoints (no self-loops)
        if self.osm_u == self.osm_v:
            raise ValueError(
                "osm_u and osm_v must be distinct node IDs (self-loops are not valid road edges)."
            )

        # 2. Consistency with attached camera_metadata
        if self.camera_metadata is not None:
            cam = self.camera_metadata
            if cam.target_osm_u is not None:
                if cam.target_osm_u != self.osm_u or cam.target_osm_v != self.osm_v:
                    raise ValueError(
                        f"Observation edge ({self.osm_u}, {self.osm_v}) conflicts with "
                        f"camera's target edge ({cam.target_osm_u}, {cam.target_osm_v})."
                    )
            if cam.camera_id != self.camera_id:
                raise ValueError(
                    f"camera_id '{self.camera_id}' conflicts with camera_metadata.camera_id '{cam.camera_id}'."
                )

        # 3. Vehicle count sanity: unique vehicles cannot be less than the count of any single class
        if self.vehicles_by_class and self.unique_vehicles_observed < max(self.vehicles_by_class.values()):
            raise ValueError(
                f"unique_vehicles_observed ({self.unique_vehicles_observed}) cannot be less than "
                f"the count for any individual vehicle class ({max(self.vehicles_by_class.values())})."
            )

        # 4. Total detection events cannot be less than unique physical vehicles
        if self.total_detection_events is not None and self.total_detection_events < self.unique_vehicles_observed:
            raise ValueError(
                f"total_detection_events ({self.total_detection_events}) cannot be less than "
                f"unique_vehicles_observed ({self.unique_vehicles_observed})."
            )

        return self

