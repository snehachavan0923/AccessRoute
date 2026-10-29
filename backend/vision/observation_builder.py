"""Observation builder module for AccessRoute computer-vision pipeline.

Converts post-processed video results (PostProcessedVideoResult) into validated
VisionEdgeObservation instances when spatial and camera metadata is provided.

DISCLAIMER AND LIMITATIONS:
1. Does NOT invent camera coordinates, OSM edge identifiers, spatial calibration,
   vehicle speeds, or congestion indices.
2. Does NOT derive congestion from vehicle counts, detection counts, track IDs,
   or bounding boxes alone.
3. For uncalibrated cameras or videos lacking empirical calibration, the builder
   does not manufacture a calibrated observation, ensuring traffic routing safeguards
   remain intact.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.vision.post_processor import PostProcessedVideoResult
from backend.vision.schemas import (
    CameraMetadata,
    ObservationTimeEnvelope,
    VisionEdgeObservation,
)


class ObservationBuildResult(BaseModel):
    """Result of attempting to build a VisionEdgeObservation from video analysis."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        description="Whether a valid VisionEdgeObservation was successfully constructed"
    )
    observation: VisionEdgeObservation | None = Field(
        default=None,
        description="The constructed VisionEdgeObservation if successful, otherwise None",
    )
    status: str = Field(
        description="Machine-readable status code explaining build outcome",
    )
    message: str = Field(
        description="Human-readable explanation of outcome and any missing requirements",
    )


def build_vision_edge_observation(
    processed_result: PostProcessedVideoResult,
    camera_metadata: CameraMetadata | None = None,
    start_time: datetime | None = None,
    time_to_live_seconds: float = 900.0,
    observation_quality_score: float | None = None,
    experimental_congestion_index: float | None = None,
    osm_u: int | None = None,
    osm_v: int | None = None,
    osm_key: int = 0,
    camera_id: str | None = None,
    duration_seconds: float | None = None,
) -> ObservationBuildResult:
    """Convert a PostProcessedVideoResult into a VisionEdgeObservation.

    Constructs a validated observation only when the necessary camera and directed
    road edge metadata are explicitly provided. Does not fabricate geographic
    coordinates or guess congestion from detection numbers.

    Args:
        processed_result: PostProcessedVideoResult containing tracklets and clean detections.
        camera_metadata: Optional CameraMetadata describing mounting location and calibration.
        start_time: Optional timezone-aware UTC datetime for observation start.
        time_to_live_seconds: TTL in seconds before observation expires (default: 900s).
        observation_quality_score: Optional explicit quality score in [0.0, 1.0].
        experimental_congestion_index: Optional explicit congestion index in [0.0, 1.0].
            Must NOT be derived from vehicle counts alone.
        osm_u: Optional explicit OpenStreetMap source node ID.
        osm_v: Optional explicit OpenStreetMap destination node ID.
        osm_key: OpenStreetMap edge key (default: 0).
        camera_id: Optional explicit camera identifier.
        duration_seconds: Optional observation duration in seconds.

    Returns:
        ObservationBuildResult indicating success/failure and containing the observation if valid.

    Raises:
        TypeError: If processed_result is not a PostProcessedVideoResult.
    """
    if not isinstance(processed_result, PostProcessedVideoResult):
        raise TypeError(
            f"Expected PostProcessedVideoResult, got {type(processed_result).__name__}."
        )

    # 1. Resolve camera identifier
    cid = camera_id or (camera_metadata.camera_id if camera_metadata else None)
    if not cid or not cid.strip():
        return ObservationBuildResult(
            success=False,
            observation=None,
            status="missing_camera_id",
            message="Camera identifier is required to construct a vision edge observation.",
        )
    cid = cid.strip()

    # 2. Resolve target directed edge (u, v, key)
    u = osm_u if osm_u is not None else (camera_metadata.target_osm_u if camera_metadata else None)
    v = osm_v if osm_v is not None else (camera_metadata.target_osm_v if camera_metadata else None)
    key = osm_key if osm_key != 0 else (camera_metadata.target_osm_key if camera_metadata else 0)

    if u is None or v is None:
        return ObservationBuildResult(
            success=False,
            observation=None,
            status="missing_edge_metadata",
            message=(
                "Directed OpenStreetMap edge endpoints (osm_u, osm_v) are required to associate "
                "video observations with a specific road segment."
            ),
        )

    if u == v:
        return ObservationBuildResult(
            success=False,
            observation=None,
            status="invalid_edge",
            message="Directed road edge endpoints osm_u and osm_v must be distinct (self-loops are not allowed).",
        )

    # 3. Check for conflict with camera metadata if both are supplied
    if camera_metadata is not None:
        if camera_metadata.target_osm_u is not None and camera_metadata.target_osm_v is not None:
            if camera_metadata.target_osm_u != u or camera_metadata.target_osm_v != v:
                return ObservationBuildResult(
                    success=False,
                    observation=None,
                    status="edge_mismatch",
                    message=(
                        f"Specified edge ({u}, {v}) conflicts with camera metadata "
                        f"target edge ({camera_metadata.target_osm_u}, {camera_metadata.target_osm_v})."
                    ),
                )
        if camera_metadata.camera_id != cid:
            return ObservationBuildResult(
                success=False,
                observation=None,
                status="camera_id_mismatch",
                message=(
                    f"Specified camera_id '{cid}' conflicts with camera_metadata.camera_id "
                    f"'{camera_metadata.camera_id}'."
                ),
            )

    # 4. Resolve observation timing window
    if duration_seconds is not None and duration_seconds > 0.0:
        dur = float(duration_seconds)
    elif (
        processed_result.frames
        and len(processed_result.frames) >= 2
        and processed_result.frames[-1].timestamp_seconds > processed_result.frames[0].timestamp_seconds
    ):
        dur = float(
            processed_result.frames[-1].timestamp_seconds
            - processed_result.frames[0].timestamp_seconds
        )
    else:
        dur = 60.0  # sensible default duration envelope in seconds

    if start_time is None:
        st = datetime.now(timezone.utc) - timedelta(seconds=dur)
    else:
        if start_time.tzinfo is None:
            return ObservationBuildResult(
                success=False,
                observation=None,
                status="naive_timestamp",
                message="start_time must be timezone-aware (e.g., UTC).",
            )
        st = start_time

    et = st + timedelta(seconds=dur)

    try:
        timing = ObservationTimeEnvelope(
            start_time=st,
            end_time=et,
            duration_seconds=dur,
            time_to_live_seconds=time_to_live_seconds,
        )
    except ValueError as val_err:
        return ObservationBuildResult(
            success=False,
            observation=None,
            status="invalid_timing",
            message=f"Invalid timing envelope: {val_err}",
        )

    # 5. Extract vehicle observation metrics from PostProcessedVideoResult
    summary = processed_result.summary
    unique_vehicles = summary.processed_total_tracked_ids
    vehicles_by_class = dict(summary.processed_tracked_id_counts)
    total_detections = summary.processed_total_detections

    # 6. Resolve quality score
    if observation_quality_score is not None:
        quality = float(observation_quality_score)
    else:
        # Heuristic quality based on tracking retention ratio
        raw_dets = summary.raw_total_detections
        quality = min(1.0, max(0.0, total_detections / raw_dets)) if raw_dets > 0 else 1.0

    # 7. Construct VisionEdgeObservation
    try:
        obs = VisionEdgeObservation(
            camera_id=cid,
            camera_metadata=camera_metadata,
            osm_u=u,
            osm_v=v,
            osm_key=key,
            timing=timing,
            unique_vehicles_observed=unique_vehicles,
            vehicles_by_class=vehicles_by_class,
            total_detection_events=total_detections,
            observation_quality_score=quality,
            experimental_congestion_index=experimental_congestion_index,
        )
    except ValueError as val_err:
        return ObservationBuildResult(
            success=False,
            observation=None,
            status="validation_error",
            message=f"Observation validation failed: {val_err}",
        )

    return ObservationBuildResult(
        success=True,
        observation=obs,
        status="success",
        message="Successfully constructed vision edge observation.",
    )
