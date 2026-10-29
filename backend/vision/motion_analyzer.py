"""Vehicle movement analysis on computer-vision YOLO and ByteTrack tracking data.

Measures image-plane displacement, elapsed time, and pixel movement rate (px/s)
for tracked vehicles across consecutive video frames.

IMPORTANT DISCLAIMER & CALIBRATION REQUIREMENTS:
1. Pixel displacement on an image sensor does NOT represent physical distance (meters)
   or vehicle speed (km/h).
2. Estimating real-world physical speed requires:
   a. Camera-to-roadway metric calibration or a planar homography matrix (H)
      mapping 2D pixel coordinates (u, v) to ground-plane coordinates (X, Y).
   b. Perspective foreshortening correction (objects farther from the camera appear
      smaller and move fewer pixels per physical meter than closer objects).
   c. Defined roadway region of interest (ROI) and lane boundary polygons.
   d. Ground-truth distance reference measurements (e.g., surveyed lane markings).
   e. Verified monotonic timestamps and accurate video frame rate (FPS).
3. Camera location metadata (latitude, longitude, heading) alone is NOT sufficient
   for physical speed estimation.
4. Pixel movement measurements must NOT be used to compute traffic congestion
   scores or route costs without validated, certified camera calibration.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.vision.schemas import (
    BoundingBox,
    DetectedObject,
    FrameDetectionResult,
)

DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX: float = 3.0
DEFAULT_STATIONARY_SPEED_THRESHOLD_PX_PER_SEC: float = 3.0
DEFAULT_FALLBACK_FPS: float = 30.0

CALIBRATION_REQUIREMENTS_SUMMARY: str = (
    "Physical vehicle speed (in m/s or km/h) cannot be determined from 2D image coordinates alone. "
    "Reliable physical speed measurement requires: (1) camera-to-roadway metric calibration or a "
    "planar homography matrix (H) mapping image pixels to ground coordinates; (2) road-region definition "
    "(region of interest / lane masks); (3) empirical ground-truth scale reference measurements; "
    "(4) perspective foreshortening correction; and (5) verified monotonic timestamps and video frame rate. "
    "Camera mounting coordinates (latitude, longitude) alone are insufficient for speed estimation."
)


class MotionMeasurementStatus(str, Enum):
    """Validity and classification status for a tracked vehicle's motion measurement."""

    VALID = "valid"
    STATIONARY = "stationary"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    INVALID_TIMING = "invalid_timing"
    NO_TRACKS = "no_tracks"


class TrackObservationPoint(BaseModel):
    """Single observation point along a tracked vehicle's trajectory."""

    model_config = ConfigDict(frozen=True)

    frame_number: int = Field(ge=0, description="Zero-based video frame index")
    timestamp_seconds: float = Field(ge=0.0, description="Elapsed timestamp in seconds from video start")
    center_x: float = Field(ge=0.0, description="Image-plane horizontal bounding box center in pixels")
    center_y: float = Field(ge=0.0, description="Image-plane vertical bounding box center in pixels")
    bounding_box: BoundingBox = Field(description="Bounding box coordinates")
    confidence: float = Field(ge=0.0, le=1.0, description="YOLO detection confidence score")
    class_name: str = Field(description="Detected object class label")

    @field_validator("timestamp_seconds", "center_x", "center_y")
    @classmethod
    def validate_finite_coords(cls, value: float) -> float:
        """Reject non-finite coordinates and timestamps."""
        if not math.isfinite(value):
            raise ValueError("Observation coordinates and timestamps must be finite numbers (NaN and Inf rejected).")
        return value


class ConsecutiveStepDisplacement(BaseModel):
    """Displacement and motion measurement between two consecutive observations of a track."""

    model_config = ConfigDict(frozen=True)

    from_frame: int = Field(ge=0, description="Initial frame index")
    to_frame: int = Field(ge=0, description="Subsequent frame index")
    from_timestamp_seconds: float = Field(ge=0.0, description="Initial timestamp in seconds")
    to_timestamp_seconds: float = Field(ge=0.0, description="Subsequent timestamp in seconds")
    elapsed_seconds: float = Field(gt=0.0, description="Elapsed time in seconds between observations")
    displacement_pixels: float = Field(ge=0.0, description="Euclidean displacement in image pixels")
    pixel_speed_px_per_sec: float = Field(
        ge=0.0,
        description="Estimated image-plane movement in pixels per second (displacement / elapsed_seconds)",
    )
    timing_source: str = Field(
        description="Source used for timing calculation ('timestamp' or 'fps_fallback')",
    )


class TrackMotionMeasurement(BaseModel):
    """Motion and displacement analysis for an individual tracked vehicle trajectory."""

    model_config = ConfigDict(frozen=True)

    track_id: int = Field(ge=0, description="ByteTrack tracking identifier")
    assigned_class: str = Field(description="Vehicle class (e.g. car, bus, truck, motorcycle)")
    usable_observation_count: int = Field(
        ge=0,
        description="Total distinct usable frame observations for this vehicle track",
    )
    first_observed_frame: int | None = Field(
        default=None,
        ge=0,
        description="Earliest observed frame index",
    )
    last_observed_frame: int | None = Field(
        default=None,
        ge=0,
        description="Latest observed frame index",
    )
    first_observed_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Earliest observation timestamp in seconds",
    )
    last_observed_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Latest observation timestamp in seconds",
    )
    total_observed_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total duration in seconds spanned by valid observations",
    )
    cumulative_pixel_displacement: float = Field(
        default=0.0,
        ge=0.0,
        description="Cumulative Euclidean displacement in pixels across all consecutive steps",
    )
    net_pixel_displacement: float = Field(
        default=0.0,
        ge=0.0,
        description="Straight-line Euclidean displacement between first and last observed centers",
    )
    average_pixel_speed_px_per_sec: float | None = Field(
        default=None,
        ge=0.0,
        description="Average image-plane speed in pixels per second across valid observed steps",
    )
    max_pixel_speed_px_per_sec: float | None = Field(
        default=None,
        ge=0.0,
        description="Peak image-plane step speed in pixels per second",
    )
    step_displacements: list[ConsecutiveStepDisplacement] = Field(
        default_factory=list,
        description="Detailed step-by-step displacements between consecutive observations",
    )
    measurement_status: MotionMeasurementStatus = Field(
        default=MotionMeasurementStatus.INSUFFICIENT_OBSERVATIONS,
        description="Status of the motion measurement",
    )
    status_message: str = Field(
        description="Human-readable explanation of measurement validity",
    )
    is_stationary: bool = Field(
        default=False,
        description="True if vehicle image movement is within sensor jitter threshold",
    )
    is_physical_speed_available: bool = Field(
        default=False,
        description="Always False until validated camera-to-roadway metric calibration is provided",
    )
    physical_speed_limitation: str = Field(
        default=CALIBRATION_REQUIREMENTS_SUMMARY,
        description="Scientific limitations explaining why physical speed in km/h is unavailable",
    )


class VideoMotionSummary(BaseModel):
    """Aggregate motion measurement summary across all tracked vehicles in a video.

    AGGREGATION METHODOLOGY AND LIMITATIONS:
    - average_scene_pixel_speed_px_per_sec: Unweighted arithmetic mean of average_pixel_speed_px_per_sec
      across all tracks with measurement_status == VALID (moving tracks). Excludes stationary tracks and
      tracks with insufficient observations to prevent zero-speed skew.
      NOTE: Unweighted aggregation weights a 2-frame transient track identically to a 300-frame persistent
      track and does NOT account for perspective scaling across depth planes.
    - weighted_scene_pixel_speed_px_per_sec: Duration-weighted aggregate calculated as:
      (sum of cumulative displacements of valid moving tracks) / (sum of total observed durations of valid moving tracks).
      This weights persistent trajectories proportionally to their duration, mitigating transient track bias.
    - Neither aggregate represents physical road traffic speed without camera-to-roadway metric calibration.
    """

    model_config = ConfigDict(frozen=True)

    total_tracks_analyzed: int = Field(ge=0, description="Total unique track IDs evaluated")
    valid_moving_tracks_count: int = Field(ge=0, description="Tracks with valid non-stationary movement")
    stationary_tracks_count: int = Field(ge=0, description="Tracks classified as stationary")
    insufficient_data_tracks_count: int = Field(
        ge=0,
        description="Tracks with fewer than 2 observations or invalid timestamps",
    )
    average_scene_pixel_speed_px_per_sec: float | None = Field(
        default=None,
        ge=0.0,
        description="Unweighted arithmetic mean of average pixel speeds across valid moving tracks (px/s)",
    )
    weighted_scene_pixel_speed_px_per_sec: float | None = Field(
        default=None,
        ge=0.0,
        description="Duration-weighted mean pixel speed across valid moving tracks: total_disp / total_dur (px/s)",
    )
    max_scene_pixel_speed_px_per_sec: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum observed pixel speed (px/s) across all tracks in the scene",
    )
    is_physical_speed_available: bool = Field(
        default=False,
        description="Always False without camera-to-roadway metric calibration",
    )
    calibration_status: str = Field(
        default=CALIBRATION_REQUIREMENTS_SUMMARY,
        description="Explanation of missing calibration prerequisites",
    )
    track_measurements: list[TrackMotionMeasurement] = Field(
        default_factory=list,
        description="Individual motion measurements for each analyzed track",
    )


def compute_bounding_box_center(bbox: BoundingBox) -> tuple[float, float]:
    """Calculate the 2D image-plane center coordinates (cx, cy) of a bounding box.

    Args:
        bbox: BoundingBox with x_min, y_min, x_max, y_max.

    Returns:
        tuple[float, float]: (center_x, center_y) in pixel coordinates.

    Raises:
        ValueError: If bounding box coordinates or calculated centers are not finite numbers.
    """
    if not (
        math.isfinite(bbox.x_min)
        and math.isfinite(bbox.y_min)
        and math.isfinite(bbox.x_max)
        and math.isfinite(bbox.y_max)
    ):
        raise ValueError(f"BoundingBox contains non-finite coordinates: {bbox}")

    center_x = (bbox.x_min + bbox.x_max) / 2.0
    center_y = (bbox.y_min + bbox.y_max) / 2.0

    if not (math.isfinite(center_x) and math.isfinite(center_y)):
        raise ValueError(f"Calculated bounding box center is non-finite: ({center_x}, {center_y})")

    return center_x, center_y


def compute_pixel_displacement(
    center_1: tuple[float, float],
    center_2: tuple[float, float],
) -> float:
    """Calculate Euclidean distance between two 2D points in pixels.

    Args:
        center_1: (x1, y1) coordinates.
        center_2: (x2, y2) coordinates.

    Returns:
        float: Euclidean distance in pixels, non-negative.

    Raises:
        ValueError: If coordinate values are non-finite.
    """
    if not (
        math.isfinite(center_1[0])
        and math.isfinite(center_1[1])
        and math.isfinite(center_2[0])
        and math.isfinite(center_2[1])
    ):
        raise ValueError(f"Point coordinates must be finite: {center_1}, {center_2}")

    dx = center_2[0] - center_1[0]
    dy = center_2[1] - center_1[1]
    dist = math.hypot(dx, dy)

    if not math.isfinite(dist):
        raise ValueError(f"Calculated pixel displacement is non-finite: {dist}")

    return dist


def analyze_single_track_motion(
    track_id: int,
    observations: Sequence[TrackObservationPoint],
    fps: float = DEFAULT_FALLBACK_FPS,
    assigned_class: str | None = None,
    stationary_displacement_threshold_px: float = DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX,
    stationary_speed_threshold_px_per_sec: float = DEFAULT_STATIONARY_SPEED_THRESHOLD_PX_PER_SEC,
) -> TrackMotionMeasurement:
    """Analyze image-plane displacement and movement rate for a single vehicle track.

    STATIONARY CLASSIFICATION HEURISTICS & LIMITATIONS:
    - stationary_displacement_threshold_px (default 3.0px) and stationary_speed_threshold_px_per_sec (default 3.0px/s)
      are image-space empirical heuristics designed to absorb bounding-box jitter and detector noise.
    - A track is classified as STATIONARY if:
      1. cumulative_pixel_displacement <= displacement_threshold (negligible total image movement), OR
      2. net_pixel_displacement <= displacement_threshold AND average_speed <= speed_threshold (the centroid remained
         in a localized cluster even over extended durations).
    - Requiring net displacement <= threshold prevents slowly-creeping or distant vehicles that traversed substantial
      image distances (e.g. 500px over 200s) from being falsely marked stationary.
    - LIMITATIONS: Perspective foreshortening means 3px in the background corresponds to many physical meters,
      while 3px in the foreground corresponds to centimeters. Vehicles moving along the camera optical axis
      may exhibit minimal image displacement despite moving physically. These heuristics do NOT certify physical immobility.

    AVERAGE PIXEL SPEED DEFINITION:
    - average_pixel_speed_px_per_sec = cumulative_pixel_displacement / valid_elapsed_seconds_total
      This is the true kinematic time-weighted average speed across the trajectory (total distance / total time),
      preventing bias from steps of varying temporal durations or frame gaps.

    Args:
        track_id: Tracking identifier.
        observations: Chronological sequence of TrackObservationPoint instances for this track.
        fps: Video frames per second, used as fallback when timestamps are identical or zero.
        assigned_class: Optional smoothed vehicle class label (e.g. from post-processor).
        stationary_displacement_threshold_px: Maximum cumulative pixels to be considered stationary.
        stationary_speed_threshold_px_per_sec: Maximum pixel speed to be considered stationary.

    Returns:
        TrackMotionMeasurement containing step displacements, image speeds, and validity status.
    """
    resolved_class = assigned_class or (observations[0].class_name if observations else "vehicle")

    if not observations:
        return TrackMotionMeasurement(
            track_id=track_id,
            assigned_class=resolved_class,
            usable_observation_count=0,
            measurement_status=MotionMeasurementStatus.INSUFFICIENT_OBSERVATIONS,
            status_message="No usable observations available for this track ID.",
        )

    effective_fps = fps if (math.isfinite(fps) and fps > 0.0) else DEFAULT_FALLBACK_FPS
    eff_disp_thresh = (
        stationary_displacement_threshold_px
        if (math.isfinite(stationary_displacement_threshold_px) and stationary_displacement_threshold_px >= 0.0)
        else DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX
    )
    eff_speed_thresh = (
        stationary_speed_threshold_px_per_sec
        if (math.isfinite(stationary_speed_threshold_px_per_sec) and stationary_speed_threshold_px_per_sec >= 0.0)
        else DEFAULT_STATIONARY_SPEED_THRESHOLD_PX_PER_SEC
    )

    # Sort observations by frame number and timestamp, removing frame duplicates if any
    sorted_obs: list[TrackObservationPoint] = sorted(
        observations,
        key=lambda obs: (obs.frame_number, obs.timestamp_seconds),
    )

    # Deduplicate observations occurring on the same frame index by retaining the highest confidence
    deduped_obs: list[TrackObservationPoint] = []
    seen_frames: dict[int, TrackObservationPoint] = {}
    for obs in sorted_obs:
        existing = seen_frames.get(obs.frame_number)
        if existing is None or obs.confidence > existing.confidence:
            seen_frames[obs.frame_number] = obs

    deduped_obs = [seen_frames[f] for f in sorted(seen_frames.keys())]

    first_obs = deduped_obs[0]
    last_obs = deduped_obs[-1]
    usable_count = len(deduped_obs)

    if usable_count < 2:
        return TrackMotionMeasurement(
            track_id=track_id,
            assigned_class=resolved_class,
            usable_observation_count=usable_count,
            first_observed_frame=first_obs.frame_number,
            last_observed_frame=last_obs.frame_number,
            first_observed_time_seconds=first_obs.timestamp_seconds,
            last_observed_time_seconds=last_obs.timestamp_seconds,
            total_observed_duration_seconds=0.0,
            cumulative_pixel_displacement=0.0,
            net_pixel_displacement=0.0,
            measurement_status=MotionMeasurementStatus.INSUFFICIENT_OBSERVATIONS,
            status_message=(
                f"Track has only {usable_count} observation. At least 2 distinct observations "
                f"are required to calculate displacement and elapsed time."
            ),
        )

    # Net displacement between initial and terminal centers
    net_displacement = compute_pixel_displacement(
        (first_obs.center_x, first_obs.center_y),
        (last_obs.center_x, last_obs.center_y),
    )

    step_displacements: list[ConsecutiveStepDisplacement] = []
    valid_elapsed_seconds_total = 0.0
    cumulative_displacement = 0.0

    for i in range(len(deduped_obs) - 1):
        prev = deduped_obs[i]
        curr = deduped_obs[i + 1]

        step_dist = compute_pixel_displacement(
            (prev.center_x, prev.center_y),
            (curr.center_x, curr.center_y),
        )

        # Elapsed time calculation: prefer timestamps, fallback to frame delta / fps
        dt = curr.timestamp_seconds - prev.timestamp_seconds
        timing_source = "timestamp"

        if dt <= 0.0 or not math.isfinite(dt):
            frame_delta = curr.frame_number - prev.frame_number
            if frame_delta > 0 and effective_fps > 0.0:
                dt = frame_delta / effective_fps
                timing_source = "fps_fallback"

        # Discard invalid or non-positive elapsed time steps safely
        if dt > 0.0 and math.isfinite(dt) and math.isfinite(step_dist):
            speed_px_s = step_dist / dt
            step_displacements.append(
                ConsecutiveStepDisplacement(
                    from_frame=prev.frame_number,
                    to_frame=curr.frame_number,
                    from_timestamp_seconds=prev.timestamp_seconds,
                    to_timestamp_seconds=curr.timestamp_seconds,
                    elapsed_seconds=round(dt, 5),
                    displacement_pixels=round(step_dist, 3),
                    pixel_speed_px_per_sec=round(speed_px_s, 3),
                    timing_source=timing_source,
                )
            )
            cumulative_displacement += step_dist
            valid_elapsed_seconds_total += dt

    if not step_displacements or valid_elapsed_seconds_total <= 0.0:
        return TrackMotionMeasurement(
            track_id=track_id,
            assigned_class=resolved_class,
            usable_observation_count=usable_count,
            first_observed_frame=first_obs.frame_number,
            last_observed_frame=last_obs.frame_number,
            first_observed_time_seconds=first_obs.timestamp_seconds,
            last_observed_time_seconds=last_obs.timestamp_seconds,
            total_observed_duration_seconds=0.0,
            cumulative_pixel_displacement=round(cumulative_displacement, 3),
            net_pixel_displacement=round(net_displacement, 3),
            measurement_status=MotionMeasurementStatus.INVALID_TIMING,
            status_message=(
                "Observations could not be analyzed because elapsed times between "
                "consecutive frames were invalid, zero, or negative."
            ),
        )

    # Kinematic time-weighted average speed: total distance / total elapsed time
    avg_speed = cumulative_displacement / valid_elapsed_seconds_total
    max_speed = max(s.pixel_speed_px_per_sec for s in step_displacements)

    # Determine stationary status based on conservative image-space heuristics
    is_stationary = (
        cumulative_displacement <= eff_disp_thresh
        or (
            net_displacement <= eff_disp_thresh
            and avg_speed <= eff_speed_thresh
        )
    )

    if is_stationary:
        status = MotionMeasurementStatus.STATIONARY
        status_msg = (
            f"Vehicle classified as stationary under image sensor jitter heuristic "
            f"(displacement: {cumulative_displacement:.2f}px, speed: {avg_speed:.2f}px/s)."
        )
    else:
        status = MotionMeasurementStatus.VALID
        status_msg = (
            f"Valid image-plane pixel movement measured across {len(step_displacements)} steps "
            f"(cumulative: {cumulative_displacement:.2f}px, average: {avg_speed:.2f}px/s)."
        )

    return TrackMotionMeasurement(
        track_id=track_id,
        assigned_class=resolved_class,
        usable_observation_count=usable_count,
        first_observed_frame=first_obs.frame_number,
        last_observed_frame=last_obs.frame_number,
        first_observed_time_seconds=first_obs.timestamp_seconds,
        last_observed_time_seconds=last_obs.timestamp_seconds,
        total_observed_duration_seconds=round(valid_elapsed_seconds_total, 4),
        cumulative_pixel_displacement=round(cumulative_displacement, 3),
        net_pixel_displacement=round(net_displacement, 3),
        average_pixel_speed_px_per_sec=round(avg_speed, 3),
        max_pixel_speed_px_per_sec=round(max_speed, 3),
        step_displacements=step_displacements,
        measurement_status=status,
        status_message=status_msg,
        is_stationary=is_stationary,
        is_physical_speed_available=False,
    )


def analyze_video_motion(
    frame_results: Sequence[FrameDetectionResult],
    fps: float = DEFAULT_FALLBACK_FPS,
    tracklet_classes: dict[int, str] | None = None,
    filtered_track_ids: Sequence[int] | None = None,
    stationary_displacement_threshold_px: float = DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX,
    stationary_speed_threshold_px_per_sec: float = DEFAULT_STATIONARY_SPEED_THRESHOLD_PX_PER_SEC,
) -> VideoMotionSummary:
    """Analyze vehicle movement across all tracks in a video evaluation sequence.

    Args:
        frame_results: Chronological sequence of FrameDetectionResult instances.
        fps: Video framerate (FPS).
        tracklet_classes: Optional mapping of track_id to post-processed majority-voted class name.
        filtered_track_ids: Optional collection of track IDs that were filtered by post-processing.
        stationary_displacement_threshold_px: Maximum cumulative pixels to be considered stationary.
        stationary_speed_threshold_px_per_sec: Maximum pixel speed to be considered stationary.

    Returns:
        VideoMotionSummary containing individual track measurements and scene-level summaries.
    """
    effective_fps = fps if (math.isfinite(fps) and fps > 0.0) else DEFAULT_FALLBACK_FPS
    class_map = tracklet_classes or {}
    filtered_set = set(filtered_track_ids or [])

    # Group frame detections by track_id
    track_observations: dict[int, list[TrackObservationPoint]] = {}

    for frame in frame_results:
        for det in frame.detections:
            if det.track_id is None:
                continue

            try:
                cx, cy = compute_bounding_box_center(det.bounding_box)
                obs = TrackObservationPoint(
                    frame_number=frame.frame_number,
                    timestamp_seconds=frame.timestamp_seconds,
                    center_x=round(cx, 3),
                    center_y=round(cy, 3),
                    bounding_box=det.bounding_box,
                    confidence=det.confidence,
                    class_name=det.class_name,
                )
            except (ValueError, TypeError):
                # Safely ignore malformed or non-finite detection instances
                continue

            if det.track_id not in track_observations:
                track_observations[det.track_id] = []
            track_observations[det.track_id].append(obs)

    if not track_observations:
        return VideoMotionSummary(
            total_tracks_analyzed=0,
            valid_moving_tracks_count=0,
            stationary_tracks_count=0,
            insufficient_data_tracks_count=0,
            average_scene_pixel_speed_px_per_sec=None,
            weighted_scene_pixel_speed_px_per_sec=None,
            max_scene_pixel_speed_px_per_sec=None,
            is_physical_speed_available=False,
            calibration_status=CALIBRATION_REQUIREMENTS_SUMMARY,
            track_measurements=[],
        )

    track_measurements: list[TrackMotionMeasurement] = []
    moving_speeds: list[float] = []
    all_speeds: list[float] = []
    valid_moving_count = 0
    stationary_count = 0
    insufficient_count = 0

    for tid in sorted(track_observations.keys()):
        assigned_cls = class_map.get(tid)
        measurement = analyze_single_track_motion(
            track_id=tid,
            observations=track_observations[tid],
            fps=effective_fps,
            assigned_class=assigned_cls,
            stationary_displacement_threshold_px=stationary_displacement_threshold_px,
            stationary_speed_threshold_px_per_sec=stationary_speed_threshold_px_per_sec,
        )
        track_measurements.append(measurement)

        if measurement.measurement_status == MotionMeasurementStatus.VALID:
            valid_moving_count += 1
            if measurement.average_pixel_speed_px_per_sec is not None:
                moving_speeds.append(measurement.average_pixel_speed_px_per_sec)
            if measurement.max_pixel_speed_px_per_sec is not None:
                all_speeds.append(measurement.max_pixel_speed_px_per_sec)
        elif measurement.measurement_status == MotionMeasurementStatus.STATIONARY:
            stationary_count += 1
            if measurement.max_pixel_speed_px_per_sec is not None:
                all_speeds.append(measurement.max_pixel_speed_px_per_sec)
        else:
            insufficient_count += 1

    # Unweighted arithmetic mean across valid moving tracks
    avg_scene_speed = (
        round(sum(moving_speeds) / len(moving_speeds), 3) if moving_speeds else None
    )

    # Duration-weighted mean across valid moving tracks: total displacement / total duration
    valid_moving_total_disp = sum(
        m.cumulative_pixel_displacement
        for m in track_measurements
        if m.measurement_status == MotionMeasurementStatus.VALID
    )
    valid_moving_total_dur = sum(
        m.total_observed_duration_seconds
        for m in track_measurements
        if m.measurement_status == MotionMeasurementStatus.VALID
    )
    weighted_scene_speed = (
        round(valid_moving_total_disp / valid_moving_total_dur, 3)
        if valid_moving_total_dur > 0.0
        else None
    )

    max_scene_speed = round(max(all_speeds), 3) if all_speeds else None

    return VideoMotionSummary(
        total_tracks_analyzed=len(track_measurements),
        valid_moving_tracks_count=valid_moving_count,
        stationary_tracks_count=stationary_count,
        insufficient_data_tracks_count=insufficient_count,
        average_scene_pixel_speed_px_per_sec=avg_scene_speed,
        weighted_scene_pixel_speed_px_per_sec=weighted_scene_speed,
        max_scene_pixel_speed_px_per_sec=max_scene_speed,
        is_physical_speed_available=False,
        calibration_status=CALIBRATION_REQUIREMENTS_SUMMARY,
        track_measurements=track_measurements,
    )
