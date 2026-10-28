"""Offline tracklet post-processor for computer-vision vehicle detections.

Provides temporal majority voting for vehicle-class consistency across tracklets
and configurable minimum track lifespan filtering for transient detection pruning.

DISCLAIMER AND LIMITATIONS:
Temporal majority voting and short-track filtering are heuristic data-cleaning techniques.
They do NOT:
1. Prove that a tracklet corresponds to exactly one physical ground-truth vehicle.
2. Eliminate identity switches or track fragmentation (e.g., if tracking
   fragmentation occurs across separate time intervals, they remain distinct tracklets).
3. Certify ground-truth traffic congestion levels or pedestrian route accessibility.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.vision.schemas import (
    DetectedObject,
    FrameDetectionResult,
    VideoAnalysisSummary,
)

DEFAULT_MIN_TRACK_LIFESPAN_FRAMES: int = 5


class Tracklet(BaseModel):
    """Aggregated temporal trajectory representation for a single tracker ID."""

    model_config = ConfigDict(frozen=True)

    track_id: int = Field(
        ge=0,
        description="Tracker identifier assigned by multi-object tracking",
    )

    assigned_class: str = Field(
        min_length=1,
        description="Winning vehicle category determined by temporal majority voting",
    )

    original_classes: dict[str, int] = Field(
        default_factory=dict,
        description="Raw occurrence count per class observed for this track ID",
    )

    detection_count: int = Field(
        ge=1,
        description="Total detection bounding boxes belonging to this tracklet",
    )

    first_frame: int = Field(
        ge=0,
        description="Earliest video frame index where this tracklet was observed",
    )

    last_frame: int = Field(
        ge=0,
        description="Latest video frame index where this tracklet was observed",
    )

    lifespan_frames: int = Field(
        ge=1,
        description="Temporal frame span: (last_frame - first_frame + 1)",
    )

    observed_frame_count: int = Field(
        ge=1,
        description="Number of distinct video frames containing detections for this tracklet",
    )

    average_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean detection confidence score across all observations of this tracklet",
    )

    is_filtered: bool = Field(
        default=False,
        description="Flag indicating whether this tracklet was pruned due to insufficient lifespan",
    )

    has_class_switch: bool = Field(
        default=False,
        description=(
            "Flag indicating whether this tracklet had multiple distinct raw class labels "
            "(class label oscillation on a single track, distinct from tracking identity switches)"
        ),
    )


class PostProcessingSummary(BaseModel):
    """Comparative summary contrasting raw tracking metrics with post-processed metrics."""

    model_config = ConfigDict(frozen=True)

    # Raw metrics before post-processing
    raw_total_detections: int = Field(
        ge=0,
        description="Total detection instances in the raw input frames",
    )

    raw_total_tracked_ids: int = Field(
        ge=0,
        description="Distinct tracker IDs in the raw input",
    )

    raw_class_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Raw per-frame detection events by class",
    )

    raw_tracked_id_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of distinct tracker IDs observed per class before smoothing",
    )

    raw_class_switches_count: int = Field(
        ge=0,
        description="Number of tracklets that exhibited multi-class assignments in raw data",
    )

    # Post-processed metrics
    processed_total_detections: int = Field(
        ge=0,
        description="Total detection instances remaining after lifespan filtering",
    )

    processed_total_tracked_ids: int = Field(
        ge=0,
        description="Distinct tracker IDs remaining after lifespan filtering",
    )

    processed_class_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-frame detection events by class after majority voting and filtering",
    )

    processed_tracked_id_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of distinct tracker IDs per class after majority voting and filtering",
    )

    # Filter & resolution counters
    min_track_lifespan_frames: int = Field(
        ge=1,
        description="Configured minimum frame count threshold for track retention",
    )

    filtered_tracks_count: int = Field(
        ge=0,
        description="Number of short-lived tracklets pruned by lifespan filtering",
    )

    filtered_detections_count: int = Field(
        ge=0,
        description="Number of detection bounding boxes removed via lifespan filtering",
    )

    untracked_detections_count: int = Field(
        ge=0,
        description="Number of detection instances that lacked a persistent tracker ID",
    )

    resolved_class_switches_count: int = Field(
        ge=0,
        description=(
            "Number of multi-class tracklets whose oscillating labels were smoothed to a single "
            "category via majority voting (NOTE: this is label smoothing on a single track, "
            "NOT trajectory re-identification or identity-switch stitching across split tracks)"
        ),
    )

    disclaimer: str = Field(
        default=(
            "Post-processing heuristics (temporal majority voting, short-track filtering) "
            "improve class consistency and prune transient detections. Temporal majority voting "
            "resolves intra-tracklet class label instability (smoothing oscillating predictions "
            "on a single tracker ID). It does NOT perform tracklet re-identification (stitching "
            "split tracklets) and does NOT correct identity switches where one physical vehicle "
            "receives multiple tracker IDs. It does NOT establish ground-truth traffic congestion "
            "or certify pedestrian route accessibility."
        ),
        description="Limitations of post-processing heuristics",
    )

    @model_validator(mode="after")
    def validate_metric_conservation(self) -> PostProcessingSummary:
        """Enforce mathematical consistency between raw and processed metrics."""
        expected_raw_detections = self.processed_total_detections + self.filtered_detections_count
        if self.raw_total_detections != expected_raw_detections:
            raise ValueError(
                f"raw_total_detections ({self.raw_total_detections}) does not equal "
                f"processed_total_detections ({self.processed_total_detections}) + "
                f"filtered_detections_count ({self.filtered_detections_count})."
            )

        expected_raw_tracks = self.processed_total_tracked_ids + self.filtered_tracks_count
        if self.raw_total_tracked_ids != expected_raw_tracks:
            raise ValueError(
                f"raw_total_tracked_ids ({self.raw_total_tracked_ids}) does not equal "
                f"processed_total_tracked_ids ({self.processed_total_tracked_ids}) + "
                f"filtered_tracks_count ({self.filtered_tracks_count})."
            )

        if sum(self.raw_class_counts.values()) != self.raw_total_detections:
            raise ValueError("Sum of raw_class_counts must equal raw_total_detections.")

        if sum(self.processed_class_counts.values()) != self.processed_total_detections:
            raise ValueError("Sum of processed_class_counts must equal processed_total_detections.")

        if sum(self.processed_tracked_id_counts.values()) != self.processed_total_tracked_ids:
            raise ValueError(
                "Sum of processed_tracked_id_counts must equal processed_total_tracked_ids "
                "because each retained track has exactly one assigned class."
            )

        return self


class PostProcessedVideoResult(BaseModel):
    """Structured container encapsulating tracklets, cleaned frames, and comparison summary."""

    model_config = ConfigDict(frozen=True)

    summary: PostProcessingSummary = Field(
        description="Comparative before-and-after tracking and detection summary",
    )

    tracklets: dict[int, Tracklet] = Field(
        default_factory=dict,
        description="All evaluated tracklets keyed by tracker ID (including filtered tracks)",
    )

    filtered_track_ids: list[int] = Field(
        default_factory=list,
        description="Tracker IDs pruned due to lifespan shorter than threshold",
    )

    frames: list[FrameDetectionResult] = Field(
        default_factory=list,
        description="Post-processed video frame detection results with smoothed classes",
    )


class TrackletPostProcessor:
    """Offline processor applying temporal majority voting and track lifespan filtering.

    Operates on sequences of FrameDetectionResult instances. Does not mutate input objects.

    Determinism Rules for Majority Voting:
    1. Primary: Class with the highest occurrence frequency across the tracklet.
    2. Secondary tie-breaker: Class with the highest cumulative confidence score.
    3. Tertiary tie-breaker: Class with the highest single-frame maximum confidence.
    4. Quaternary tie-breaker: Lexicographical order of class name (e.g. 'bus' < 'car').
    """

    def __init__(
        self,
        min_track_lifespan_frames: int = DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
        filter_short_tracks: bool = True,
    ) -> None:
        """Initialize post-processor settings.

        Args:
            min_track_lifespan_frames: Minimum number of distinct frames a track must
                appear in to be retained. Defaults to 5 (~0.16s at 30 FPS). Must be >= 1.
            filter_short_tracks: If True, tracks with observed_frame_count < threshold
                are marked as filtered and excluded from the post-processed frame results.
                If False, short tracks are retained with their majority-voted class.

        Raises:
            ValueError: If min_track_lifespan_frames < 1.
        """
        if min_track_lifespan_frames < 1:
            raise ValueError(
                f"min_track_lifespan_frames must be at least 1, got {min_track_lifespan_frames}."
            )

        self._min_track_lifespan_frames = min_track_lifespan_frames
        self._filter_short_tracks = filter_short_tracks

    @property
    def min_track_lifespan_frames(self) -> int:
        """Minimum frame lifespan required for track retention."""
        return self._min_track_lifespan_frames

    @property
    def filter_short_tracks(self) -> bool:
        """Whether short-lived tracks are pruned from processed frames."""
        return self._filter_short_tracks

    def process_frames(
        self,
        frame_results: Sequence[FrameDetectionResult],
        raw_summary: VideoAnalysisSummary | None = None,
    ) -> PostProcessedVideoResult:
        """Process frame detection results and return a PostProcessedVideoResult.

        Args:
            frame_results: Sequence of FrameDetectionResult instances (unmodified).
            raw_summary: Optional VideoAnalysisSummary for cross-validation.

        Returns:
            PostProcessedVideoResult containing tracklets, comparative metrics,
            and post-processed frame detection results.
        """
        # 1. Group detections by track_id
        # Structure: track_id -> list of (frame_index, frame_number, DetectedObject)
        track_observations: dict[int, list[tuple[int, int, DetectedObject]]] = defaultdict(list)
        untracked_count = 0
        raw_class_counts: dict[str, int] = defaultdict(int)
        raw_tracked_ids_by_class: dict[str, set[int]] = defaultdict(set)
        raw_total_detections = 0

        for frame_idx, frame in enumerate(frame_results):
            for det in frame.detections:
                raw_total_detections += 1
                raw_class_counts[det.class_name] += 1

                if det.track_id is None:
                    untracked_count += 1
                else:
                    track_observations[det.track_id].append(
                        (frame_idx, frame.frame_number, det)
                    )
                    raw_tracked_ids_by_class[det.class_name].add(det.track_id)

        raw_total_tracked_ids = len(track_observations)
        raw_tracked_id_counts = {
            cls: len(ids) for cls, ids in raw_tracked_ids_by_class.items()
        }

        # 2. Build Tracklets with temporal majority voting and tie-breaking
        tracklets: dict[int, Tracklet] = {}
        filtered_track_ids: list[int] = []
        raw_class_switches_count = 0

        for track_id, observations in track_observations.items():
            # Tally occurrences, confidence sums, and max confidence per class
            class_counts: dict[str, int] = defaultdict(int)
            class_conf_sums: dict[str, float] = defaultdict(float)
            class_conf_maxes: dict[str, float] = defaultdict(float)
            total_conf = 0.0

            appearance_frames: set[int] = set()

            for _, frame_no, det in observations:
                class_counts[det.class_name] += 1
                class_conf_sums[det.class_name] += det.confidence
                if det.confidence > class_conf_maxes[det.class_name]:
                    class_conf_maxes[det.class_name] = det.confidence
                total_conf += det.confidence
                appearance_frames.add(frame_no)

            # Deterministic majority voting sorting key:
            # (-occurrence_count, -sum_confidence, -max_confidence, class_name)
            sorted_classes = sorted(
                class_counts.keys(),
                key=lambda c: (
                    -class_counts[c],
                    -class_conf_sums[c],
                    -class_conf_maxes[c],
                    c,
                ),
            )
            winning_class = sorted_classes[0]

            detection_count = len(observations)
            avg_confidence = total_conf / detection_count if detection_count > 0 else 0.0
            first_frame = min(r[1] for r in observations)
            last_frame = max(r[1] for r in observations)
            lifespan_frames = last_frame - first_frame + 1
            observed_frame_count = len(appearance_frames)

            has_switch = len(class_counts) > 1
            if has_switch:
                raw_class_switches_count += 1

            # Check lifespan threshold
            is_filtered = (
                self._filter_short_tracks
                and observed_frame_count < self._min_track_lifespan_frames
            )

            if is_filtered:
                filtered_track_ids.append(track_id)

            tracklets[track_id] = Tracklet(
                track_id=track_id,
                assigned_class=winning_class,
                original_classes=dict(class_counts),
                detection_count=detection_count,
                first_frame=first_frame,
                last_frame=last_frame,
                lifespan_frames=lifespan_frames,
                observed_frame_count=observed_frame_count,
                average_confidence=avg_confidence,
                is_filtered=is_filtered,
                has_class_switch=has_switch,
            )

        filtered_track_ids.sort()

        # 3. Generate cleaned post-processed frames (input frames remain immutable)
        post_processed_frames: list[FrameDetectionResult] = []
        filtered_detections_count = 0
        processed_total_detections = 0
        processed_class_counts: dict[str, int] = defaultdict(int)
        processed_tracked_ids_by_class: dict[str, set[int]] = defaultdict(set)
        resolved_class_switches_count = 0

        for frame in frame_results:
            cleaned_detections: list[DetectedObject] = []

            for det in frame.detections:
                if det.track_id is None:
                    # Untracked detections are preserved as-is
                    cleaned_detections.append(det)
                    processed_total_detections += 1
                    processed_class_counts[det.class_name] += 1
                else:
                    tracklet = tracklets[det.track_id]
                    if tracklet.is_filtered:
                        # Exclude filtered transient detections
                        filtered_detections_count += 1
                    else:
                        processed_total_detections += 1
                        processed_class_counts[tracklet.assigned_class] += 1
                        processed_tracked_ids_by_class[tracklet.assigned_class].add(det.track_id)

                        if det.class_name != tracklet.assigned_class:
                            # Reclassify to winning class
                            cleaned_detections.append(
                                DetectedObject(
                                    class_name=tracklet.assigned_class,
                                    confidence=det.confidence,
                                    bounding_box=det.bounding_box,
                                    track_id=det.track_id,
                                )
                            )
                        else:
                            cleaned_detections.append(det)

            post_processed_frames.append(
                FrameDetectionResult.from_detections(
                    frame_number=frame.frame_number,
                    timestamp_seconds=frame.timestamp_seconds,
                    detections=cleaned_detections,
                )
            )

        # Count resolved multi-class tracklets among retained tracks
        for t in tracklets.values():
            if not t.is_filtered and t.has_class_switch:
                resolved_class_switches_count += 1

        retained_tracked_ids = [t for t in tracklets.values() if not t.is_filtered]
        processed_total_tracked_ids = len(retained_tracked_ids)
        processed_tracked_id_counts = {
            cls: len(ids) for cls, ids in processed_tracked_ids_by_class.items()
        }

        summary = PostProcessingSummary(
            raw_total_detections=raw_total_detections,
            raw_total_tracked_ids=raw_total_tracked_ids,
            raw_class_counts=dict(raw_class_counts),
            raw_tracked_id_counts=raw_tracked_id_counts,
            raw_class_switches_count=raw_class_switches_count,
            processed_total_detections=processed_total_detections,
            processed_total_tracked_ids=processed_total_tracked_ids,
            processed_class_counts=dict(processed_class_counts),
            processed_tracked_id_counts=processed_tracked_id_counts,
            min_track_lifespan_frames=self._min_track_lifespan_frames,
            filtered_tracks_count=len(filtered_track_ids),
            filtered_detections_count=filtered_detections_count,
            untracked_detections_count=untracked_count,
            resolved_class_switches_count=resolved_class_switches_count,
        )

        return PostProcessedVideoResult(
            summary=summary,
            tracklets=tracklets,
            filtered_track_ids=filtered_track_ids,
            frames=post_processed_frames,
        )
