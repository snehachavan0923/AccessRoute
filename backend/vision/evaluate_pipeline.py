"""Offline evaluation script for the AccessRoute computer-vision pipeline.

Compares raw YOLO11n + ByteTrack detections against TrackletPostProcessor
cleaned results for a configurable, limited number of video frames.

DISCLAIMER AND LIMITATIONS:
1. Detection events count instantaneous per-frame bounding boxes. A stationary
   car at 30 FPS generates 30 detection events per second.
2. Tracker IDs represent algorithmic trajectories assigned by ByteTrack.
   They are subject to track fragmentation and ID switches.
3. Neither detection events nor tracker IDs represent verified ground-truth vehicles.
4. Temporal majority voting resolves intra-tracklet class label instability
   (smoothing oscillating predictions on a single tracker ID). It does NOT
   merge fragmented tracklets or correct identity switches where one physical
   vehicle receives multiple tracker IDs across separate time intervals.
5. Computer vision observations do not establish road congestion or certify
   pedestrian route accessibility without geometric camera-to-roadway calibration
   and empirical capacity modeling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from backend.vision.motion_analyzer import (
    CALIBRATION_REQUIREMENTS_SUMMARY,
    TrackMotionMeasurement,
    VideoMotionSummary,
    analyze_video_motion,
)
from backend.vision.post_processor import (
    DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
    PostProcessedVideoResult,
    PostProcessingSummary,
    Tracklet,
    TrackletPostProcessor,
)
from backend.vision.schemas import (
    BoundingBox,
    DetectedObject,
    FrameDetectionResult,
    VideoAnalysisSummary,
)
from backend.vision.service import (
    DEFAULT_MODEL_NAME,
    VideoAnalysisService,
)

# Conservative default frame count to prevent heavy full-video execution
DEFAULT_EVAL_MAX_FRAMES: int = 30


class ReclassifiedDetection(BaseModel):
    """Record of a per-frame detection event whose class was smoothed by majority voting."""

    model_config = ConfigDict(frozen=True)

    frame_number: int = Field(ge=0, description="Zero-based video frame index")
    timestamp_seconds: float = Field(ge=0.0, description="Elapsed timestamp in seconds")
    track_id: int = Field(ge=0, description="Tracker ID assigned by ByteTrack")
    raw_class: str = Field(description="Raw class predicted on this frame by YOLO")
    assigned_class: str = Field(description="Winning class assigned by temporal majority voting")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")


class TrackletDetail(BaseModel):
    """Detailed summary of a single vehicle tracklet evaluated by the pipeline."""

    model_config = ConfigDict(frozen=True)

    track_id: int = Field(ge=0, description="Tracker identifier")
    assigned_class: str = Field(description="Majority-voted vehicle class")
    original_classes: dict[str, int] = Field(description="Raw occurrence count per class")
    detection_count: int = Field(ge=1, description="Total detection boxes for this tracklet")
    first_frame: int = Field(ge=0, description="Earliest frame index observed")
    last_frame: int = Field(ge=0, description="Latest frame index observed")
    lifespan_frames: int = Field(ge=1, description="Frame span: last_frame - first_frame + 1")
    observed_frame_count: int = Field(ge=1, description="Distinct frames with detections")
    average_confidence: float = Field(ge=0.0, le=1.0, description="Mean detection confidence")
    is_filtered: bool = Field(description="True if pruned by minimum lifespan threshold")
    has_class_switch: bool = Field(description="True if tracklet had multiple raw classes")
    motion: TrackMotionMeasurement | None = Field(
        default=None,
        description="Optional image-plane motion and pixel displacement measurement",
    )


class PipelineEvaluationReport(BaseModel):
    """Comprehensive evaluation report contrasting raw vs post-processed vision results."""

    model_config = ConfigDict(frozen=True)

    evaluated_at_utc: str = Field(description="ISO-8601 UTC timestamp of evaluation run")
    video_name: str = Field(description="Name of the video file")
    video_path: str = Field(description="Path to the evaluated video file")
    total_frames_in_video: int = Field(ge=0, description="Total frame count of source video")
    processed_frames: int = Field(ge=0, description="Number of frames evaluated in this run")
    video_duration_seconds: float = Field(ge=0.0, description="Total duration of source video")
    video_fps: float = Field(ge=0.0, description="Framerate of source video")

    # Evaluation hyperparameters
    conf_threshold: float = Field(ge=0.0, le=1.0, description="YOLO detection confidence threshold")
    iou_threshold: float = Field(ge=0.0, le=1.0, description="NMS IoU threshold")
    min_track_lifespan_frames: int = Field(ge=1, description="Minimum observed frames to retain track")

    # Raw metrics
    raw_total_detections: int = Field(ge=0, description="Total raw detection events across processed frames")
    raw_total_tracked_ids: int = Field(ge=0, description="Total distinct tracker IDs in raw detections")
    raw_detection_events_by_class: dict[str, int] = Field(default_factory=dict, description="Raw detections by class")
    raw_tracked_ids_by_class: dict[str, int] = Field(default_factory=dict, description="Distinct tracker IDs per class")
    raw_track_observed_frames: dict[int, int] = Field(default_factory=dict, description="Observed frames per tracker ID")
    raw_class_switches_count: int = Field(ge=0, description="Number of tracks with multi-class raw predictions")

    # Processed metrics
    processed_total_detections: int = Field(ge=0, description="Total detections remaining after filtering")
    processed_total_tracked_ids: int = Field(ge=0, description="Total distinct tracker IDs retained")
    processed_detection_events_by_class: dict[str, int] = Field(default_factory=dict, description="Processed detections by class")
    processed_tracked_ids_by_class: dict[str, int] = Field(default_factory=dict, description="Processed tracker IDs by class")
    filtered_tracks_count: int = Field(ge=0, description="Number of short-lived tracks pruned")
    filtered_detections_count: int = Field(ge=0, description="Number of detection boxes removed via filtering")
    filtered_track_ids: list[int] = Field(default_factory=list, description="List of pruned tracker IDs")
    untracked_detections_count: int = Field(ge=0, description="Detections lacking a tracker ID")
    resolved_class_switches_count: int = Field(ge=0, description="Multi-class tracks smoothed to single class")

    # Granular details
    tracklets: list[TrackletDetail] = Field(default_factory=list, description="Detailed tracklet records")
    reclassified_detections: list[ReclassifiedDetection] = Field(default_factory=list, description="Specific detections reclassified")
    motion_summary: VideoMotionSummary | None = Field(
        default=None,
        description="Aggregate image-plane motion analysis across tracked vehicles",
    )

    disclaimer: str = Field(
        default=(
            "Offline vision evaluation tool for research purposes. "
            "Detection events measure per-frame bounding boxes; tracker IDs measure spatial trajectories. "
            "Neither metric represents verified ground-truth physical vehicle counts. "
            "Temporal majority voting resolves intra-tracklet label oscillation on a single tracker ID; "
            "it does NOT stitch fragmented tracklets or resolve identity switches where one vehicle receives multiple IDs. "
            "Vision observations do not certify traffic congestion levels or pedestrian route accessibility."
        ),
        description="Limitations and academic disclaimer",
    )


def evaluate_frames(
    frame_results: Sequence[FrameDetectionResult],
    video_metadata: dict[str, Any] | None = None,
    min_track_lifespan_frames: int = DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
    filter_short_tracks: bool = True,
    conf_threshold: float = 0.50,
    iou_threshold: float = 0.30,
    video_path: str = "",
) -> PipelineEvaluationReport:
    """Evaluate frame detection results comparing raw tracking against post-processing.

    This function operates purely on extracted FrameDetectionResult objects without
    requiring video decoding or neural network inference, making it fully unit-testable.

    Args:
        frame_results: Sequence of FrameDetectionResult instances.
        video_metadata: Optional dictionary with video attributes (fps, duration, etc.).
        min_track_lifespan_frames: Minimum frame count to retain a tracklet.
        filter_short_tracks: Whether to prune short-lived tracklets.
        conf_threshold: Configured confidence threshold used during detection.
        iou_threshold: Configured IoU threshold used during detection.
        video_path: Source video path for reporting purposes.

    Returns:
        PipelineEvaluationReport containing side-by-side comparative analysis.
    """
    post_processor = TrackletPostProcessor(
        min_track_lifespan_frames=min_track_lifespan_frames,
        filter_short_tracks=filter_short_tracks,
    )
    processed_result: PostProcessedVideoResult = post_processor.process_frames(frame_results)
    summary: PostProcessingSummary = processed_result.summary

    meta = video_metadata or {}
    total_frames = meta.get("frame_count", len(frame_results))
    fps = meta.get("fps", 29.97)
    duration_seconds = meta.get(
        "duration_seconds",
        (total_frames / fps) if fps > 0.0 else 0.0,
    )
    video_name = meta.get("video_name", Path(video_path).name if video_path else "in_memory_frames")

    # Map raw observed frame count per tracker ID
    raw_track_observed_frames: dict[int, int] = {
        t.track_id: t.observed_frame_count for t in processed_result.tracklets.values()
    }

    # Analyze vehicle image-plane motion across frames
    tracklet_classes = {
        t.track_id: t.assigned_class for t in processed_result.tracklets.values()
    }
    motion_summary = analyze_video_motion(
        frame_results=frame_results,
        fps=fps,
        tracklet_classes=tracklet_classes,
        filtered_track_ids=processed_result.filtered_track_ids,
    )
    motion_by_track = {m.track_id: m for m in motion_summary.track_measurements}

    # Identify individual detection events whose class was reclassified
    reclassified_detections: list[ReclassifiedDetection] = []
    for frame in frame_results:
        for det in frame.detections:
            if det.track_id is not None:
                tracklet = processed_result.tracklets.get(det.track_id)
                if (
                    tracklet is not None
                    and not tracklet.is_filtered
                    and det.class_name != tracklet.assigned_class
                ):
                    reclassified_detections.append(
                        ReclassifiedDetection(
                            frame_number=frame.frame_number,
                            timestamp_seconds=frame.timestamp_seconds,
                            track_id=det.track_id,
                            raw_class=det.class_name,
                            assigned_class=tracklet.assigned_class,
                            confidence=det.confidence,
                        )
                    )

    # Convert tracklets to serializable TrackletDetail records
    tracklet_details: list[TrackletDetail] = []
    for tid in sorted(processed_result.tracklets.keys()):
        t = processed_result.tracklets[tid]
        tracklet_details.append(
            TrackletDetail(
                track_id=t.track_id,
                assigned_class=t.assigned_class,
                original_classes=t.original_classes,
                detection_count=t.detection_count,
                first_frame=t.first_frame,
                last_frame=t.last_frame,
                lifespan_frames=t.lifespan_frames,
                observed_frame_count=t.observed_frame_count,
                average_confidence=t.average_confidence,
                is_filtered=t.is_filtered,
                has_class_switch=t.has_class_switch,
                motion=motion_by_track.get(t.track_id),
            )
        )

    return PipelineEvaluationReport(
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        video_name=video_name,
        video_path=str(video_path),
        total_frames_in_video=total_frames,
        processed_frames=len(frame_results),
        video_duration_seconds=duration_seconds,
        video_fps=fps,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        min_track_lifespan_frames=min_track_lifespan_frames,
        raw_total_detections=summary.raw_total_detections,
        raw_total_tracked_ids=summary.raw_total_tracked_ids,
        raw_detection_events_by_class=summary.raw_class_counts,
        raw_tracked_ids_by_class=summary.raw_tracked_id_counts,
        raw_track_observed_frames=raw_track_observed_frames,
        raw_class_switches_count=summary.raw_class_switches_count,
        processed_total_detections=summary.processed_total_detections,
        processed_total_tracked_ids=summary.processed_total_tracked_ids,
        processed_detection_events_by_class=summary.processed_class_counts,
        processed_tracked_ids_by_class=summary.processed_tracked_id_counts,
        filtered_tracks_count=summary.filtered_tracks_count,
        filtered_detections_count=summary.filtered_detections_count,
        filtered_track_ids=processed_result.filtered_track_ids,
        untracked_detections_count=summary.untracked_detections_count,
        resolved_class_switches_count=summary.resolved_class_switches_count,
        tracklets=tracklet_details,
        reclassified_detections=reclassified_detections,
        motion_summary=motion_summary,
    )


def evaluate_video(
    video_path: str | Path,
    service: VideoAnalysisService | None = None,
    max_frames: int | None = None,
    min_track_lifespan_frames: int = DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
    conf: float = 0.50,
    iou: float = 0.30,
    tracker: str = "bytetrack.yaml",
) -> PipelineEvaluationReport:
    """Run offline evaluation on a video file.

    Args:
        video_path: Path to the target video file.
        service: Optional pre-configured VideoAnalysisService instance.
        max_frames: Upper limit on frames to evaluate (default: None, processes entire video).
            When specified, must be >= 1.
        min_track_lifespan_frames: Minimum frame count for tracklet retention (default: 5).
        conf: YOLO confidence threshold (default: 0.50).
        iou: NMS IoU threshold (default: 0.30).
        tracker: Tracker configuration filename (default: 'bytetrack.yaml').

    Returns:
        PipelineEvaluationReport comparing raw and processed results.

    Raises:
        FileNotFoundError: If video file or model weights do not exist.
        ValueError: If max_frames < 1 or video cannot be opened.
    """
    if not str(video_path).strip():
        raise ValueError("video_path must not be empty or whitespace-only.")

    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found at '{v_path}'.")
    if not v_path.is_file():
        raise ValueError(f"Video path '{v_path}' is not a regular file.")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be at least 1, got {max_frames}.")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"conf must be between 0.0 and 1.0, got {conf}.")
    if not (0.0 <= iou <= 1.0):
        raise ValueError(f"iou must be between 0.0 and 1.0, got {iou}.")
    if min_track_lifespan_frames < 1:
        raise ValueError(f"min_track_lifespan_frames must be at least 1, got {min_track_lifespan_frames}.")

    if service is None:
        service = VideoAnalysisService()

    # Extract source video parameters
    metadata = service.extract_video_metadata(v_path)

    # Run detection on limited frames
    _, frame_results = service.process_video(
        video_path=v_path,
        conf=conf,
        iou=iou,
        tracker=tracker,
        max_frames=max_frames,
    )

    if len(frame_results) == 0:
        raise ValueError(
            f"No frames could be read or decoded from '{v_path}'. "
            "Please verify that the video is valid and not corrupted."
        )

    return evaluate_frames(
        frame_results=frame_results,
        video_metadata=metadata,
        min_track_lifespan_frames=min_track_lifespan_frames,
        conf_threshold=conf,
        iou_threshold=iou,
        video_path=str(v_path),
    )


def format_evaluation_report(report: PipelineEvaluationReport) -> str:
    """Format a PipelineEvaluationReport into a structured, human-readable text document.

    Args:
        report: PipelineEvaluationReport instance.

    Returns:
        Multi-line formatted text report suitable for terminal viewing or file saving.
    """
    lines: list[str] = []
    w = 80
    sep = "=" * w
    sub_sep = "-" * w

    lines.append(sep)
    lines.append("ACCESSROUTE COMPUTER-VISION PIPELINE EVALUATION REPORT")
    lines.append(sep)
    lines.append(f"Evaluated At (UTC) : {report.evaluated_at_utc}")
    lines.append(f"Video Source       : {report.video_path}")
    lines.append(f"Video Filename     : {report.video_name}")
    pct_processed = (
        (report.processed_frames / report.total_frames_in_video * 100.0)
        if report.total_frames_in_video > 0
        else 0.0
    )
    mode_tag = (
        "[Full Video Mode]"
        if report.processed_frames >= report.total_frames_in_video
        else "[Conservative Sampling Mode]"
    )
    lines.append(
        f"Video Profile      : {report.total_frames_in_video} total frames | "
        f"{report.video_duration_seconds:.2f}s duration | {report.video_fps:.2f} FPS"
    )
    lines.append(
        f"Evaluation Sample  : {report.processed_frames} frames processed "
        f"({pct_processed:.1f}% of video) {mode_tag}"
    )
    lines.append(
        f"Detector Settings  : conf={report.conf_threshold:.2f}, "
        f"iou={report.iou_threshold:.2f}"
    )
    lines.append(
        f"Post-Processor     : min_track_lifespan_frames={report.min_track_lifespan_frames}"
    )
    lines.append("")

    # Section 1: Executive Comparison Summary
    lines.append(sub_sep)
    lines.append("1. EXECUTIVE COMPARISON SUMMARY (RAW vs. POST-PROCESSED)")
    lines.append(sub_sep)
    lines.append(f"{'Metric':<36} {'Raw Input':<14} {'Processed':<14} {'Delta / Change'}")
    lines.append("-" * w)

    det_delta = report.processed_total_detections - report.raw_total_detections
    lines.append(
        f"{'Total Detection Events':<36} {report.raw_total_detections:<14} "
        f"{report.processed_total_detections:<14} {det_delta:+d} (filtered boxes)"
    )

    track_delta = report.processed_total_tracked_ids - report.raw_total_tracked_ids
    lines.append(
        f"{'Total Distinct Tracker IDs':<36} {report.raw_total_tracked_ids:<14} "
        f"{report.processed_total_tracked_ids:<14} {track_delta:+d} (pruned tracks)"
    )

    lines.append(
        f"{'Filtered Short-Lived Tracks':<36} {'0':<14} "
        f"{report.filtered_tracks_count:<14} +{report.filtered_tracks_count} pruned"
    )

    lines.append(
        f"{'Filtered Transient Detections':<36} {'0':<14} "
        f"{report.filtered_detections_count:<14} +{report.filtered_detections_count} pruned"
    )

    lines.append(
        f"{'Untracked Detections (track_id=None)':<36} {report.untracked_detections_count:<14} "
        f"{report.untracked_detections_count:<14} preserved as-is"
    )

    filtered_class_switches = (
        report.raw_class_switches_count - report.resolved_class_switches_count
    )
    if filtered_class_switches > 0:
        switches_desc = (
            f"-{report.resolved_class_switches_count} smoothed "
            f"(-{filtered_class_switches} pruned with short tracks)"
        )
    else:
        switches_desc = f"-{report.resolved_class_switches_count} smoothed"

    lines.append(
        f"{'Multi-Class Oscillating Tracks':<36} {report.raw_class_switches_count:<14} "
        f"{'0':<14} {switches_desc}"
    )

    lines.append("")
    lines.append("Detection Events by Vehicle Class (Per-Frame Bounding Boxes):")
    all_classes = sorted(
        set(report.raw_detection_events_by_class.keys())
        | set(report.processed_detection_events_by_class.keys())
    )
    if not all_classes:
        lines.append("  (No vehicle detections observed)")
    else:
        for c in all_classes:
            r_c = report.raw_detection_events_by_class.get(c, 0)
            p_c = report.processed_detection_events_by_class.get(c, 0)
            diff = p_c - r_c
            lines.append(f"  - {c:<15}: Raw = {r_c:<5} | Processed = {p_c:<5} (delta: {diff:+d})")

    lines.append("")
    lines.append("Distinct Tracker IDs by Vehicle Class (Unique Tracks per Class):")
    tracked_classes = sorted(
        set(report.raw_tracked_ids_by_class.keys())
        | set(report.processed_tracked_ids_by_class.keys())
    )
    if not tracked_classes:
        lines.append("  (No persistent tracks observed)")
    else:
        for c in tracked_classes:
            r_t = report.raw_tracked_ids_by_class.get(c, 0)
            p_t = report.processed_tracked_ids_by_class.get(c, 0)
            diff = p_t - r_t
            lines.append(f"  - {c:<15}: Raw = {r_t:<5} | Processed = {p_t:<5} (delta: {diff:+d})")

    lines.append("")

    # Section 2: Raw Tracker IDs & Observed Frame Counts
    lines.append(sub_sep)
    lines.append("2. RAW TRACKER IDS & OBSERVED FRAME COUNTS")
    lines.append(sub_sep)
    lines.append(
        f"{'Track ID':<10} {'Assigned Class':<16} {'Observed Frames':<18} "
        f"{'Frame Span':<18} {'Avg Conf':<10} {'Filter Status'}"
    )
    lines.append("-" * w)

    if not report.tracklets:
        lines.append("  (No tracklets recorded in this evaluation run)")
    else:
        for t in report.tracklets:
            status = "FILTERED (< threshold)" if t.is_filtered else "RETAINED"
            span_str = f"f{t.first_frame}-f{t.last_frame} ({t.lifespan_frames}f)"
            obs_str = f"{t.observed_frame_count} frames"
            lines.append(
                f"{t.track_id:<10} {t.assigned_class:<16} {obs_str:<18} "
                f"{span_str:<18} {t.average_confidence:.2f}{'':<6} {status}"
            )

    lines.append("")

    # Section 3: Tracklet Filtering Details
    lines.append(sub_sep)
    lines.append("3. TRACKLET POST-PROCESSING & FILTERING DETAILS")
    lines.append(sub_sep)
    lines.append(
        f"- Min Track Lifespan Threshold: {report.min_track_lifespan_frames} frames "
        f"(tracks with observed_frame_count < {report.min_track_lifespan_frames} are pruned)"
    )
    lines.append(f"- Filtered Track IDs Count    : {report.filtered_tracks_count}")
    lines.append(
        f"- Filtered Tracker ID List    : "
        f"{report.filtered_track_ids if report.filtered_track_ids else 'None (all tracks retained)'}"
    )
    lines.append(
        f"- Filtered Detections Count   : {report.filtered_detections_count} bounding boxes"
    )
    lines.append("")

    # Section 4: Class Consistency & Label Smoothing
    lines.append(sub_sep)
    lines.append("4. TEMPORAL MAJORITY VOTING & LABEL SMOOTHING")
    lines.append(sub_sep)
    filtered_switches = report.raw_class_switches_count - report.resolved_class_switches_count
    lines.append(
        f"- Multi-Class Oscillating Tracks: {report.raw_class_switches_count} raw "
        f"({report.resolved_class_switches_count} smoothed on retained tracks, "
        f"{filtered_switches} pruned with short-lived tracks)"
    )

    multi_class_tracks = [t for t in report.tracklets if t.has_class_switch]
    if not multi_class_tracks:
        lines.append("  (No multi-class oscillating tracklets observed in this run)")
    else:
        for t in multi_class_tracks:
            classes_str = ", ".join(f"{k}: {v}" for k, v in t.original_classes.items())
            if t.is_filtered:
                lines.append(
                    f"  * Track #{t.track_id} [PRUNED / FILTERED]: Raw class counts = {{{classes_str}}} "
                    f"(Observed in {t.observed_frame_count} frames < min {report.min_track_lifespan_frames}; "
                    f"pruned along with track, NOT included in processed detections)"
                )
            else:
                lines.append(
                    f"  * Track #{t.track_id} [RETAINED / SMOOTHED]: Raw class counts = {{{classes_str}}} "
                    f"-> Smoothed to '{t.assigned_class}' via majority voting"
                )

    lines.append("")
    lines.append(
        f"- Reclassified Detection Instances: {len(report.reclassified_detections)} events"
    )
    if not report.reclassified_detections:
        lines.append("  (No detection instances required reclassification)")
    else:
        for rd in report.reclassified_detections:
            lines.append(
                f"  * Frame {rd.frame_number:03d} (t={rd.timestamp_seconds:.2f}s) | "
                f"Track #{rd.track_id}: '{rd.raw_class}' -> '{rd.assigned_class}' "
                f"(conf={rd.confidence:.2f})"
            )

    lines.append("")

    # Section 5: Metric Definitions & Educational Guide
    lines.append(sub_sep)
    lines.append("5. METRIC DEFINITIONS & EDUCATIONAL GUIDE")
    lines.append(sub_sep)
    lines.append(
        "1. Frame Detection Event:\n"
        "   An instantaneous bounding box output by YOLO on a single frame.\n"
        "   A stationary vehicle observed across 30 consecutive frames generates\n"
        "   30 detection events. Comparing instantaneous per-frame detection events\n"
        "   directly to tracker IDs is a dimensional category error."
    )
    lines.append(
        "2. Tracker ID:\n"
        "   A spatial trajectory identifier assigned by ByteTrack using spatial\n"
        "   IoU association and Kalman filter state estimation across frames.\n"
        "   Tracker IDs are susceptible to fragmentation and occlusion."
    )
    lines.append(
        "3. Observed Frame Count:\n"
        "   The count of distinct video frames in which a tracklet was actually\n"
        "   detected. Lifespan filtering uses observed_frame_count, NOT the frame\n"
        "   span (last_frame - first_frame + 1), to prevent temporal gaps from\n"
        "   falsely inflating tracklet persistence."
    )
    lines.append(
        "4. Temporal Majority Voting:\n"
        "   Resolves intra-tracklet class label instability by assigning the\n"
        "   statistically dominant category to all observations of that tracklet.\n"
        "   Tie-breaking is strictly deterministic: frequency -> confidence sum\n"
        "   -> max confidence -> alphabetical class name."
    )
    lines.append(
        "5. Minimum Lifespan Filtering:\n"
        "   Prunes transient, low-persistence detections (often false positives or\n"
        "   edge glitches) that appear in fewer than N distinct frames."
    )
    lines.append(
        "6. Verified Ground-Truth Vehicles:\n"
        "   The actual number of distinct physical motor vehicles present in the\n"
        "   real world. Neither YOLO nor ByteTrack outputs verified ground truth;\n"
        "   ground truth requires human manual audit or calibrated sensor benchmarks."
    )
    lines.append("")

    # Section 6: Research Limitations & Non-Claims
    lines.append(sub_sep)
    lines.append("6. RESEARCH LIMITATIONS & ACADEMIC NON-CLAIMS")
    lines.append(sub_sep)
    lines.append(
        "- Identity Switches NOT Corrected:\n"
        "  Temporal majority voting operates strictly within a single tracklet.\n"
        "  If tracking fragmentation occurs across disjoint intervals, those\n"
        "  tracklets remain separate. The post-processor does NOT perform\n"
        "  cross-tracklet visual re-identification or assert that separate\n"
        "  tracker IDs represent the same physical vehicle without independent\n"
        "  ground-truth verification."
    )
    lines.append(
        "- No Camera Georeferencing:\n"
        "  Camera mount coordinates, heading, tilt, and target OpenStreetMap road\n"
        "  segments are uncalibrated for the sample video. Visual detections\n"
        "  cannot be associated with road graph edges without ground surveys."
    )
    lines.append(
        "- No Traffic Congestion Claims:\n"
        "  Vehicle counts in video coordinates do not establish roadway congestion.\n"
        "  Congestion requires empirical road capacity values (Capacity_ref)\n"
        "  and calibrated traffic flow formulations."
    )
    lines.append(
        "- No Sidewalk Accessibility Sensing:\n"
        "  Motorized vehicle detection provides zero information regarding\n"
        "  pedestrian accessibility barriers, curb ramps, stairs, or walkway widths."
    )
    lines.append(
        "- No Physical Vehicle Speed Estimation:\n"
        "  Pixel displacement measures 2D movement in image sensor coordinates only.\n"
        "  Estimating real-world speed (m/s or km/h) requires metric camera-to-roadway\n"
        "  homography calibration, perspective distortion rectification, and verified\n"
        "  ground control points. Pixel displacement is NOT vehicle speed."
    )
    lines.append(sep)

    return "\n".join(lines)


def save_evaluation_outputs(
    report: PipelineEvaluationReport,
    formatted_report: str,
    output_txt: str | Path | None = None,
    output_json: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path | None, Path | None]:
    """Save formatted text and JSON reports to disk, protecting against accidental overwrites.

    Args:
        report: PipelineEvaluationReport instance.
        formatted_report: Pre-rendered string version of the text report.
        output_txt: Optional file path to save the formatted text report.
        output_json: Optional file path to save the JSON data report.
        overwrite: If True, existing files are replaced. If False and a target
            file already exists, raises FileExistsError to protect experiment artifacts.

    Returns:
        Tuple of (saved_txt_path, saved_json_path).

    Raises:
        FileExistsError: If a target path already exists and overwrite is False.
    """
    saved_txt: Path | None = None
    saved_json: Path | None = None

    if output_txt:
        txt_path = Path(output_txt)
        if txt_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output text file already exists at '{txt_path}'. "
                f"Use --overwrite / -f to allow replacing existing files."
            )
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(formatted_report, encoding="utf-8")
        saved_txt = txt_path

    if output_json:
        json_path = Path(output_json)
        if json_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output JSON file already exists at '{json_path}'. "
                f"Use --overwrite / -f to allow replacing existing files."
            )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        saved_json = json_path

    return saved_txt, saved_json


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m backend.vision.evaluate_pipeline",
        description=(
            "AccessRoute Computer-Vision Offline Evaluation Script.\n\n"
            "Evaluates raw YOLO11n + ByteTrack detections against TrackletPostProcessor\n"
            "for a configurable, conservative sample of video frames.\n\n"
            "Usage Examples:\n"
            "  # Evaluate first 30 frames (default conservative sample):\n"
            '  python -m backend.vision.evaluate_pipeline --video-path "data/vision/WhatsApp Video 2026-10-26 at 9.29.19 AM.mp4"\n\n'
            "  # Evaluate 60 frames with custom lifespan filter and save text report:\n"
            '  python -m backend.vision.evaluate_pipeline --video-path "data/vision/sample.mp4" --max-frames 60 --min-lifespan 10 --output-txt report.txt\n\n'
            "  # Export structured evaluation report to JSON without overwriting existing files:\n"
            '  python -m backend.vision.evaluate_pipeline --video-path "data/vision/sample.mp4" --output-json report.json\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--video-path",
        "-v",
        required=True,
        type=str,
        help="Path to the video file to evaluate (REQUIRED).",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help=(
            "Maximum number of video frames to process (default: None, processes entire video). "
            "Set to a positive integer (e.g. 30) to evaluate a limited sample."
        ),
    )
    parser.add_argument(
        "--min-lifespan",
        "-l",
        type=int,
        default=DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
        help=(
            f"Minimum observed frames required to retain a tracklet "
            f"(default: {DEFAULT_MIN_TRACK_LIFESPAN_FRAMES})."
        ),
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.50,
        help="YOLO detection confidence threshold (default: 0.50).",
    )
    parser.add_argument(
        "--iou",
        "-i",
        type=float,
        default=0.30,
        help="NMS IoU suppression threshold (default: 0.30).",
    )
    parser.add_argument(
        "--tracker",
        "-t",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration filename (default: 'bytetrack.yaml').",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Optional path to write the formatted human-readable text report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        default=False,
        help="Overwrite existing output files if they already exist (default: False).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for running the offline vision evaluation script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = evaluate_video(
            video_path=args.video_path,
            max_frames=args.max_frames,
            min_track_lifespan_frames=args.min_lifespan,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
        )

        formatted_report = format_evaluation_report(report)
        print(formatted_report)

        saved_txt, saved_json = save_evaluation_outputs(
            report=report,
            formatted_report=formatted_report,
            output_txt=args.output_txt,
            output_json=args.output_json,
            overwrite=args.overwrite,
        )
        if saved_txt:
            print(f"\n[INFO] Text report saved to '{saved_txt}'.")
        if saved_json:
            print(f"[INFO] JSON report saved to '{saved_json}'.")

        return 0

    except FileNotFoundError as exc:
        print(f"[ERROR] File not found: {exc}", file=sys.stderr)
        return 1
    except FileExistsError as exc:
        print(f"[ERROR] Output file conflict: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[ERROR] Invalid configuration or input: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
