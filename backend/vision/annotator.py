"""Annotator for drawing bounding boxes and tracking IDs on video footage.

Reuses detection and tracking results produced by VideoAnalysisService and
TrackletPostProcessor. Does not run independent object detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from backend.vision.schemas import FrameDetectionResult

logger = logging.getLogger(__name__)

# BGR color palette for vehicle classes
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (240, 150, 40),       # Light Blue / Cyan
    "bus": (30, 160, 245),       # Amber / Orange
    "motorcycle": (200, 70, 180), # Purple / Magenta
    "truck": (70, 190, 100),     # Emerald Green
}
DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)


def annotate_frame(
    frame: np.ndarray,
    detection_result: FrameDetectionResult,
) -> np.ndarray:
    """Draw bounding boxes, class labels, and tracking IDs on a single video frame.

    Args:
        frame: OpenCV BGR image array (mutated in-place or copied).
        detection_result: FrameDetectionResult containing detections for this frame.

    Returns:
        Annotated BGR frame image array.
    """
    height, width = frame.shape[:2]

    for det in detection_result.detections:
        box = det.bounding_box
        x1 = max(0, min(width - 1, int(round(box.x_min))))
        y1 = max(0, min(height - 1, int(round(box.y_min))))
        x2 = max(0, min(width - 1, int(round(box.x_max))))
        y2 = max(0, min(height - 1, int(round(box.y_max))))

        if x2 <= x1 or y2 <= y1:
            continue

        color = CLASS_COLORS.get(det.class_name.lower(), DEFAULT_COLOR)

        # Draw bounding box outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label text: e.g. "car #3 0.88"
        if det.track_id is not None:
            label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
        else:
            label = f"{det.class_name} {det.confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Label background pill
        bg_y1 = max(0, y1 - label_h - baseline - 4)
        bg_y2 = y1
        bg_x2 = min(width, x1 + label_w + 6)
        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # High-contrast label text (white or dark depending on background)
        text_y = bg_y2 - baseline - 2
        cv2.putText(
            frame,
            label,
            (x1 + 3, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def annotate_video(
    source_video_path: str | Path,
    output_video_path: str | Path,
    frame_results: Sequence[FrameDetectionResult],
    max_frames: int | None = None,
) -> Path:
    """Render an annotated video file from existing frame detection results.

    Reuses existing detections and post-processed tracklet labels. Does not
    instantiate or run any neural network inference.

    Args:
        source_video_path: Path to the input video file.
        output_video_path: Destination path for the annotated video.
        frame_results: Sequence of FrameDetectionResult instances corresponding to sequential frames.
        max_frames: Optional upper limit on processed frames.

    Returns:
        Path to the successfully written annotated video file.

    Raises:
        FileNotFoundError: If source video is not found.
        ValueError: If video cannot be read or writer cannot be opened.
    """
    src_path = Path(source_video_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source video file not found at '{src_path}'.")

    dst_path = Path(output_video_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file at '{src_path}'.")

    writer = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}.")

        # Try H.264 (avc1) first for native browser HTML5 video playback, fallback to mp4v
        writer = None
        for fourcc_code in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            candidate = cv2.VideoWriter(str(dst_path), fourcc, fps, (width, height))
            if candidate.isOpened():
                writer = candidate
                break

        if writer is None or not writer.isOpened():
            raise ValueError(f"OpenCV could not open VideoWriter for '{dst_path}'.")

        total_to_process = len(frame_results)
        if max_frames is not None:
            total_to_process = min(total_to_process, max_frames)

        for i in range(total_to_process):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            annotated = annotate_frame(frame, frame_results[i])
            writer.write(annotated)

        writer.release()
        writer = None

        if not dst_path.is_file() or dst_path.stat().st_size == 0:
            raise ValueError(f"Failed to generate annotated video at '{dst_path}'.")

        return dst_path

    finally:
        cap.release()
        if writer is not None:
            writer.release()
