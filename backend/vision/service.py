"""Service layer for computer-vision video analysis using Ultralytics YOLO.

This service is dedicated to vehicle and object detection from video footage.
It operates independently from the pedestrian routing engine and simulated traffic
dataset. It does not validate physical accessibility obstacles or compute verified
traffic congestion indexes.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Sequence

from backend.vision.schemas import (
    BoundingBox,
    DetectedObject,
    FrameDetectionResult,
    VideoAnalysisSummary,
)

# Standard COCO vehicle classes used in the teammate's experiment
TARGET_VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

EXPECTED_WEIGHTS_PATH = Path("backend/vision/weights/yolo11n.pt")
DEFAULT_MODEL_NAME = str(EXPECTED_WEIGHTS_PATH)


class VideoAnalysisService:
    """Service providing lazy-loaded YOLO model management and video analysis summaries.

    Reuses a single loaded YOLO instance across operations rather than reloading
    model weights per frame or per request. Does not trigger weight downloads
    during service instantiation.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize service settings without triggering immediate model downloads.

        Args:
            model_path: Path or filename for model weights. Defaults to the
                VISION_MODEL_PATH environment variable or 'backend/vision/weights/yolo11n.pt'.
            device: Target compute device (e.g., 'cpu', 'mps'). Defaults to the
                VISION_DEVICE environment variable or 'cpu'.
        """
        env_model_path = os.environ.get("VISION_MODEL_PATH")
        self._model_path = Path(model_path or env_model_path or DEFAULT_MODEL_NAME)
        self._device = device or os.environ.get("VISION_DEVICE", "cpu")
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        """Configured path to model weights."""
        return self._model_path

    @property
    def device(self) -> str:
        """Configured execution device ('cpu' or 'mps')."""
        return self._device

    @property
    def is_model_loaded(self) -> bool:
        """Return True if model weights have been instantiated in memory."""
        return self._model is not None

    @classmethod
    def resolve_model_path(cls, path: str | Path) -> Path:
        """Resolve model weight path, checking configured, local, and expected locations.

        Evaluation order:
        1. If path exists as-is (absolute or relative to current working directory).
        2. If path is a bare filename (e.g. 'yolo11n.pt') and exists in 'backend/vision/weights/'.
        3. If path exists relative to this module directory in 'weights/'.
        4. Otherwise returns the normalized Path as requested.
        """
        p = Path(path)
        if p.is_file():
            return p
        weights_candidate = EXPECTED_WEIGHTS_PATH.parent / p.name
        if weights_candidate.is_file():
            return weights_candidate
        module_candidate = Path(__file__).parent / "weights" / p.name
        if module_candidate.is_file():
            return module_candidate
        return p

    def get_model(self) -> Any:
        """Lazily load and return the shared YOLO model instance.

        Thread-safe singleton loader. Reuses the loaded instance once initialized.
        Verifies that model weights exist locally before initializing YOLO to prevent
        unintended automatic weight downloads.

        Raises:
            FileNotFoundError: If the configured weights file does not exist on disk.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    resolved_path = self.resolve_model_path(self._model_path)
                    if not resolved_path.is_file():
                        raise FileNotFoundError(
                            f"Model weights file not found at '{resolved_path}'. "
                            f"Expected local path is '{EXPECTED_WEIGHTS_PATH}' or configured path '{self._model_path}'. "
                            f"Automatic downloads are disabled to prevent unintended network operations. "
                            f"Please ensure valid weights are placed at '{EXPECTED_WEIGHTS_PATH}' or configure "
                            f"the VISION_MODEL_PATH environment variable."
                        )

                    from ultralytics import YOLO

                    self._model = YOLO(str(resolved_path))
        return self._model

    def reset_tracker(self) -> None:
        """Reset internal tracker state on the loaded model if supported.

        Ensures that subsequent video processing does not inherit tracking IDs or
        Kalman filter state from a previously processed video. Safely handles
        cases where predictor or trackers are uninitialized or mocked.
        """
        if self._model is not None:
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                try:
                    predictor.trackers = []
                except Exception:
                    pass

    def extract_video_metadata(self, video_path: str | Path) -> dict[str, Any]:
        """Extract metadata from a video file without performing object detection.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing 'video_name', 'frame_count', 'fps',
            'width', 'height', and 'duration_seconds'.

        Raises:
            FileNotFoundError: If the video path does not exist.
            ValueError: If the path is not a regular file or cannot be opened by OpenCV.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found at '{video_file}'.")
        if not video_file.is_file():
            raise ValueError(f"Path '{video_file}' is not a regular file.")

        import cv2

        cap = cv2.VideoCapture(str(video_file))
        try:
            if not cap.isOpened():
                raise ValueError(
                    f"OpenCV could not open video file at '{video_file}'."
                )

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = (
                (frame_count / fps) if (fps > 0.0 and frame_count > 0) else 0.0
            )

            return {
                "video_name": video_file.name,
                "frame_count": max(0, frame_count),
                "fps": max(0.0, fps),
                "width": max(0, width),
                "height": max(0, height),
                "duration_seconds": max(0.0, duration_seconds),
            }
        finally:
            cap.release()

    def extract_frame_detections(
        self,
        result: Any,
        frame_number: int,
        timestamp_seconds: float,
    ) -> FrameDetectionResult:
        """Extract and validate vehicle detections from a YOLO result object.

        Filters detections to target vehicle classes (car, motorcycle, bus, truck),
        converts bounding box coordinates, and preserves tracking IDs when present.

        Args:
            result: Ultralytics Results object or equivalent inference output.
            frame_number: Zero-based sequential frame index.
            timestamp_seconds: Elapsed video time in seconds for this frame.

        Returns:
            FrameDetectionResult instance with validated vehicle detections.
        """
        if result is None or getattr(result, "boxes", None) is None:
            return FrameDetectionResult.from_detections(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                detections=[],
            )

        boxes = result.boxes
        if len(boxes) == 0:
            return FrameDetectionResult.from_detections(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                detections=[],
            )

        names: dict[int, str] = getattr(result, "names", {}) or {}

        # Safely convert tensors/lists to Python sequences
        raw_xyxy = getattr(boxes, "xyxy", [])
        xyxy_list = raw_xyxy.tolist() if hasattr(raw_xyxy, "tolist") else list(raw_xyxy)

        raw_conf = getattr(boxes, "conf", [])
        conf_list = raw_conf.tolist() if hasattr(raw_conf, "tolist") else list(raw_conf)

        raw_cls = getattr(boxes, "cls", [])
        cls_list = raw_cls.tolist() if hasattr(raw_cls, "tolist") else list(raw_cls)

        id_attr = getattr(boxes, "id", None)
        id_list: list[Any] | None = None
        if id_attr is not None:
            if callable(getattr(id_attr, "tolist", None)):
                converted = id_attr.tolist()
                if isinstance(converted, (list, tuple)):
                    id_list = list(converted)
            elif isinstance(id_attr, (list, tuple)):
                id_list = list(id_attr)

        detections: list[DetectedObject] = []

        for i in range(len(xyxy_list)):
            raw_cls_val = int(cls_list[i])
            class_name = TARGET_VEHICLE_CLASSES.get(raw_cls_val)
            if not class_name:
                named = str(names.get(raw_cls_val, "")).lower()
                if named in TARGET_VEHICLE_CLASSES.values():
                    class_name = named
                else:
                    # Filter out non-vehicle categories (e.g. person, dog, traffic light)
                    continue

            raw_conf_val = float(conf_list[i])
            confidence = max(0.0, min(1.0, raw_conf_val))

            box_coords = xyxy_list[i]
            x1 = max(0.0, float(box_coords[0]))
            y1 = max(0.0, float(box_coords[1]))
            x2 = max(0.0, float(box_coords[2]))
            y2 = max(0.0, float(box_coords[3]))

            # Discard degenerate zero-width or zero-height detections
            if x2 <= x1 or y2 <= y1:
                continue

            bounding_box = BoundingBox(
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2,
            )

            track_id: int | None = None
            if id_list is not None and i < len(id_list) and id_list[i] is not None:
                try:
                    parsed_id = int(id_list[i])
                    if parsed_id >= 0:
                        track_id = parsed_id
                except (ValueError, TypeError):
                    track_id = None

            detections.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    track_id=track_id,
                )
            )

        return FrameDetectionResult.from_detections(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            detections=detections,
        )

    def process_video(
        self,
        video_path: str | Path,
        conf: float = 0.50,
        iou: float = 0.30,
        tracker: str = "bytetrack.yaml",
        max_frames: int | None = None,
    ) -> tuple[VideoAnalysisSummary, list[FrameDetectionResult]]:
        """Process a video file sequentially with YOLO vehicle detection and tracking.

        Maintains persistent tracker state across sequential frames. Does not reset
        the tracker per frame. Video resources are guaranteed to be released.

        Args:
            video_path: Path to the target video file.
            conf: Minimum confidence threshold for vehicle detections (default: 0.50).
            iou: IoU threshold for NMS and association (default: 0.30).
            tracker: Tracker configuration file (default: 'bytetrack.yaml').
            max_frames: Optional upper limit on processed frames for sampling.

        Returns:
            Tuple of (VideoAnalysisSummary, list of FrameDetectionResult).

        Raises:
            FileNotFoundError: If the video or model weights file does not exist.
            ValueError: If the video cannot be opened by OpenCV.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found at '{video_file}'.")
        if not video_file.is_file():
            raise ValueError(f"Path '{video_file}' is not a regular file.")

        # Require local model weights before proceeding with video decoding
        model = self.get_model()

        # Reset internal tracker state before processing a new video
        self.reset_tracker()

        import cv2

        cap = cv2.VideoCapture(str(video_file))
        try:
            if not cap.isOpened():
                raise ValueError(
                    f"OpenCV could not open video file at '{video_file}'."
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0:
                fps = 30.0  # sensible default when metadata reports 0 FPS

            duration_seconds = (
                (total_frames / fps) if (fps > 0.0 and total_frames > 0) else 0.0
            )

            frame_results: list[FrameDetectionResult] = []
            frame_number = 0

            while True:
                if max_frames is not None and frame_number >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_seconds = frame_number / fps if fps > 0.0 else 0.0

                # On the first frame of a new video, persist=False forces Ultralytics
                # to initialize a fresh tracker. Subsequent frames use persist=True
                # to preserve tracking continuity across consecutive frames.
                persist_tracking = (frame_number > 0)

                results = model.track(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    persist=persist_tracking,
                    verbose=False,
                )

                result = results[0] if results else None
                frame_detection = self.extract_frame_detections(
                    result=result,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                )
                frame_results.append(frame_detection)
                frame_number += 1

            processed_frames = len(frame_results)
            actual_total_frames = max(total_frames, processed_frames)

            summary = self.build_summary(
                video_name=video_file.name,
                total_frames=actual_total_frames,
                processed_frames=processed_frames,
                duration_seconds=duration_seconds,
                fps=fps,
                frame_results=frame_results,
            )

            return summary, frame_results
        finally:
            cap.release()

    def analyze_video(
        self,
        video_path: str | Path,
        conf: float = 0.50,
        iou: float = 0.30,
        tracker: str = "bytetrack.yaml",
        max_frames: int | None = None,
    ) -> VideoAnalysisSummary:
        """Run video analysis and return the aggregated VideoAnalysisSummary.

        Convenience wrapper around process_video discarding detailed frame results.

        Args:
            video_path: Path to the target video file.
            conf: Minimum confidence threshold (default: 0.50).
            iou: IoU threshold (default: 0.30).
            tracker: Tracker configuration (default: 'bytetrack.yaml').
            max_frames: Optional frame processing limit.

        Returns:
            VideoAnalysisSummary with aggregated detection counts.
        """
        summary, _ = self.process_video(
            video_path=video_path,
            conf=conf,
            iou=iou,
            tracker=tracker,
            max_frames=max_frames,
        )
        return summary

    def build_summary(
        self,
        video_name: str,
        total_frames: int,
        processed_frames: int,
        duration_seconds: float,
        fps: float,
        frame_results: Sequence[FrameDetectionResult],
    ) -> VideoAnalysisSummary:
        """Aggregate per-frame detection results into a standardized video summary.

        Computes cumulative per-frame detection events across processed frames.
        Also tracks observed unique tracking IDs per class when available.
        Does not estimate verified unique vehicles or classify traffic congestion levels.

        Args:
            video_name: Identifier or filename of the video.
            total_frames: Total frame count in the source video.
            processed_frames: Number of frames evaluated.
            duration_seconds: Length of the video in seconds.
            fps: Frame rate of the video.
            frame_results: Sequence of FrameDetectionResult instances.

        Returns:
            VideoAnalysisSummary with aggregated detection counts by class.
        """
        class_counts: dict[str, int] = {}
        total_detections = 0
        tracked_ids_by_class: dict[str, set[int]] = {}
        has_tracking_ids = False

        for frame in frame_results:
            for item in frame.detections:
                class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
                total_detections += 1

                if item.track_id is not None:
                    has_tracking_ids = True
                    if item.class_name not in tracked_ids_by_class:
                        tracked_ids_by_class[item.class_name] = set()
                    tracked_ids_by_class[item.class_name].add(item.track_id)

        tracked_id_counts: dict[str, int] | None = None
        total_tracked_ids: int | None = None

        if has_tracking_ids:
            tracked_id_counts = {
                cls: len(ids) for cls, ids in tracked_ids_by_class.items()
            }
            total_tracked_ids = (
                len(set().union(*tracked_ids_by_class.values()))
                if tracked_ids_by_class
                else 0
            )

        return VideoAnalysisSummary(
            video_name=video_name,
            total_frames=total_frames,
            processed_frames=processed_frames,
            duration_seconds=duration_seconds,
            fps=fps,
            class_counts=class_counts,
            total_detections=total_detections,
            tracked_id_counts=tracked_id_counts,
            total_tracked_ids=total_tracked_ids,
        )
