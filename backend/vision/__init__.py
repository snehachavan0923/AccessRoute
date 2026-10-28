"""Vision module for vehicle and object detection from video footage."""

from backend.vision.post_processor import (
    DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
    PostProcessedVideoResult,
    PostProcessingSummary,
    Tracklet,
    TrackletPostProcessor,
)
from backend.vision.schemas import (
    BoundingBox,
    CameraMetadata,
    DetectedObject,
    FrameDetectionResult,
    ObservationTimeEnvelope,
    VideoAnalysisSummary,
    VisionEdgeObservation,
)
from backend.vision.service import (
    DEFAULT_MODEL_NAME,
    EXPECTED_WEIGHTS_PATH,
    TARGET_VEHICLE_CLASSES,
    VideoAnalysisService,
)

__all__ = [
    "BoundingBox",
    "CameraMetadata",
    "DetectedObject",
    "FrameDetectionResult",
    "ObservationTimeEnvelope",
    "VideoAnalysisSummary",
    "VisionEdgeObservation",
    "DEFAULT_MODEL_NAME",
    "EXPECTED_WEIGHTS_PATH",
    "TARGET_VEHICLE_CLASSES",
    "VideoAnalysisService",
    "DEFAULT_MIN_TRACK_LIFESPAN_FRAMES",
    "Tracklet",
    "PostProcessingSummary",
    "PostProcessedVideoResult",
    "TrackletPostProcessor",
]
