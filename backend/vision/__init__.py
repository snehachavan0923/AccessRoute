"""Vision module for vehicle and object detection from video footage."""

from backend.vision.schemas import (
    BoundingBox,
    DetectedObject,
    FrameDetectionResult,
    VideoAnalysisSummary,
)
from backend.vision.service import (
    DEFAULT_MODEL_NAME,
    EXPECTED_WEIGHTS_PATH,
    TARGET_VEHICLE_CLASSES,
    VideoAnalysisService,
)

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "FrameDetectionResult",
    "VideoAnalysisSummary",
    "DEFAULT_MODEL_NAME",
    "EXPECTED_WEIGHTS_PATH",
    "TARGET_VEHICLE_CLASSES",
    "VideoAnalysisService",
]
