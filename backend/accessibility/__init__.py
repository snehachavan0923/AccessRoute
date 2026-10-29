"""Accessibility subsystem for pedestrian pathway obstacle and mobility routing."""

from backend.accessibility.adapter import (
    AccessibilityAdapter,
    get_default_accessibility_adapter,
)
from backend.accessibility.contracts import (
    AccessibilityObstacleObservation,
    AccessibilityObstacleType,
    AccessibilitySeverity,
    ObstacleDetectionPipelineProtocol,
)

__all__ = [
    "AccessibilityAdapter",
    "AccessibilityObstacleObservation",
    "AccessibilityObstacleType",
    "AccessibilitySeverity",
    "ObstacleDetectionPipelineProtocol",
    "get_default_accessibility_adapter",
]
