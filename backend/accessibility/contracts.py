"""Data contracts and schemas for pedestrian accessibility observations and obstacles.

Defines the formal interface between computer-vision obstacle perception
and the graph-based pedestrian routing engine.

DISCLAIMER & SAFETY NOTICE:
1. Obstacle observations represent localized physical barriers (e.g., blocked
   sidewalks, construction work, missing curb ramps, or severe surface degradation).
2. Missing accessibility information MUST be treated as UNKNOWN, not as proof
   that a pathway is accessible or wheelchair-compliant.
3. AccessRoute does NOT claim that a route is fully wheelchair-accessible
   unless validated accessibility survey data or certified obstacle observations
   confirm compliance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AccessibilityObstacleType(str, Enum):
    """Categorical classification of pedestrian pathway barriers."""

    BLOCKED_SIDEWALK = "blocked_sidewalk"
    CONSTRUCTION_BARRIER = "construction_barrier"
    MISSING_CURB_RAMP = "missing_curb_ramp"
    STEEP_INCLINE = "steep_incline"
    SURFACE_OBSTRUCTION = "surface_obstruction"
    STEPS_BARRIER = "steps_barrier"


class AccessibilitySeverity(str, Enum):
    """Severity level indicating traversal impact for pedestrians and mobility devices."""

    IMPASSABLE = "impassable"  # Completely blocks traversal (e.g. deep excavation, locked gate)
    HIGH_DIFFICULTY = "high_difficulty"  # Severe obstacle (e.g. high curb, broken pavers, stairs)
    CAUTION = "caution"  # Traversable with care (e.g. narrow path, temporary debris)


DEFAULT_PENALTY_MULTIPLIERS: dict[AccessibilitySeverity, float] = {
    AccessibilitySeverity.IMPASSABLE: 100.0,
    AccessibilitySeverity.HIGH_DIFFICULTY: 10.0,
    AccessibilitySeverity.CAUTION: 2.0,
}


class AccessibilityObstacleObservation(BaseModel):
    """Validated observation of a pedestrian obstacle mapped to a directed road/path edge."""

    model_config = ConfigDict(frozen=True)

    obstacle_id: str = Field(
        min_length=1,
        max_length=100,
        description="Unique identifier for the reported obstacle",
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

    obstacle_type: AccessibilityObstacleType = Field(
        description="Type of accessibility barrier or pathway obstruction",
    )

    severity: AccessibilitySeverity = Field(
        default=AccessibilitySeverity.HIGH_DIFFICULTY,
        description="Traversal impact severity",
    )

    penalty_multiplier: float | None = Field(
        default=None,
        gt=0.0,
        le=1000.0,
        description="Routing impedance cost multiplier applied to physical segment length",
    )

    description: str = Field(
        default="Verified pedestrian pathway obstacle",
        min_length=1,
        max_length=500,
        description="Human-readable explanation of the obstacle and location context",
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Perception model detection confidence in [0.0, 1.0]",
    )

    reported_at_utc: datetime = Field(
        description="Timezone-aware UTC timestamp when obstacle was observed",
    )

    time_to_live_seconds: float = Field(
        default=86400.0,  # 24 hours
        ge=60.0,
        description="Validity window in seconds before obstacle observation is considered stale",
    )

    is_verified: bool = Field(
        default=True,
        description="Whether this obstacle report has been verified by perception or human survey",
    )

    disclaimer: str = Field(
        default=(
            "Accessibility observations reflect localized surveyed or detected barriers. "
            "Unmonitored pathway segments are of unknown accessibility and are not guaranteed accessible."
        ),
        description="Safety and non-guarantee disclaimer",
    )

    @field_validator("reported_at_utc")
    @classmethod
    def require_timezone_aware(cls, dt: datetime) -> datetime:
        """Require explicit timezone awareness to prevent ambiguous cross-server comparisons."""
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError(
                "reported_at_utc must be timezone-aware (e.g., UTC with tzinfo=timezone.utc)."
            )
        return dt

    @field_validator("penalty_multiplier")
    @classmethod
    def validate_finite_penalty(cls, value: float | None) -> float | None:
        """Reject non-finite numbers for penalty multiplier."""
        if value is not None and not math.isfinite(value):
            raise ValueError("penalty_multiplier must be a finite number.")
        return value

    @model_validator(mode="after")
    def validate_and_set_defaults(self) -> AccessibilityObstacleObservation:
        """Ensure osm_u and osm_v are distinct endpoints and assign severity default multiplier."""
        if self.osm_u == self.osm_v:
            raise ValueError(
                "osm_u and osm_v must be distinct node IDs (self-loops are not allowed)."
            )
        if self.penalty_multiplier is None:
            default_mult = DEFAULT_PENALTY_MULTIPLIERS.get(self.severity, 10.0)
            object.__setattr__(self, "penalty_multiplier", default_mult)
        return self



class ObstacleDetectionPipelineProtocol(Protocol):
    """Formal protocol for the perception team's obstacle detection pipeline.
    
    The computer-vision obstacle model is currently in active development.
    When ready, the perception service will implement this protocol to register
    verified obstacle observations into the AccessibilityAdapter.
    """

    def detect_obstacles(self, source_image_or_video: str) -> list[AccessibilityObstacleObservation]:
        """Detect and classify pedestrian obstacles in an image or video."""
        ...
