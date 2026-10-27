"""Data models and schemas for video-based vehicle and object detection."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixels or normalized coordinates."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(ge=0.0, description="Left coordinate")
    y_min: float = Field(ge=0.0, description="Top coordinate")
    x_max: float = Field(ge=0.0, description="Right coordinate")
    y_max: float = Field(ge=0.0, description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        """Require strictly positive width and height."""
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be strictly greater than x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("y_max must be strictly greater than y_min.")

        return self


class DetectedObject(BaseModel):
    """A single object detected in a video frame."""

    model_config = ConfigDict(frozen=True)

    class_name: str = Field(
        min_length=1,
        max_length=50,
        description="Detected category, such as car, bus, truck, or motorcycle",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Detection confidence between 0 and 1",
    )

    bounding_box: BoundingBox

    track_id: int | None = Field(
        default=None,
        ge=0,
        description="Optional tracking identifier across frames",
    )


class FrameDetectionResult(BaseModel):
    """Detection results for one processed video frame."""

    model_config = ConfigDict(frozen=True)

    frame_number: int = Field(
        ge=0,
        description="Zero-based frame index",
    )

    timestamp_seconds: float = Field(
        ge=0.0,
        description="Time offset from the beginning of the video",
    )

    detections: list[DetectedObject] = Field(
        default_factory=list,
        description="Objects detected in this frame",
    )

    detection_count: int = Field(
        ge=0,
        description="Number of detections in this frame",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_or_validate_detection_count(cls, data: Any) -> Any:
        """Populate the count when omitted and reject inconsistent values."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        detections = data.get("detections")

        if detections is None:
            detections = []
            data["detections"] = detections

        if not isinstance(detections, (list, tuple)):
            # Let Pydantic report the appropriate list validation error.
            return data

        actual_count = len(detections)
        provided_count = data.get("detection_count")

        if provided_count is None:
            data["detection_count"] = actual_count
        elif provided_count != actual_count:
            raise ValueError(
                f"detection_count ({provided_count}) does not match "
                f"the number of detections ({actual_count})."
            )

        return data

    @classmethod
    def from_detections(
        cls,
        frame_number: int,
        timestamp_seconds: float,
        detections: list[DetectedObject],
    ) -> FrameDetectionResult:
        """Create a frame result and calculate its detection count."""
        return cls(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            detections=detections,
            detection_count=len(detections),
        )


class VideoAnalysisSummary(BaseModel):
    """Summary of object detections across a video.

    class_counts and total_detections count per-frame detection events,
    not unique physical vehicles. This summary does not verify traffic
    congestion or identify accessibility barriers.
    """

    model_config = ConfigDict(frozen=True)

    video_name: str = Field(
        min_length=1,
        description="Video filename or identifier",
    )

    total_frames: int = Field(
        ge=0,
        description="Total frames in the source video",
    )

    processed_frames: int = Field(
        ge=0,
        description="Number of frames processed",
    )

    duration_seconds: float = Field(
        ge=0.0,
        description="Video duration in seconds",
    )

    fps: float = Field(
        ge=0.0,
        description="Source video frame rate",
    )

    class_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-frame detection events by object class; "
            "these are not unique vehicle counts."
        ),
    )

    total_detections: int = Field(
        ge=0,
        description="Total detection events across processed frames",
    )

    tracked_id_counts: dict[str, int] | None = Field(
        default=None,
        description=(
            "Optional count of distinct tracking IDs observed per class. "
            "These are tracker IDs, not verified unique physical vehicles."
        ),
    )

    total_tracked_ids: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total distinct tracking IDs observed across all target classes. "
            "Subject to ID switches, occlusions, and multi-class assignments; "
            "does not represent verified ground-truth vehicles."
        ),
    )

    disclaimer: str = Field(
        default=(
            "These are preliminary per-frame detection events, "
            "not verified unique vehicle counts. They do not establish "
            "traffic congestion levels or detect accessibility barriers."
        ),
        description="Limitations of the analysis",
    )

    @field_validator("tracked_id_counts")
    @classmethod
    def validate_tracked_id_counts(
        cls,
        counts: dict[str, int] | None,
    ) -> dict[str, int] | None:
        """Require non-empty class names and non-negative integer counts when provided."""
        if counts is None:
            return None
        for class_name, count in counts.items():
            if not class_name.strip():
                raise ValueError("Class names must not be empty.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"Tracked count for class '{class_name}' must be an integer."
                )
            if count < 0:
                raise ValueError(
                    f"Tracked count for class '{class_name}' must be non-negative."
                )
        return counts

    @field_validator("class_counts")
    @classmethod
    def validate_class_counts(
        cls,
        counts: dict[str, int],
    ) -> dict[str, int]:
        """Require non-empty class names and non-negative integer counts."""
        for class_name, count in counts.items():
            if not class_name.strip():
                raise ValueError("Class names must not be empty.")

            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"Count for class '{class_name}' must be an integer."
                )

            if count < 0:
                raise ValueError(
                    f"Count for class '{class_name}' must be non-negative."
                )

        return counts

    @model_validator(mode="after")
    def validate_summary_consistency(self) -> VideoAnalysisSummary:
        """Ensure frame totals and detection counts are consistent."""
        if self.processed_frames > self.total_frames:
            raise ValueError(
                "processed_frames cannot exceed total_frames."
            )

        summed_counts = sum(self.class_counts.values())

        if self.total_detections != summed_counts:
            raise ValueError(
                f"total_detections ({self.total_detections}) does not match the sum of class_counts ({summed_counts})."
            )

        return self
