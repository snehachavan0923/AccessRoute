"""FastAPI router for computer vision vehicle detection, tracking, and evaluation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.vision.annotator import annotate_video
from backend.vision.evaluate_pipeline import (
    DEFAULT_EVAL_MAX_FRAMES,
    PipelineEvaluationReport,
    evaluate_frames,
)
from backend.vision.post_processor import (
    DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
    PostProcessedVideoResult,
    TrackletPostProcessor,
)
from backend.vision.schemas import VideoAnalysisSummary
from backend.vision.service import (
    EXPECTED_WEIGHTS_PATH,
    TARGET_VEHICLE_CLASSES,
    VideoAnalysisService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])

# Thread lock protecting YOLO and ByteTrack state against concurrent requests
_analysis_lock = threading.Lock()

# Shared singleton service instance
_vision_service = VideoAnalysisService()

# Paths for video assets and temporary cache
DATA_VISION_DIR = Path("data/vision")
SAMPLE_VIDEO_FILENAME = "WhatsApp Video 2026-10-26 at 9.29.19 AM.mp4"
SAMPLE_VIDEO_PATH = DATA_VISION_DIR / SAMPLE_VIDEO_FILENAME

UPLOADS_DIR = DATA_VISION_DIR / "uploads"
ANNOTATED_CACHE_DIR = DATA_VISION_DIR / "cache"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# In-memory mapping of uploaded video identifiers to filesystem paths
_uploaded_videos: dict[str, Path] = {}


class SampleVideoInfo(BaseModel):
    """Metadata describing a pre-loaded sample video."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    filename: str
    duration_seconds: float
    total_frames: int
    fps: float
    width: int
    height: int
    video_url: str


class VisionStatusResponse(BaseModel):
    """Vision service operational status and available models/videos."""

    model_config = ConfigDict(frozen=True)

    weights_available: bool
    is_model_loaded: bool
    model_name: str
    device: str
    target_classes: dict[int, str]
    sample_videos: list[SampleVideoInfo]


class AnalyzeVideoRequest(BaseModel):
    """Parameters for running computer vision analysis on a video."""

    video_id: str = Field(
        default="sample_traffic_video",
        description="Identifier of the video to evaluate (e.g. 'sample_traffic_video' or an uploaded ID)",
    )
    max_frames: int | None = Field(
        default=DEFAULT_EVAL_MAX_FRAMES,
        ge=1,
        le=2000,
        description="Limit on evaluated frames (e.g. 30 for quick sample, or None/301 for full video)",
    )
    conf: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        description="YOLO detection confidence threshold",
    )
    iou: float = Field(
        default=0.30,
        ge=0.10,
        le=0.90,
        description="NMS IoU threshold",
    )
    min_track_lifespan_frames: int = Field(
        default=DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
        ge=1,
        le=60,
        description="Minimum observed frames required to retain a vehicle tracklet",
    )
    generate_annotated_video: bool = Field(
        default=True,
        description="Whether to generate an annotated MP4 video with bounding boxes and track IDs",
    )


class AnalyzeVideoResponse(BaseModel):
    """Detailed results of computer vision detection and tracklet post-processing."""

    job_id: str
    video_id: str = Field(default="sample_traffic_video", description="Identifier of the evaluated video")
    video_width: int = Field(default=640, ge=0, description="Source video width in pixels")
    video_height: int = Field(default=360, ge=0, description="Source video height in pixels")
    processing_time_seconds: float
    annotated_video_url: str | None = None
    source_video_url: str
    report: PipelineEvaluationReport


class VideoUploadResponse(BaseModel):
    """Confirmation and metadata for an uploaded video file."""

    video_id: str
    filename: str
    duration_seconds: float
    total_frames: int
    fps: float
    width: int
    height: int
    size_bytes: int
    video_url: str


def _resolve_video_path(video_id: str) -> Path:
    """Safely resolve video path from identifier without arbitrary path traversal."""
    if video_id in ("sample", "sample_traffic_video", "default"):
        if not SAMPLE_VIDEO_PATH.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Sample traffic video not found on disk at '{SAMPLE_VIDEO_FILENAME}'.",
            )
        return SAMPLE_VIDEO_PATH

    if video_id in _uploaded_videos:
        path = _uploaded_videos[video_id]
        if path.is_file():
            return path
        raise HTTPException(
            status_code=404,
            detail="Uploaded video file is no longer available on disk.",
        )

    # Check if video_id matches a file in uploads directory with safe UUID format
    safe_id_match = re.match(r"^[0-9a-fA-F\-]{36}$", video_id)
    if safe_id_match:
        for file in UPLOADS_DIR.glob(f"{video_id}_*"):
            if file.is_file():
                _uploaded_videos[video_id] = file
                return file

    raise HTTPException(
        status_code=404,
        detail=f"Video with ID '{video_id}' was not found. Please select a valid sample video or upload a new file.",
    )


@router.get("/status", response_model=VisionStatusResponse)
def get_vision_status() -> VisionStatusResponse:
    """Return operational status of the vision pipeline, loaded weights, and sample videos."""
    weights_path = _vision_service.resolve_model_path(_vision_service.model_path)
    weights_available = weights_path.is_file()

    sample_videos: list[SampleVideoInfo] = []
    if SAMPLE_VIDEO_PATH.is_file():
        try:
            meta = _vision_service.extract_video_metadata(SAMPLE_VIDEO_PATH)
            sample_videos.append(
                SampleVideoInfo(
                    id="sample_traffic_video",
                    name="Electronic City Traffic Footage (Sample)",
                    filename=SAMPLE_VIDEO_PATH.name,
                    duration_seconds=round(meta.get("duration_seconds", 0.0), 2),
                    total_frames=meta.get("frame_count", 0),
                    fps=round(meta.get("fps", 0.0), 2),
                    width=meta.get("width", 0),
                    height=meta.get("height", 0),
                    video_url="/api/vision/video/sample_traffic_video",
                )
            )
        except Exception as err:
            logger.warning("Could not extract sample video metadata: %s", err)

    return VisionStatusResponse(
        weights_available=weights_available,
        is_model_loaded=_vision_service.is_model_loaded,
        model_name=weights_path.name,
        device=_vision_service.device,
        target_classes=TARGET_VEHICLE_CLASSES,
        sample_videos=sample_videos,
    )


@router.api_route("/sample-video", methods=["GET", "HEAD"])
def get_sample_video() -> FileResponse:
    """Stream the raw sample traffic video for browser playback."""
    if not SAMPLE_VIDEO_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Sample traffic video file not found on disk.",
        )
    return FileResponse(
        path=str(SAMPLE_VIDEO_PATH),
        media_type="video/mp4",
        filename=SAMPLE_VIDEO_PATH.name,
    )


@router.api_route("/video/{video_id}", methods=["GET", "HEAD"])
def get_video_stream(video_id: str) -> FileResponse:
    """Stream a specified source video file for browser playback."""
    path = _resolve_video_path(video_id)
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=path.name,
    )


@router.api_route("/annotated-video/{job_id}", methods=["GET", "HEAD"])
def get_annotated_video(job_id: str) -> FileResponse:
    """Stream a generated annotated video file with bounding boxes and tracking IDs."""
    if not re.match(r"^[0-9a-fA-F\-]{36}$", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format.")

    annotated_path = ANNOTATED_CACHE_DIR / f"annotated_{job_id}.mp4"
    if not annotated_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Annotated video not found or has expired.",
        )

    return FileResponse(
        path=str(annotated_path),
        media_type="video/mp4",
        filename=f"annotated_{job_id}.mp4",
    )


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    """Upload a custom video file for computer vision analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Supported formats: {allowed}.",
        )

    # Ensure uploads directory exists
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    upload_id = str(uuid.uuid4())
    safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", Path(file.filename).name)
    destination = UPLOADS_DIR / f"{upload_id}_{safe_name}"

    total_bytes = 0
    try:
        with destination.open("wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.",
                    )
                out_file.write(chunk)
    except Exception:
        if destination.is_file():
            destination.unlink(missing_ok=True)
        raise

    # Verify that the uploaded file can be opened by OpenCV
    try:
        metadata = _vision_service.extract_video_metadata(destination)
    except Exception as err:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file could not be decoded as a valid video: {err}",
        )

    _uploaded_videos[upload_id] = destination

    return VideoUploadResponse(
        video_id=upload_id,
        filename=file.filename,
        duration_seconds=round(metadata.get("duration_seconds", 0.0), 2),
        total_frames=metadata.get("frame_count", 0),
        fps=round(metadata.get("fps", 0.0), 2),
        width=metadata.get("width", 0),
        height=metadata.get("height", 0),
        size_bytes=total_bytes,
        video_url=f"/api/vision/video/{upload_id}",
    )


@router.post("/analyze", response_model=AnalyzeVideoResponse)
def analyze_video(request: AnalyzeVideoRequest) -> AnalyzeVideoResponse:
    """Run real YOLO detection and ByteTrack tracking on the selected video."""
    video_path = _resolve_video_path(request.video_id)

    # Protect shared model and tracker against concurrent simultaneous inference
    acquired = _analysis_lock.acquire(timeout=60.0)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="The computer vision pipeline is currently busy processing another request. Please try again shortly.",
        )

    t_start = time.time()
    job_id = str(uuid.uuid4())

    try:
        # 1. Extract metadata from target video
        metadata = _vision_service.extract_video_metadata(video_path)

        # 2. Sequential frame-by-frame YOLO detection and ByteTrack tracking
        _, frame_results = _vision_service.process_video(
            video_path=video_path,
            conf=request.conf,
            iou=request.iou,
            max_frames=request.max_frames,
        )

        if not frame_results:
            raise HTTPException(
                status_code=422,
                detail="No video frames could be read or decoded from the source video.",
            )

        # 3. Post-process tracklets with temporal majority voting and lifespan filtering
        report = evaluate_frames(
            frame_results=frame_results,
            video_metadata=metadata,
            min_track_lifespan_frames=request.min_track_lifespan_frames,
            conf_threshold=request.conf,
            iou_threshold=request.iou,
            video_path=video_path.name,
        )

        # 4. Generate annotated video if requested
        annotated_video_url = None
        if request.generate_annotated_video:
            post_processor = TrackletPostProcessor(
                min_track_lifespan_frames=request.min_track_lifespan_frames,
                filter_short_tracks=True,
            )
            processed_result: PostProcessedVideoResult = post_processor.process_frames(frame_results)

            output_annotated_path = ANNOTATED_CACHE_DIR / f"annotated_{job_id}.mp4"
            try:
                annotate_video(
                    source_video_path=video_path,
                    output_video_path=output_annotated_path,
                    frame_results=processed_result.frames,
                    max_frames=request.max_frames,
                )
                annotated_video_url = f"/api/vision/annotated-video/{job_id}"
            except Exception as annot_err:
                logger.warning("Failed to render annotated video: %s", annot_err)
                # Soft failure: analysis report remains available even if video encoding fails

        t_end = time.time()
        elapsed = round(t_end - t_start, 2)

        source_video_url = (
            f"/api/vision/video/{request.video_id}"
            if request.video_id != "sample_traffic_video"
            else "/api/vision/sample-video"
        )

        return AnalyzeVideoResponse(
            job_id=job_id,
            video_id=request.video_id,
            video_width=int(metadata.get("width", 640)),
            video_height=int(metadata.get("height", 360)),
            processing_time_seconds=elapsed,
            annotated_video_url=annotated_video_url,
            source_video_url=source_video_url,
            report=report,
        )

    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except Exception as err:
        logger.exception("Unexpected error during video analysis: %s", err)
        raise HTTPException(
            status_code=500,
            detail=f"Computer vision analysis failed: {err}",
        )
    finally:
        _analysis_lock.release()
