"""Unit and integration tests for Computer Vision REST API endpoints."""

import io
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.traffic import get_default_vision_adapter, set_default_vision_adapter
from backend.vision.schemas import BoundingBox, DetectedObject, FrameDetectionResult


class TestVisionAPI(unittest.TestCase):
    """Tests for vision REST API endpoints."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        set_default_vision_adapter(None)

    def test_get_vision_status(self) -> None:
        """Verify GET /api/vision/status returns operational parameters and sample video info."""
        response = self.client.get("/api/vision/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()


        self.assertIn("weights_available", data)
        self.assertIn("is_model_loaded", data)
        self.assertIn("model_name", data)
        self.assertIn("device", data)
        self.assertIn("target_classes", data)
        self.assertIn("sample_videos", data)

        # Check target vehicle classes
        self.assertEqual(data["target_classes"]["2"], "car")
        self.assertEqual(data["target_classes"]["3"], "motorcycle")
        self.assertEqual(data["target_classes"]["5"], "bus")
        self.assertEqual(data["target_classes"]["7"], "truck")

        # Verify sample videos list is non-empty when sample exists
        if Path("data/vision/WhatsApp Video 2026-10-26 at 9.29.19 AM.mp4").is_file():
            self.assertGreater(len(data["sample_videos"]), 0)
            sample = data["sample_videos"][0]
            self.assertEqual(sample["id"], "sample_traffic_video")
            self.assertIn("WhatsApp Video", sample["filename"])
            self.assertGreater(sample["duration_seconds"], 0)
            self.assertGreater(sample["total_frames"], 0)

    def test_get_sample_video_stream(self) -> None:
        """Verify GET /api/vision/sample-video returns video/mp4 stream."""
        if not Path("data/vision/WhatsApp Video 2026-10-26 at 9.29.19 AM.mp4").is_file():
            self.skipTest("Sample video file not available.")

        response = self.client.get("/api/vision/sample-video")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "video/mp4")
        self.assertGreater(len(response.content), 1000)

    def test_analyze_video_invalid_id(self) -> None:
        """Verify analyze returns 404 for non-existent video identifier."""
        response = self.client.post(
            "/api/vision/analyze",
            json={"video_id": "non_existent_video_12345"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json().get("detail", "").lower())

    def test_analyze_video_validation_errors(self) -> None:
        """Verify analyze returns 422 for out-of-range hyperparameters."""
        # conf must be between 0.05 and 0.95
        res1 = self.client.post("/api/vision/analyze", json={"conf": 1.5})
        self.assertEqual(res1.status_code, 422)

        res2 = self.client.post("/api/vision/analyze", json={"iou": -0.1})
        self.assertEqual(res2.status_code, 422)

        res3 = self.client.post("/api/vision/analyze", json={"max_frames": 0})
        self.assertEqual(res3.status_code, 422)

    def test_annotated_video_invalid_job_id(self) -> None:
        """Verify invalid job ID formats are rejected with 400."""
        # Path traversal attempt
        response = self.client.get("/api/vision/annotated-video/../../etc/passwd")
        self.assertIn(response.status_code, (400, 404))

        response = self.client.get("/api/vision/annotated-video/invalid-job-id")
        self.assertEqual(response.status_code, 400)

    def test_annotated_video_nonexistent(self) -> None:
        """Verify 404 for valid UUID job_id that doesn't exist on disk."""
        response = self.client.get("/api/vision/annotated-video/00000000-0000-0000-0000-000000000000")
        self.assertEqual(response.status_code, 404)

    def test_upload_invalid_extension(self) -> None:
        """Verify upload rejects unsupported file extensions."""
        response = self.client.post(
            "/api/vision/upload",
            files={"file": ("test.exe", io.BytesIO(b"executable content"), "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json().get("detail", ""))

    def test_upload_invalid_video_payload(self) -> None:
        """Verify upload rejects corrupted/non-video files even with .mp4 extension."""
        response = self.client.post(
            "/api/vision/upload",
            files={"file": ("corrupted.mp4", io.BytesIO(b"not a real video"), "video/mp4")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not be decoded", response.json().get("detail", "").lower())

    def test_analyze_video_mocked_pipeline(self) -> None:
        """Verify full /api/vision/analyze response schema using mocked inference results."""
        det1 = DetectedObject(
            class_name="car",
            confidence=0.88,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        det2 = DetectedObject(
            class_name="bus",
            confidence=0.92,
            bounding_box=BoundingBox(x_min=150, y_min=40, x_max=300, y_max=250),
            track_id=2,
        )
        mock_frames = [
            FrameDetectionResult.from_detections(0, 0.0, [det1, det2]),
            FrameDetectionResult.from_detections(1, 0.033, [det1, det2]),
            FrameDetectionResult.from_detections(2, 0.066, [det1, det2]),
            FrameDetectionResult.from_detections(3, 0.100, [det1, det2]),
            FrameDetectionResult.from_detections(4, 0.133, [det1, det2]),
        ]

        with patch("backend.vision.router._vision_service.process_video") as mock_proc, \
             patch("backend.vision.router.annotate_video") as mock_annot:
            mock_proc.return_value = (None, mock_frames)
            mock_annot.return_value = Path("data/vision/cache/mock_annot.mp4")

            response = self.client.post(
                "/api/vision/analyze",
                json={
                    "video_id": "sample_traffic_video",
                    "max_frames": 5,
                    "conf": 0.50,
                    "iou": 0.30,
                    "min_track_lifespan_frames": 5,
                    "generate_annotated_video": True,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("job_id", data)
            self.assertEqual(data["video_id"], "sample_traffic_video")
            self.assertIn("video_width", data)
            self.assertIn("video_height", data)
            self.assertIn("processing_time_seconds", data)
            self.assertIn("report", data)
            report = data["report"]

            self.assertEqual(report["raw_total_detections"], 10)
            self.assertEqual(report["raw_total_tracked_ids"], 2)
            self.assertEqual(report["raw_detection_events_by_class"]["car"], 5)
            self.assertEqual(report["raw_detection_events_by_class"]["bus"], 5)
            self.assertEqual(report["processed_total_tracked_ids"], 2)
            self.assertEqual(report["filtered_tracks_count"], 0)
            self.assertEqual(report["filtered_detections_count"], 0)
            self.assertEqual(
                report["raw_total_detections"],
                report["processed_total_detections"] + report["filtered_detections_count"],
            )
            self.assertEqual(
                report["raw_total_tracked_ids"],
                report["processed_total_tracked_ids"] + report["filtered_tracks_count"],
            )
            self.assertIn("disclaimer", report)

    def test_analyze_video_transient_filtering_and_class_smoothing(self) -> None:
        """Verify metric calculations for transient track filtering and temporal majority voting."""
        # Track 1: 5 frames, 4 'car' and 1 'motorcycle' (class switch resolved to car)
        # Track 2: 2 frames, 2 'truck' (transient noise, filtered because lifespan < 5)
        det_car = DetectedObject(
            class_name="car",
            confidence=0.85,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        det_moto = DetectedObject(
            class_name="motorcycle",
            confidence=0.60,
            bounding_box=BoundingBox(x_min=12, y_min=22, x_max=102, y_max=202),
            track_id=1,
        )
        det_truck = DetectedObject(
            class_name="truck",
            confidence=0.75,
            bounding_box=BoundingBox(x_min=200, y_min=50, x_max=350, y_max=300),
            track_id=2,
        )

        mock_frames = [
            FrameDetectionResult.from_detections(0, 0.0, [det_car, det_truck]),
            FrameDetectionResult.from_detections(1, 0.033, [det_car, det_truck]),
            FrameDetectionResult.from_detections(2, 0.066, [det_car]),
            FrameDetectionResult.from_detections(3, 0.100, [det_moto]),  # switch instance
            FrameDetectionResult.from_detections(4, 0.133, [det_car]),
        ]

        with patch("backend.vision.router._vision_service.process_video") as mock_proc, \
             patch("backend.vision.router.annotate_video") as mock_annot:
            mock_proc.return_value = (None, mock_frames)
            mock_annot.return_value = Path("data/vision/cache/mock_annot.mp4")

            response = self.client.post(
                "/api/vision/analyze",
                json={
                    "video_id": "sample_traffic_video",
                    "max_frames": 5,
                    "conf": 0.50,
                    "iou": 0.30,
                    "min_track_lifespan_frames": 5,
                    "generate_annotated_video": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            report = data["report"]

            # Raw counts
            self.assertEqual(report["raw_total_detections"], 7)  # 5 for track 1 + 2 for track 2
            self.assertEqual(report["raw_total_tracked_ids"], 2)

            # Filtering verification
            self.assertEqual(report["filtered_tracks_count"], 1)  # Track 2 has lifespan 2 < 5
            self.assertEqual(report["filtered_detections_count"], 2)  # Track 2 had 2 detections
            self.assertEqual(report["processed_total_tracked_ids"], 1)  # Only Track 1 retained
            self.assertEqual(report["processed_total_detections"], 5)  # 7 raw - 2 filtered = 5

            # Conservation properties
            self.assertEqual(
                report["raw_total_detections"],
                report["processed_total_detections"] + report["filtered_detections_count"],
            )
            self.assertEqual(
                report["raw_total_tracked_ids"],
                report["processed_total_tracked_ids"] + report["filtered_tracks_count"],
            )

            # Class smoothing verification:
            # 1 track had a class switch resolved (resolved_class_switches_count == 1)
            # 1 individual detection instance was reclassified (len(reclassified_detections) == 1)
            self.assertEqual(report["resolved_class_switches_count"], 1)
            self.assertEqual(len(report["reclassified_detections"]), 1)
            reclass = report["reclassified_detections"][0]
            self.assertEqual(reclass["track_id"], 1)
            self.assertEqual(reclass["raw_class"], "motorcycle")
            self.assertEqual(reclass["assigned_class"], "car")
            self.assertEqual(reclass["frame_number"], 3)

    def test_analyze_video_publish_false_by_default(self) -> None:
        """Verify publish_to_traffic_router defaults to False and traffic_publication is None."""
        det = DetectedObject(
            class_name="car",
            confidence=0.88,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        mock_frames = [FrameDetectionResult.from_detections(i, float(i), [det]) for i in range(5)]

        with patch("backend.vision.router._vision_service.process_video") as mock_proc, \
             patch("backend.vision.router.annotate_video"):
            mock_proc.return_value = (None, mock_frames)

            response = self.client.post(
                "/api/vision/analyze",
                json={
                    "video_id": "sample_traffic_video",
                    "max_frames": 5,
                    "generate_annotated_video": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsNone(data.get("traffic_publication"))

    def test_analyze_video_publish_uncalibrated_returns_clear_explanation(self) -> None:
        """Verify publishing an uncalibrated video fails cleanly with an explanatory message."""
        det = DetectedObject(
            class_name="car",
            confidence=0.88,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        mock_frames = [FrameDetectionResult.from_detections(i, float(i), [det]) for i in range(5)]

        with patch("backend.vision.router._vision_service.process_video") as mock_proc, \
             patch("backend.vision.router.annotate_video"):
            mock_proc.return_value = (None, mock_frames)

            # Request publication without camera metadata
            response = self.client.post(
                "/api/vision/analyze",
                json={
                    "video_id": "sample_traffic_video",
                    "max_frames": 5,
                    "generate_annotated_video": False,
                    "publish_to_traffic_router": True,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            pub = data.get("traffic_publication")
            self.assertIsNotNone(pub)
            self.assertFalse(pub["published"])
            self.assertEqual(pub["status"], "uncalibrated_video")
            self.assertIn("Calibrated camera-to-road mapping and a valid congestion estimate are required", pub["message"])

            # Verify no observation was registered in the shared adapter
            adapter = get_default_vision_adapter()
            self.assertEqual(len(adapter), 0)

    def test_analyze_video_publish_missing_congestion_estimate_returns_explanation(self) -> None:
        """Verify calibrated camera without congestion estimate refuses to publish without error."""
        det = DetectedObject(
            class_name="car",
            confidence=0.88,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        mock_frames = [FrameDetectionResult.from_detections(i, float(i), [det]) for i in range(5)]

        with patch("backend.vision.router._vision_service.process_video") as mock_proc, \
             patch("backend.vision.router.annotate_video"):
            mock_proc.return_value = (None, mock_frames)

            # Calibrated camera, but no explicit congestion index
            response = self.client.post(
                "/api/vision/analyze",
                json={
                    "video_id": "sample_traffic_video",
                    "max_frames": 5,
                    "generate_annotated_video": False,
                    "publish_to_traffic_router": True,
                    "camera_metadata": {
                        "camera_id": "cam_ec_01",
                        "latitude": 12.84,
                        "longitude": 77.67,
                        "heading_degrees": 90.0,
                        "target_osm_u": 101,
                        "target_osm_v": 102,
                        "target_osm_key": 0,
                        "is_spatially_calibrated": True,
                    },
                    "experimental_congestion_index": None,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            pub = data.get("traffic_publication")
            self.assertIsNotNone(pub)
            self.assertFalse(pub["published"])
            self.assertEqual(pub["status"], "missing_congestion_estimate")
            self.assertIn("Vehicle counts and tracking IDs cannot establish congestion levels alone", pub["message"])

            adapter = get_default_vision_adapter()
            self.assertEqual(len(adapter), 0)

    def test_analyze_video_publish_valid_observation_ingested_in_adapter(self) -> None:
        """Verify calibrated video with valid congestion index publishes observation to shared adapter."""
        det = DetectedObject(
            class_name="car",
            confidence=0.88,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        mock_frames = [FrameDetectionResult.from_detections(i, float(i), [det]) for i in range(5)]

        with patch("backend.vision.router._vision_service.process_video") as mock_proc, \
             patch("backend.vision.router.annotate_video"):
            mock_proc.return_value = (None, mock_frames)

            response = self.client.post(
                "/api/vision/analyze",
                json={
                    "video_id": "sample_traffic_video",
                    "max_frames": 5,
                    "generate_annotated_video": False,
                    "publish_to_traffic_router": True,
                    "camera_metadata": {
                        "camera_id": "cam_ec_01",
                        "latitude": 12.84,
                        "longitude": 77.67,
                        "heading_degrees": 90.0,
                        "target_osm_u": 101,
                        "target_osm_v": 102,
                        "target_osm_key": 0,
                        "is_spatially_calibrated": True,
                    },
                    "experimental_congestion_index": 0.85,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            pub = data.get("traffic_publication")
            self.assertIsNotNone(pub)
            self.assertTrue(pub["published"])
            self.assertEqual(pub["status"], "published")
            self.assertEqual(pub["target_edge"], [101, 102, 0])
            self.assertEqual(pub["congestion_index"], 0.85)

            # Confirm shared adapter now has this observation
            adapter = get_default_vision_adapter()
            from backend.traffic import TrafficEdgeKey
            edge_key = TrafficEdgeKey(osm_u=101, osm_v=102, key=0)
            self.assertEqual(adapter.get_congestion_index(edge_key), 0.85)
