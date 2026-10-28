"""Unit tests for the vision schemas and service foundation."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from backend.vision.evaluate_pipeline import (
    DEFAULT_EVAL_MAX_FRAMES,
    PipelineEvaluationReport,
    ReclassifiedDetection,
    TrackletDetail,
    build_arg_parser,
    evaluate_frames,
    evaluate_video,
    format_evaluation_report,
    main as eval_main,
    save_evaluation_outputs,
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


class TestBoundingBox(unittest.TestCase):
    """Tests for BoundingBox coordinate validation."""

    def test_valid_bounding_box(self) -> None:
        box = BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0)
        self.assertEqual(box.x_min, 10.0)
        self.assertEqual(box.y_min, 20.0)
        self.assertEqual(box.x_max, 100.0)
        self.assertEqual(box.y_max, 200.0)

    def test_negative_coordinates_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BoundingBox(x_min=-1.0, y_min=0.0, x_max=10.0, y_max=10.0)
        with self.assertRaises(ValidationError):
            BoundingBox(x_min=0.0, y_min=-1.0, x_max=10.0, y_max=10.0)

    def test_zero_width_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            BoundingBox(x_min=50.0, y_min=10.0, x_max=50.0, y_max=60.0)
        self.assertIn("strictly greater than x_min", str(context.exception))

    def test_zero_height_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            BoundingBox(x_min=10.0, y_min=40.0, x_max=50.0, y_max=40.0)
        self.assertIn("strictly greater than y_min", str(context.exception))

    def test_inverted_coordinates_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BoundingBox(x_min=100.0, y_min=10.0, x_max=50.0, y_max=60.0)
        with self.assertRaises(ValidationError):
            BoundingBox(x_min=10.0, y_min=100.0, x_max=50.0, y_max=60.0)


class TestDetectedObject(unittest.TestCase):
    """Tests for DetectedObject validation."""

    def setUp(self) -> None:
        self.box = BoundingBox(x_min=0.0, y_min=0.0, x_max=50.0, y_max=50.0)

    def test_valid_detected_object(self) -> None:
        obj = DetectedObject(
            class_name="car",
            confidence=0.92,
            bounding_box=self.box,
            track_id=10,
        )
        self.assertEqual(obj.class_name, "car")
        self.assertAlmostEqual(obj.confidence, 0.92)
        self.assertEqual(obj.track_id, 10)

    def test_confidence_boundary_values(self) -> None:
        obj_min = DetectedObject(class_name="car", confidence=0.0, bounding_box=self.box)
        obj_max = DetectedObject(class_name="bus", confidence=1.0, bounding_box=self.box)
        self.assertEqual(obj_min.confidence, 0.0)
        self.assertEqual(obj_max.confidence, 1.0)

    def test_invalid_confidence_scores_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DetectedObject(class_name="car", confidence=-0.01, bounding_box=self.box)
        with self.assertRaises(ValidationError):
            DetectedObject(class_name="car", confidence=1.001, bounding_box=self.box)

    def test_empty_class_name_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DetectedObject(class_name="", confidence=0.5, bounding_box=self.box)

    def test_negative_track_id_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DetectedObject(
                class_name="truck",
                confidence=0.8,
                bounding_box=self.box,
                track_id=-1,
            )


class TestFrameDetectionResult(unittest.TestCase):
    """Tests for FrameDetectionResult consistency."""

    def setUp(self) -> None:
        box = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
        self.det1 = DetectedObject(class_name="car", confidence=0.9, bounding_box=box)
        self.det2 = DetectedObject(class_name="motorcycle", confidence=0.8, bounding_box=box)

    def test_detection_count_automatically_populated(self) -> None:
        frame = FrameDetectionResult(
            frame_number=1,
            timestamp_seconds=0.033,
            detections=[self.det1, self.det2],
        )
        self.assertEqual(frame.detection_count, 2)

    def test_detection_count_consistency_enforced(self) -> None:
        with self.assertRaises(ValidationError) as context:
            FrameDetectionResult(
                frame_number=1,
                timestamp_seconds=0.033,
                detections=[self.det1],
                detection_count=99,
            )
        self.assertIn("does not match", str(context.exception))

    def test_empty_detections_yields_zero_count(self) -> None:
        frame = FrameDetectionResult(frame_number=0, timestamp_seconds=0.0)
        self.assertEqual(frame.detection_count, 0)
        self.assertEqual(frame.detections, [])

    def test_from_detections_helper(self) -> None:
        frame = FrameDetectionResult.from_detections(
            frame_number=5,
            timestamp_seconds=0.2,
            detections=[self.det1],
        )
        self.assertEqual(frame.detection_count, 1)
        self.assertEqual(frame.frame_number, 5)

    def test_negative_frame_or_timestamp_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            FrameDetectionResult(frame_number=-1, timestamp_seconds=0.0)
        with self.assertRaises(ValidationError):
            FrameDetectionResult(frame_number=0, timestamp_seconds=-0.1)


class TestVideoAnalysisSummary(unittest.TestCase):
    """Tests for VideoAnalysisSummary metrics validation."""

    def test_valid_video_summary(self) -> None:
        summary = VideoAnalysisSummary(
            video_name="test_traffic.mp4",
            total_frames=300,
            processed_frames=30,
            duration_seconds=10.0,
            fps=30.0,
            class_counts={"car": 25, "bus": 5},
            total_detections=30,
        )
        self.assertEqual(summary.video_name, "test_traffic.mp4")
        self.assertEqual(summary.class_counts["car"], 25)
        self.assertEqual(summary.total_detections, 30)
        self.assertIn("per-frame detection events", summary.disclaimer)

    def test_processed_frames_cannot_exceed_total_frames(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VideoAnalysisSummary(
                video_name="test.mp4",
                total_frames=100,
                processed_frames=101,
                duration_seconds=5.0,
                fps=20.0,
                class_counts={},
                total_detections=0,
            )
        self.assertIn("cannot exceed total_frames", str(context.exception))

    def test_negative_class_count_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VideoAnalysisSummary(
                video_name="test.mp4",
                total_frames=100,
                processed_frames=50,
                duration_seconds=5.0,
                fps=20.0,
                class_counts={"car": -2},
                total_detections=0,
            )
        self.assertIn("must be non-negative", str(context.exception))

    def test_total_detections_must_match_sum_of_class_counts(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VideoAnalysisSummary(
                video_name="test.mp4",
                total_frames=100,
                processed_frames=50,
                duration_seconds=5.0,
                fps=20.0,
                class_counts={"car": 5, "bus": 2},
                total_detections=10,  # Mismatched (5 + 2 = 7 != 10)
            )
        self.assertIn("does not match the sum of class_counts", str(context.exception))


class TestVideoAnalysisService(unittest.TestCase):
    """Tests for VideoAnalysisService foundation."""

    def test_service_initialization_does_not_load_model(self) -> None:
        service = VideoAnalysisService()
        self.assertFalse(service.is_model_loaded)
        self.assertEqual(service.model_path, Path(DEFAULT_MODEL_NAME))
        self.assertEqual(service.device, "cpu")

    def test_missing_model_weights_raises_file_not_found(self) -> None:
        non_existent_weights = Path("/tmp/definitely_missing_yolo_weights_12345.pt")
        service = VideoAnalysisService(model_path=non_existent_weights)

        with self.assertRaises(FileNotFoundError) as context:
            service.get_model()

        self.assertIn("Model weights file not found", str(context.exception))
        self.assertIn("Automatic downloads are disabled", str(context.exception))
        self.assertFalse(service.is_model_loaded)

    @patch("pathlib.Path.is_file")
    def test_model_loading_and_reuse_when_file_exists(self, mock_is_file) -> None:
        mock_is_file.return_value = True
        service = VideoAnalysisService(model_path="/dummy/path/weights.pt")

        mock_yolo_cls = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            # First call should instantiate YOLO
            model1 = service.get_model()
            self.assertEqual(model1, mock_yolo_instance)
            self.assertTrue(service.is_model_loaded)
            self.assertEqual(mock_yolo_cls.call_count, 1)

            # Second call should reuse the same instance without re-instantiation
            model2 = service.get_model()
            self.assertEqual(model2, mock_yolo_instance)
            self.assertEqual(mock_yolo_cls.call_count, 1)

    def test_build_summary_aggregates_cumulative_detection_events(self) -> None:
        box = BoundingBox(x_min=1.0, y_min=1.0, x_max=10.0, y_max=10.0)
        frame1 = FrameDetectionResult(
            frame_number=0,
            timestamp_seconds=0.0,
            detections=[
                DetectedObject(class_name="car", confidence=0.9, bounding_box=box),
                DetectedObject(class_name="bus", confidence=0.85, bounding_box=box),
            ],
        )
        frame2 = FrameDetectionResult(
            frame_number=1,
            timestamp_seconds=0.5,
            detections=[
                DetectedObject(class_name="car", confidence=0.92, bounding_box=box),
                DetectedObject(class_name="car", confidence=0.88, bounding_box=box),
                DetectedObject(class_name="motorcycle", confidence=0.78, bounding_box=box),
            ],
        )

        service = VideoAnalysisService()
        summary = service.build_summary(
            video_name="junction_feed.mp4",
            total_frames=100,
            processed_frames=2,
            duration_seconds=10.0,
            fps=10.0,
            frame_results=[frame1, frame2],
        )

        self.assertEqual(summary.video_name, "junction_feed.mp4")
        self.assertEqual(summary.processed_frames, 2)
        self.assertEqual(summary.total_frames, 100)
        self.assertEqual(summary.class_counts, {"car": 3, "bus": 1, "motorcycle": 1})
        self.assertEqual(summary.total_detections, 5)
        self.assertIn("per-frame detection events", summary.disclaimer)


class TestVideoMetadataExtraction(unittest.TestCase):
    """Tests for VideoAnalysisService.extract_video_metadata."""

    def setUp(self) -> None:
        self.service = VideoAnalysisService()

    def test_missing_video_path_raises_file_not_found(self) -> None:
        non_existent = "/tmp/definitely_missing_video_12345.mp4"
        with self.assertRaises(FileNotFoundError) as context:
            self.service.extract_video_metadata(non_existent)
        self.assertIn("Video file not found", str(context.exception))

    def test_directory_path_raises_value_error(self) -> None:
        # A directory exists but is not a regular file
        with self.assertRaises(ValueError) as context:
            self.service.extract_video_metadata(".")
        self.assertIn("not a regular file", str(context.exception))

    def test_opencv_cannot_open_raises_value_error(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with self.assertRaises(ValueError) as context:
                self.service.extract_video_metadata(__file__)

            self.assertIn("OpenCV could not open video file", str(context.exception))
            mock_cap.release.assert_called_once()

    def test_successful_metadata_extraction(self) -> None:
        import cv2

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        def mock_get(prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 301.0
            if prop == cv2.CAP_PROP_FPS:
                return 29.97
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 360.0
            return 0.0

        mock_cap.get.side_effect = mock_get

        with patch("cv2.VideoCapture", return_value=mock_cap):
            metadata = self.service.extract_video_metadata(__file__)

            self.assertEqual(metadata["video_name"], Path(__file__).name)
            self.assertEqual(metadata["frame_count"], 301)
            self.assertAlmostEqual(metadata["fps"], 29.97)
            self.assertEqual(metadata["width"], 640)
            self.assertEqual(metadata["height"], 360)
            self.assertAlmostEqual(metadata["duration_seconds"], 301 / 29.97, places=2)
            mock_cap.release.assert_called_once()

    def test_video_capture_released_on_error(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = RuntimeError("Hardware decoding fault")

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with self.assertRaises(RuntimeError):
                self.service.extract_video_metadata(__file__)

            mock_cap.release.assert_called_once()


class TestFrameDetectionExtraction(unittest.TestCase):
    """Tests for VideoAnalysisService.extract_frame_detections."""

    def setUp(self) -> None:
        self.service = VideoAnalysisService()

    def test_empty_frame_returns_zero_detections(self) -> None:
        # None result
        res1 = self.service.extract_frame_detections(None, 0, 0.0)
        self.assertEqual(res1.detection_count, 0)
        self.assertEqual(res1.detections, [])

        # Result with None boxes
        mock_none_boxes = MagicMock()
        mock_none_boxes.boxes = None
        res2 = self.service.extract_frame_detections(mock_none_boxes, 1, 0.033)
        self.assertEqual(res2.detection_count, 0)

        # Result with empty boxes list
        mock_empty_boxes = MagicMock()
        mock_empty_boxes.boxes = []
        res3 = self.service.extract_frame_detections(mock_empty_boxes, 2, 0.066)
        self.assertEqual(res3.detection_count, 0)

    def test_filtering_to_four_vehicle_classes(self) -> None:
        # COCO IDs: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 15=cat
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 7
        mock_boxes.xyxy = [
            [10.0, 10.0, 20.0, 20.0],
            [20.0, 20.0, 30.0, 30.0],
            [30.0, 30.0, 40.0, 40.0],
            [40.0, 40.0, 50.0, 50.0],
            [50.0, 50.0, 60.0, 60.0],
            [60.0, 60.0, 70.0, 70.0],
            [70.0, 70.0, 80.0, 80.0],
        ]
        mock_boxes.conf = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.95]
        mock_boxes.cls = [0, 1, 2, 3, 5, 7, 15]
        mock_boxes.id = None

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            15: "cat",
        }

        res = self.service.extract_frame_detections(mock_result, 0, 0.0)
        self.assertEqual(res.detection_count, 4)
        detected_classes = [det.class_name for det in res.detections]
        self.assertEqual(detected_classes, ["car", "motorcycle", "bus", "truck"])

    def test_bounding_box_conversion_and_clamping(self) -> None:
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xyxy = [[-10.0, -5.0, 50.0, 60.0]]
        mock_boxes.conf = [0.85]
        mock_boxes.cls = [2]
        mock_boxes.id = None

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        res = self.service.extract_frame_detections(mock_result, 0, 0.0)
        self.assertEqual(res.detection_count, 1)
        box = res.detections[0].bounding_box
        self.assertEqual(box.x_min, 0.0)
        self.assertEqual(box.y_min, 0.0)
        self.assertEqual(box.x_max, 50.0)
        self.assertEqual(box.y_max, 60.0)

    def test_degenerate_bounding_box_discarded(self) -> None:
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 2
        # First box has zero width (50 -> 50), second is valid
        mock_boxes.xyxy = [
            [50.0, 20.0, 50.0, 60.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
        mock_boxes.conf = [0.9, 0.8]
        mock_boxes.cls = [2, 2]
        mock_boxes.id = None

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        res = self.service.extract_frame_detections(mock_result, 0, 0.0)
        self.assertEqual(res.detection_count, 1)
        self.assertEqual(res.detections[0].bounding_box.x_min, 10.0)

    def test_confidence_handling_and_clamping(self) -> None:
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 2
        mock_boxes.xyxy = [[1.0, 1.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0]]
        mock_boxes.conf = [1.2, -0.1]
        mock_boxes.cls = [2, 2]
        mock_boxes.id = None

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        res = self.service.extract_frame_detections(mock_result, 0, 0.0)
        self.assertEqual(res.detection_count, 2)
        self.assertEqual(res.detections[0].confidence, 1.0)
        self.assertEqual(res.detections[1].confidence, 0.0)

    def test_tracking_id_preserved_when_present(self) -> None:
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xyxy = [[10.0, 10.0, 50.0, 50.0]]
        mock_boxes.conf = [0.95]
        mock_boxes.cls = [5]  # bus
        mock_boxes.id = [19]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        res = self.service.extract_frame_detections(mock_result, 58, 1.935)
        self.assertEqual(res.detection_count, 1)
        self.assertEqual(res.detections[0].track_id, 19)
        self.assertEqual(res.detections[0].class_name, "bus")

    def test_missing_tracking_id_remains_none(self) -> None:
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xyxy = [[10.0, 10.0, 50.0, 50.0]]
        mock_boxes.conf = [0.9]
        mock_boxes.cls = [2]
        mock_boxes.id = None

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        res = self.service.extract_frame_detections(mock_result, 0, 0.0)
        self.assertEqual(res.detection_count, 1)
        self.assertIsNone(res.detections[0].track_id)

    def test_frame_numbers_and_timestamps(self) -> None:
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 0

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        res = self.service.extract_frame_detections(mock_result, 150, 5.005)
        self.assertEqual(res.frame_number, 150)
        self.assertAlmostEqual(res.timestamp_seconds, 5.005)


class TestVideoProcessingAndSummary(unittest.TestCase):
    """Tests for VideoAnalysisService tracking, processing, and summary creation."""

    def setUp(self) -> None:
        self.service = VideoAnalysisService()
        self.box = BoundingBox(x_min=5.0, y_min=5.0, x_max=50.0, y_max=50.0)

    def test_build_summary_with_tracked_ids(self) -> None:
        frame1 = FrameDetectionResult.from_detections(
            frame_number=0,
            timestamp_seconds=0.0,
            detections=[
                DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=1),
                DetectedObject(class_name="bus", confidence=0.85, bounding_box=self.box, track_id=2),
            ],
        )
        frame2 = FrameDetectionResult.from_detections(
            frame_number=1,
            timestamp_seconds=0.033,
            detections=[
                DetectedObject(class_name="car", confidence=0.91, bounding_box=self.box, track_id=1),
                DetectedObject(class_name="car", confidence=0.88, bounding_box=self.box, track_id=3),
                DetectedObject(class_name="motorcycle", confidence=0.75, bounding_box=self.box, track_id=4),
            ],
        )

        summary = self.service.build_summary(
            video_name="test_feed.mp4",
            total_frames=10,
            processed_frames=2,
            duration_seconds=0.33,
            fps=30.0,
            frame_results=[frame1, frame2],
        )

        self.assertEqual(summary.class_counts, {"car": 3, "bus": 1, "motorcycle": 1})
        self.assertEqual(summary.total_detections, 5)
        self.assertIsNotNone(summary.tracked_id_counts)
        self.assertEqual(summary.tracked_id_counts, {"car": 2, "bus": 1, "motorcycle": 1})
        self.assertEqual(summary.total_tracked_ids, 4)
        self.assertIn("per-frame detection events", summary.disclaimer)

    def test_build_summary_multi_class_tracked_id_union(self) -> None:
        # Reproduces the Colab notebook behavior where a single track ID was assigned across multiple classes
        frame1 = FrameDetectionResult.from_detections(
            frame_number=0,
            timestamp_seconds=0.0,
            detections=[
                DetectedObject(class_name="car", confidence=0.7, bounding_box=self.box, track_id=2),
            ],
        )
        frame2 = FrameDetectionResult.from_detections(
            frame_number=1,
            timestamp_seconds=0.033,
            detections=[
                DetectedObject(class_name="bus", confidence=0.85, bounding_box=self.box, track_id=2),
            ],
        )

        summary = self.service.build_summary(
            video_name="id_switch.mp4",
            total_frames=2,
            processed_frames=2,
            duration_seconds=0.066,
            fps=30.0,
            frame_results=[frame1, frame2],
        )

        self.assertEqual(summary.class_counts, {"car": 1, "bus": 1})
        self.assertEqual(summary.total_detections, 2)
        # Class counts sum to 2
        self.assertEqual(summary.tracked_id_counts, {"car": 1, "bus": 1})
        # Distinct IDs union is {2}, length 1
        self.assertEqual(summary.total_tracked_ids, 1)

    def test_process_video_missing_weights_raises_file_not_found(self) -> None:
        service = VideoAnalysisService(model_path="/tmp/missing_weights.pt")
        with self.assertRaises(FileNotFoundError) as context:
            service.process_video(__file__)
        self.assertIn("Model weights file not found", str(context.exception))
        self.assertIn("Automatic downloads are disabled", str(context.exception))

    def test_process_video_and_analyze_video_mocked(self) -> None:
        service = VideoAnalysisService(model_path="/mock/weights.pt")

        mock_model = MagicMock()
        service._model = mock_model

        mock_boxes_f0 = MagicMock()
        mock_boxes_f0.__len__.return_value = 1
        mock_boxes_f0.xyxy = [[10.0, 10.0, 50.0, 50.0]]
        mock_boxes_f0.conf = [0.9]
        mock_boxes_f0.cls = [2]
        mock_boxes_f0.id = [1]
        mock_res_f0 = MagicMock(boxes=mock_boxes_f0)

        mock_boxes_f1 = MagicMock()
        mock_boxes_f1.__len__.return_value = 0
        mock_res_f1 = MagicMock(boxes=mock_boxes_f1)

        mock_boxes_f2 = MagicMock()
        mock_boxes_f2.__len__.return_value = 1
        mock_boxes_f2.xyxy = [[20.0, 20.0, 60.0, 60.0]]
        mock_boxes_f2.conf = [0.85]
        mock_boxes_f2.cls = [5]
        mock_boxes_f2.id = [2]
        mock_res_f2 = MagicMock(boxes=mock_boxes_f2)

        mock_model.track.side_effect = [
            [mock_res_f0],
            [mock_res_f1],
            [mock_res_f2],
        ]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        import cv2

        def mock_get(prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 3.0
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        mock_cap.get.side_effect = mock_get

        import numpy as np
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
            (False, None),
        ]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            summary, frames = service.process_video(
                __file__,
                conf=0.50,
                iou=0.30,
                tracker="bytetrack.yaml",
            )

            self.assertEqual(len(frames), 3)
            self.assertEqual(summary.processed_frames, 3)
            self.assertEqual(summary.total_frames, 3)
            self.assertEqual(summary.total_detections, 2)
            self.assertEqual(summary.class_counts, {"car": 1, "bus": 1})
            self.assertEqual(summary.tracked_id_counts, {"car": 1, "bus": 1})
            self.assertEqual(summary.total_tracked_ids, 2)
            self.assertEqual(mock_model.track.call_count, 3)

            # Frame 0 must pass persist=False to reset any stale tracker state
            self.assertFalse(mock_model.track.call_args_list[0].kwargs["persist"])
            # Consecutive frames 1 and 2 must pass persist=True for trajectory continuity
            self.assertTrue(mock_model.track.call_args_list[1].kwargs["persist"])
            self.assertTrue(mock_model.track.call_args_list[2].kwargs["persist"])

            mock_cap.release.assert_called_once()

    def test_process_video_resets_tracker_between_distinct_videos(self) -> None:
        """Verify that processing a second video on the same service resets tracker state."""
        service = VideoAnalysisService(model_path="/mock/weights.pt")

        mock_predictor = MagicMock()
        mock_predictor.trackers = ["active_tracker_video_1"]

        mock_model = MagicMock()
        mock_model.predictor = mock_predictor
        service._model = mock_model

        mock_res = MagicMock(boxes=[])
        mock_model.track.return_value = [mock_res]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        import cv2
        mock_cap.get.side_effect = lambda prop: 10.0 if prop == cv2.CAP_PROP_FRAME_COUNT else 30.0

        import numpy as np
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            # Process Video 1 (2 frames)
            summary1, frames1 = service.process_video(__file__, max_frames=2)
            self.assertEqual(len(frames1), 2)
            # Frame 0 of video 1 must have persist=False
            self.assertFalse(mock_model.track.call_args_list[0].kwargs["persist"])
            # Frame 1 of video 1 must have persist=True
            self.assertTrue(mock_model.track.call_args_list[1].kwargs["persist"])

            # Simulate tracker state leftover on predictor after video 1
            mock_predictor.trackers = ["stale_tracker_video_1"]

            # Process Video 2 on the SAME service instance (2 frames)
            summary2, frames2 = service.process_video(__file__, max_frames=2)
            self.assertEqual(len(frames2), 2)

            # Predictor trackers should have been reset to an empty list
            self.assertEqual(mock_predictor.trackers, [])
            # Frame 0 of video 2 (call index 2) must pass persist=False to reset tracker
            self.assertFalse(mock_model.track.call_args_list[2].kwargs["persist"])
            # Frame 1 of video 2 (call index 3) must pass persist=True
            self.assertTrue(mock_model.track.call_args_list[3].kwargs["persist"])

    def test_reset_tracker_handles_uninitialized_and_mock_predictors(self) -> None:
        """Verify reset_tracker is safe when model or predictor lacks trackers."""
        service = VideoAnalysisService()
        # When model is None, should not raise
        service.reset_tracker()

        # When model has no predictor attribute, should not raise
        service._model = object()
        service.reset_tracker()

        # When predictor has no trackers attribute, should not raise
        mock_model = MagicMock(spec=[])
        service._model = mock_model
        service.reset_tracker()

    def test_git_ignores_vision_weights_and_pt_files(self) -> None:
        """Verify that .gitignore properly excludes backend/vision/weights/ and *.pt."""
        gitignore_path = Path(__file__).parents[2] / ".gitignore"
        self.assertTrue(gitignore_path.is_file(), ".gitignore must exist")
        content = gitignore_path.read_text(encoding="utf-8")
        self.assertIn("backend/vision/weights/", content)
        self.assertIn("*.pt", content)

    def test_max_frames_limits_processing(self) -> None:
        service = VideoAnalysisService(model_path="/mock/weights.pt")
        mock_model = MagicMock()
        service._model = mock_model

        mock_res = MagicMock(boxes=[])
        mock_model.track.return_value = [mock_res]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        import cv2
        mock_cap.get.side_effect = lambda prop: 100.0 if prop == cv2.CAP_PROP_FRAME_COUNT else 30.0

        import numpy as np
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            summary, frames = service.process_video(
                __file__,
                max_frames=2,
            )

            self.assertEqual(len(frames), 2)
            self.assertEqual(summary.processed_frames, 2)
            self.assertEqual(summary.total_frames, 100)
            self.assertEqual(frames[0].frame_number, 0)
            self.assertEqual(frames[1].frame_number, 1)
            self.assertEqual(mock_model.track.call_count, 2)
            mock_cap.release.assert_called_once()

    def test_process_video_releases_capture_on_error(self) -> None:
        service = VideoAnalysisService(model_path="/mock/weights.pt")
        service._model = MagicMock()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = RuntimeError("OpenCV frame decode crash")

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with self.assertRaises(RuntimeError):
                service.process_video(__file__)

            mock_cap.release.assert_called_once()

    def test_resolve_model_path_resolution_order(self) -> None:
        # Bare name should resolve
        p1 = VideoAnalysisService.resolve_model_path("yolo11n.pt")
        self.assertIsInstance(p1, Path)

        # Expected weights path constant
        self.assertEqual(EXPECTED_WEIGHTS_PATH, Path("backend/vision/weights/yolo11n.pt"))
        self.assertEqual(DEFAULT_MODEL_NAME, str(EXPECTED_WEIGHTS_PATH))


class TestCameraMetadata(unittest.TestCase):
    """Tests for CameraMetadata validation and constraints."""

    def test_valid_minimal_camera_metadata(self) -> None:
        camera = CameraMetadata(
            camera_id="cam_main_st_01",
            latitude=19.0760,
            longitude=72.8777,
            heading_degrees=45.0,
        )
        self.assertEqual(camera.camera_id, "cam_main_st_01")
        self.assertEqual(camera.latitude, 19.0760)
        self.assertEqual(camera.longitude, 72.8777)
        self.assertEqual(camera.heading_degrees, 45.0)
        self.assertIsNone(camera.elevation_meters)
        self.assertIsNone(camera.tilt_degrees)
        self.assertIsNone(camera.target_osm_u)
        self.assertIsNone(camera.target_osm_v)
        self.assertEqual(camera.target_osm_key, 0)
        self.assertFalse(camera.is_spatially_calibrated)

    def test_valid_full_camera_metadata(self) -> None:
        camera = CameraMetadata(
            camera_id="cam_junction_02",
            latitude=19.0760,
            longitude=72.8777,
            elevation_meters=8.5,
            heading_degrees=180.0,
            tilt_degrees=-25.0,
            target_osm_u=1001,
            target_osm_v=1002,
            target_osm_key=0,
            is_spatially_calibrated=True,
        )
        self.assertEqual(camera.elevation_meters, 8.5)
        self.assertEqual(camera.tilt_degrees, -25.0)
        self.assertEqual(camera.target_osm_u, 1001)
        self.assertEqual(camera.target_osm_v, 1002)
        self.assertTrue(camera.is_spatially_calibrated)

    def test_valid_boundary_coordinates_and_heading(self) -> None:
        cam_min = CameraMetadata(
            camera_id="cam_min",
            latitude=-90.0,
            longitude=-180.0,
            heading_degrees=0.0,
        )
        self.assertEqual(cam_min.latitude, -90.0)
        self.assertEqual(cam_min.longitude, -180.0)
        self.assertEqual(cam_min.heading_degrees, 0.0)

        cam_max = CameraMetadata(
            camera_id="cam_max",
            latitude=90.0,
            longitude=180.0,
            heading_degrees=359.99,
        )
        self.assertEqual(cam_max.latitude, 90.0)
        self.assertEqual(cam_max.longitude, 180.0)
        self.assertEqual(cam_max.heading_degrees, 359.99)

    def test_invalid_latitude_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=90.001, longitude=0.0, heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=-90.001, longitude=0.0, heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=float("nan"), longitude=0.0, heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=float("inf"), longitude=0.0, heading_degrees=0.0)

    def test_invalid_longitude_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=180.001, heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=-180.001, heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=float("nan"), heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=float("-inf"), heading_degrees=0.0)

    def test_invalid_heading_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=-0.01)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=360.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=float("nan"))

    def test_invalid_elevation_and_tilt_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(
                camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=0.0,
                elevation_meters=-0.1,
            )
        with self.assertRaises(ValidationError):
            CameraMetadata(
                camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=0.0,
                elevation_meters=float("nan"),
            )
        with self.assertRaises(ValidationError):
            CameraMetadata(
                camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=0.0,
                tilt_degrees=-90.1,
            )
        with self.assertRaises(ValidationError):
            CameraMetadata(
                camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=0.0,
                tilt_degrees=90.1,
            )
        with self.assertRaises(ValidationError):
            CameraMetadata(
                camera_id="c", latitude=0.0, longitude=0.0, heading_degrees=0.0,
                tilt_degrees=float("inf"),
            )

    def test_empty_or_whitespace_camera_id_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="", latitude=0.0, longitude=0.0, heading_degrees=0.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="   ", latitude=0.0, longitude=0.0, heading_degrees=0.0)

    def test_missing_required_camera_fields_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(latitude=0.0, longitude=0.0, heading_degrees=0.0)  # missing camera_id
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", longitude=0.0, heading_degrees=0.0)  # missing latitude
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, heading_degrees=0.0)  # missing longitude
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=0.0, longitude=0.0)  # missing heading_degrees

    def test_immutability_and_serialization(self) -> None:
        camera = CameraMetadata(
            camera_id="cam_immutability",
            latitude=19.0,
            longitude=72.0,
            heading_degrees=90.0,
        )
        with self.assertRaises(ValidationError):
            camera.latitude = 20.0

        dumped = camera.model_dump()
        self.assertEqual(dumped["camera_id"], "cam_immutability")
        self.assertEqual(dumped["latitude"], 19.0)
        json_str = camera.model_dump_json()
        self.assertIn('"cam_immutability"', json_str)

    def test_spatial_calibration_requires_target_osm_nodes(self) -> None:
        with self.assertRaises(ValidationError) as context:
            CameraMetadata(
                camera_id="cam_uncal",
                latitude=19.0,
                longitude=72.0,
                heading_degrees=90.0,
                is_spatially_calibrated=True,  # Missing target_osm_u and target_osm_v
            )
        self.assertIn("is_spatially_calibrated cannot be True", str(context.exception))

    def test_partial_target_osm_nodes_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            CameraMetadata(
                camera_id="c",
                latitude=19.0,
                longitude=72.0,
                heading_degrees=90.0,
                target_osm_u=1001,  # target_osm_v is None
            )
        self.assertIn("must both be specified together", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            CameraMetadata(
                camera_id="c",
                latitude=19.0,
                longitude=72.0,
                heading_degrees=90.0,
                target_osm_v=1002,  # target_osm_u is None
            )
        self.assertIn("must both be specified together", str(context.exception))

    def test_self_loop_target_edge_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            CameraMetadata(
                camera_id="c",
                latitude=19.0,
                longitude=72.0,
                heading_degrees=90.0,
                target_osm_u=1001,
                target_osm_v=1001,  # Self loop u == v
            )
        self.assertIn("self-loops are not allowed", str(context.exception))

    def test_boolean_coordinates_and_nodes_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraMetadata(camera_id="c", latitude=True, longitude=72.0, heading_degrees=90.0)
        with self.assertRaises(ValidationError):
            CameraMetadata(
                camera_id="c",
                latitude=19.0,
                longitude=72.0,
                heading_degrees=90.0,
                target_osm_u=True,
                target_osm_v=1002,
            )

    def test_camera_target_aliases(self) -> None:
        cam = CameraMetadata(
            camera_id="c_alias",
            latitude=19.0,
            longitude=72.0,
            heading_degrees=90.0,
            target_u=1001,
            target_v=1002,
            target_key=1,
        )
        self.assertEqual(cam.target_osm_u, 1001)
        self.assertEqual(cam.target_osm_v, 1002)
        self.assertEqual(cam.target_osm_key, 1)


class TestObservationTimeEnvelope(unittest.TestCase):
    """Tests for ObservationTimeEnvelope validation and constraints."""

    def setUp(self) -> None:
        self.start_utc = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
        self.end_utc = datetime(2026, 10, 27, 10, 5, 0, tzinfo=timezone.utc)

    def test_valid_observation_time_envelope(self) -> None:
        envelope = ObservationTimeEnvelope(
            start_time=self.start_utc,
            end_time=self.end_utc,
            duration_seconds=300.0,
        )
        self.assertEqual(envelope.start_time, self.start_utc)
        self.assertEqual(envelope.end_time, self.end_utc)
        self.assertEqual(envelope.duration_seconds, 300.0)
        self.assertEqual(envelope.time_to_live_seconds, 900.0)

    def test_auto_duration_calculation_when_omitted(self) -> None:
        envelope = ObservationTimeEnvelope(
            start_time=self.start_utc,
            end_time=self.end_utc,
        )
        self.assertEqual(envelope.duration_seconds, 300.0)

    def test_captured_at_utc_alias_support(self) -> None:
        envelope = ObservationTimeEnvelope(
            captured_at_utc=self.start_utc,
            end_time=self.end_utc,
        )
        self.assertEqual(envelope.start_time, self.start_utc)
        self.assertEqual(envelope.duration_seconds, 300.0)

    def test_naive_timestamp_rejected(self) -> None:
        naive_start = datetime(2026, 10, 27, 10, 0, 0)
        naive_end = datetime(2026, 10, 27, 10, 5, 0)

        with self.assertRaises(ValidationError) as context:
            ObservationTimeEnvelope(
                start_time=naive_start,
                end_time=self.end_utc,
                duration_seconds=300.0,
            )
        self.assertIn("must be timezone-aware", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=naive_end,
                duration_seconds=300.0,
            )
        self.assertIn("must be timezone-aware", str(context.exception))

    def test_end_time_not_strictly_after_start_time_rejected(self) -> None:
        # Equal timestamps
        with self.assertRaises(ValidationError) as context:
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.start_utc,
                duration_seconds=1.0,
            )
        self.assertIn("strictly after start_time", str(context.exception))

        # Inverted timestamps (end earlier than start)
        with self.assertRaises(ValidationError) as context:
            ObservationTimeEnvelope(
                start_time=self.end_utc,
                end_time=self.start_utc,
                duration_seconds=300.0,
            )
        self.assertIn("strictly after start_time", str(context.exception))

    def test_duration_mismatch_rejected(self) -> None:
        # Actual elapsed is 300.0s, but user specified 100.0s
        with self.assertRaises(ValidationError) as context:
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.end_utc,
                duration_seconds=100.0,
            )
        self.assertIn("does not match the elapsed time", str(context.exception))

    def test_invalid_duration_and_ttl_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.end_utc,
                duration_seconds=0.0,
            )
        with self.assertRaises(ValidationError):
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.end_utc,
                duration_seconds=-10.0,
            )
        with self.assertRaises(ValidationError):
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.end_utc,
                duration_seconds=300.0,
                time_to_live_seconds=30.0,  # Below 60.0s minimum
            )

    def test_serialization_and_immutability(self) -> None:
        envelope = ObservationTimeEnvelope(
            start_time=self.start_utc,
            end_time=self.end_utc,
            duration_seconds=300.0,
        )
        with self.assertRaises(ValidationError):
            envelope.duration_seconds = 500.0

        dumped = envelope.model_dump()
        reconstituted = ObservationTimeEnvelope.model_validate(dumped)
        self.assertEqual(reconstituted.start_time, self.start_utc)
        self.assertEqual(reconstituted.end_time, self.end_utc)

    def test_captured_at_utc_and_duration_auto_computes_end_time(self) -> None:
        envelope = ObservationTimeEnvelope(
            captured_at_utc=self.start_utc,
            duration_seconds=60.0,
        )
        self.assertEqual(envelope.start_time, self.start_utc)
        self.assertEqual(envelope.end_time, self.start_utc + timedelta(seconds=60.0))
        self.assertEqual(envelope.duration_seconds, 60.0)
        self.assertEqual(envelope.captured_at_utc, self.start_utc)

    def test_string_iso_timestamps_auto_compute_duration(self) -> None:
        envelope = ObservationTimeEnvelope.model_validate({
            "start_time": "2026-10-27T10:00:00Z",
            "end_time": "2026-10-27T10:05:00Z",
        })
        self.assertEqual(envelope.duration_seconds, 300.0)
        self.assertEqual(envelope.start_time, self.start_utc)
        self.assertEqual(envelope.end_time, self.end_utc)

    def test_boolean_timing_numbers_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.end_utc,
                duration_seconds=True,
            )
        with self.assertRaises(ValidationError):
            ObservationTimeEnvelope(
                start_time=self.start_utc,
                end_time=self.end_utc,
                duration_seconds=300.0,
                time_to_live_seconds=False,
            )


class TestVisionEdgeObservation(unittest.TestCase):
    """Tests for VisionEdgeObservation payload validation and constraints."""

    def setUp(self) -> None:
        start_utc = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
        end_utc = datetime(2026, 10, 27, 10, 5, 0, tzinfo=timezone.utc)
        self.timing = ObservationTimeEnvelope(
            start_time=start_utc,
            end_time=end_utc,
            duration_seconds=300.0,
        )
        self.camera = CameraMetadata(
            camera_id="cam_main_01",
            latitude=19.0760,
            longitude=72.8777,
            heading_degrees=90.0,
        )

    def test_valid_vision_edge_observation(self) -> None:
        obs = VisionEdgeObservation(
            camera_id="cam_main_01",
            osm_u=1001,
            osm_v=1002,
            osm_key=0,
            timing=self.timing,
            unique_vehicles_observed=15,
            vehicles_by_class={"car": 12, "bus": 2, "motorcycle": 1},
            total_detection_events=450,
            average_speed_kmh=28.4,
            observation_quality_score=0.92,
            experimental_congestion_index=0.35,
        )
        self.assertEqual(obs.camera_id, "cam_main_01")
        self.assertEqual(obs.osm_u, 1001)
        self.assertEqual(obs.osm_v, 1002)
        self.assertEqual(obs.osm_key, 0)
        self.assertEqual(obs.unique_vehicles_observed, 15)
        self.assertEqual(obs.vehicles_by_class["car"], 12)
        self.assertEqual(obs.total_detection_events, 450)
        self.assertAlmostEqual(obs.average_speed_kmh, 28.4)
        self.assertAlmostEqual(obs.observation_quality_score, 0.92)
        self.assertAlmostEqual(obs.experimental_congestion_index, 0.35)
        self.assertIn("experimental", obs.disclaimer.lower())

    def test_camera_metadata_and_alias_inputs(self) -> None:
        # Pass CameraMetadata instance as 'camera'
        obs1 = VisionEdgeObservation(
            camera=self.camera,
            osm_u=1001,
            osm_v=1002,
            timing=self.timing,
            unique_vehicles_observed=5,
        )
        self.assertEqual(obs1.camera_id, "cam_main_01")
        self.assertEqual(obs1.camera_metadata, self.camera)

        # Pass dict as 'camera'
        obs2 = VisionEdgeObservation(
            camera={"camera_id": "cam_dict_02"},
            osm_u=2001,
            osm_v=2002,
            timing=self.timing,
            unique_vehicles_observed=0,
        )
        self.assertEqual(obs2.camera_id, "cam_dict_02")

        # Pass string as 'camera'
        obs3 = VisionEdgeObservation(
            camera="cam_str_03",
            osm_u=3001,
            osm_v=3002,
            timing=self.timing,
            unique_vehicles_observed=0,
        )
        self.assertEqual(obs3.camera_id, "cam_str_03")

    def test_missing_required_identifiers_rejected(self) -> None:
        # Missing camera_id
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        # Empty camera_id
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="   ",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        # Missing osm_u
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        # Missing osm_v
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        # Missing timing
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                unique_vehicles_observed=5,
            )

    def test_negative_vehicle_counts_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=-1,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                total_detection_events=-1,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                vehicles_by_class={"car": -3},
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                vehicles_by_class={"": 3},
            )

    def test_quality_and_congestion_index_range_validation(self) -> None:
        # Quality score out of [0.0, 1.0]
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                observation_quality_score=-0.01,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                observation_quality_score=1.01,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                observation_quality_score=float("nan"),
            )

        # Experimental congestion index out of [0.0, 1.0]
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                experimental_congestion_index=-0.1,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                experimental_congestion_index=1.01,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                experimental_congestion_index=float("nan"),
            )

        # Default is None
        obs = VisionEdgeObservation(
            camera_id="cam_01",
            osm_u=1001,
            osm_v=1002,
            timing=self.timing,
            unique_vehicles_observed=5,
        )
        self.assertIsNone(obs.experimental_congestion_index)

    def test_serialization_and_immutability(self) -> None:
        obs = VisionEdgeObservation(
            camera_id="cam_main_01",
            osm_u=1001,
            osm_v=1002,
            timing=self.timing,
            unique_vehicles_observed=10,
        )
        with self.assertRaises(ValidationError):
            obs.unique_vehicles_observed = 20

        dumped = obs.model_dump()
        reconstituted = VisionEdgeObservation.model_validate(dumped)
        self.assertEqual(reconstituted.camera_id, "cam_main_01")
        self.assertEqual(reconstituted.unique_vehicles_observed, 10)

    def test_self_loop_road_edge_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1001,  # Self loop
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        self.assertIn("self-loops are not valid road edges", str(context.exception))

    def test_edge_and_camera_property_aliases(self) -> None:
        obs = VisionEdgeObservation(
            camera=self.camera,
            osm_u=1001,
            osm_v=1002,
            osm_key=1,
            timing=self.timing,
            unique_vehicles_observed=5,
        )
        self.assertEqual(obs.u, 1001)
        self.assertEqual(obs.v, 1002)
        self.assertEqual(obs.key, 1)
        self.assertEqual(obs.camera, self.camera)
        self.assertEqual(obs.edge_tuple(), (1001, 1002, 1))

    def test_edge_aliases_in_constructor(self) -> None:
        obs = VisionEdgeObservation(
            camera_id="cam_01",
            u=2001,
            v=2002,
            key=2,
            timing=self.timing,
            unique_vehicles_observed=5,
        )
        self.assertEqual(obs.osm_u, 2001)
        self.assertEqual(obs.osm_v, 2002)
        self.assertEqual(obs.osm_key, 2)

    def test_auto_populate_edge_from_calibrated_camera(self) -> None:
        calibrated_cam = CameraMetadata(
            camera_id="cam_calibrated_01",
            latitude=19.0760,
            longitude=72.8777,
            heading_degrees=90.0,
            target_osm_u=5001,
            target_osm_v=5002,
            target_osm_key=0,
            is_spatially_calibrated=True,
        )
        obs = VisionEdgeObservation(
            camera=calibrated_cam,
            timing=self.timing,
            unique_vehicles_observed=3,
        )
        self.assertEqual(obs.osm_u, 5001)
        self.assertEqual(obs.osm_v, 5002)
        self.assertEqual(obs.osm_key, 0)
        self.assertEqual(obs.camera_id, "cam_calibrated_01")

    def test_conflicting_camera_target_edge_rejected(self) -> None:
        cam_with_target = CameraMetadata(
            camera_id="cam_calibrated_02",
            latitude=19.0760,
            longitude=72.8777,
            heading_degrees=90.0,
            target_osm_u=5001,
            target_osm_v=5002,
        )
        with self.assertRaises(ValidationError) as context:
            VisionEdgeObservation(
                camera=cam_with_target,
                osm_u=9999,  # Mismatched edge
                osm_v=8888,
                timing=self.timing,
                unique_vehicles_observed=3,
            )
        self.assertIn("conflicts with camera's target edge", str(context.exception))

    def test_conflicting_camera_id_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VisionEdgeObservation(
                camera_id="cam_declared",
                camera_metadata=self.camera,  # self.camera has camera_id="cam_main_01"
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        self.assertIn("conflicts with camera_metadata.camera_id", str(context.exception))

    def test_vehicle_count_sanity_checks(self) -> None:
        # unique_vehicles_observed cannot be less than the count for any single class
        with self.assertRaises(ValidationError) as context:
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=3,
                vehicles_by_class={"car": 10},
            )
        self.assertIn("cannot be less than the count for any individual vehicle class", str(context.exception))

        # total_detection_events cannot be less than unique_vehicles_observed
        with self.assertRaises(ValidationError) as context:
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=10,
                total_detection_events=4,
            )
        self.assertIn("cannot be less than unique_vehicles_observed", str(context.exception))

    def test_empty_disclaimer_rejected(self) -> None:
        with self.assertRaises(ValidationError) as context:
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
                disclaimer="   ",
            )
        self.assertIn("disclaimer must not be empty", str(context.exception))

    def test_boolean_node_identifiers_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=True,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=5,
            )
        with self.assertRaises(ValidationError):
            VisionEdgeObservation(
                camera_id="cam_01",
                osm_u=1001,
                osm_v=1002,
                timing=self.timing,
                unique_vehicles_observed=True,
            )


class TestVisionSchemasBackwardCompatibility(unittest.TestCase):
    """Verify existing vision schemas and detection outputs remain 100% backward compatible."""

    def test_existing_schemas_preserve_backward_compatibility(self) -> None:
        box = BoundingBox(x_min=5.0, y_min=10.0, x_max=50.0, y_max=60.0)
        obj = DetectedObject(class_name="car", confidence=0.88, bounding_box=box, track_id=3)
        frame = FrameDetectionResult.from_detections(
            frame_number=0,
            timestamp_seconds=0.0,
            detections=[obj],
        )
        summary = VideoAnalysisSummary(
            video_name="test_traffic.mp4",
            total_frames=10,
            processed_frames=1,
            duration_seconds=0.33,
            fps=30.0,
            class_counts={"car": 1},
            total_detections=1,
            tracked_id_counts={"car": 1},
            total_tracked_ids=1,
        )

        self.assertEqual(box.x_min, 5.0)
        self.assertEqual(obj.track_id, 3)
        self.assertEqual(frame.detection_count, 1)
        self.assertEqual(summary.class_counts["car"], 1)
        self.assertEqual(summary.total_detections, 1)
        self.assertEqual(summary.total_tracked_ids, 1)
        self.assertIn("per-frame detection events", summary.disclaimer)


class TestTrackletPostProcessor(unittest.TestCase):
    """Unit tests for offline TrackletPostProcessor and post-processing schemas."""

    def setUp(self) -> None:
        self.box = BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=80.0)

    def test_empty_input(self) -> None:
        processor = TrackletPostProcessor()
        result = processor.process_frames([])
        self.assertIsInstance(result, PostProcessedVideoResult)
        self.assertEqual(result.summary.raw_total_detections, 0)
        self.assertEqual(result.summary.raw_total_tracked_ids, 0)
        self.assertEqual(result.summary.processed_total_detections, 0)
        self.assertEqual(result.summary.processed_total_tracked_ids, 0)
        self.assertEqual(len(result.tracklets), 0)
        self.assertEqual(len(result.filtered_track_ids), 0)
        self.assertEqual(len(result.frames), 0)
        self.assertIn("heuristics", result.summary.disclaimer.lower())

    def test_normal_operation_and_metrics(self) -> None:
        frames: list[FrameDetectionResult] = []
        for fn in range(6):
            det1 = DetectedObject(class_name="car", confidence=0.90, bounding_box=self.box, track_id=1)
            det2 = DetectedObject(class_name="bus", confidence=0.85, bounding_box=self.box, track_id=2)
            frames.append(
                FrameDetectionResult.from_detections(
                    frame_number=fn,
                    timestamp_seconds=fn / 30.0,
                    detections=[det1, det2],
                )
            )

        processor = TrackletPostProcessor(min_track_lifespan_frames=5)
        result = processor.process_frames(frames)

        self.assertEqual(result.summary.raw_total_tracked_ids, 2)
        self.assertEqual(result.summary.processed_total_tracked_ids, 2)
        self.assertEqual(result.summary.raw_total_detections, 12)
        self.assertEqual(result.summary.processed_total_detections, 12)
        self.assertEqual(result.summary.filtered_tracks_count, 0)
        self.assertEqual(result.summary.raw_class_switches_count, 0)

        # Inspect Tracklet 1
        t1 = result.tracklets[1]
        self.assertEqual(t1.track_id, 1)
        self.assertEqual(t1.assigned_class, "car")
        self.assertEqual(t1.detection_count, 6)
        self.assertEqual(t1.observed_frame_count, 6)
        self.assertEqual(t1.first_frame, 0)
        self.assertEqual(t1.last_frame, 5)
        self.assertEqual(t1.lifespan_frames, 6)
        self.assertAlmostEqual(t1.average_confidence, 0.90)
        self.assertFalse(t1.is_filtered)
        self.assertFalse(t1.has_class_switch)

        # Inspect Tracklet 2
        t2 = result.tracklets[2]
        self.assertEqual(t2.track_id, 2)
        self.assertEqual(t2.assigned_class, "bus")
        self.assertFalse(t2.is_filtered)

    def test_temporal_majority_voting_and_reclassification(self) -> None:
        # Track 10: 4 frames as "car" and 1 frame as "truck" (class switch)
        frames: list[FrameDetectionResult] = []
        for fn in range(4):
            det = DetectedObject(class_name="car", confidence=0.88, bounding_box=self.box, track_id=10)
            frames.append(FrameDetectionResult.from_detections(frame_number=fn, timestamp_seconds=fn * 0.1, detections=[det]))

        # Frame 4 has class switch to "truck"
        det_switched = DetectedObject(class_name="truck", confidence=0.75, bounding_box=self.box, track_id=10)
        frames.append(FrameDetectionResult.from_detections(frame_number=4, timestamp_seconds=0.4, detections=[det_switched]))

        processor = TrackletPostProcessor(min_track_lifespan_frames=3)
        result = processor.process_frames(frames)

        self.assertEqual(result.summary.raw_class_switches_count, 1)
        self.assertEqual(result.summary.resolved_class_switches_count, 1)

        t10 = result.tracklets[10]
        self.assertEqual(t10.assigned_class, "car")
        self.assertTrue(t10.has_class_switch)
        self.assertEqual(t10.original_classes, {"car": 4, "truck": 1})

        # Frame 4 in post-processed frames must now be reclassified to "car"
        frame_4_processed = result.frames[4]
        self.assertEqual(len(frame_4_processed.detections), 1)
        self.assertEqual(frame_4_processed.detections[0].class_name, "car")
        self.assertEqual(frame_4_processed.detections[0].track_id, 10)
        self.assertEqual(frame_4_processed.detections[0].confidence, 0.75)

        # In summary, track 10 counts under "car", not "truck"
        self.assertEqual(result.summary.processed_tracked_id_counts, {"car": 1})
        self.assertEqual(result.summary.processed_class_counts, {"car": 5})

    def test_deterministic_tie_breaking_confidence_sum(self) -> None:
        # Track 20: 2 "car" detections (conf 0.90, 0.90 -> sum 1.80)
        #           2 "truck" detections (conf 0.60, 0.60 -> sum 1.20)
        # Frequency is tied (2 vs 2); "car" wins due to higher sum of confidence
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0, timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.90, bounding_box=self.box, track_id=20)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=1, timestamp_seconds=0.1,
                detections=[DetectedObject(class_name="car", confidence=0.90, bounding_box=self.box, track_id=20)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=2, timestamp_seconds=0.2,
                detections=[DetectedObject(class_name="truck", confidence=0.60, bounding_box=self.box, track_id=20)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=3, timestamp_seconds=0.3,
                detections=[DetectedObject(class_name="truck", confidence=0.60, bounding_box=self.box, track_id=20)],
            ),
        ]
        processor = TrackletPostProcessor(min_track_lifespan_frames=2)
        result = processor.process_frames(frames)
        self.assertEqual(result.tracklets[20].assigned_class, "car")

    def test_deterministic_tie_breaking_max_confidence(self) -> None:
        # Track 30: 2 "car" (conf 0.80, 0.40 -> sum 1.20, max 0.80)
        #           2 "truck" (conf 0.60, 0.60 -> sum 1.20, max 0.60)
        # Frequency tied, sum tied; "car" wins due to peak max confidence (0.80 > 0.60)
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0, timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.80, bounding_box=self.box, track_id=30)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=1, timestamp_seconds=0.1,
                detections=[DetectedObject(class_name="car", confidence=0.40, bounding_box=self.box, track_id=30)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=2, timestamp_seconds=0.2,
                detections=[DetectedObject(class_name="truck", confidence=0.60, bounding_box=self.box, track_id=30)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=3, timestamp_seconds=0.3,
                detections=[DetectedObject(class_name="truck", confidence=0.60, bounding_box=self.box, track_id=30)],
            ),
        ]
        processor = TrackletPostProcessor(min_track_lifespan_frames=2)
        result = processor.process_frames(frames)
        self.assertEqual(result.tracklets[30].assigned_class, "car")

    def test_deterministic_tie_breaking_alphabetical(self) -> None:
        # Track 40: 2 "bus" (conf 0.70, 0.70 -> sum 1.40, max 0.70)
        #           2 "car" (conf 0.70, 0.70 -> sum 1.40, max 0.70)
        # Frequency tied, sum tied, max tied; "bus" wins alphabetically ('bus' < 'car')
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0, timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="bus", confidence=0.70, bounding_box=self.box, track_id=40)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=1, timestamp_seconds=0.1,
                detections=[DetectedObject(class_name="bus", confidence=0.70, bounding_box=self.box, track_id=40)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=2, timestamp_seconds=0.2,
                detections=[DetectedObject(class_name="car", confidence=0.70, bounding_box=self.box, track_id=40)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=3, timestamp_seconds=0.3,
                detections=[DetectedObject(class_name="car", confidence=0.70, bounding_box=self.box, track_id=40)],
            ),
        ]
        processor = TrackletPostProcessor(min_track_lifespan_frames=2)
        result = processor.process_frames(frames)
        self.assertEqual(result.tracklets[40].assigned_class, "bus")

    def test_short_track_lifespan_filtering(self) -> None:
        # Track 100: appears in 5 frames (retained)
        # Track 101: appears in 2 frames (filtered)
        # Track 102: appears in 1 frame (filtered)
        frames: list[FrameDetectionResult] = []
        for fn in range(5):
            dets = [DetectedObject(class_name="car", confidence=0.90, bounding_box=self.box, track_id=100)]
            if fn < 2:
                dets.append(DetectedObject(class_name="motorcycle", confidence=0.80, bounding_box=self.box, track_id=101))
            if fn == 0:
                dets.append(DetectedObject(class_name="truck", confidence=0.85, bounding_box=self.box, track_id=102))
            frames.append(FrameDetectionResult.from_detections(frame_number=fn, timestamp_seconds=fn * 0.1, detections=dets))

        processor = TrackletPostProcessor(min_track_lifespan_frames=5)
        result = processor.process_frames(frames)

        self.assertEqual(result.summary.raw_total_tracked_ids, 3)
        self.assertEqual(result.summary.processed_total_tracked_ids, 1)
        self.assertEqual(result.summary.filtered_tracks_count, 2)
        self.assertEqual(result.filtered_track_ids, [101, 102])

        self.assertFalse(result.tracklets[100].is_filtered)
        self.assertTrue(result.tracklets[101].is_filtered)
        self.assertTrue(result.tracklets[102].is_filtered)

        # In processed frames, only Track 100 should remain
        for pf in result.frames:
            for det in pf.detections:
                self.assertEqual(det.track_id, 100)

    def test_lifespan_filtering_disabled(self) -> None:
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0, timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=50)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=1, timestamp_seconds=0.1,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=50)],
            ),
        ]
        processor = TrackletPostProcessor(min_track_lifespan_frames=5, filter_short_tracks=False)
        result = processor.process_frames(frames)

        self.assertFalse(result.tracklets[50].is_filtered)
        self.assertEqual(result.summary.filtered_tracks_count, 0)
        self.assertEqual(result.summary.processed_total_tracked_ids, 1)
        self.assertEqual(len(result.frames[0].detections), 1)

    def test_missing_tracker_ids_handled_safely(self) -> None:
        # Two detections without track_id, and one detection with track_id=5
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0, timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=None)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=1, timestamp_seconds=0.1,
                detections=[DetectedObject(class_name="bus", confidence=0.8, bounding_box=self.box, track_id=None)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=2, timestamp_seconds=0.2,
                detections=[DetectedObject(class_name="truck", confidence=0.85, bounding_box=self.box, track_id=5)],
            ),
        ]
        processor = TrackletPostProcessor(min_track_lifespan_frames=1)
        result = processor.process_frames(frames)

        # Missing tracker IDs must NOT be grouped into a single tracklet
        self.assertEqual(len(result.tracklets), 1)
        self.assertIn(5, result.tracklets)
        self.assertEqual(result.summary.untracked_detections_count, 2)
        self.assertEqual(result.summary.raw_total_tracked_ids, 1)

        # Untracked detections must be preserved in post-processed frames
        self.assertEqual(len(result.frames[0].detections), 1)
        self.assertIsNone(result.frames[0].detections[0].track_id)
        self.assertEqual(result.frames[0].detections[0].class_name, "car")
        self.assertEqual(len(result.frames[1].detections), 1)
        self.assertIsNone(result.frames[1].detections[0].track_id)
        self.assertEqual(result.frames[1].detections[0].class_name, "bus")

    def test_input_immutability(self) -> None:
        det = DetectedObject(class_name="truck", confidence=0.75, bounding_box=self.box, track_id=10)
        orig_frame = FrameDetectionResult.from_detections(frame_number=0, timestamp_seconds=0.0, detections=[det])
        input_frames = [orig_frame]

        processor = TrackletPostProcessor(min_track_lifespan_frames=1)
        result = processor.process_frames(input_frames)

        # Input objects must not be altered
        self.assertEqual(orig_frame.detections[0].class_name, "truck")
        self.assertEqual(orig_frame.detections[0].track_id, 10)
        self.assertIsNot(input_frames[0], result.frames[0])

    def test_repeated_frames_and_frame_indices(self) -> None:
        # Two frame detection records with the same frame number
        det = DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=7)
        f1 = FrameDetectionResult.from_detections(frame_number=2, timestamp_seconds=0.066, detections=[det])
        f2 = FrameDetectionResult.from_detections(frame_number=2, timestamp_seconds=0.066, detections=[det])

        processor = TrackletPostProcessor(min_track_lifespan_frames=1)
        result = processor.process_frames([f1, f2])

        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.tracklets[7].observed_frame_count, 1)
        self.assertEqual(result.tracklets[7].detection_count, 2)

    def test_invalid_min_lifespan_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            TrackletPostProcessor(min_track_lifespan_frames=0)
        self.assertIn("must be at least 1", str(context.exception))

    def test_post_processing_metric_conservation_enforced(self) -> None:
        """Verify that PostProcessingSummary enforces mathematical conservation."""
        with self.assertRaises(ValidationError) as ctx:
            PostProcessingSummary(
                raw_total_detections=10,
                raw_total_tracked_ids=2,
                raw_class_counts={"car": 10},
                raw_tracked_id_counts={"car": 2},
                raw_class_switches_count=0,
                processed_total_detections=8,  # 8 + 1 != 10 (violates detection conservation)
                processed_total_tracked_ids=2,
                processed_class_counts={"car": 8},
                processed_tracked_id_counts={"car": 2},
                min_track_lifespan_frames=5,
                filtered_tracks_count=0,
                filtered_detections_count=1,
                untracked_detections_count=0,
                resolved_class_switches_count=0,
            )
        self.assertIn("raw_total_detections", str(ctx.exception))

    def test_identity_switches_not_falsely_claimed_as_resolved(self) -> None:
        """Demonstrate that fragmented tracklets remain distinct tracklets after post-processing."""
        frames = []
        for fn in range(5):
            frames.append(
                FrameDetectionResult.from_detections(
                    frame_number=fn,
                    timestamp_seconds=fn * 0.1,
                    detections=[DetectedObject(class_name="bus", confidence=0.9, bounding_box=self.box, track_id=2)],
                )
            )
        for fn in range(10, 15):
            frames.append(
                FrameDetectionResult.from_detections(
                    frame_number=fn,
                    timestamp_seconds=fn * 0.1,
                    detections=[DetectedObject(class_name="bus", confidence=0.9, bounding_box=self.box, track_id=19)],
                )
            )

        processor = TrackletPostProcessor(min_track_lifespan_frames=3)
        result = processor.process_frames(frames)

        # Track 2 and Track 19 must NOT be merged; they remain 2 separate tracklets
        self.assertEqual(len(result.tracklets), 2)
        self.assertIn(2, result.tracklets)
        self.assertIn(19, result.tracklets)
        self.assertEqual(result.summary.processed_total_tracked_ids, 2)
        # Identity switch is NOT resolved (only label smoothing within a single tracklet is performed)
        self.assertEqual(result.summary.resolved_class_switches_count, 0)


class TestPipelineEvaluation(unittest.TestCase):
    """Unit tests for offline pipeline evaluation script and reporting."""

    def setUp(self) -> None:
        self.box = BoundingBox(x_min=10.0, y_min=20.0, x_max=60.0, y_max=80.0)

    def test_evaluate_frames_basic(self) -> None:
        frames = []
        for fn in range(5):
            frames.append(
                FrameDetectionResult.from_detections(
                    frame_number=fn,
                    timestamp_seconds=fn * 0.1,
                    detections=[
                        DetectedObject(class_name="car", confidence=0.90, bounding_box=self.box, track_id=1),
                        DetectedObject(class_name="bus", confidence=0.85, bounding_box=self.box, track_id=2),
                    ],
                )
            )

        report = evaluate_frames(
            frame_results=frames,
            video_metadata={"video_name": "test_video.mp4", "frame_count": 50, "fps": 25.0},
            min_track_lifespan_frames=3,
        )

        self.assertIsInstance(report, PipelineEvaluationReport)
        self.assertEqual(report.video_name, "test_video.mp4")
        self.assertEqual(report.total_frames_in_video, 50)
        self.assertEqual(report.processed_frames, 5)
        self.assertEqual(report.raw_total_detections, 10)
        self.assertEqual(report.processed_total_detections, 10)
        self.assertEqual(report.raw_total_tracked_ids, 2)
        self.assertEqual(report.processed_total_tracked_ids, 2)
        self.assertEqual(report.filtered_tracks_count, 0)
        self.assertEqual(len(report.tracklets), 2)
        self.assertEqual(report.raw_track_observed_frames, {1: 5, 2: 5})
        self.assertEqual(len(report.reclassified_detections), 0)

    def test_evaluate_frames_with_class_switch_and_filtering(self) -> None:
        frames = []
        # Track 10: 4 frames "car", 1 frame "truck" -> majority "car" (retained)
        # Track 20: 2 frames "motorcycle" -> filtered (< 5 frames)
        for fn in range(4):
            dets = [
                DetectedObject(class_name="car", confidence=0.90, bounding_box=self.box, track_id=10),
            ]
            if fn < 2:
                dets.append(
                    DetectedObject(class_name="motorcycle", confidence=0.75, bounding_box=self.box, track_id=20)
                )
            frames.append(
                FrameDetectionResult.from_detections(frame_number=fn, timestamp_seconds=fn * 0.1, detections=dets)
            )

        # Frame 4 has class switch for Track 10 to "truck"
        frames.append(
            FrameDetectionResult.from_detections(
                frame_number=4,
                timestamp_seconds=0.4,
                detections=[
                    DetectedObject(class_name="truck", confidence=0.70, bounding_box=self.box, track_id=10),
                ],
            )
        )

        report = evaluate_frames(
            frame_results=frames,
            min_track_lifespan_frames=5,
        )

        self.assertEqual(report.filtered_tracks_count, 1)
        self.assertEqual(report.filtered_track_ids, [20])
        self.assertEqual(report.filtered_detections_count, 2)
        self.assertEqual(report.raw_class_switches_count, 1)
        self.assertEqual(report.resolved_class_switches_count, 1)

        # Frame 4 detection for track 10 should be reclassified
        self.assertEqual(len(report.reclassified_detections), 1)
        reclass = report.reclassified_detections[0]
        self.assertEqual(reclass.frame_number, 4)
        self.assertEqual(reclass.track_id, 10)
        self.assertEqual(reclass.raw_class, "truck")
        self.assertEqual(reclass.assigned_class, "car")

        # Track 10 in tracklets list
        t10 = next(t for t in report.tracklets if t.track_id == 10)
        self.assertEqual(t10.assigned_class, "car")
        self.assertFalse(t10.is_filtered)
        self.assertTrue(t10.has_class_switch)

        # Track 20 in tracklets list
        t20 = next(t for t in report.tracklets if t.track_id == 20)
        self.assertTrue(t20.is_filtered)

    def test_format_evaluation_report(self) -> None:
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=1)],
            )
        ]
        report = evaluate_frames(frames, min_track_lifespan_frames=1)
        text = format_evaluation_report(report)

        self.assertIn("ACCESSROUTE COMPUTER-VISION PIPELINE EVALUATION REPORT", text)
        self.assertIn("1. EXECUTIVE COMPARISON SUMMARY", text)
        self.assertIn("2. RAW TRACKER IDS & OBSERVED FRAME COUNTS", text)
        self.assertIn("3. TRACKLET POST-PROCESSING & FILTERING DETAILS", text)
        self.assertIn("4. TEMPORAL MAJORITY VOTING & LABEL SMOOTHING", text)
        self.assertIn("5. METRIC DEFINITIONS & EDUCATIONAL GUIDE", text)
        self.assertIn("6. RESEARCH LIMITATIONS & ACADEMIC NON-CLAIMS", text)
        self.assertIn("Identity Switches NOT Corrected", text)
        self.assertIn("ground-truth verification", text)
        self.assertNotIn("tracker ID 19", text)
        self.assertNotIn("Comparing 1,631 detection events", text)

    def test_format_evaluation_report_no_unverified_identity_claims(self) -> None:
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[
                    DetectedObject(class_name="car", confidence=0.8, bounding_box=self.box, track_id=2),
                    DetectedObject(class_name="bus", confidence=0.9, bounding_box=self.box, track_id=18),
                ],
            )
        ]
        report = evaluate_frames(frames, min_track_lifespan_frames=1)
        text = format_evaluation_report(report)
        self.assertNotIn("ID 2 and tracker ID 19", text)
        self.assertIn("ground-truth verification", text)

    def test_evaluate_video_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate_video("non_existent_sample_video_path.mp4")

    def test_evaluate_video_invalid_max_frames(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_video(__file__, max_frames=0)
        self.assertIn("must be at least 1", str(ctx.exception))

    def test_build_arg_parser_and_defaults(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--video-path", "sample.mp4"])
        self.assertEqual(args.video_path, "sample.mp4")
        self.assertIsNone(args.max_frames)
        self.assertEqual(args.min_lifespan, DEFAULT_MIN_TRACK_LIFESPAN_FRAMES)
        self.assertEqual(args.conf, 0.50)
        self.assertEqual(args.iou, 0.30)
        self.assertEqual(args.tracker, "bytetrack.yaml")

        # Test explicit max_frames parsing
        sample_args = parser.parse_args(["--video-path", "sample.mp4", "--max-frames", "30"])
        self.assertEqual(sample_args.max_frames, 30)

    def test_cli_main_missing_required_arg(self) -> None:
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                eval_main([])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_evaluate_video_empty_path(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_video("")
        self.assertIn("must not be empty", str(ctx.exception))

    def test_evaluate_video_invalid_conf(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_video(__file__, conf=-0.1)
        self.assertIn("conf must be between", str(ctx.exception))

    def test_evaluate_video_invalid_iou(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_video(__file__, iou=1.5)
        self.assertIn("iou must be between", str(ctx.exception))

    def test_format_evaluation_report_with_filtered_class_switch(self) -> None:
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.8, bounding_box=self.box, track_id=10)],
            ),
            FrameDetectionResult.from_detections(
                frame_number=1,
                timestamp_seconds=0.1,
                detections=[DetectedObject(class_name="truck", confidence=0.7, bounding_box=self.box, track_id=10)],
            ),
        ]
        report = evaluate_frames(frames, min_track_lifespan_frames=5)
        text = format_evaluation_report(report)
        self.assertIn("pruned with short tracks", text)
        self.assertIn("Track #10 [PRUNED / FILTERED]", text)

    def test_save_evaluation_outputs_protection(self) -> None:
        import tempfile
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=1)],
            )
        ]
        report = evaluate_frames(frames, min_track_lifespan_frames=1)
        text = format_evaluation_report(report)

        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "report.txt"
            json_file = Path(tmpdir) / "report.json"

            # 1. Normal save to new paths
            saved_txt, saved_json = save_evaluation_outputs(
                report=report,
                formatted_report=text,
                output_txt=txt_file,
                output_json=json_file,
                overwrite=False,
            )
            self.assertEqual(saved_txt, txt_file)
            self.assertEqual(saved_json, json_file)
            self.assertTrue(txt_file.exists())
            self.assertTrue(json_file.exists())

            # 2. Saving again without overwrite=True must raise FileExistsError
            with self.assertRaises(FileExistsError):
                save_evaluation_outputs(
                    report=report,
                    formatted_report=text,
                    output_txt=txt_file,
                    overwrite=False,
                )

            with self.assertRaises(FileExistsError):
                save_evaluation_outputs(
                    report=report,
                    formatted_report=text,
                    output_json=json_file,
                    overwrite=False,
                )

            # 3. Saving with overwrite=True succeeds
            saved_txt, saved_json = save_evaluation_outputs(
                report=report,
                formatted_report="Updated report",
                output_txt=txt_file,
                output_json=json_file,
                overwrite=True,
            )
            self.assertEqual(txt_file.read_text(encoding="utf-8"), "Updated report")

    @patch("backend.vision.evaluate_pipeline.evaluate_video")
    def test_cli_main_success_flow(self, mock_eval_video: MagicMock) -> None:
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=1)],
            )
        ]
        mock_report = evaluate_frames(frames, min_track_lifespan_frames=1)
        mock_eval_video.return_value = mock_report

        with patch("builtins.print"):
            exit_code = eval_main(["--video-path", "mock_video.mp4"])
        self.assertEqual(exit_code, 0)
        mock_eval_video.assert_called_once()

    @patch("backend.vision.evaluate_pipeline.evaluate_video")
    def test_cli_main_file_exists_error(self, mock_eval_video: MagicMock) -> None:
        import tempfile
        frames = [
            FrameDetectionResult.from_detections(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[DetectedObject(class_name="car", confidence=0.9, bounding_box=self.box, track_id=1)],
            )
        ]
        mock_report = evaluate_frames(frames, min_track_lifespan_frames=1)
        mock_eval_video.return_value = mock_report

        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "exists.txt"
            existing_file.write_text("prior content", encoding="utf-8")

            with patch("sys.stderr"), patch("builtins.print"):
                exit_code = eval_main([
                    "--video-path", "mock_video.mp4",
                    "--output-txt", str(existing_file),
                ])
            # Exits with error code 1 when attempting to overwrite without --overwrite
            self.assertEqual(exit_code, 1)
            self.assertEqual(existing_file.read_text(encoding="utf-8"), "prior content")


if __name__ == "__main__":
    unittest.main()


