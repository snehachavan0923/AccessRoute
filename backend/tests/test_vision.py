"""Unit tests for the vision schemas and service foundation."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

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


if __name__ == "__main__":
    unittest.main()
