"""Unit and integration tests for ObservationBuilder and vision-to-traffic pipeline connection."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from backend.pedestrian_network import PedestrianNetworkProvider
from backend.routing.contracts import Coordinate
from backend.routing.shortest_path import NetworkAreaSettings
from backend.routing.traffic_aware_path import (
    TrafficAwarePathService,
    TrafficRouteComparisonService,
)
from backend.traffic.loader import TrafficDataset, TrafficEdgeKey
from backend.traffic.vision_adapter import (
    ObservationValidationStatus,
    VisionTrafficAdapter,
    get_default_vision_adapter,
    set_default_vision_adapter,
)
from backend.vision.observation_builder import (
    ObservationBuildResult,
    build_vision_edge_observation,
)
from backend.vision.post_processor import (
    DEFAULT_MIN_TRACK_LIFESPAN_FRAMES,
    PostProcessedVideoResult,
    PostProcessingSummary,
    TrackletPostProcessor,
)
from backend.vision.schemas import (
    BoundingBox,
    CameraMetadata,
    DetectedObject,
    FrameDetectionResult,
    VisionEdgeObservation,
)
from backend.tests.test_traffic_aware_path import (
    create_synthetic_dataset,
    make_forked_graph,
)


class TestObservationBuilder(unittest.TestCase):
    """Unit tests for the Observation Builder module."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        # Create a sample PostProcessedVideoResult
        det_car = DetectedObject(
            class_name="car",
            confidence=0.85,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        det_bus = DetectedObject(
            class_name="bus",
            confidence=0.92,
            bounding_box=BoundingBox(x_min=150, y_min=40, x_max=300, y_max=250),
            track_id=2,
        )
        self.frames = [
            FrameDetectionResult.from_detections(0, 0.0, [det_car, det_bus]),
            FrameDetectionResult.from_detections(1, 1.0, [det_car, det_bus]),
            FrameDetectionResult.from_detections(2, 2.0, [det_car, det_bus]),
            FrameDetectionResult.from_detections(3, 3.0, [det_car, det_bus]),
            FrameDetectionResult.from_detections(4, 4.0, [det_car, det_bus]),
        ]
        processor = TrackletPostProcessor(min_track_lifespan_frames=5)
        self.processed_result = processor.process_frames(self.frames)

        self.calibrated_camera = CameraMetadata(
            camera_id="cam_ec_01",
            latitude=12.8476,
            longitude=77.6712,
            heading_degrees=90.0,
            target_osm_u=100,
            target_osm_v=200,
            target_osm_key=0,
            is_spatially_calibrated=True,
        )
        self.uncalibrated_camera = CameraMetadata(
            camera_id="cam_ec_uncalib",
            latitude=12.8476,
            longitude=77.6712,
            heading_degrees=90.0,
            target_osm_u=100,
            target_osm_v=200,
            target_osm_key=0,
            is_spatially_calibrated=False,
        )

    def test_missing_camera_id_fails_cleanly(self) -> None:
        """Builder fails gracefully when no camera identifier is available."""
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            osm_u=100,
            osm_v=200,
            camera_id=None,
            camera_metadata=None,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.observation)
        self.assertEqual(result.status, "missing_camera_id")
        self.assertIn("Camera identifier is required", result.message)

    def test_missing_edge_metadata_fails_cleanly(self) -> None:
        """Builder fails gracefully when no target road edge (osm_u, osm_v) is specified."""
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_id="cam_test",
            osm_u=None,
            osm_v=None,
            camera_metadata=None,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.observation)
        self.assertEqual(result.status, "missing_edge_metadata")
        self.assertIn("osm_u, osm_v", result.message)

    def test_self_loop_edge_fails_cleanly(self) -> None:
        """Builder rejects self-loops where osm_u == osm_v."""
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_id="cam_test",
            osm_u=100,
            osm_v=100,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.observation)
        self.assertEqual(result.status, "invalid_edge")

    def test_edge_mismatch_with_camera_metadata_fails_cleanly(self) -> None:
        """Builder rejects conflicting edge definitions between arguments and camera metadata."""
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_metadata=self.calibrated_camera,
            osm_u=999,  # Conflicts with target_osm_u=100
            osm_v=200,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.observation)
        self.assertEqual(result.status, "edge_mismatch")

    def test_naive_timestamp_fails_cleanly(self) -> None:
        """Builder rejects timezone-naive start_time to prevent ambiguous timestamp comparisons."""
        naive_dt = datetime(2026, 1, 1, 12, 0, 0)  # No tzinfo
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_metadata=self.calibrated_camera,
            start_time=naive_dt,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.observation)
        self.assertEqual(result.status, "naive_timestamp")

    def test_valid_observation_building_with_calibrated_camera(self) -> None:
        """Builder successfully constructs a VisionEdgeObservation when required metadata is present."""
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_metadata=self.calibrated_camera,
            start_time=self.now - timedelta(seconds=60),
            experimental_congestion_index=0.75,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.observation)

        obs = result.observation
        self.assertEqual(obs.camera_id, "cam_ec_01")
        self.assertEqual(obs.osm_u, 100)
        self.assertEqual(obs.osm_v, 200)
        self.assertEqual(obs.osm_key, 0)
        self.assertEqual(obs.unique_vehicles_observed, 2)
        self.assertEqual(obs.vehicles_by_class, {"car": 1, "bus": 1})
        self.assertEqual(obs.total_detection_events, 10)
        self.assertEqual(obs.experimental_congestion_index, 0.75)
        self.assertTrue(obs.camera_metadata.is_spatially_calibrated)

    def test_uncalibrated_observation_preserves_safeguard(self) -> None:
        """Uncalibrated camera builds an observation that is rejected by VisionTrafficAdapter calibration check."""
        result = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_metadata=self.uncalibrated_camera,
            start_time=self.now - timedelta(seconds=60),
            experimental_congestion_index=None,  # No manufactured congestion index
        )
        self.assertTrue(result.success)
        obs = result.observation
        self.assertIsNotNone(obs)
        self.assertFalse(obs.camera_metadata.is_spatially_calibrated)
        self.assertIsNone(obs.experimental_congestion_index)

        # Ingest into adapter with calibration safeguard enabled
        adapter = VisionTrafficAdapter(require_spatial_calibration=True)
        adapter.set_observation(obs)

        # Adapter returns None because camera is uncalibrated
        edge_key = TrafficEdgeKey(osm_u=100, osm_v=200, key=0)
        self.assertEqual(
            adapter.evaluate_observation_status(edge_key, current_time=self.now),
            ObservationValidationStatus.UNCALIBRATED,
        )
        self.assertIsNone(adapter.get_congestion_index(edge_key, current_time=self.now))


class TestVisionTrafficRoutingWithBuilder(unittest.TestCase):
    """Integration test verifying that built observations interact safely with the routing engine."""

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Deterministic graph: Path A (1 -> 2 -> 3, 200m), Path B (1 -> 4 -> 3, 240m)
        nodes = [
            {"node_id": "0", "osm_node_id": "1", "x": "0.0", "y": "0.0"},
            {"node_id": "1", "osm_node_id": "2", "x": "0.001", "y": "0.0"},
            {"node_id": "2", "osm_node_id": "3", "x": "0.002", "y": "0.0"},
            {"node_id": "3", "osm_node_id": "4", "x": "0.001", "y": "0.001"},
        ]
        edges = [
            {"edge_id": "0", "u": "0", "v": "1", "osm_u": "1", "osm_v": "2", "key": "0", "length_m": "100.0", "highway": "primary", "maxspeed": "50"},
            {"edge_id": "1", "u": "1", "v": "2", "osm_u": "2", "osm_v": "3", "key": "0", "length_m": "100.0", "highway": "primary", "maxspeed": "50"},
            {"edge_id": "2", "u": "0", "v": "3", "osm_u": "1", "osm_v": "4", "key": "0", "length_m": "120.0", "highway": "footway", "maxspeed": "10"},
            {"edge_id": "3", "u": "3", "v": "2", "osm_u": "4", "osm_v": "3", "key": "0", "length_m": "120.0", "highway": "footway", "maxspeed": "10"},
            {"edge_id": "4", "u": "2", "v": "1", "osm_u": "3", "osm_v": "2", "key": "0", "length_m": "100.0", "highway": "primary", "maxspeed": "50"},
            {"edge_id": "5", "u": "1", "v": "0", "osm_u": "2", "osm_v": "1", "key": "0", "length_m": "100.0", "highway": "primary", "maxspeed": "50"},
        ]
        timestamps = ["2026-01-01 08:30:00", "2026-01-01 02:00:00"]
        # Index 1: low congestion (0.05 on direct path edges)
        congestion_matrix = np.array(
            [
                [0.85, 0.85, 0.0, 0.0, 0.0, 0.0],
                [0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        speed_matrix = np.ones((2, 6), dtype=np.float32) * 30.0
        vehicle_matrix = np.ones((2, 6), dtype=np.float32) * 15.0

        create_synthetic_dataset(
            self.dataset_dir,
            nodes=nodes,
            edges=edges,
            timestamps=timestamps,
            congestion_matrix=congestion_matrix,
            speed_matrix=speed_matrix,
            vehicle_matrix=vehicle_matrix,
        )
        self.dataset = TrafficDataset.load(self.dataset_dir)

        self.mock_network_provider = Mock(spec=PedestrianNetworkProvider)
        self.mock_network_provider.load.return_value = make_forked_graph()

        self.settings = NetworkAreaSettings(buffer_meters=500, max_diagonal_meters=5000)
        self.origin = Coordinate(0.0, 0.0)
        self.destination = Coordinate(0.0, 0.002)

        self.vision_adapter = VisionTrafficAdapter(require_spatial_calibration=True)

        # Create dummy processed result
        det = DetectedObject(
            class_name="car",
            confidence=0.90,
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200),
            track_id=1,
        )
        frames = [FrameDetectionResult.from_detections(i, float(i), [det]) for i in range(5)]
        self.processed_result = TrackletPostProcessor().process_frames(frames)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        set_default_vision_adapter(None)

    def test_uncalibrated_built_observation_does_not_influence_route_costs(self) -> None:
        """An observation built from an uncalibrated camera falls back to simulated traffic and does not alter routes."""
        uncalib_cam = CameraMetadata(
            camera_id="cam_uncalib",
            latitude=12.84,
            longitude=77.67,
            heading_degrees=90.0,
            target_osm_u=1,
            target_osm_v=2,
            target_osm_key=0,
            is_spatially_calibrated=False,
        )
        build_res = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_metadata=uncalib_cam,
            start_time=self.now - timedelta(seconds=60),
            experimental_congestion_index=None,
        )
        self.assertTrue(build_res.success)
        self.vision_adapter.set_observation(build_res.observation)

        service = TrafficRouteComparisonService(
            network_provider=self.mock_network_provider,
            traffic_dataset=self.dataset,
            settings=self.settings,
            vision_adapter=self.vision_adapter,
        )

        res = service.compare(
            origin=self.origin,
            destination=self.destination,
            time_index=1,
            traffic_weight=2.0,
        )

        # Route cost remains baseline simulated 220.0m along direct path (200m)
        self.assertEqual(res.status, "identical_routes")
        self.assertEqual(res.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertAlmostEqual(res.traffic_accessibility_aware.cost, 220.0, places=1)

    def test_calibrated_built_observation_with_congestion_alters_route(self) -> None:
        """A validly calibrated built observation with congestion overrides simulated traffic and diverts route."""
        calib_cam = CameraMetadata(
            camera_id="cam_calib",
            latitude=12.84,
            longitude=77.67,
            heading_degrees=90.0,
            target_osm_u=1,
            target_osm_v=2,
            target_osm_key=0,
            is_spatially_calibrated=True,
        )
        build_res = build_vision_edge_observation(
            processed_result=self.processed_result,
            camera_metadata=calib_cam,
            start_time=self.now - timedelta(seconds=60),
            experimental_congestion_index=0.90,  # Explicit heavy congestion
        )
        self.assertTrue(build_res.success)
        self.vision_adapter.set_observation(build_res.observation)

        service = TrafficRouteComparisonService(
            network_provider=self.mock_network_provider,
            traffic_dataset=self.dataset,
            settings=self.settings,
            vision_adapter=self.vision_adapter,
        )

        res = service.compare(
            origin=self.origin,
            destination=self.destination,
            time_index=1,
            traffic_weight=2.0,
        )

        # Route diverts to 240m detour to avoid high vision congestion
        self.assertEqual(res.status, "alternative_found")
        self.assertEqual(res.shortest_distance.distance_meters, 200.0)
        self.assertEqual(res.traffic_accessibility_aware.distance_meters, 240.0)
        self.assertAlmostEqual(res.traffic_accessibility_aware.cost, 240.0, places=1)


if __name__ == "__main__":
    unittest.main()
