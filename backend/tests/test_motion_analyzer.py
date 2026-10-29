"""Unit tests for vehicle movement analysis and image-plane motion measurement."""

from __future__ import annotations

import unittest

from backend.vision.evaluate_pipeline import evaluate_frames
from backend.vision.motion_analyzer import (
    CALIBRATION_REQUIREMENTS_SUMMARY,
    ConsecutiveStepDisplacement,
    DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_PX,
    DEFAULT_STATIONARY_SPEED_THRESHOLD_PX_PER_SEC,
    MotionMeasurementStatus,
    TrackMotionMeasurement,
    TrackObservationPoint,
    VideoMotionSummary,
    analyze_single_track_motion,
    analyze_video_motion,
    compute_bounding_box_center,
    compute_pixel_displacement,
)
from backend.vision.schemas import (
    BoundingBox,
    DetectedObject,
    FrameDetectionResult,
)


class TestMotionAnalyzer(unittest.TestCase):
    """Deterministic unit tests for motion analyzer algorithms and data structures."""

    def test_compute_bounding_box_center(self) -> None:
        """Verify 2D center coordinate computation."""
        bbox = BoundingBox(x_min=100.0, y_min=200.0, x_max=300.0, y_max=400.0)
        cx, cy = compute_bounding_box_center(bbox)
        self.assertEqual(cx, 200.0)
        self.assertEqual(cy, 300.0)

    def test_compute_pixel_displacement(self) -> None:
        """Verify Euclidean distance between pixel centers."""
        p1 = (100.0, 100.0)
        p2 = (130.0, 140.0)
        # 30-40-50 right triangle
        disp = compute_pixel_displacement(p1, p2)
        self.assertAlmostEqual(disp, 50.0, places=4)

    def test_stationary_vehicle(self) -> None:
        """A vehicle with identical bounding box centers across frames is classified stationary."""
        observations = [
            TrackObservationPoint(
                frame_number=i,
                timestamp_seconds=i * 0.1,
                center_x=150.0,
                center_y=250.0,
                bounding_box=BoundingBox(x_min=100.0, y_min=200.0, x_max=200.0, y_max=300.0),
                confidence=0.9,
                class_name="car",
            )
            for i in range(5)
        ]

        result = analyze_single_track_motion(track_id=1, observations=observations, fps=10.0)

        self.assertEqual(result.track_id, 1)
        self.assertEqual(result.usable_observation_count, 5)
        self.assertEqual(result.measurement_status, MotionMeasurementStatus.STATIONARY)
        self.assertTrue(result.is_stationary)
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 0.0, places=2)
        self.assertAlmostEqual(result.net_pixel_displacement, 0.0, places=2)
        self.assertAlmostEqual(result.average_pixel_speed_px_per_sec, 0.0, places=2)
        self.assertFalse(result.is_physical_speed_available)
        self.assertIn("stationary", result.status_message.lower())

    def test_stationary_vehicle_within_sensor_jitter(self) -> None:
        """A vehicle moving <= 1.0 pixel due to detection jitter remains classified stationary."""
        # Jitter of 0.5px total
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=150.0,
            center_y=250.0,
            bounding_box=BoundingBox(x_min=100.0, y_min=200.0, x_max=200.0, y_max=300.0),
            confidence=0.9,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=10,
            timestamp_seconds=1.0,
            center_x=150.5,
            center_y=250.0,
            bounding_box=BoundingBox(x_min=100.5, y_min=200.0, x_max=200.5, y_max=300.0),
            confidence=0.9,
            class_name="car",
        )

        result = analyze_single_track_motion(track_id=2, observations=[obs1, obs2], fps=10.0)

        self.assertEqual(result.measurement_status, MotionMeasurementStatus.STATIONARY)
        self.assertTrue(result.is_stationary)
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 0.5, places=2)
        self.assertAlmostEqual(result.average_pixel_speed_px_per_sec, 0.5, places=2)

    def test_moving_vehicle_between_frames(self) -> None:
        """A vehicle displacing across frames produces valid image movement rates in px/s."""
        # Moves 50 pixels in 0.5 seconds -> 100 pixels / second
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=100.0,
            center_y=100.0,
            bounding_box=BoundingBox(x_min=80.0, y_min=80.0, x_max=120.0, y_max=120.0),
            confidence=0.88,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=15,
            timestamp_seconds=0.5,
            center_x=140.0,
            center_y=130.0,  # dx=40, dy=30 => dist=50.0 px
            bounding_box=BoundingBox(x_min=120.0, y_min=110.0, x_max=160.0, y_max=150.0),
            confidence=0.91,
            class_name="car",
        )
        obs3 = TrackObservationPoint(
            frame_number=30,
            timestamp_seconds=1.0,
            center_x=180.0,
            center_y=160.0,  # dx=40, dy=30 => dist=50.0 px
            bounding_box=BoundingBox(x_min=160.0, y_min=140.0, x_max=200.0, y_max=180.0),
            confidence=0.92,
            class_name="car",
        )

        result = analyze_single_track_motion(track_id=10, observations=[obs1, obs2, obs3], fps=30.0)

        self.assertEqual(result.measurement_status, MotionMeasurementStatus.VALID)
        self.assertFalse(result.is_stationary)
        self.assertEqual(result.usable_observation_count, 3)
        self.assertAlmostEqual(result.total_observed_duration_seconds, 1.0, places=2)
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 100.0, places=2)
        self.assertAlmostEqual(result.net_pixel_displacement, 100.0, places=2)
        self.assertAlmostEqual(result.average_pixel_speed_px_per_sec, 100.0, places=2)
        self.assertAlmostEqual(result.max_pixel_speed_px_per_sec, 100.0, places=2)
        self.assertEqual(len(result.step_displacements), 2)
        self.assertFalse(result.is_physical_speed_available)

    def test_elapsed_time_calculations_with_timestamps(self) -> None:
        """Verify elapsed time uses timestamp difference when positive."""
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=1.25,
            center_x=50.0,
            center_y=50.0,
            bounding_box=BoundingBox(x_min=40.0, y_min=40.0, x_max=60.0, y_max=60.0),
            confidence=0.85,
            class_name="truck",
        )
        obs2 = TrackObservationPoint(
            frame_number=6,
            timestamp_seconds=1.45,
            center_x=70.0,
            center_y=50.0,
            bounding_box=BoundingBox(x_min=60.0, y_min=40.0, x_max=80.0, y_max=60.0),
            confidence=0.87,
            class_name="truck",
        )

        result = analyze_single_track_motion(track_id=5, observations=[obs1, obs2], fps=30.0)

        self.assertEqual(len(result.step_displacements), 1)
        step = result.step_displacements[0]
        self.assertAlmostEqual(step.elapsed_seconds, 0.20, places=3)
        self.assertEqual(step.timing_source, "timestamp")
        # 20 px in 0.20s => 100 px/s
        self.assertAlmostEqual(step.pixel_speed_px_per_sec, 100.0, places=2)

    def test_elapsed_time_fallback_to_fps(self) -> None:
        """When timestamps are zero or identical, video FPS is used for elapsed time calculation."""
        obs1 = TrackObservationPoint(
            frame_number=10,
            timestamp_seconds=0.0,  # Unset timestamp
            center_x=100.0,
            center_y=100.0,
            bounding_box=BoundingBox(x_min=80.0, y_min=80.0, x_max=120.0, y_max=120.0),
            confidence=0.8,
            class_name="bus",
        )
        obs2 = TrackObservationPoint(
            frame_number=25,
            timestamp_seconds=0.0,  # Unset timestamp
            center_x=130.0,
            center_y=140.0,  # 50 px
            bounding_box=BoundingBox(x_min=110.0, y_min=120.0, x_max=150.0, y_max=160.0),
            confidence=0.8,
            class_name="bus",
        )

        # 15 frames at 30 FPS = 0.5 seconds
        result = analyze_single_track_motion(track_id=8, observations=[obs1, obs2], fps=30.0)

        self.assertEqual(len(result.step_displacements), 1)
        step = result.step_displacements[0]
        self.assertAlmostEqual(step.elapsed_seconds, 0.50, places=3)
        self.assertEqual(step.timing_source, "fps_fallback")
        # 50 px in 0.5s = 100 px/s
        self.assertAlmostEqual(step.pixel_speed_px_per_sec, 100.0, places=2)

    def test_different_video_fps_values(self) -> None:
        """Frame delta elapsed time scales inversely with FPS."""
        # 10 frame delta with 20 px displacement
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=0.0,
            center_y=0.0,
            bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0),
            confidence=0.9,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=10,
            timestamp_seconds=0.0,
            center_x=20.0,
            center_y=0.0,
            bounding_box=BoundingBox(x_min=15.0, y_min=0.0, x_max=25.0, y_max=10.0),
            confidence=0.9,
            class_name="car",
        )

        res_10fps = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=10.0)
        res_20fps = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=20.0)
        res_60fps = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=60.0)

        # At 10 fps, 10 frames = 1.0s -> 20 px / 1.0s = 20.0 px/s
        self.assertAlmostEqual(res_10fps.step_displacements[0].elapsed_seconds, 1.0, places=2)
        self.assertAlmostEqual(res_10fps.average_pixel_speed_px_per_sec, 20.0, places=2)

        # At 20 fps, 10 frames = 0.5s -> 20 px / 0.5s = 40.0 px/s
        self.assertAlmostEqual(res_20fps.step_displacements[0].elapsed_seconds, 0.5, places=2)
        self.assertAlmostEqual(res_20fps.average_pixel_speed_px_per_sec, 40.0, places=2)

        # At 60 fps, 10 frames = 1/6 s -> 20 px / (1/6)s = 120.0 px/s
        self.assertAlmostEqual(res_60fps.step_displacements[0].elapsed_seconds, 10.0 / 60.0, places=2)
        self.assertAlmostEqual(res_60fps.average_pixel_speed_px_per_sec, 120.0, places=2)

    def test_missing_observations_and_frame_gaps(self) -> None:
        """Tracking gaps between frames (e.g. frames 0, 5, 12) are handled safely."""
        obs = [
            TrackObservationPoint(
                frame_number=0,
                timestamp_seconds=0.0,
                center_x=10.0,
                center_y=10.0,
                bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
                confidence=0.9,
                class_name="car",
            ),
            TrackObservationPoint(
                frame_number=5,
                timestamp_seconds=0.25,
                center_x=20.0,
                center_y=10.0,  # 10 px
                bounding_box=BoundingBox(x_min=10.0, y_min=0.0, x_max=30.0, y_max=20.0),
                confidence=0.85,
                class_name="car",
            ),
            TrackObservationPoint(
                frame_number=12,
                timestamp_seconds=0.60,
                center_x=48.0,
                center_y=10.0,  # 28 px
                bounding_box=BoundingBox(x_min=38.0, y_min=0.0, x_max=58.0, y_max=20.0),
                confidence=0.82,
                class_name="car",
            ),
        ]

        result = analyze_single_track_motion(track_id=3, observations=obs, fps=20.0)

        self.assertEqual(result.usable_observation_count, 3)
        self.assertEqual(len(result.step_displacements), 2)
        self.assertEqual(result.step_displacements[0].from_frame, 0)
        self.assertEqual(result.step_displacements[0].to_frame, 5)
        self.assertEqual(result.step_displacements[1].from_frame, 5)
        self.assertEqual(result.step_displacements[1].to_frame, 12)
        # Total displacement = 10 + 28 = 38 px
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 38.0, places=2)
        self.assertAlmostEqual(result.net_pixel_displacement, 38.0, places=2)

    def test_invalid_or_non_positive_elapsed_times(self) -> None:
        """Steps with non-positive elapsed times are safely rejected without crashing."""
        obs = [
            TrackObservationPoint(
                frame_number=0,
                timestamp_seconds=2.0,  # Decreasing or non-positive timestamps
                center_x=10.0,
                center_y=10.0,
                bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
                confidence=0.9,
                class_name="car",
            ),
            TrackObservationPoint(
                frame_number=0,  # Same frame number
                timestamp_seconds=1.0,  # Earlier timestamp
                center_x=20.0,
                center_y=10.0,
                bounding_box=BoundingBox(x_min=10.0, y_min=0.0, x_max=30.0, y_max=20.0),
                confidence=0.8,
                class_name="car",
            ),
        ]

        result = analyze_single_track_motion(track_id=7, observations=obs, fps=0.0)

        # When sorted by frame_number and deduplicated on the same frame, highest confidence (obs[0]) is kept
        self.assertEqual(result.usable_observation_count, 1)
        self.assertEqual(result.measurement_status, MotionMeasurementStatus.INSUFFICIENT_OBSERVATIONS)

    def test_short_tracks_single_observation(self) -> None:
        """A track with only 1 usable observation returns INSUFFICIENT_OBSERVATIONS."""
        obs = [
            TrackObservationPoint(
                frame_number=0,
                timestamp_seconds=0.0,
                center_x=100.0,
                center_y=100.0,
                bounding_box=BoundingBox(x_min=80.0, y_min=80.0, x_max=120.0, y_max=120.0),
                confidence=0.9,
                class_name="motorcycle",
            )
        ]

        result = analyze_single_track_motion(track_id=99, observations=obs, fps=30.0)

        self.assertEqual(result.track_id, 99)
        self.assertEqual(result.usable_observation_count, 1)
        self.assertEqual(result.measurement_status, MotionMeasurementStatus.INSUFFICIENT_OBSERVATIONS)
        self.assertFalse(result.is_stationary)
        self.assertFalse(result.is_physical_speed_available)
        self.assertIsNone(result.average_pixel_speed_px_per_sec)
        self.assertIn("at least 2 distinct observations", result.status_message.lower())

    def test_missing_calibration_information(self) -> None:
        """Physical speed is unavailable and required calibration prerequisites are documented."""
        obs = [
            TrackObservationPoint(
                frame_number=0,
                timestamp_seconds=0.0,
                center_x=10.0,
                center_y=10.0,
                bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
                confidence=0.9,
                class_name="car",
            ),
            TrackObservationPoint(
                frame_number=10,
                timestamp_seconds=1.0,
                center_x=50.0,
                center_y=10.0,
                bounding_box=BoundingBox(x_min=40.0, y_min=0.0, x_max=60.0, y_max=20.0),
                confidence=0.9,
                class_name="car",
            ),
        ]

        result = analyze_single_track_motion(track_id=1, observations=obs, fps=10.0)

        self.assertFalse(result.is_physical_speed_available)
        self.assertEqual(result.physical_speed_limitation, CALIBRATION_REQUIREMENTS_SUMMARY)
        self.assertIn("homography matrix", result.physical_speed_limitation.lower())
        self.assertIn("perspective foreshortening", result.physical_speed_limitation.lower())
        self.assertIn("ground-truth scale", result.physical_speed_limitation.lower())

    def test_pixel_movement_without_falsely_claiming_physical_speed(self) -> None:
        """No output field claims km/h or meters without ground calibration."""
        obs = [
            TrackObservationPoint(
                frame_number=0,
                timestamp_seconds=0.0,
                center_x=0.0,
                center_y=0.0,
                bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0),
                confidence=0.9,
                class_name="car",
            ),
            TrackObservationPoint(
                frame_number=1,
                timestamp_seconds=0.1,
                center_x=10.0,
                center_y=0.0,
                bounding_box=BoundingBox(x_min=5.0, y_min=0.0, x_max=15.0, y_max=10.0),
                confidence=0.9,
                class_name="car",
            ),
        ]

        result = analyze_single_track_motion(track_id=1, observations=obs, fps=10.0)

        # Confirm fields specify pixels and px_per_sec, not km/h
        dump = result.model_dump()
        self.assertIn("cumulative_pixel_displacement", dump)
        self.assertIn("average_pixel_speed_px_per_sec", dump)
        self.assertNotIn("speed_kmh", dump)
        self.assertNotIn("speed_mph", dump)
        self.assertNotIn("distance_meters", dump)

    def test_multiple_vehicles_with_separate_track_ids(self) -> None:
        """VideoMotionSummary correctly aggregates multiple vehicles with different behaviors."""
        # Frame 0: Track 1 (at 100, 100), Track 2 (at 300, 300), Track 3 (at 500, 500)
        # Frame 1: Track 1 (at 150, 100) [moving], Track 2 (at 300, 300) [stationary]
        # Frame 2: Track 1 (at 200, 100) [moving], Track 2 (at 300, 300) [stationary]
        frames = [
            FrameDetectionResult(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.9,
                        bounding_box=BoundingBox(x_min=80.0, y_min=80.0, x_max=120.0, y_max=120.0),
                        track_id=1,
                    ),
                    DetectedObject(
                        class_name="bus",
                        confidence=0.85,
                        bounding_box=BoundingBox(x_min=280.0, y_min=280.0, x_max=320.0, y_max=320.0),
                        track_id=2,
                    ),
                    DetectedObject(
                        class_name="motorcycle",
                        confidence=0.75,
                        bounding_box=BoundingBox(x_min=490.0, y_min=490.0, x_max=510.0, y_max=510.0),
                        track_id=3,
                    ),
                ],
            ),
            FrameDetectionResult(
                frame_number=1,
                timestamp_seconds=0.5,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.92,
                        bounding_box=BoundingBox(x_min=130.0, y_min=80.0, x_max=170.0, y_max=120.0),
                        track_id=1,
                    ),
                    DetectedObject(
                        class_name="bus",
                        confidence=0.88,
                        bounding_box=BoundingBox(x_min=280.0, y_min=280.0, x_max=320.0, y_max=320.0),
                        track_id=2,
                    ),
                ],
            ),
            FrameDetectionResult(
                frame_number=2,
                timestamp_seconds=1.0,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.93,
                        bounding_box=BoundingBox(x_min=180.0, y_min=80.0, x_max=220.0, y_max=120.0),
                        track_id=1,
                    ),
                    DetectedObject(
                        class_name="bus",
                        confidence=0.87,
                        bounding_box=BoundingBox(x_min=280.0, y_min=280.0, x_max=320.0, y_max=320.0),
                        track_id=2,
                    ),
                ],
            ),
        ]

        summary = analyze_video_motion(
            frame_results=frames,
            fps=2.0,
            tracklet_classes={1: "car", 2: "bus", 3: "motorcycle"},
        )

        self.assertEqual(summary.total_tracks_analyzed, 3)
        self.assertEqual(summary.valid_moving_tracks_count, 1)  # Track 1
        self.assertEqual(summary.stationary_tracks_count, 1)    # Track 2
        self.assertEqual(summary.insufficient_data_tracks_count, 1)  # Track 3 (1 observation)
        self.assertFalse(summary.is_physical_speed_available)

        # Track 1 moved 100 px over 1.0s -> 100 px/s
        self.assertAlmostEqual(summary.average_scene_pixel_speed_px_per_sec, 100.0, places=2)

    def test_pipeline_evaluation_integration(self) -> None:
        """evaluate_frames integrates motion analysis into report and tracklets."""
        frames = [
            FrameDetectionResult(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.9,
                        bounding_box=BoundingBox(x_min=10.0, y_min=10.0, x_max=30.0, y_max=30.0),
                        track_id=1,
                    )
                ],
            ),
            FrameDetectionResult(
                frame_number=1,
                timestamp_seconds=0.1,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.91,
                        bounding_box=BoundingBox(x_min=30.0, y_min=10.0, x_max=50.0, y_max=30.0),
                        track_id=1,
                    )
                ],
            ),
        ]

        report = evaluate_frames(frames, min_track_lifespan_frames=1)

        self.assertIsNotNone(report.motion_summary)
        self.assertEqual(report.motion_summary.total_tracks_analyzed, 1)
        self.assertEqual(len(report.tracklets), 1)
        t = report.tracklets[0]
        self.assertIsNotNone(t.motion)
        self.assertEqual(t.motion.track_id, 1)
        self.assertEqual(t.motion.measurement_status, MotionMeasurementStatus.VALID)
        self.assertFalse(t.motion.is_physical_speed_available)

    def test_slow_moving_vehicle_across_distance_not_stationary(self) -> None:
        """A vehicle that moved 200px over 100s (2 px/s) is NOT stationary because net displacement is large."""
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=100.0,
            center_y=100.0,
            bounding_box=BoundingBox(x_min=80.0, y_min=80.0, x_max=120.0, y_max=120.0),
            confidence=0.9,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=1000,
            timestamp_seconds=100.0,
            center_x=300.0,  # 200 px displacement
            center_y=100.0,
            bounding_box=BoundingBox(x_min=280.0, y_min=80.0, x_max=320.0, y_max=120.0),
            confidence=0.9,
            class_name="car",
        )

        result = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=10.0)

        self.assertEqual(result.measurement_status, MotionMeasurementStatus.VALID)
        self.assertFalse(result.is_stationary)
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 200.0, places=2)
        self.assertAlmostEqual(result.net_pixel_displacement, 200.0, places=2)
        self.assertAlmostEqual(result.average_pixel_speed_px_per_sec, 2.0, places=2)

    def test_looped_trajectory_net_vs_cumulative_displacement(self) -> None:
        """A vehicle driving in a loop has near-zero net displacement but large cumulative displacement."""
        # Loop of 4 corners: (100, 100) -> (200, 100) -> (200, 200) -> (100, 200) -> (100, 100)
        coords = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0), (100.0, 100.0)]
        obs = [
            TrackObservationPoint(
                frame_number=i * 10,
                timestamp_seconds=float(i),
                center_x=c[0],
                center_y=c[1],
                bounding_box=BoundingBox(x_min=c[0] - 10.0, y_min=c[1] - 10.0, x_max=c[0] + 10.0, y_max=c[1] + 10.0),
                confidence=0.9,
                class_name="car",
            )
            for i, c in enumerate(coords)
        ]

        result = analyze_single_track_motion(track_id=1, observations=obs, fps=10.0)

        self.assertEqual(result.measurement_status, MotionMeasurementStatus.VALID)
        self.assertFalse(result.is_stationary)
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 400.0, places=2)
        self.assertAlmostEqual(result.net_pixel_displacement, 0.0, places=2)
        self.assertAlmostEqual(result.average_pixel_speed_px_per_sec, 100.0, places=2)

    def test_out_of_order_frame_observations(self) -> None:
        """Observations provided out of chronological order are sorted by frame number."""
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=10.0,
            center_y=10.0,
            bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
            confidence=0.9,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=10,
            timestamp_seconds=1.0,
            center_x=30.0,
            center_y=10.0,
            bounding_box=BoundingBox(x_min=20.0, y_min=0.0, x_max=40.0, y_max=20.0),
            confidence=0.9,
            class_name="car",
        )
        obs3 = TrackObservationPoint(
            frame_number=20,
            timestamp_seconds=2.0,
            center_x=60.0,
            center_y=10.0,
            bounding_box=BoundingBox(x_min=50.0, y_min=0.0, x_max=70.0, y_max=20.0),
            confidence=0.9,
            class_name="car",
        )

        # Pass in reverse order: obs3, obs1, obs2
        result = analyze_single_track_motion(track_id=1, observations=[obs3, obs1, obs2], fps=10.0)

        self.assertEqual(result.first_observed_frame, 0)
        self.assertEqual(result.last_observed_frame, 20)
        self.assertEqual(len(result.step_displacements), 2)
        self.assertEqual(result.step_displacements[0].from_frame, 0)
        self.assertEqual(result.step_displacements[0].to_frame, 10)
        self.assertEqual(result.step_displacements[1].from_frame, 10)
        self.assertEqual(result.step_displacements[1].to_frame, 20)

    def test_duplicate_frame_observations_deduplication(self) -> None:
        """Duplicate observations on the same frame retain the higher confidence observation."""
        obs_low = TrackObservationPoint(
            frame_number=5,
            timestamp_seconds=0.5,
            center_x=20.0,
            center_y=10.0,
            bounding_box=BoundingBox(x_min=10.0, y_min=0.0, x_max=30.0, y_max=20.0),
            confidence=0.50,
            class_name="car",
        )
        obs_high = TrackObservationPoint(
            frame_number=5,
            timestamp_seconds=0.5,
            center_x=25.0,
            center_y=10.0,
            bounding_box=BoundingBox(x_min=15.0, y_min=0.0, x_max=35.0, y_max=20.0),
            confidence=0.95,
            class_name="car",
        )
        obs_start = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=0.0,
            center_y=10.0,
            bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
            confidence=0.90,
            class_name="car",
        )

        result = analyze_single_track_motion(track_id=1, observations=[obs_start, obs_low, obs_high], fps=10.0)

        self.assertEqual(result.usable_observation_count, 2)
        # Displacement should be from obs_start (0, 10) to obs_high (25, 10) = 25.0 px
        self.assertAlmostEqual(result.cumulative_pixel_displacement, 25.0, places=2)

    def test_non_finite_coordinate_rejection(self) -> None:
        """Non-finite coordinates raise validation errors and cannot slip into calculations."""
        with self.assertRaises(ValueError):
            BoundingBox(x_min=float("nan"), y_min=0.0, x_max=10.0, y_max=10.0)

        with self.assertRaises(ValueError):
            BoundingBox(x_min=0.0, y_min=0.0, x_max=float("inf"), y_max=10.0)

        with self.assertRaises(ValueError):
            compute_pixel_displacement((float("nan"), 0.0), (10.0, 10.0))

        with self.assertRaises(ValueError):
            TrackObservationPoint(
                frame_number=0,
                timestamp_seconds=float("nan"),
                center_x=10.0,
                center_y=10.0,
                bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
                confidence=0.9,
                class_name="car",
            )

    def test_negative_or_non_finite_fps_fallback(self) -> None:
        """Negative, zero, or non-finite FPS values safely fallback without division by zero."""
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=0.0,
            center_y=0.0,
            bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0),
            confidence=0.9,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=30,
            timestamp_seconds=0.0,  # Zero timestamp forces FPS fallback
            center_x=30.0,
            center_y=0.0,
            bounding_box=BoundingBox(x_min=20.0, y_min=0.0, x_max=40.0, y_max=10.0),
            confidence=0.9,
            class_name="car",
        )

        # Negative FPS falls back to DEFAULT_FALLBACK_FPS (30.0)
        res_neg = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=-10.0)
        self.assertEqual(res_neg.measurement_status, MotionMeasurementStatus.VALID)
        self.assertAlmostEqual(res_neg.step_displacements[0].elapsed_seconds, 1.0, places=2)

        # NaN FPS falls back to DEFAULT_FALLBACK_FPS (30.0)
        res_nan = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=float("nan"))
        self.assertEqual(res_nan.measurement_status, MotionMeasurementStatus.VALID)
        self.assertAlmostEqual(res_nan.step_displacements[0].elapsed_seconds, 1.0, places=2)

    def test_duration_weighted_vs_unweighted_scene_speed(self) -> None:
        """VideoMotionSummary reports both unweighted track-mean and duration-weighted scene speed."""
        # Track 1: short transient track: moves 10 px over 0.1s -> speed = 100 px/s
        # Track 2: long persistent track: moves 100 px over 10.0s -> speed = 10 px/s
        frames = [
            FrameDetectionResult(
                frame_number=0,
                timestamp_seconds=0.0,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.9,
                        bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
                        track_id=1,
                    ),
                    DetectedObject(
                        class_name="truck",
                        confidence=0.9,
                        bounding_box=BoundingBox(x_min=100.0, y_min=0.0, x_max=120.0, y_max=20.0),
                        track_id=2,
                    ),
                ],
            ),
            FrameDetectionResult(
                frame_number=1,
                timestamp_seconds=0.1,
                detections=[
                    DetectedObject(
                        class_name="car",
                        confidence=0.9,
                        bounding_box=BoundingBox(x_min=10.0, y_min=0.0, x_max=30.0, y_max=20.0),  # 10 px
                        track_id=1,
                    ),
                ],
            ),
            FrameDetectionResult(
                frame_number=100,
                timestamp_seconds=10.0,
                detections=[
                    DetectedObject(
                        class_name="truck",
                        confidence=0.9,
                        bounding_box=BoundingBox(x_min=200.0, y_min=0.0, x_max=220.0, y_max=20.0),  # 100 px
                        track_id=2,
                    ),
                ],
            ),
        ]

        summary = analyze_video_motion(frame_results=frames, fps=10.0)

        self.assertEqual(summary.valid_moving_tracks_count, 2)
        # Unweighted mean: (100.0 + 10.0) / 2 = 55.0 px/s
        self.assertAlmostEqual(summary.average_scene_pixel_speed_px_per_sec, 55.0, places=2)
        # Weighted mean: (10 + 100) / (0.1 + 10.0) = 110 / 10.1 = 10.891 px/s
        self.assertAlmostEqual(summary.weighted_scene_pixel_speed_px_per_sec, 10.891, places=2)

    def test_custom_stationary_thresholds(self) -> None:
        """Custom stationary displacement and speed thresholds are respected."""
        obs1 = TrackObservationPoint(
            frame_number=0,
            timestamp_seconds=0.0,
            center_x=0.0,
            center_y=0.0,
            bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0),
            confidence=0.9,
            class_name="car",
        )
        obs2 = TrackObservationPoint(
            frame_number=10,
            timestamp_seconds=1.0,
            center_x=8.0,
            center_y=0.0,  # 8px displacement at 8 px/s
            bounding_box=BoundingBox(x_min=3.0, y_min=0.0, x_max=13.0, y_max=10.0),
            confidence=0.9,
            class_name="car",
        )

        # Under default threshold (3px), 8px is VALID (moving)
        res_default = analyze_single_track_motion(track_id=1, observations=[obs1, obs2], fps=10.0)
        self.assertEqual(res_default.measurement_status, MotionMeasurementStatus.VALID)
        self.assertFalse(res_default.is_stationary)

        # Under custom threshold (10px, 10px/s), 8px is STATIONARY
        res_custom = analyze_single_track_motion(
            track_id=1,
            observations=[obs1, obs2],
            fps=10.0,
            stationary_displacement_threshold_px=10.0,
            stationary_speed_threshold_px_per_sec=10.0,
        )
        self.assertEqual(res_custom.measurement_status, MotionMeasurementStatus.STATIONARY)
        self.assertTrue(res_custom.is_stationary)


if __name__ == "__main__":
    unittest.main()
