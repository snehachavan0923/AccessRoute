"""Unit tests for VisionTrafficAdapter in backend/traffic/vision_adapter.py."""

from datetime import datetime, timedelta, timezone
import threading
import unittest

from backend.traffic.loader import TrafficEdgeKey
from backend.traffic.vision_adapter import (
    ObservationValidationStatus,
    VisionTrafficAdapter,
)
from backend.vision.schemas import (
    CameraMetadata,
    ObservationTimeEnvelope,
    VisionEdgeObservation,
)


class TestVisionTrafficAdapter(unittest.TestCase):
    """Test suite for VisionTrafficAdapter functionality, safety checks, and edge cases."""

    def setUp(self) -> None:
        self.now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        self.calibrated_camera = CameraMetadata(
            camera_id="cam_main_01",
            latitude=12.8476,
            longitude=77.6712,
            heading_degrees=90.0,
            target_osm_u=1001,
            target_osm_v=1002,
            target_osm_key=0,
            is_spatially_calibrated=True,
        )
        self.uncalibrated_camera = CameraMetadata(
            camera_id="cam_uncalibrated",
            latitude=12.8476,
            longitude=77.6712,
            heading_degrees=90.0,
            target_osm_u=1001,
            target_osm_v=1002,
            target_osm_key=0,
            is_spatially_calibrated=False,
        )

    def _create_observation(
        self,
        osm_u: int = 1001,
        osm_v: int = 1002,
        osm_key: int = 0,
        camera: CameraMetadata | None = None,
        start_time: datetime | None = None,
        duration_seconds: float = 60.0,
        ttl_seconds: float = 900.0,
        quality_score: float = 0.95,
        congestion_index: float | None = 0.45,
        unique_vehicles: int = 12,
    ) -> VisionEdgeObservation:
        """Helper to create a valid VisionEdgeObservation."""
        start = start_time or (self.now - timedelta(seconds=120))
        end = start + timedelta(seconds=duration_seconds)
        timing = ObservationTimeEnvelope(
            start_time=start,
            end_time=end,
            duration_seconds=duration_seconds,
            time_to_live_seconds=ttl_seconds,
        )

        if camera is not None and (
            (camera.target_osm_u is not None and camera.target_osm_u != osm_u)
            or (camera.target_osm_v is not None and camera.target_osm_v != osm_v)
            or (camera.target_osm_key != osm_key)
        ):
            cam_data = camera.model_dump()
            cam_data["camera_id"] = f"{camera.camera_id}_{osm_u}_{osm_v}_{osm_key}"
            cam_data["target_osm_u"] = osm_u
            cam_data["target_osm_v"] = osm_v
            cam_data["target_osm_key"] = osm_key
            camera = CameraMetadata(**cam_data)

        return VisionEdgeObservation(
            camera_id=camera.camera_id if camera else "cam_default",
            camera_metadata=camera,
            osm_u=osm_u,
            osm_v=osm_v,
            osm_key=osm_key,
            timing=timing,
            unique_vehicles_observed=unique_vehicles,
            vehicles_by_class={"car": unique_vehicles},
            observation_quality_score=quality_score,
            experimental_congestion_index=congestion_index,
        )

    # -------------------------------------------------------------------------
    # 1. Basic Ingestion & Directed-Edge Matching
    # -------------------------------------------------------------------------

    def test_ingest_and_retrieve_valid_observation(self) -> None:
        adapter = VisionTrafficAdapter()
        obs = self._create_observation(camera=self.calibrated_camera)
        edge_key = adapter.ingest(obs)

        self.assertEqual(edge_key, TrafficEdgeKey(1001, 1002, 0))
        self.assertTrue(adapter.has_edge(edge_key))
        self.assertEqual(len(adapter), 1)

        retrieved = adapter.get_valid_observation(edge_key, current_time=self.now)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.osm_u, 1001)
        self.assertEqual(retrieved.osm_v, 1002)

        # Raw observation access
        raw = adapter.get_raw_observation(edge_key)
        self.assertEqual(raw, obs)

    def test_directed_edge_matching_and_reverse_edge_separation(self) -> None:
        """Verify directed edges (u -> v) and reverse edges (v -> u) are strictly separated."""
        adapter = VisionTrafficAdapter()
        forward_obs = self._create_observation(
            osm_u=1001,
            osm_v=1002,
            osm_key=0,
            camera=self.calibrated_camera,
            congestion_index=0.60,
        )
        forward_key = adapter.ingest(forward_obs)

        reverse_key = TrafficEdgeKey(osm_u=1002, osm_v=1001, key=0)
        self.assertTrue(adapter.has_edge(forward_key))
        self.assertFalse(adapter.has_edge(reverse_key))

        # Querying forward edge returns the observation
        self.assertIsNotNone(adapter.get_valid_observation(forward_key, current_time=self.now))
        self.assertEqual(adapter.get_congestion_index(forward_key, current_time=self.now), 0.60)

        # Querying reverse edge returns None (never leaks across directions)
        self.assertIsNone(adapter.get_valid_observation(reverse_key, current_time=self.now))
        self.assertIsNone(adapter.get_congestion_index(reverse_key, current_time=self.now))
        self.assertEqual(
            adapter.evaluate_observation_status(reverse_key, current_time=self.now),
            ObservationValidationStatus.NOT_FOUND,
        )

    def test_parallel_edges_distinguished_by_key(self) -> None:
        """Verify parallel edges with different keys are stored separately."""
        adapter = VisionTrafficAdapter()
        obs_key0 = self._create_observation(osm_u=1001, osm_v=1002, osm_key=0, camera=self.calibrated_camera, congestion_index=0.3)
        obs_key1 = self._create_observation(osm_u=1001, osm_v=1002, osm_key=1, camera=self.calibrated_camera, congestion_index=0.7)

        key0 = adapter.ingest(obs_key0)
        key1 = adapter.ingest(obs_key1)

        self.assertEqual(len(adapter), 2)
        self.assertEqual(adapter.get_congestion_index(key0, current_time=self.now), 0.3)
        self.assertEqual(adapter.get_congestion_index(key1, current_time=self.now), 0.7)

    # -------------------------------------------------------------------------
    # 2. Spatial Calibration Filtering
    # -------------------------------------------------------------------------

    def test_uncalibrated_observation_rejected_when_calibration_required(self) -> None:
        """Uncalibrated cameras must be rejected when require_spatial_calibration=True."""
        adapter = VisionTrafficAdapter(require_spatial_calibration=True)
        obs = self._create_observation(camera=self.uncalibrated_camera)
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.UNCALIBRATED)

        self.assertIsNone(adapter.get_valid_observation(key, current_time=self.now))
        self.assertIsNone(adapter.get_congestion_index(key, current_time=self.now))

        # Raw observation still accessible for inspection
        self.assertIsNotNone(adapter.get_raw_observation(key))

    def test_observation_without_camera_metadata_rejected_as_uncalibrated(self) -> None:
        adapter = VisionTrafficAdapter(require_spatial_calibration=True)
        obs = self._create_observation(camera=None)  # camera_metadata is None
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.UNCALIBRATED)
        self.assertIsNone(adapter.get_valid_observation(key, current_time=self.now))

    def test_uncalibrated_observation_accepted_when_calibration_not_required(self) -> None:
        adapter = VisionTrafficAdapter(require_spatial_calibration=False)
        obs = self._create_observation(camera=self.uncalibrated_camera, congestion_index=0.4)
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.VALID)
        self.assertEqual(adapter.get_congestion_index(key, current_time=self.now), 0.4)

    # -------------------------------------------------------------------------
    # 3. Temporal Validity & TTL Expiration
    # -------------------------------------------------------------------------

    def test_expired_observation_rejected(self) -> None:
        """Observation older than TTL must be marked EXPIRED and return None."""
        adapter = VisionTrafficAdapter(default_ttl_seconds=900.0)  # 15 mins
        # Observation ended 20 minutes ago
        old_start = self.now - timedelta(minutes=21)
        obs = self._create_observation(
            camera=self.calibrated_camera,
            start_time=old_start,
            duration_seconds=60.0,
            ttl_seconds=900.0,
        )
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.EXPIRED)

        self.assertIsNone(adapter.get_valid_observation(key, current_time=self.now))
        self.assertIsNone(adapter.get_congestion_index(key, current_time=self.now))

    def test_fresh_observation_within_ttl_accepted(self) -> None:
        adapter = VisionTrafficAdapter(default_ttl_seconds=900.0)
        # Observation ended 5 minutes ago (well within 15 min TTL)
        recent_start = self.now - timedelta(minutes=6)
        obs = self._create_observation(
            camera=self.calibrated_camera,
            start_time=recent_start,
            duration_seconds=60.0,
            ttl_seconds=900.0,
            congestion_index=0.55,
        )
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.VALID)
        self.assertEqual(adapter.get_congestion_index(key, current_time=self.now), 0.55)

    def test_future_clock_skew_rejected(self) -> None:
        """Observation ending far in the future (> 5s skew) must be rejected."""
        adapter = VisionTrafficAdapter(max_future_skew_seconds=5.0)
        # Observation ended 1 minute in the future
        future_start = self.now + timedelta(seconds=30)
        obs = self._create_observation(
            camera=self.calibrated_camera,
            start_time=future_start,
            duration_seconds=30.0,
        )
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.FUTURE_SKEW)
        self.assertIsNone(adapter.get_valid_observation(key, current_time=self.now))

    def test_timezone_naive_datetime_rejected(self) -> None:
        adapter = VisionTrafficAdapter()
        obs = self._create_observation(camera=self.calibrated_camera)
        key = adapter.ingest(obs)

        naive_time = datetime(2026, 10, 27, 12, 0, 0)  # No tzinfo
        with self.assertRaises(ValueError) as ctx:
            adapter.get_valid_observation(key, current_time=naive_time)
        self.assertIn("timezone-aware", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Quality Threshold Filtering
    # -------------------------------------------------------------------------

    def test_low_quality_observation_rejected(self) -> None:
        """Observation with quality below min_quality_score (e.g. 0.50 vs 0.70) is rejected."""
        adapter = VisionTrafficAdapter(min_quality_score=0.70)
        obs = self._create_observation(
            camera=self.calibrated_camera,
            quality_score=0.65,  # Below 0.70
        )
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.LOW_QUALITY)
        self.assertIsNone(adapter.get_valid_observation(key, current_time=self.now))
        self.assertIsNone(adapter.get_congestion_index(key, current_time=self.now))

    def test_high_quality_observation_accepted(self) -> None:
        adapter = VisionTrafficAdapter(min_quality_score=0.70)
        obs = self._create_observation(
            camera=self.calibrated_camera,
            quality_score=0.70,  # Exactly at threshold
            congestion_index=0.25,
        )
        key = adapter.ingest(obs)

        status = adapter.evaluate_observation_status(key, current_time=self.now)
        self.assertEqual(status, ObservationValidationStatus.VALID)
        self.assertEqual(adapter.get_congestion_index(key, current_time=self.now), 0.25)

    # -------------------------------------------------------------------------
    # 5. Congestion Formula Safeguard (No Invented Formula)
    # -------------------------------------------------------------------------

    def test_no_congestion_formula_invented_when_index_is_none(self) -> None:
        """When experimental_congestion_index is None, get_congestion_index MUST return None.

        It must NEVER invent a formula from unique_vehicles_observed or tracker IDs.
        """
        adapter = VisionTrafficAdapter()
        obs = self._create_observation(
            camera=self.calibrated_camera,
            unique_vehicles=45,
            congestion_index=None,  # No congestion indicator
        )
        key = adapter.ingest(obs)

        # Observation itself is valid and retrievable
        self.assertIsNotNone(adapter.get_valid_observation(key, current_time=self.now))

        # But congestion index is None (no uncalibrated number returned)
        self.assertIsNone(adapter.get_congestion_index(key, current_time=self.now))

    # -------------------------------------------------------------------------
    # 6. Management Operations (Purge, Remove, Clear)
    # -------------------------------------------------------------------------

    def test_purge_expired_removes_only_expired_records(self) -> None:
        adapter = VisionTrafficAdapter(default_ttl_seconds=300.0)

        # Fresh observation (ended 1 minute ago)
        fresh_obs = self._create_observation(
            osm_u=1001,
            osm_v=1002,
            camera=self.calibrated_camera,
            start_time=self.now - timedelta(minutes=2),
            ttl_seconds=300.0,
        )
        fresh_key = adapter.ingest(fresh_obs)

        # Expired observation (ended 10 minutes ago, TTL is 5 minutes)
        expired_obs = self._create_observation(
            osm_u=2001,
            osm_v=2002,
            camera=self.calibrated_camera,
            start_time=self.now - timedelta(minutes=11),
            ttl_seconds=300.0,
        )
        expired_key = adapter.ingest(expired_obs)

        self.assertEqual(len(adapter), 2)
        purged = adapter.purge_expired(current_time=self.now)
        self.assertEqual(purged, 1)

        self.assertEqual(len(adapter), 1)
        self.assertTrue(adapter.has_edge(fresh_key))
        self.assertFalse(adapter.has_edge(expired_key))

    def test_remove_and_clear(self) -> None:
        adapter = VisionTrafficAdapter()
        obs = self._create_observation(camera=self.calibrated_camera)
        key = adapter.ingest(obs)

        self.assertTrue(adapter.has_edge(key))
        self.assertTrue(adapter.remove_observation(key))
        self.assertFalse(adapter.has_edge(key))
        self.assertFalse(adapter.remove_observation(key))

        # Test clear
        adapter.ingest(obs)
        self.assertEqual(len(adapter), 1)
        adapter.clear()
        self.assertEqual(len(adapter), 0)

    # -------------------------------------------------------------------------
    # 7. Thread-Safety / Concurrent Access
    # -------------------------------------------------------------------------

    def test_concurrent_ingest_and_reads(self) -> None:
        """Verify thread-safety when multiple threads ingest and read concurrently."""
        adapter = VisionTrafficAdapter()
        errors: list[Exception] = []

        def worker_ingest(thread_id: int) -> None:
            try:
                for i in range(50):
                    u = 1000 + thread_id
                    v = 2000 + i
                    obs = self._create_observation(
                        osm_u=u,
                        osm_v=v,
                        camera=self.calibrated_camera,
                        congestion_index=round((i % 10) * 0.1, 2),
                    )
                    adapter.ingest(obs)
            except Exception as exc:
                errors.append(exc)

        def worker_read(thread_id: int) -> None:
            try:
                for i in range(50):
                    u = 1000 + thread_id
                    v = 2000 + i
                    key = TrafficEdgeKey(u, v, 0)
                    adapter.get_valid_observation(key, current_time=self.now)
                    adapter.get_congestion_index(key, current_time=self.now)
                    adapter.get_all_edge_keys()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker_ingest, args=(t,)) for t in range(5)
        ] + [
            threading.Thread(target=worker_read, args=(t,)) for t in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors encountered: {errors}")
        self.assertGreater(len(adapter), 0)

    # -------------------------------------------------------------------------
    # 8. Initialization Validation
    # -------------------------------------------------------------------------

    def test_invalid_init_parameters_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            VisionTrafficAdapter(default_ttl_seconds=30.0)  # min 60.0
        with self.assertRaises(ValueError):
            VisionTrafficAdapter(min_quality_score=-0.1)
        with self.assertRaises(ValueError):
            VisionTrafficAdapter(min_quality_score=1.5)
        with self.assertRaises(ValueError):
            VisionTrafficAdapter(max_future_skew_seconds=-1.0)

    def test_type_error_on_invalid_observation_type(self) -> None:
        adapter = VisionTrafficAdapter()
        with self.assertRaises(TypeError):
            adapter.ingest({"not": "a VisionEdgeObservation"})  # type: ignore

    # -------------------------------------------------------------------------
    # 9. Additional Regression & Boundary Tests
    # -------------------------------------------------------------------------

    def test_observation_replacement_on_same_edge(self) -> None:
        """Verify that ingesting a newer observation on the same edge overwrites the previous one."""
        adapter = VisionTrafficAdapter()
        obs1 = self._create_observation(
            camera=self.calibrated_camera,
            unique_vehicles=5,
            congestion_index=0.20,
        )
        obs2 = self._create_observation(
            camera=self.calibrated_camera,
            unique_vehicles=20,
            congestion_index=0.80,
        )

        key1 = adapter.ingest(obs1)
        self.assertEqual(len(adapter), 1)
        self.assertEqual(adapter.get_congestion_index(key1, current_time=self.now), 0.20)

        key2 = adapter.ingest(obs2)
        self.assertEqual(key1, key2)
        self.assertEqual(len(adapter), 1)
        retrieved = adapter.get_valid_observation(key1, current_time=self.now)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.unique_vehicles_observed, 20)
        self.assertEqual(adapter.get_congestion_index(key1, current_time=self.now), 0.80)

    def test_exact_ttl_and_skew_boundary_conditions(self) -> None:
        """Verify boundary precision: age == TTL is valid, age > TTL is expired, age == -skew is valid."""
        adapter = VisionTrafficAdapter(default_ttl_seconds=300.0, max_future_skew_seconds=5.0)
        end_time = self.now

        obs = self._create_observation(
            camera=self.calibrated_camera,
            start_time=end_time - timedelta(seconds=60),
            duration_seconds=60.0,
            ttl_seconds=300.0,
        )
        key = adapter.ingest(obs)

        # 1. Exactly at TTL limit (age == 300.0s) -> VALID
        at_ttl = end_time + timedelta(seconds=300.0)
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=at_ttl),
            ObservationValidationStatus.VALID,
        )

        # 2. Just past TTL limit (age == 300.01s) -> EXPIRED
        past_ttl = end_time + timedelta(seconds=300.01)
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=past_ttl),
            ObservationValidationStatus.EXPIRED,
        )

        # 3. Exactly at future skew limit (age == -5.0s) -> VALID
        at_skew = end_time - timedelta(seconds=5.0)
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=at_skew),
            ObservationValidationStatus.VALID,
        )

        # 4. Farther into future (age == -5.01s) -> FUTURE_SKEW
        past_skew = end_time - timedelta(seconds=5.01)
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=past_skew),
            ObservationValidationStatus.FUTURE_SKEW,
        )

    def test_timezone_conversion_consistency(self) -> None:
        """Verify that timezone-aware datetimes with different offsets evaluate consistently."""
        adapter = VisionTrafficAdapter()
        # Observation timestamped in UTC: 12:00:00 UTC
        utc_time = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        obs = self._create_observation(
            camera=self.calibrated_camera,
            start_time=utc_time - timedelta(seconds=60),
            duration_seconds=60.0,
            ttl_seconds=300.0,
        )
        key = adapter.ingest(obs)

        # Reference time in IST (+05:30): 17:30:00 IST is the EXACT same moment as 12:00:00 UTC
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        ist_time = datetime(2026, 10, 27, 17, 30, 0, tzinfo=ist_tz)

        # Age should be 0.0s elapsed from end_time
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=ist_time),
            ObservationValidationStatus.VALID,
        )
        self.assertIsNotNone(adapter.get_valid_observation(key, current_time=ist_time))

    def test_no_congestion_indicator_status_and_contracts(self) -> None:
        """Check API contracts for observations lacking experimental_congestion_index."""
        adapter = VisionTrafficAdapter()
        obs = self._create_observation(
            camera=self.calibrated_camera,
            congestion_index=None,
        )
        key = adapter.ingest(obs)

        # evaluate_observation_status without requiring congestion returns VALID
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=self.now, require_congestion_index=False),
            ObservationValidationStatus.VALID,
        )

        # evaluate_observation_status with require_congestion_index returns NO_CONGESTION_INDICATOR
        self.assertEqual(
            adapter.evaluate_observation_status(key, current_time=self.now, require_congestion_index=True),
            ObservationValidationStatus.NO_CONGESTION_INDICATOR,
        )

        # get_valid_observation returns the observation
        self.assertIsNotNone(adapter.get_valid_observation(key, current_time=self.now))

        # get_congestion_index returns None
        self.assertIsNone(adapter.get_congestion_index(key, current_time=self.now))

    def test_boolean_congestion_and_quality_rejected(self) -> None:
        """Verify that boolean values for float fields are rejected by schemas and adapter."""
        from pydantic import ValidationError

        # Schema rejects boolean for experimental_congestion_index
        with self.assertRaises(ValidationError):
            self._create_observation(congestion_index=True)  # type: ignore

        # Schema rejects boolean for observation_quality_score
        with self.assertRaises(ValidationError):
            self._create_observation(quality_score=False)  # type: ignore

    def test_tuple_edge_key_normalization(self) -> None:
        """Verify that tuples (u, v) and (u, v, key) are accepted and normalized."""
        adapter = VisionTrafficAdapter()
        obs = self._create_observation(
            osm_u=1001,
            osm_v=1002,
            osm_key=0,
            camera=self.calibrated_camera,
            congestion_index=0.65,
        )
        adapter.ingest(obs)

        # Query using 3-tuple (1001, 1002, 0)
        self.assertTrue(adapter.has_edge((1001, 1002, 0)))
        self.assertIsNotNone(adapter.get_valid_observation((1001, 1002, 0), current_time=self.now))
        self.assertEqual(adapter.get_congestion_index((1001, 1002, 0), current_time=self.now), 0.65)

        # Query using 2-tuple (1001, 1002) - defaults key to 0
        self.assertTrue(adapter.has_edge((1001, 1002)))
        self.assertIsNotNone(adapter.get_valid_observation((1001, 1002), current_time=self.now))
        self.assertEqual(adapter.get_congestion_index((1001, 1002), current_time=self.now), 0.65)

        # Invalid tuple raises TypeError or ValueError
        with self.assertRaises((TypeError, ValueError)):
            adapter.has_edge("invalid_key")  # type: ignore


if __name__ == "__main__":
    unittest.main()

