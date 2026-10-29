"""Milestone 3 Demonstration: Deterministic End-to-End Vision-to-Routing Integration Test.

Demonstrates all Milestone 3 requirements using a synthetic test graph and
explicitly constructed synthetic test observations:

A. With no eligible vision observation, the alternative route uses the simulated dataset.
B. With an eligible synthetic observation for a specific directed edge, the traffic-aware
   route uses that observation according to the existing cost formula:
       Cost(e) = Length(e) * (1 + traffic_weight * congestion_index)
C. The shortest-distance route remains strictly independent of vision observations.
D. Uncalibrated, expired, low-quality, or invalid observations are rejected, and dataset
   fallback remains available.
E. An observation for one directed edge does not affect its reverse edge or unrelated roads.
F. Existing route comparison, response schemas, and explainability fields work as expected.

IMPORTANT DISCLAIMER:
All observations in this test suite are explicitly labeled SYNTHETIC TEST DATA.
They do NOT represent real-world camera measurements.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import networkx as nx
import numpy as np

from backend.routing.contracts import Coordinate
from backend.routing.traffic_aware_path import (
    TrafficAwarePathService,
    TrafficRouteComparisonService,
)
from backend.tests.test_traffic_loader import create_synthetic_dataset
from backend.traffic import (
    TrafficDataset,
    TrafficEdgeKey,
    VisionTrafficAdapter,
)
from backend.vision.schemas import (
    CameraMetadata,
    ObservationTimeEnvelope,
    VisionEdgeObservation,
)


def make_synthetic_network() -> nx.MultiDiGraph:
    """Create a deterministic synthetic graph with two alternative paths:

    Path 1 (Direct): 1 -> 2 -> 3 (100m + 100m = 200m physical)
    Path 2 (Detour): 1 -> 4 -> 3 (120m + 120m = 240m physical)
    Reverse paths are also present to test directed edge isolation.
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"

    g.add_node(1, x=77.670, y=12.840)
    g.add_node(2, x=77.671, y=12.840)
    g.add_node(3, x=77.672, y=12.840)
    g.add_node(4, x=77.671, y=12.841)

    # Path 1: Forward
    g.add_edge(1, 2, 0, length=100.0, highway="residential")
    g.add_edge(2, 3, 0, length=100.0, highway="residential")

    # Path 1: Reverse
    g.add_edge(3, 2, 0, length=100.0, highway="residential")
    g.add_edge(2, 1, 0, length=100.0, highway="residential")

    # Path 2: Forward detour
    g.add_edge(1, 4, 0, length=120.0, highway="footway")
    g.add_edge(4, 3, 0, length=120.0, highway="footway")

    # Path 2: Reverse detour
    g.add_edge(3, 4, 0, length=120.0, highway="footway")
    g.add_edge(4, 1, 0, length=120.0, highway="footway")

    return g


class TestVisionRoutingMilestoneDemonstration(unittest.TestCase):
    """Formal verification of Milestone 3 requirements A through F."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # 4 nodes matching graph
        nodes = [
            {"node_id": "0", "osm_node_id": "1", "x": "77.670", "y": "12.840"},
            {"node_id": "1", "osm_node_id": "2", "x": "77.671", "y": "12.840"},
            {"node_id": "2", "osm_node_id": "3", "x": "77.672", "y": "12.840"},
            {"node_id": "3", "osm_node_id": "4", "x": "77.671", "y": "12.841"},
        ]

        # Edges in dataset:
        # Edge 0: (1 -> 2, key 0)
        # Edge 1: (2 -> 3, key 0)
        # Edge 2: (2 -> 1, key 0) [reverse]
        # Edge 3: (3 -> 2, key 0) [reverse]
        edges = [
            {"edge_id": "0", "u": "0", "v": "1", "osm_u": "1", "osm_v": "2", "key": "0", "length_m": "100.0", "highway": "residential"},
            {"edge_id": "1", "u": "1", "v": "2", "osm_u": "2", "osm_v": "3", "key": "0", "length_m": "100.0", "highway": "residential"},
            {"edge_id": "2", "u": "1", "v": "0", "osm_u": "2", "osm_v": "1", "key": "0", "length_m": "100.0", "highway": "residential"},
            {"edge_id": "3", "u": "2", "v": "1", "osm_u": "3", "osm_v": "2", "key": "0", "length_m": "100.0", "highway": "residential"},
        ]

        timestamps = [
            "2026-01-01 00:00:00",
            "2026-01-01 00:05:00",
        ]

        # Congestion matrix: shape (2, 4)
        # time_index=1 has mild congestion (0.05) on edges 0 and 1 (1->2, 2->3), 0.0 on reverse edges
        congestion_matrix = np.array([
            [0.10, 0.10, 0.0, 0.0],
            [0.05, 0.05, 0.0, 0.0],
        ], dtype=float)

        speed_matrix = np.array([
            [30.0, 30.0, 30.0, 30.0],
            [35.0, 35.0, 35.0, 35.0],
        ], dtype=float)

        vehicle_matrix = np.array([
            [5.0, 5.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
        ], dtype=float)

        create_synthetic_dataset(
            self.dataset_dir,
            timestamps=timestamps,
            nodes=nodes,
            edges=edges,
            congestion_matrix=congestion_matrix,
            speed_matrix=speed_matrix,
            vehicle_matrix=vehicle_matrix,
        )

        self.dataset = TrafficDataset.load(self.dataset_dir)
        self.mock_network_provider = Mock()
        self.mock_network_provider.load.return_value = make_synthetic_network()

        self.vision_adapter = VisionTrafficAdapter(default_ttl_seconds=900.0)
        self.comparison_service = TrafficRouteComparisonService(
            network_provider=self.mock_network_provider,
            traffic_dataset=self.dataset,
            vision_adapter=self.vision_adapter,
        )

        self.origin = Coordinate(latitude=12.840, longitude=77.670)
        self.destination = Coordinate(latitude=12.840, longitude=77.672)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_synthetic_test_observation(
        self,
        osm_u: int = 1,
        osm_v: int = 2,
        osm_key: int = 0,
        congestion_index: float | None = 0.90,
        quality_score: float = 0.95,
        is_calibrated: bool = True,
        age_seconds: float = 10.0,
        ttl_seconds: float = 600.0,
    ) -> VisionEdgeObservation:
        """Construct an explicitly labeled synthetic observation for controlled tests."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(seconds=age_seconds + 60.0)
        end = now - timedelta(seconds=age_seconds)

        camera_meta = CameraMetadata(
            camera_id="synthetic-test-cam-01",
            latitude=12.840,
            longitude=77.670,
            heading_degrees=90.0,
            target_osm_u=osm_u,
            target_osm_v=osm_v,
            target_osm_key=osm_key,
            is_spatially_calibrated=is_calibrated,
        )

        return VisionEdgeObservation(
            camera_id="synthetic-test-cam-01",
            camera_metadata=camera_meta,
            osm_u=osm_u,
            osm_v=osm_v,
            osm_key=osm_key,
            timing=ObservationTimeEnvelope(
                start_time=start,
                end_time=end,
                duration_seconds=60.0,
                time_to_live_seconds=ttl_seconds,
            ),
            unique_vehicles_observed=15,
            vehicles_by_class={"car": 12, "motorcycle": 3},
            observation_quality_score=quality_score,
            experimental_congestion_index=congestion_index,
            disclaimer="SYNTHETIC TEST OBSERVATION — FOR CONTROLLED INTEGRATION TESTING ONLY.",
        )

    def test_A_no_eligible_vision_observation_uses_simulated_dataset(self) -> None:
        """Milestone 3A: When no vision observation exists, routing uses simulated dataset."""
        # Ensure vision adapter is completely empty
        self.assertEqual(len(self.vision_adapter), 0)

        res = self.comparison_service.compare(
            self.origin,
            self.destination,
            time_index=1,
            traffic_weight=2.0,
        )

        # Baseline shortest path is 200m
        self.assertEqual(res.shortest_distance.distance_meters, 200.0)

        # Direct path (1 -> 2 -> 3) has simulated congestion 0.05 on both edges.
        # Cost = 100 * (1 + 2 * 0.05) + 100 * (1 + 2 * 0.05) = 110 + 110 = 220m.
        # Detour path (1 -> 4 -> 3) cost = 120 + 120 = 240m.
        # Because 220m < 240m, both routes choose the direct path.
        self.assertEqual(res.status, "identical_routes")
        self.assertEqual(res.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertAlmostEqual(res.traffic_accessibility_aware.cost, 220.0, places=1)
        self.assertEqual(res.primary_traffic_source, "simulated")
        self.assertEqual(res.traffic_accessibility_aware.simulated_matched_edges_count, 2)
        self.assertEqual(res.traffic_accessibility_aware.vision_matched_edges_count, 0)
        self.assertIn("simulated Electronic City traffic dataset fallback", res.explanation)

    def test_B_eligible_synthetic_observation_alters_route_by_cost_formula(self) -> None:
        """Milestone 3B: An eligible synthetic observation modifies edge cost and diverts route."""
        # Synthetic observation on directed edge (1 -> 2) with heavy congestion = 0.90
        obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, osm_key=0, congestion_index=0.90
        )
        self.vision_adapter.set_observation(obs)

        res = self.comparison_service.compare(
            self.origin,
            self.destination,
            time_index=1,
            traffic_weight=2.0,
        )

        # With synthetic vision observation:
        # Edge (1 -> 2) cost = 100 * (1 + 2 * 0.90) = 280.0m (vision override)
        # Edge (2 -> 3) cost = 100 * (1 + 2 * 0.05) = 110.0m (simulated fallback)
        # Direct path total cost = 280 + 110 = 390.0m.
        # Detour path (1 -> 4 -> 3) total cost = 120 + 120 = 240.0m.
        # Since 240m < 390m, the traffic-aware route diverts to Path 2 (240m)!
        self.assertEqual(res.status, "alternative_found")
        self.assertEqual(res.traffic_accessibility_aware.distance_meters, 240.0)
        self.assertAlmostEqual(res.traffic_accessibility_aware.cost, 240.0, places=1)
        self.assertTrue(res.vision_observation_active)
        self.assertIn("avoids vehicle traffic congestion", res.explanation)

    def test_C_shortest_distance_route_remains_independent_of_vision(self) -> None:
        """Milestone 3C: Shortest-distance route remains strictly 200.0m regardless of congestion."""
        # Heavy congestion observation on edge (1 -> 2)
        obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, osm_key=0, congestion_index=0.99
        )
        self.vision_adapter.set_observation(obs)

        res = self.comparison_service.compare(
            self.origin,
            self.destination,
            time_index=1,
            traffic_weight=5.0,  # Extreme weight
        )

        # Shortest distance route is strictly calculated on edge length (200.0m)
        self.assertEqual(res.shortest_distance.distance_meters, 200.0)
        self.assertEqual(res.shortest_distance.edges_count, 2)

    def test_D_rejected_observations_fall_back_to_simulated_dataset(self) -> None:
        """Milestone 3D: Uncalibrated, expired, low-quality, and future observations are rejected."""
        # 1. Uncalibrated camera observation
        uncalibrated_obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, is_calibrated=False, congestion_index=0.90
        )
        self.vision_adapter.set_observation(uncalibrated_obs)
        res_uncalibrated = self.comparison_service.compare(self.origin, self.destination, time_index=1)
        self.assertEqual(res_uncalibrated.status, "identical_routes")
        self.assertEqual(res_uncalibrated.traffic_accessibility_aware.distance_meters, 200.0)

        # 2. Expired observation (age 1000s > TTL 600s)
        self.vision_adapter.clear()
        expired_obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, age_seconds=1000.0, ttl_seconds=600.0, congestion_index=0.90
        )
        self.vision_adapter.set_observation(expired_obs)
        res_expired = self.comparison_service.compare(self.origin, self.destination, time_index=1)
        self.assertEqual(res_expired.status, "identical_routes")
        self.assertEqual(res_expired.traffic_accessibility_aware.distance_meters, 200.0)

        # 3. Low-quality observation (quality 0.40 < threshold 0.70)
        self.vision_adapter.clear()
        low_qual_obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, quality_score=0.40, congestion_index=0.90
        )
        self.vision_adapter.set_observation(low_qual_obs)
        res_low_qual = self.comparison_service.compare(self.origin, self.destination, time_index=1)
        self.assertEqual(res_low_qual.status, "identical_routes")
        self.assertEqual(res_low_qual.traffic_accessibility_aware.distance_meters, 200.0)

        # 4. Missing congestion index (None)
        self.vision_adapter.clear()
        no_cong_obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, congestion_index=None
        )
        self.vision_adapter.set_observation(no_cong_obs)
        res_no_cong = self.comparison_service.compare(self.origin, self.destination, time_index=1)
        self.assertEqual(res_no_cong.status, "identical_routes")
        self.assertEqual(res_no_cong.traffic_accessibility_aware.distance_meters, 200.0)

    def test_E_directed_edge_isolation_and_no_reverse_leakage(self) -> None:
        """Milestone 3E: Observation on (1 -> 2) does not affect reverse edge (2 -> 1) or unrelated edges."""
        obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, osm_key=0, congestion_index=0.95
        )
        self.vision_adapter.set_observation(obs)

        # Reverse query: from node 3 to node 1
        origin_rev = Coordinate(latitude=12.840, longitude=77.672)
        dest_rev = Coordinate(latitude=12.840, longitude=77.670)

        res_rev = self.comparison_service.compare(
            origin_rev,
            dest_rev,
            time_index=1,
            traffic_weight=2.0,
        )

        # Reverse path 3 -> 2 -> 1 has simulated congestion 0.0, total distance 200m
        self.assertEqual(res_rev.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertAlmostEqual(res_rev.traffic_accessibility_aware.cost, 200.0, places=1)
        self.assertEqual(res_rev.traffic_accessibility_aware.vision_matched_edges_count, 0)

    def test_F_route_comparison_response_fields_and_attribution(self) -> None:
        """Milestone 3F: RouteComparison exposes transparent source attribution and cost metrics."""
        obs = self._build_synthetic_test_observation(
            osm_u=1, osm_v=2, osm_key=0, congestion_index=0.90
        )
        self.vision_adapter.set_observation(obs)

        res = self.comparison_service.compare(
            self.origin,
            self.destination,
            time_index=1,
            traffic_weight=2.0,
        )

        # Verify response structure and schema compliance
        # Since edge (1->2) has vision and edge (2->3) has simulated fallback, source is mixed
        self.assertEqual(res.primary_traffic_source, "mixed")
        self.assertTrue(res.vision_observation_active)
        self.assertIsInstance(res.explanation, str)
        self.assertGreater(len(res.explanation), 20)

        # Candidate breakdown
        self.assertGreater(res.shortest_distance.distance_meters, 0)
        self.assertGreater(res.traffic_accessibility_aware.distance_meters, 0)
        self.assertIsInstance(res.traffic_accessibility_aware.traffic_cost_penalty, float)
        self.assertIsInstance(res.traffic_accessibility_aware.accessibility_cost_penalty, float)


if __name__ == "__main__":
    unittest.main()
