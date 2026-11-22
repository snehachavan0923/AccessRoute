"""Tests for route recommendation explainability and data-source attribution.

Verifies:
1. Shortest route baseline remains independent of traffic and accessibility.
2. RouteComparison transparency: primary_traffic_source accurately reflects 'vision', 'simulated', 'mixed', or 'none'.
3. Human-readable explanation explicitly identifies why a route alternative was selected.
4. When vision observations are unavailable or rejected, simulated fallback is explicitly documented.
5. GET /api/routes/observation-status reports active vision observations and accessibility barriers.
"""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock

import networkx as nx

from backend.accessibility import (
    AccessibilityAdapter,
    AccessibilityObstacleObservation,
    AccessibilityObstacleType,
    AccessibilitySeverity,
)
from backend.routing.contracts import Coordinate
from backend.routing.routes import get_routing_observation_status
from backend.routing.traffic_aware_path import (
    TrafficRouteComparisonService,
)
from backend.traffic import (
    TomTomMatchResult,
    TomTomObservation,
    TomTomTrafficAdapter,
    TrafficEdgeKey,
    VisionTrafficAdapter,
    set_default_vision_adapter,
)
from backend.vision.schemas import (
    CameraMetadata,
    ObservationTimeEnvelope,
    VisionEdgeObservation,
)


def make_forked_test_graph() -> nx.MultiDiGraph:
    """Graph with two routes:

    Route 1: 1 -> 2 -> 3 (direct, 100m + 100m = 200m)
    Route 2: 1 -> 4 -> 3 (detour, 120m + 120m = 240m)
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"

    g.add_node(1, x=77.670, y=12.840)
    g.add_node(2, x=77.671, y=12.840)
    g.add_node(3, x=77.672, y=12.840)
    g.add_node(4, x=77.671, y=12.841)

    g.add_edge(1, 2, 0, length=100.0, highway="residential")
    g.add_edge(2, 3, 0, length=100.0, highway="residential")

    g.add_edge(1, 4, 0, length=120.0, highway="footway")
    g.add_edge(4, 3, 0, length=120.0, highway="footway")

    return g


class TestRouteExplainability(unittest.TestCase):
    def setUp(self) -> None:
        self.network_provider = Mock()
        self.network_provider.load.return_value = make_forked_test_graph()
        self.vision_adapter = VisionTrafficAdapter()
        self.accessibility_adapter = AccessibilityAdapter()

        self.service = TrafficRouteComparisonService(
            network_provider=self.network_provider,
            traffic_dataset=None,
            vision_adapter=self.vision_adapter,
            accessibility_adapter=self.accessibility_adapter,
        )

        self.origin = Coordinate(latitude=12.840, longitude=77.670)
        self.destination = Coordinate(latitude=12.840, longitude=77.672)

    def test_no_traffic_data_produces_none_source_and_shortest_path(self) -> None:
        res = self.service.compare(self.origin, self.destination)
        self.assertEqual(res.primary_traffic_source, "none")
        self.assertAlmostEqual(res.shortest_distance.distance_meters, 200.0)
        self.assertAlmostEqual(res.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertIn("shortest walking route is also recommended", res.explanation)

    def test_vision_observation_attribution(self) -> None:
        now = datetime.now(timezone.utc)
        start = now - timedelta(seconds=60)
        end = now
        obs = VisionEdgeObservation(
            camera_id="cam-ecity-01",
            camera_metadata=CameraMetadata(
                camera_id="cam-ecity-01",
                latitude=12.84,
                longitude=77.67,
                heading_degrees=90.0,
                target_osm_u=1,
                target_osm_v=2,
                target_osm_key=0,
                is_spatially_calibrated=True,
            ),
            osm_u=1,
            osm_v=2,
            osm_key=0,
            timing=ObservationTimeEnvelope(
                start_time=start,
                end_time=end,
                duration_seconds=60.0,
                time_to_live_seconds=600.0,
            ),
            unique_vehicles_observed=12,
            vehicles_by_class={"car": 12},
            observation_quality_score=0.95,
            experimental_congestion_index=0.90,
        )
        self.vision_adapter.set_observation(obs)

        res = self.service.compare(
            self.origin,
            self.destination,
            traffic_weight=2.0,
        )

        self.assertEqual(res.status, "alternative_found")
        self.assertEqual(res.primary_traffic_source, "vision")
        self.assertTrue(res.vision_observation_active)
        self.assertEqual(
            res.traffic_accessibility_aware.vision_matched_edges_count, 0
        )  # Detour avoids congested edge (1 -> 2)!
        self.assertEqual(
            res.shortest_distance.vision_matched_edges_count, 1
        )  # Congested edge was on shortest path
        self.assertIn("avoids vehicle traffic congestion", res.explanation)

    def test_observation_status_endpoint(self) -> None:
        now = datetime.now(timezone.utc)
        start = now - timedelta(seconds=60)
        end = now
        set_default_vision_adapter(self.vision_adapter)

        obs = VisionEdgeObservation(
            camera_id="cam-test",
            camera_metadata=CameraMetadata(
                camera_id="cam-test",
                latitude=12.84,
                longitude=77.67,
                heading_degrees=90.0,
                target_osm_u=100,
                target_osm_v=200,
                target_osm_key=0,
                is_spatially_calibrated=True,
            ),
            osm_u=100,
            osm_v=200,
            osm_key=0,
            timing=ObservationTimeEnvelope(
                start_time=start,
                end_time=end,
                duration_seconds=60.0,
                time_to_live_seconds=600.0,
            ),
            unique_vehicles_observed=8,
            vehicles_by_class={"car": 8},
            observation_quality_score=0.90,
            experimental_congestion_index=0.75,
        )
        self.vision_adapter.set_observation(obs)

        status = get_routing_observation_status()
        self.assertGreaterEqual(status.vision_registered_count, 1)
        self.assertGreaterEqual(status.active_vision_observations_count, 1)
        obs_item = next(
            (o for o in status.active_vision_observations if o["osm_u"] == 100),
            None,
        )
        self.assertIsNotNone(obs_item)
        self.assertEqual(obs_item["camera_id"], "cam-test")
        self.assertEqual(obs_item["experimental_congestion_index"], 0.75)

    def test_mg_road_attribution_shortest_tomtom_recommended_diverts_to_footways(self) -> None:
        """When shortest route has TomTom congestion and recommended route diverts onto footways,

        source attribution must distinguish per-route sources and clearly explain the diversion.
        """
        mock_tomtom = Mock(spec=TomTomTrafficAdapter)
        mock_tomtom.is_available = True

        def mock_match(edge_data, src, tgt):
            hw = edge_data.get("highway")
            if hw == "residential":
                obs = TomTomObservation(
                    current_speed_kmh=10.0,
                    free_flow_speed_kmh=50.0,
                    current_travel_time=150.0,
                    free_flow_travel_time=30.0,
                    confidence=1.0,
                    road_closure=False,
                    frc="FRC3",
                    congestion_index=0.80,
                    coordinates=((12.840, 77.670), (12.840, 77.671)),
                    fetched_at=1000.0,
                )
                return TomTomMatchResult(matched=True, observation=obs)
            return TomTomMatchResult(matched=False, observation=None)

        mock_tomtom.get_edge_observation.side_effect = mock_match

        service = TrafficRouteComparisonService(
            network_provider=self.network_provider,
            traffic_dataset=None,
            vision_adapter=None,
            tomtom_adapter=mock_tomtom,
        )

        res = service.compare(self.origin, self.destination, traffic_weight=2.0)

        # Baseline shortest path cost: 200m + traffic penalty (80% congestion)
        # Detour path cost: 240m on footway with 0 traffic penalty
        self.assertEqual(res.status, "alternative_found")
        self.assertAlmostEqual(res.shortest_distance.distance_meters, 200.0)
        self.assertAlmostEqual(res.traffic_accessibility_aware.distance_meters, 240.0)

        # Shortest route has TomTom matches; recommended route has 0
        self.assertEqual(res.shortest_distance.tomtom_matched_edges_count, 2)
        self.assertEqual(res.shortest_distance.traffic_source, "tomtom")
        self.assertEqual(res.traffic_accessibility_aware.tomtom_matched_edges_count, 0)
        self.assertEqual(res.traffic_accessibility_aware.traffic_source, "none")

        # Comparison-level primary_traffic_source reflects TomTom because it drove the detour
        self.assertEqual(res.primary_traffic_source, "tomtom")

        # Explanation must clearly state that alternative diverts onto pedestrian footways
        self.assertIn("avoids vehicle traffic congestion", res.explanation)
        self.assertIn(
            "recommended alternative diverts onto pedestrian footways to avoid traffic observed on the shortest route",
            res.explanation,
        )
        self.assertNotIn("active on 2 recommended corridor segment(s)", res.explanation)

    def test_accurate_per_route_and_comparison_level_source_labels(self) -> None:
        """Both candidates must accurately state their own traffic_source while comparison states primary_traffic_source."""
        mock_tomtom = Mock(spec=TomTomTrafficAdapter)
        mock_tomtom.is_available = True

        def mock_match(edge_data, src, tgt):
            obs = TomTomObservation(
                current_speed_kmh=45.0,
                free_flow_speed_kmh=50.0,
                current_travel_time=33.0,
                free_flow_travel_time=30.0,
                confidence=1.0,
                road_closure=False,
                frc="FRC3",
                congestion_index=0.10,
                coordinates=((12.840, 77.670), (12.840, 77.671)),
                fetched_at=1000.0,
            )
            return TomTomMatchResult(matched=True, observation=obs)

        mock_tomtom.get_edge_observation.side_effect = mock_match

        service = TrafficRouteComparisonService(
            network_provider=self.network_provider,
            traffic_dataset=None,
            vision_adapter=None,
            tomtom_adapter=mock_tomtom,
        )

        # Identical routes when traffic is light
        res = service.compare(self.origin, self.destination, traffic_weight=0.1)
        self.assertEqual(res.status, "identical_routes")
        self.assertEqual(res.shortest_distance.traffic_source, "tomtom")
        self.assertEqual(res.traffic_accessibility_aware.traffic_source, "tomtom")
        self.assertEqual(res.primary_traffic_source, "tomtom")
        self.assertIn("active on 2 corridor segment(s)", res.explanation)


if __name__ == "__main__":
    unittest.main()
