import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
from fastapi import HTTPException

from backend.pedestrian_network import PedestrianNetworkProvider
from backend.routing.contracts import Coordinate, RouteCandidate, RouteComparison
from backend.routing.routes import (
    CompareRoutesRequest,
    ShortestRouteRequest,
    calculate_shortest_route,
    compare_routes,
)
from backend.routing.shortest_path import (
    NetworkAreaSettings,
    NetworkAreaTooLargeError,
    RouteNetworkUnavailableError,
    RouteNotFoundError,
    ShortestPathService,
)
from backend.routing.traffic_aware_path import (
    TrafficAwarePathService,
    TrafficRouteComparisonService,
)
from backend.tests.test_traffic_loader import create_synthetic_dataset
from backend.traffic import TrafficDataset


def make_forked_graph() -> nx.MultiDiGraph:
    """Graph with two alternative paths between node 1 and node 3:
    - Path A: 1 -> 2 -> 3 (direct: 100m + 100m = 200m)
    - Path B: 1 -> 4 -> 3 (detour: 120m + 120m = 240m)
    """
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.001, y=0.0)
    graph.add_node(3, x=0.002, y=0.0)
    graph.add_node(4, x=0.001, y=0.001)

    # Path A: 1 -> 2 -> 3
    graph.add_edge(1, 2, key=0, length=100.0)
    graph.add_edge(2, 3, key=0, length=100.0)

    # Path B: 1 -> 4 -> 3
    graph.add_edge(1, 4, key=0, length=120.0)
    graph.add_edge(4, 3, key=0, length=120.0)

    # Also reverse edges for bidirectional graph
    graph.add_edge(3, 2, key=0, length=100.0)
    graph.add_edge(2, 1, key=0, length=100.0)
    graph.add_edge(3, 4, key=0, length=120.0)
    graph.add_edge(4, 1, key=0, length=120.0)

    return graph


def make_parallel_edge_graph() -> nx.MultiDiGraph:
    """Graph with two parallel edges between node 1 and node 2:
    - key 0: length 100m
    - key 1: length 110m
    """
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.001, y=0.0)

    graph.add_edge(1, 2, key=0, length=100.0)
    graph.add_edge(1, 2, key=1, length=110.0)
    return graph


class TrafficAwareRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Nodes matching graph node IDs: 1, 2, 3, 4
        nodes = [
            {"node_id": "0", "osm_node_id": "1", "x": "0.0", "y": "0.0"},
            {"node_id": "1", "osm_node_id": "2", "x": "0.001", "y": "0.0"},
            {"node_id": "2", "osm_node_id": "3", "x": "0.002", "y": "0.0"},
            {"node_id": "3", "osm_node_id": "4", "x": "0.001", "y": "0.001"},
        ]

        # 4 edges in dataset:
        # edge 0: 1 -> 2 (key 0)
        # edge 1: 2 -> 3 (key 0)
        # edge 2: 1 -> 4 (key 0)
        # edge 3: 4 -> 3 (key 0)
        edges = [
            {
                "edge_id": "0",
                "u": "0",
                "v": "1",
                "osm_u": "1",
                "osm_v": "2",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "1",
                "u": "1",
                "v": "2",
                "osm_u": "2",
                "osm_v": "3",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "2",
                "u": "0",
                "v": "3",
                "osm_u": "1",
                "osm_v": "4",
                "key": "0",
                "length_m": "120.0",
                "highway": "footway",
                "maxspeed": "10",
            },
            {
                "edge_id": "3",
                "u": "3",
                "v": "2",
                "osm_u": "4",
                "osm_v": "3",
                "key": "0",
                "length_m": "120.0",
                "highway": "footway",
                "maxspeed": "10",
            },
            {
                "edge_id": "4",
                "u": "2",
                "v": "1",
                "osm_u": "3",
                "osm_v": "2",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "5",
                "u": "1",
                "v": "0",
                "osm_u": "2",
                "osm_v": "1",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
        ]

        # Timestamps: 2 timestamps
        timestamps = ["2026-01-01 08:30:00", "2026-01-01 02:00:00"]

        # Congestion matrix: shape (2, 6)
        # At time 0 (08:30 AM):
        # Forward edges 0 and 1 (direct path 1->2->3) are heavily congested: 0.85
        # Detour edges 2 and 3 (1->4->3) and reverse edges 4 and 5 (3->2->1) have zero congestion: 0.0
        # At time 1 (02:00 AM):
        # All edges have low congestion: 0.05
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
            timestamps=timestamps,
            nodes=nodes,
            edges=edges,
            congestion_matrix=congestion_matrix,
            speed_matrix=speed_matrix,
            vehicle_matrix=vehicle_matrix,
        )
        self.dataset = TrafficDataset.load(self.dataset_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_diverges_to_alternative_route_when_traffic_penalty_justifies(self) -> None:
        provider = Mock()
        provider.load.return_value = make_forked_graph()

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=self.dataset,
        )

        origin = Coordinate(latitude=0.0, longitude=0.0)
        destination = Coordinate(latitude=0.0, longitude=0.002)

        comparison = service.compare(
            origin,
            destination,
            timestamp="2026-01-01 08:30:00",
            traffic_weight=2.0,
        )

        # Shortest distance path takes 1 -> 2 -> 3 (distance = 200m)
        self.assertAlmostEqual(comparison.shortest_distance.distance_meters, 200.0)
        # Cost of shortest route: 100*(1 + 2*0.85) + 100*(1 + 2*0.85) = 270 + 270 = 540
        self.assertAlmostEqual(comparison.shortest_distance.cost, 540.0, places=2)

        # Traffic-aware path takes 1 -> 4 -> 3 (distance = 240m, cost = 240m)
        self.assertAlmostEqual(
            comparison.traffic_accessibility_aware.distance_meters, 240.0
        )
        self.assertAlmostEqual(comparison.traffic_accessibility_aware.cost, 240.0, places=2)

        # Status must indicate alternative route was found
        self.assertEqual(comparison.status, "alternative_found")
        self.assertEqual(comparison.selected_timestamp, "2026-01-01 08:30:00")
        self.assertEqual(comparison.time_index, 0)

        # Verify coordinates differ (traffic-aware goes via node 4 at lat 0.001)
        traffic_lats = [
            c.latitude for c in comparison.traffic_accessibility_aware.coordinates
        ]
        self.assertIn(0.001, traffic_lats)

    def test_routes_are_identical_when_traffic_is_low(self) -> None:
        provider = Mock()
        provider.load.return_value = make_forked_graph()

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=self.dataset,
        )

        origin = Coordinate(latitude=0.0, longitude=0.0)
        destination = Coordinate(latitude=0.0, longitude=0.002)

        # At 02:00 AM (time index 1), congestion is 0.05
        # Cost of path A = 200 * (1 + 2*0.05) = 220 < 240 (Path B)
        comparison = service.compare(
            origin,
            destination,
            time_index=1,
            traffic_weight=2.0,
        )

        self.assertAlmostEqual(comparison.shortest_distance.distance_meters, 200.0)
        self.assertAlmostEqual(
            comparison.traffic_accessibility_aware.distance_meters, 200.0
        )
        self.assertEqual(comparison.status, "identical_routes")

    def test_directed_edge_distinction_preserves_reverse_path(self) -> None:
        provider = Mock()
        provider.load.return_value = make_forked_graph()

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=self.dataset,
        )

        # Reverse direction: 3 -> 1
        # Edges (3->2) and (2->1) are NOT in dataset (only forward 1->2 and 2->3 were in dataset)
        # Unmatched edges incur 0 penalty, so 3->2->1 has cost 200m!
        origin = Coordinate(latitude=0.0, longitude=0.002)
        destination = Coordinate(latitude=0.0, longitude=0.0)

        comparison = service.compare(origin, destination, time_index=0)
        self.assertAlmostEqual(
            comparison.traffic_accessibility_aware.distance_meters, 200.0
        )
        self.assertEqual(comparison.status, "identical_routes")

    def test_parallel_edge_selection_favors_less_congested_edge(self) -> None:
        # Create dataset where (1, 2, 0) has congestion 0.9 and (1, 2, 1) has congestion 0.0
        with tempfile.TemporaryDirectory() as p_dir:
            p_path = Path(p_dir)
            p_nodes = [
                {"node_id": "0", "osm_node_id": "1", "x": "0.0", "y": "0.0"},
                {"node_id": "1", "osm_node_id": "2", "x": "0.001", "y": "0.0"},
            ]
            p_edges = [
                {
                    "edge_id": "0",
                    "u": "0",
                    "v": "1",
                    "osm_u": "1",
                    "osm_v": "2",
                    "key": "0",
                    "length_m": "100.0",
                    "highway": "primary",
                    "maxspeed": "50",
                },
                {
                    "edge_id": "1",
                    "u": "0",
                    "v": "1",
                    "osm_u": "1",
                    "osm_v": "2",
                    "key": "1",
                    "length_m": "110.0",
                    "highway": "service",
                    "maxspeed": "30",
                },
            ]
            p_cong = np.array([[0.9, 0.0]], dtype=np.float32)
            p_spd = np.ones((1, 2), dtype=np.float32) * 30.0
            p_veh = np.ones((1, 2), dtype=np.float32) * 10.0

            create_synthetic_dataset(
                p_path,
                timestamps=["2026-01-01 00:00:00"],
                nodes=p_nodes,
                edges=p_edges,
                congestion_matrix=p_cong,
                speed_matrix=p_spd,
                vehicle_matrix=p_veh,
            )
            p_dataset = TrafficDataset.load(p_path)

            provider = Mock()
            provider.load.return_value = make_parallel_edge_graph()

            service = TrafficRouteComparisonService(
                network_provider=provider,
                traffic_dataset=p_dataset,
            )

            comparison = service.compare(
                Coordinate(0.0, 0.0), Coordinate(0.0, 0.001), time_index=0
            )

            # Shortest path chose key 0 (100m)
            self.assertAlmostEqual(comparison.shortest_distance.distance_meters, 100.0)
            # Traffic-aware chose key 1 (110m, cost 110 < 100 * (1 + 2*0.9) = 280)
            self.assertAlmostEqual(
                comparison.traffic_accessibility_aware.distance_meters, 110.0
            )
            self.assertAlmostEqual(
                comparison.traffic_accessibility_aware.cost, 110.0
            )

    def test_zero_traffic_coverage_defaults_safely(self) -> None:
        # Create an empty or disjoint dataset
        with tempfile.TemporaryDirectory() as empty_dir:
            empty_path = Path(empty_dir)
            create_synthetic_dataset(empty_path)  # nodes 101, 102, 103 (disjoint from 1,2,3)
            disjoint_dataset = TrafficDataset.load(empty_path)

            provider = Mock()
            provider.load.return_value = make_forked_graph()

            service = TrafficRouteComparisonService(
                network_provider=provider,
                traffic_dataset=disjoint_dataset,
            )

            comparison = service.compare(
                Coordinate(0.0, 0.0), Coordinate(0.0, 0.002)
            )

            self.assertEqual(comparison.status, "no_traffic_coverage_in_area")
            self.assertEqual(
                comparison.traffic_accessibility_aware.matched_traffic_edges_count, 0
            )
            self.assertEqual(
                comparison.traffic_accessibility_aware.traffic_coverage_percent, 0.0
            )
            self.assertAlmostEqual(
                comparison.traffic_accessibility_aware.distance_meters, 200.0
            )

    def test_missing_dataset_service_fallback(self) -> None:
        provider = Mock()
        provider.load.return_value = make_forked_graph()

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=None,
        )
        # Explicitly simulate dataset being None
        service._traffic_service._dataset = None

        comparison = service.compare(
            Coordinate(0.0, 0.0), Coordinate(0.0, 0.002)
        )
        self.assertEqual(comparison.status, "traffic_dataset_unavailable")
        self.assertAlmostEqual(
            comparison.traffic_accessibility_aware.distance_meters, 200.0
        )

    def test_invalid_time_index_raises_error(self) -> None:
        provider = Mock()
        provider.load.return_value = make_forked_graph()

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=self.dataset,
        )

        with self.assertRaises(ValueError):
            service.compare(
                Coordinate(0.0, 0.0), Coordinate(0.0, 0.002), time_index=99
            )

    def test_negative_traffic_weight_raises_error(self) -> None:
        provider = Mock()
        provider.load.return_value = make_forked_graph()

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=self.dataset,
        )

        with self.assertRaises(ValueError):
            service.compare(
                Coordinate(0.0, 0.0), Coordinate(0.0, 0.002), traffic_weight=-1.0
            )


class RouteComparisonEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compare_request = CompareRoutesRequest.model_validate(
            {
                "origin": {"latitude": 0.0, "longitude": 0.0},
                "destination": {"latitude": 0.0, "longitude": 0.002},
                "timestamp": "2026-01-01 08:30:00",
                "traffic_weight": 2.0,
            }
        )
        self.shortest_request = ShortestRouteRequest.model_validate(
            {
                "origin": {"latitude": 0.0, "longitude": 0.0},
                "destination": {"latitude": 0.0, "longitude": 0.002},
            }
        )

    @patch("backend.routing.routes.comparison_service")
    def test_compare_endpoint_returns_both_routes_and_metadata(
        self, service: Mock
    ) -> None:
        cand_short = RouteCandidate(
            coordinates=(Coordinate(0.0, 0.0), Coordinate(0.0, 0.002)),
            distance_meters=200.0,
            cost=500.0,
            edges_count=2,
            matched_traffic_edges_count=2,
            traffic_coverage_percent=100.0,
            average_congestion=0.75,
        )
        cand_traffic = RouteCandidate(
            coordinates=(
                Coordinate(0.0, 0.0),
                Coordinate(0.001, 0.001),
                Coordinate(0.0, 0.002),
            ),
            distance_meters=240.0,
            cost=240.0,
            edges_count=2,
            matched_traffic_edges_count=2,
            traffic_coverage_percent=100.0,
            average_congestion=0.0,
        )
        service.compare.return_value = RouteComparison(
            shortest_distance=cand_short,
            traffic_accessibility_aware=cand_traffic,
            snapped_origin=Coordinate(0.0, 0.0),
            snapped_destination=Coordinate(0.0, 0.002),
            selected_timestamp="2026-01-01 08:30:00",
            time_index=0,
            cost_formula="Cost formula...",
            status="alternative_found",
            disclaimer="Simulated traffic...",
        )

        response = compare_routes(self.compare_request)

        self.assertEqual(response.status, "alternative_found")
        self.assertEqual(response.shortest_distance.distance_meters, 200.0)
        self.assertEqual(
            response.traffic_accessibility_aware.distance_meters, 240.0
        )
        self.assertEqual(response.selected_timestamp, "2026-01-01 08:30:00")
        self.assertEqual(len(response.traffic_accessibility_aware.coordinates), 3)

    @patch("backend.routing.routes.comparison_service")
    def test_compare_endpoint_handles_invalid_value_error(
        self, service: Mock
    ) -> None:
        service.compare.side_effect = ValueError("Invalid time index")
        with self.assertRaises(HTTPException) as raised:
            compare_routes(self.compare_request)
        self.assertEqual(raised.exception.status_code, 422)

    @patch("backend.routing.routes.shortest_path_service")
    def test_shortest_route_endpoint_still_works_unchanged(
        self, service: Mock
    ) -> None:
        from backend.routing.contracts import ShortestRoute

        service.calculate.return_value = ShortestRoute(
            coordinates=(Coordinate(0.0, 0.0), Coordinate(0.0, 0.002)),
            distance_meters=200.0,
            snapped_origin=Coordinate(0.0, 0.0),
            snapped_destination=Coordinate(0.0, 0.002),
        )

        response = calculate_shortest_route(self.shortest_request)
        self.assertEqual(response.distance_meters, 200.0)
        self.assertEqual(len(response.coordinates), 2)

    def test_traffic_info_endpoint_returns_bounds_and_presets(self) -> None:
        from backend.routing.routes import get_traffic_info

        info = get_traffic_info()
        self.assertIn("min_lat", info.bounds)
        self.assertIn("max_lat", info.bounds)
        self.assertIn("min_lon", info.bounds)
        self.assertIn("max_lon", info.bounds)
        self.assertIn("latitude", info.center)
        self.assertIn("longitude", info.center)
        if info.available:
            self.assertGreater(len(info.sample_presets), 0)
            self.assertIn("Cost(e)", info.cost_formula)

    def test_geocoding_search_finds_local_landmarks(self) -> None:
        from backend.api.routes import search_places

        response = search_places(q="toll gate")
        self.assertEqual(response.status, "success")
        self.assertGreater(len(response.results), 0)
        first = response.results[0]
        self.assertEqual(first.source, "verified_landmark")
        self.assertAlmostEqual(first.latitude, 12.847657, places=4)
        self.assertAlmostEqual(first.longitude, 77.671263, places=4)


if __name__ == "__main__":
    unittest.main()

