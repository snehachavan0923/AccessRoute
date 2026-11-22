import unittest
from unittest.mock import Mock, patch

import networkx as nx
from pydantic import ValidationError
from shapely.geometry import LineString

from backend.pedestrian_network import (
    NetworkBounds,
    NetworkDownloadError,
    OSMnxPedestrianNetworkProvider,
    is_inaccessible_elevated_motor_edge,
    prune_inaccessible_elevated_edges,
)
from backend.routing.contracts import Coordinate, ShortestRoute
from backend.routing.routes import (
    ShortestRouteRequest,
    calculate_shortest_route,
)
from backend.routing.shortest_path import (
    NetworkAreaSettings,
    NetworkAreaTooLargeError,
    RouteNetworkUnavailableError,
    RouteNotFoundError,
    ShortestPathService,
    calculate_network_bounds,
    is_node_pedestrian_preferred,
    nearest_network_node,
)


def make_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.001, y=0.0)
    graph.add_node(3, x=0.002, y=0.0)
    graph.add_node(4, x=0.001, y=0.001)
    graph.add_edge(1, 2, length=100.0)
    graph.add_edge(2, 3, length=100.0)
    graph.add_edge(1, 4, length=80.0)
    graph.add_edge(4, 3, length=180.0)
    graph.add_edge(1, 3, length=350.0)
    return graph


def make_graph_with_curved_edge() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.002, y=0.0)
    graph.add_edge(
        1,
        2,
        length=200.0,
        geometry=LineString([(0, 0), (0.001, 0.001), (0.002, 0)]),
    )
    return graph


class ShortestPathServiceTests(unittest.TestCase):
    def test_returns_shortest_distance_path_and_snapped_coordinates(self) -> None:
        provider = Mock()
        provider.load.return_value = make_graph()
        service = ShortestPathService(provider)

        route = service.calculate(
            Coordinate(latitude=0, longitude=0),
            Coordinate(latitude=0, longitude=0.002),
        )

        self.assertEqual(route.distance_meters, 200.0)
        self.assertEqual(
            route.coordinates,
            (
                Coordinate(latitude=0.0, longitude=0.0),
                Coordinate(latitude=0.0, longitude=0.001),
                Coordinate(latitude=0.0, longitude=0.002),
            ),
        )
        self.assertEqual(route.snapped_origin, route.coordinates[0])
        self.assertEqual(route.snapped_destination, route.coordinates[-1])
        provider.load.assert_called_once()

    def test_preserves_osm_edge_geometry(self) -> None:
        provider = Mock()
        provider.load.return_value = make_graph_with_curved_edge()
        route = ShortestPathService(provider).calculate(
            Coordinate(latitude=0, longitude=0),
            Coordinate(latitude=0, longitude=0.002),
        )

        self.assertEqual(
            route.coordinates,
            (
                Coordinate(latitude=0.0, longitude=0.0),
                Coordinate(latitude=0.001, longitude=0.001),
                Coordinate(latitude=0.0, longitude=0.002),
            ),
        )

    def test_bounds_include_buffer_and_limit_download_area(self) -> None:
        bounds = calculate_network_bounds(
            Coordinate(latitude=0, longitude=0),
            Coordinate(latitude=0.01, longitude=0.01),
            NetworkAreaSettings(buffer_meters=500, max_diagonal_meters=20_000),
        )

        self.assertLess(bounds.south, 0)
        self.assertGreater(bounds.north, 0.01)
        self.assertLess(bounds.west, 0)
        self.assertGreater(bounds.east, 0.01)

    def test_rejects_excessive_area_before_downloading(self) -> None:
        provider = Mock()
        service = ShortestPathService(
            provider,
            NetworkAreaSettings(buffer_meters=1_000, max_diagonal_meters=2_000),
        )

        with self.assertRaises(NetworkAreaTooLargeError):
            service.calculate(
                Coordinate(latitude=0, longitude=0),
                Coordinate(latitude=0, longitude=0.1),
            )

        provider.load.assert_not_called()

    def test_reports_missing_path(self) -> None:
        graph = nx.MultiDiGraph(crs="epsg:4326")
        graph.add_node(1, x=0.0, y=0.0)
        graph.add_node(2, x=0.01, y=0.01)
        provider = Mock()
        provider.load.return_value = graph

        with self.assertRaises(RouteNotFoundError):
            ShortestPathService(provider).calculate(
                Coordinate(latitude=0, longitude=0),
                Coordinate(latitude=0.01, longitude=0.01),
            )

    def test_reports_download_failure(self) -> None:
        provider = Mock()
        provider.load.side_effect = NetworkDownloadError("OpenStreetMap unavailable")

        with self.assertRaisesRegex(
            RouteNetworkUnavailableError, "OpenStreetMap unavailable"
        ):
            ShortestPathService(provider).calculate(
                Coordinate(latitude=0, longitude=0),
                Coordinate(latitude=0, longitude=0.001),
            )

    def test_rejects_invalid_coordinate_request(self) -> None:
        with self.assertRaises(ValidationError):
            ShortestRouteRequest.model_validate(
                {
                    "origin": {"latitude": 91, "longitude": 0},
                    "destination": {"latitude": 0, "longitude": 0},
                }
            )


class OSMnxProviderTests(unittest.TestCase):
    @patch("backend.pedestrian_network.provider.ox.graph_from_bbox")
    def test_requests_walkable_network_for_bounded_area(
        self,
        graph_from_bbox: Mock,
    ) -> None:
        graph_from_bbox.return_value = make_graph()
        provider = OSMnxPedestrianNetworkProvider()
        bounds = NetworkBounds(north=2, south=1, east=4, west=3)

        provider.load(bounds)

        graph_from_bbox.assert_called_once_with(
            (3, 1, 4, 2),
            network_type="walk",
        )

    @patch("backend.pedestrian_network.provider.ox.graph_from_bbox")
    def test_translates_empty_osm_response(
        self,
        graph_from_bbox: Mock,
    ) -> None:
        from osmnx._errors import InsufficientResponseError

        graph_from_bbox.side_effect = InsufficientResponseError("No data")

        with self.assertRaises(NetworkDownloadError):
            OSMnxPedestrianNetworkProvider().load(
                NetworkBounds(north=2, south=1, east=4, west=3)
            )


class ShortestRouteEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = ShortestRouteRequest.model_validate(
            {
                "origin": {"latitude": 0, "longitude": 0},
                "destination": {"latitude": 0, "longitude": 0.002},
            }
        )

    @patch("backend.routing.routes.shortest_path_service")
    def test_returns_route_geometry_distance_and_snapped_endpoints(
        self,
        service: Mock,
    ) -> None:
        service.calculate.return_value = ShortestRoute(
            coordinates=(
                Coordinate(latitude=0, longitude=0),
                Coordinate(latitude=0, longitude=0.002),
            ),
            distance_meters=200,
            snapped_origin=Coordinate(latitude=0, longitude=0),
            snapped_destination=Coordinate(latitude=0, longitude=0.002),
        )

        response = calculate_shortest_route(self.request)

        self.assertEqual(response.distance_meters, 200)
        self.assertEqual(len(response.coordinates), 2)
        self.assertEqual(response.snapped_origin.latitude, 0)
        self.assertEqual(response.snapped_destination.longitude, 0.002)

    @patch("backend.routing.routes.shortest_path_service")
    def test_translates_unavailable_network_to_service_unavailable(
        self,
        service: Mock,
    ) -> None:
        from fastapi import HTTPException

        service.calculate.side_effect = RouteNetworkUnavailableError(
            "OpenStreetMap unavailable"
        )

        with self.assertRaises(HTTPException) as raised:
            calculate_shortest_route(self.request)

        self.assertEqual(raised.exception.status_code, 503)

    @patch("backend.routing.routes.shortest_path_service")
    def test_translates_missing_path_to_not_found(self, service: Mock) -> None:
        from fastapi import HTTPException

        service.calculate.side_effect = RouteNotFoundError("No path")

        with self.assertRaises(HTTPException) as raised:
            calculate_shortest_route(self.request)

        self.assertEqual(raised.exception.status_code, 404)

    def test_ground_level_snapping_preferred_over_elevated_flyover(self) -> None:
        """Ground-level pedestrian node must be preferred over a closer elevated motor flyover."""
        graph = nx.MultiDiGraph(crs="epsg:4326")
        # Elevated flyover node: only 5 meters away in 2D
        graph.add_node(1, x=77.62330, y=12.91764)
        graph.add_node(2, x=77.62340, y=12.91764)
        graph.add_edge(1, 2, 0, length=20.0, highway="primary_link", bridge="yes", name="Double Decker Flyover")

        # Ground-level node: 20 meters away in 2D
        graph.add_node(3, x=77.62330, y=12.91780)
        graph.add_node(4, x=77.62340, y=12.91780)
        graph.add_edge(3, 4, 0, length=20.0, highway="residential", name="Service Road")

        query_coord = Coordinate(latitude=12.91760, longitude=77.62330)

        # Node 1 is closer in 2D Euclidean distance
        # But node 3 is a ground-level pedestrian road and must be selected
        selected = nearest_network_node(graph, query_coord)
        self.assertEqual(selected, 3)

    def test_preservation_of_pedestrian_accessible_bridges(self) -> None:
        """Explicitly pedestrian-accessible bridges (footbridges, sidewalks) must remain preferred."""
        graph = nx.MultiDiGraph(crs="epsg:4326")
        # Footbridge node
        graph.add_node(10, x=77.6000, y=12.9700)
        graph.add_node(11, x=77.6001, y=12.9700)
        graph.add_edge(10, 11, 0, length=15.0, highway="footway", bridge="yes")

        # Node on bridge with sidewalk
        graph.add_node(20, x=77.6100, y=12.9700)
        graph.add_node(21, x=77.6101, y=12.9700)
        graph.add_edge(20, 21, 0, length=15.0, highway="primary", bridge="yes", sidewalk="both")

        # Node on bridge with foot=yes
        graph.add_node(30, x=77.6200, y=12.9700)
        graph.add_node(31, x=77.6201, y=12.9700)
        graph.add_edge(30, 31, 0, length=15.0, highway="primary", bridge="yes", foot="yes")

        self.assertTrue(is_node_pedestrian_preferred(graph, 10))
        self.assertTrue(is_node_pedestrian_preferred(graph, 20))
        self.assertTrue(is_node_pedestrian_preferred(graph, 30))

        # Snapping to footbridge coordinate selects footbridge
        selected = nearest_network_node(graph, Coordinate(latitude=12.9700, longitude=77.6000))
        self.assertEqual(selected, 10)

    def test_prune_inaccessible_elevated_edges_behavior(self) -> None:
        """prune_inaccessible_elevated_edges removes inaccessible highway bridges but preserves pedestrian bridges."""
        graph = nx.MultiDiGraph(crs="epsg:4326")
        # Inaccessible flyovers
        graph.add_node(1, x=0.0, y=0.0)
        graph.add_node(2, x=0.001, y=0.0)
        graph.add_edge(1, 2, 0, highway="primary_link", bridge="yes")

        graph.add_node(3, x=0.002, y=0.0)
        graph.add_node(4, x=0.003, y=0.0)
        graph.add_edge(3, 4, 0, highway="trunk", layer="1")

        # Accessible bridges
        graph.add_node(5, x=0.004, y=0.0)
        graph.add_node(6, x=0.005, y=0.0)
        graph.add_edge(5, 6, 0, highway="footway", bridge="yes")

        graph.add_node(7, x=0.006, y=0.0)
        graph.add_node(8, x=0.007, y=0.0)
        graph.add_edge(7, 8, 0, highway="primary", bridge="yes", foot="yes")

        graph.add_node(9, x=0.008, y=0.0)
        graph.add_node(10, x=0.009, y=0.0)
        graph.add_edge(9, 10, 0, highway="residential", bridge="yes")

        pruned = prune_inaccessible_elevated_edges(graph)

        self.assertFalse(pruned.has_edge(1, 2))
        self.assertFalse(pruned.has_edge(3, 4))
        self.assertTrue(pruned.has_edge(5, 6))
        self.assertTrue(pruned.has_edge(7, 8))
        self.assertTrue(pruned.has_edge(9, 10))

    def test_silk_board_routing_avoids_inaccessible_flyover(self) -> None:
        """Calculates route on a graph with an elevated flyover and ground road, verifying flyover avoidance."""
        graph = nx.MultiDiGraph(crs="epsg:4326")
        # Origin and Destination on ground
        # Elevated flyover node very close to origin (2m away)
        graph.add_node(100, x=77.62331, y=12.91761)  # flyover start
        graph.add_node(101, x=77.60000, y=12.91700)  # flyover 3km away
        graph.add_node(102, x=77.62750, y=12.92050)  # destination
        # Flyover connects 100 -> 101 (3000m) -> 102 (3000m) = 6000m detour
        graph.add_edge(100, 101, 0, length=3000.0, highway="primary_link", bridge="yes", name="Double Decker Flyover")
        graph.add_edge(101, 102, 0, length=3000.0, highway="primary", bridge="yes", name="Flyover Ramp")

        # Ground road connects near origin to destination directly (500m)
        graph.add_node(200, x=77.62335, y=12.91770)  # ground start near orig
        graph.add_edge(200, 102, 0, length=500.0, highway="residential", name="Hosur Service Road")

        provider = Mock()
        provider.load.return_value = graph

        service = ShortestPathService(network_provider=provider)
        route = service.calculate(
            Coordinate(latitude=12.91760, longitude=77.62330),
            Coordinate(latitude=12.92050, longitude=77.62750),
        )

        # Route must snap to ground node 200, not flyover node 100
        self.assertEqual(route.snapped_origin.latitude, 12.91770)
        self.assertAlmostEqual(route.distance_meters, 500.0)


if __name__ == "__main__":
    unittest.main()
