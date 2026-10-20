import unittest
from unittest.mock import Mock, patch

import networkx as nx
from pydantic import ValidationError
from shapely.geometry import LineString

from backend.pedestrian_network import (
    NetworkBounds,
    NetworkDownloadError,
    OSMnxPedestrianNetworkProvider,
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


if __name__ == "__main__":
    unittest.main()
