"""Unit and integration tests for Routing REST API endpoints."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.routing.contracts import (
    Coordinate,
    RouteCandidate,
    RouteComparison,
    ShortestRoute,
)
from backend.routing.shortest_path import RouteNotFoundError


class TestRoutingAPI(unittest.TestCase):
    """Verifies FastAPI routing endpoints, prefixes, method matching, and error handling."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.origin = Coordinate(12.847657, 77.671263)
        self.destination = Coordinate(12.8398, 77.6775)
        self.sample_candidate = RouteCandidate(
            coordinates=(self.origin, self.destination),
            distance_meters=1200.0,
            cost=1250.0,
            edges_count=12,
            matched_traffic_edges_count=3,
            traffic_coverage_percent=25.0,
            average_congestion=0.15,
            traffic_source="simulated",
            vision_matched_edges_count=0,
            simulated_matched_edges_count=3,
            tomtom_matched_edges_count=0,
            unmatched_edges_count=9,
            road_closures_count=0,
            steps_count=0,
            unpaved_edges_count=0,
            obstacles_count=0,
            traffic_cost_penalty=50.0,
            accessibility_cost_penalty=0.0,
            accessibility_constraints_applied=(),
            network_is_stale=False,
        )

    def test_post_routes_compare_registered_and_succeeds(self) -> None:
        """Verify POST /api/routes/compare reaches handler and returns 200 with schema."""
        mock_comparison = RouteComparison(
            shortest_distance=self.sample_candidate,
            traffic_accessibility_aware=self.sample_candidate,
            snapped_origin=self.origin,
            snapped_destination=self.destination,
            selected_timestamp="2026-01-01 08:30:00",
            time_index=17,
            cost_formula="Cost(e) = Length(e) + ...",
            status="identical_routes",
            disclaimer="Disclaimer text",
            primary_traffic_source="simulated",
            vision_observation_active=False,
            accessibility_mode="standard",
            explanation="Both routes are identical under free flow conditions.",
            network_is_stale=False,
        )

        with patch("backend.routing.routes.comparison_service.compare", return_value=mock_comparison):
            response = self.client.post(
                "/api/routes/compare",
                json={
                    "origin": {"latitude": 12.847657, "longitude": 77.671263},
                    "destination": {"latitude": 12.8398, "longitude": 77.6775},
                    "traffic_weight": 2.0,
                    "accessibility_mode": "standard",
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("shortest_distance", data)
        self.assertIn("traffic_accessibility_aware", data)
        self.assertEqual(data["status"], "identical_routes")
        self.assertEqual(data["shortest_distance"]["distance_meters"], 1200.0)

    def test_post_routes_compare_trailing_slash_variant(self) -> None:
        """Verify POST /api/routes/compare/ also matches and succeeds without redirection failure."""
        mock_comparison = RouteComparison(
            shortest_distance=self.sample_candidate,
            traffic_accessibility_aware=self.sample_candidate,
            snapped_origin=self.origin,
            snapped_destination=self.destination,
            selected_timestamp="2026-01-01 08:30:00",
            time_index=17,
            cost_formula="Cost(e) = ...",
            status="identical_routes",
            disclaimer="Disclaimer",
            primary_traffic_source="simulated",
            vision_observation_active=False,
            accessibility_mode="standard",
            explanation="Explanation",
            network_is_stale=False,
        )

        with patch("backend.routing.routes.comparison_service.compare", return_value=mock_comparison):
            response = self.client.post(
                "/api/routes/compare/",
                json={
                    "origin": {"latitude": 12.847657, "longitude": 77.671263},
                    "destination": {"latitude": 12.8398, "longitude": 77.6775},
                },
            )

        self.assertEqual(response.status_code, 200)

    def test_post_routes_compare_invalid_body_returns_422(self) -> None:
        """Verify POST /api/routes/compare with invalid body returns 422, not 404."""
        response = self.client.post("/api/routes/compare", json={})
        self.assertEqual(response.status_code, 422)

    def test_post_routes_compare_not_found_returns_404_with_detail(self) -> None:
        """Verify RouteNotFoundError returns HTTP 404 with meaningful detail."""
        with patch(
            "backend.routing.routes.comparison_service.compare",
            side_effect=RouteNotFoundError("No pedestrian path connects the selected locations."),
        ):
            response = self.client.post(
                "/api/routes/compare",
                json={
                    "origin": {"latitude": 12.847657, "longitude": 77.671263},
                    "destination": {"latitude": 12.8398, "longitude": 77.6775},
                },
            )

        self.assertEqual(response.status_code, 404)
        self.assertIn("No pedestrian path connects", response.json()["detail"])

    def test_post_routes_shortest_registered_and_succeeds(self) -> None:
        """Verify POST /api/routes/shortest reaches handler and returns 200."""
        mock_shortest = ShortestRoute(
            coordinates=(self.origin, self.destination),
            distance_meters=1200.0,
            snapped_origin=self.origin,
            snapped_destination=self.destination,
        )

        with patch("backend.routing.routes.shortest_path_service.calculate", return_value=mock_shortest):
            response = self.client.post(
                "/api/routes/shortest",
                json={
                    "origin": {"latitude": 12.847657, "longitude": 77.671263},
                    "destination": {"latitude": 12.8398, "longitude": 77.6775},
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["distance_meters"], 1200.0)

    def test_get_routes_traffic_info_registered(self) -> None:
        """Verify GET /api/routes/traffic-info returns dataset and bounds information."""
        response = self.client.get("/api/routes/traffic-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("available", data)
        self.assertIn("area_name", data)

    def test_get_routes_observation_status_registered(self) -> None:
        """Verify GET /api/routes/observation-status returns observation counters."""
        response = self.client.get("/api/routes/observation-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vision_registered_count", data)
        self.assertIn("accessibility_registered_count", data)

    def test_get_geocoding_search_registered(self) -> None:
        """Verify GET /api/geocoding/search returns geocoding matches."""
        response = self.client.get("/api/geocoding/search?q=silk%20board")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["results"]), 0)

    def test_router_prefixes_integrity(self) -> None:
        """Verify all routes in routing router start with /api/routes."""
        from backend.routing.routes import router as routing_router

        self.assertEqual(routing_router.prefix, "/api/routes")
        for route in routing_router.routes:
            self.assertTrue(
                route.path.startswith("/api/routes"),
                f"Route {route.path} does not start with /api/routes prefix",
            )
