"""Unit and integration tests for TomTom Traffic Flow API adapter and routing integration.

All HTTP requests are strictly mocked to ensure tests never consume real API quota.
Verifies normalization, error handling, rate limiting, caching, road matching,
fallback, and priority cascades.
"""

from __future__ import annotations

import math
import os
import unittest
from unittest.mock import MagicMock, patch

import networkx as nx
import requests

from backend.pedestrian_network import NetworkBounds
from backend.routing.contracts import Coordinate
from backend.routing.shortest_path import ShortestPathService
from backend.routing.traffic_aware_path import (
    TrafficAwarePathService,
    TrafficRouteComparisonService,
)
from backend.traffic.loader import TrafficDataset, TrafficEdgeKey
from backend.traffic.tomtom_adapter import (
    TomTomMatchResult,
    TomTomObservation,
    TomTomTrafficAdapter,
    calculate_bearing_degrees,
    calculate_normalized_congestion,
    distance_point_to_polyline_meters,
    haversine_distance_meters,
    is_suitable_motor_road,
)
from backend.traffic.vision_adapter import VisionTrafficAdapter
from backend.vision.schemas import CameraMetadata, ObservationTimeEnvelope, VisionEdgeObservation


def make_sample_tomtom_payload(
    current_speed: float = 25.0,
    free_flow_speed: float = 50.0,
    confidence: float = 0.95,
    road_closure: bool = False,
    frc: str = "FRC1",
    coordinates: list[dict[str, float]] | None = None,
) -> dict:
    """Helper to generate a mock TomTom flowSegmentData response."""
    if coordinates is None:
        coordinates = [
            {"latitude": 12.9175, "longitude": 77.6230},
            {"latitude": 12.9177, "longitude": 77.6235},
        ]
    return {
        "flowSegmentData": {
            "frc": frc,
            "currentSpeed": current_speed,
            "freeFlowSpeed": free_flow_speed,
            "currentTravelTime": 120,
            "freeFlowTravelTime": 60,
            "confidence": confidence,
            "roadClosure": road_closure,
            "coordinates": {"coordinate": coordinates},
        }
    }


class TestTomTomCongestionNormalization(unittest.TestCase):
    """Tests 2, 3, 4: Speed and congestion normalization math."""

    def test_standard_congestion_calculation(self):
        # Current = 25, Free-flow = 50 -> 1 - 25/50 = 0.50
        c = calculate_normalized_congestion(25.0, 50.0)
        self.assertAlmostEqual(c, 0.50, places=4)

    def test_free_flow_speed_equal_to_current_speed(self):
        # Test 3: Free-flow == current -> congestion = 0.0
        c = calculate_normalized_congestion(60.0, 60.0)
        self.assertAlmostEqual(c, 0.0, places=4)

    def test_current_speed_faster_than_free_flow_speed(self):
        # Current > Free-flow -> clamp to 0.0
        c = calculate_normalized_congestion(70.0, 50.0)
        self.assertAlmostEqual(c, 0.0, places=4)

    def test_zero_free_flow_speed(self):
        # Test 4: Free-flow <= 0 -> 0.0 without division by zero
        c = calculate_normalized_congestion(10.0, 0.0)
        self.assertEqual(c, 0.0)
        c_neg = calculate_normalized_congestion(10.0, -5.0)
        self.assertEqual(c_neg, 0.0)

    def test_zero_current_speed_gridlock(self):
        # Current = 0 -> maximum congestion 1.0
        c = calculate_normalized_congestion(0.0, 50.0)
        self.assertAlmostEqual(c, 1.0, places=4)

    def test_non_finite_inputs(self):
        self.assertEqual(calculate_normalized_congestion(float("nan"), 50.0), 0.0)
        self.assertEqual(calculate_normalized_congestion(25.0, float("nan")), 0.0)
        self.assertEqual(calculate_normalized_congestion(float("inf"), 50.0), 0.0)
        self.assertEqual(calculate_normalized_congestion(25.0, float("inf")), 0.0)


class TestTomTomAdapterParsingAndErrors(unittest.TestCase):
    """Tests 1, 5, 6, 7, 8, 9, 10, 13, 14: Parsing, error handling, confidence, closures."""

    def setUp(self):
        self.adapter = TomTomTrafficAdapter(
            api_key="mock_test_key",
            enabled=True,
            ttl_seconds=300,
            max_daily_requests=50,
            min_confidence=0.70,
            timeout_seconds=2.0,
        )

    @patch("requests.get")
    def test_valid_tomtom_response(self, mock_get):
        # Test 1: Valid response returns matching observation
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload(
            current_speed=30.0, free_flow_speed=60.0, confidence=0.90
        )
        mock_get.return_value = mock_resp

        edge_data = {"highway": "primary", "oneway": False}
        src = (12.9175, 77.6230)
        tgt = (12.9177, 77.6235)

        res = self.adapter.get_edge_observation(edge_data, src, tgt)
        self.assertTrue(res.matched)
        self.assertIsNotNone(res.observation)
        self.assertAlmostEqual(res.observation.congestion_index, 0.50, places=2)
        self.assertEqual(res.observation.current_speed_kmh, 30.0)
        self.assertEqual(res.observation.free_flow_speed_kmh, 60.0)
        self.assertFalse(res.observation.road_closure)

    @patch("requests.get")
    def test_missing_fields_in_response(self, mock_get):
        # Test 5: Missing fields handled safely
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"flowSegmentData": {"frc": "FRC1"}}  # Missing speeds
        mock_get.return_value = mock_resp

        edge_data = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge_data, (12.91, 77.62), (12.92, 77.63))
        self.assertFalse(res.matched)
        self.assertIn("missing", res.rejection_reason or "")

    @patch("requests.get")
    def test_invalid_json_response(self, mock_get):
        # Test 6: Invalid response / malformed JSON
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_resp

        edge_data = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge_data, (12.91, 77.62), (12.92, 77.63))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "malformed_json_response")

    @patch("requests.get")
    def test_low_confidence_rejection(self, mock_get):
        # Test 7: Confidence below min_confidence (0.70) is rejected
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload(confidence=0.45)
        mock_get.return_value = mock_resp

        edge_data = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge_data, (12.9175, 77.6230), (12.9177, 77.6235))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "low_confidence")

    @patch("requests.get")
    def test_road_closure_representation(self, mock_get):
        # Test 8: Road closure is preserved as a distinct condition
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload(road_closure=True)
        mock_get.return_value = mock_resp

        edge_data = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge_data, (12.9175, 77.6230), (12.9177, 77.6235))
        self.assertTrue(res.matched)
        self.assertIsNotNone(res.observation)
        self.assertTrue(res.observation.road_closure)

    @patch("requests.get")
    def test_timeout_handling(self, mock_get):
        # Test 9: Timeout handled gracefully
        mock_get.side_effect = requests.Timeout("Connection timed out")

        edge_data = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge_data, (12.9175, 77.6230), (12.9177, 77.6235))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "timeout")

    @patch("requests.get")
    def test_http_429_rate_limiting(self, mock_get):
        # Test 10: HTTP 429 Too Many Requests
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        edge_data = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge_data, (12.9175, 77.6230), (12.9177, 77.6235))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "http_429_rate_limited")

    def test_missing_api_key(self):
        # Test 13: Missing API key disables provider
        adapter = TomTomTrafficAdapter(api_key="", enabled=True)
        self.assertFalse(adapter.is_available)
        edge_data = {"highway": "primary"}
        res = adapter.get_edge_observation(edge_data, (12.91, 77.62), (12.92, 77.63))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "provider_disabled_or_no_key")

    def test_disabled_provider(self):
        # Test 14: Provider explicitly disabled
        adapter = TomTomTrafficAdapter(api_key="valid_key", enabled=False)
        self.assertFalse(adapter.is_available)
        edge_data = {"highway": "primary"}
        res = adapter.get_edge_observation(edge_data, (12.91, 77.62), (12.92, 77.63))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "provider_disabled_or_no_key")


class TestTomTomCacheAndRateLimiting(unittest.TestCase):
    """Tests 11, 12: Caching, TTL expiry, and daily request limit."""

    @patch("requests.get")
    def test_cache_hit_and_expiry(self, mock_get):
        # Test 11: First call queries API, second uses cache; expired call re-queries
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload()
        mock_get.return_value = mock_resp

        # TTL = 2 seconds
        adapter = TomTomTrafficAdapter(api_key="key", enabled=True, ttl_seconds=2)
        edge = {"highway": "primary"}
        src = (12.9175, 77.6230)
        tgt = (12.9177, 77.6235)

        # 1st call -> API
        r1 = adapter.get_edge_observation(edge, src, tgt)
        self.assertTrue(r1.matched)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(adapter.cache_hits, 0)

        # 2nd call -> Cache hit
        r2 = adapter.get_edge_observation(edge, src, tgt)
        self.assertTrue(r2.matched)
        self.assertEqual(mock_get.call_count, 1)  # No extra HTTP call
        self.assertEqual(adapter.cache_hits, 1)

        # Simulate TTL expiry by manually altering cached timestamp
        cache_key = next(iter(adapter._cache))
        obs, _ = adapter._cache[cache_key]
        adapter._cache[cache_key] = (obs, 0.0)  # ancient timestamp

        # 3rd call -> Cache miss due to TTL expiry -> new API call
        r3 = adapter.get_edge_observation(edge, src, tgt)
        self.assertTrue(r3.matched)
        self.assertEqual(mock_get.call_count, 2)

    @patch("requests.get")
    def test_daily_request_limit(self, mock_get):
        # Test 12: Internal daily limit strictly enforced
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload()
        mock_get.return_value = mock_resp

        # Limit to 2 requests
        adapter = TomTomTrafficAdapter(
            api_key="key", enabled=True, max_daily_requests=2, ttl_seconds=0
        )
        edge = {"highway": "primary"}

        # Request 1: Allowed
        adapter.get_edge_observation(edge, (12.01, 77.01), (12.02, 77.02))
        self.assertEqual(adapter.daily_requests_count, 1)

        # Request 2: Allowed
        adapter.get_edge_observation(edge, (12.11, 77.11), (12.12, 77.12))
        self.assertEqual(adapter.daily_requests_count, 2)

        # Request 3: Budget reached -> no HTTP call
        r3 = adapter.get_edge_observation(edge, (12.21, 77.21), (12.22, 77.22))
        self.assertFalse(r3.matched)
        self.assertEqual(r3.rejection_reason, "daily_budget_reached")
        self.assertEqual(mock_get.call_count, 2)


class TestSafeOSMRoadMatching(unittest.TestCase):
    """Tests 15, 16, 17: Geometric matching, bearing comparison, pedestrian exclusion."""

    def setUp(self):
        self.adapter = TomTomTrafficAdapter(
            api_key="test_key",
            enabled=True,
            match_max_distance_meters=30.0,
            match_max_bearing_diff_degrees=45.0,
        )

    def test_pure_pedestrian_path_exclusion(self):
        # Test 17: Footways, steps, cycleways excluded from querying
        footway = {"highway": "footway"}
        steps = {"highway": "steps"}
        pedestrian = {"highway": "pedestrian"}
        cycleway = {"highway": "cycleway"}

        self.assertFalse(is_suitable_motor_road(footway))
        self.assertFalse(is_suitable_motor_road(steps))
        self.assertFalse(is_suitable_motor_road(pedestrian))
        self.assertFalse(is_suitable_motor_road(cycleway))

        # Check adapter returns rejection without HTTP call
        res = self.adapter.get_edge_observation(footway, (12.91, 77.62), (12.92, 77.63))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "non_motor_road_class")

    def test_motor_vehicle_road_inclusion(self):
        self.assertTrue(is_suitable_motor_road({"highway": "primary"}))
        self.assertTrue(is_suitable_motor_road({"highway": "residential"}))
        self.assertTrue(is_suitable_motor_road({"highway": "secondary"}))
        self.assertTrue(is_suitable_motor_road({"highway": "tertiary"}))

    @patch("requests.get")
    def test_ambiguous_geometry_distance_rejection(self, mock_get):
        # Test 15: Displaced segment > 30m away is rejected
        # Requested point is at (12.9175, 77.6230), but returned coordinates are ~200m away
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload(
            coordinates=[
                {"latitude": 12.9200, "longitude": 77.6250},
                {"latitude": 12.9205, "longitude": 77.6255},
            ]
        )
        mock_get.return_value = mock_resp

        edge = {"highway": "primary"}
        res = self.adapter.get_edge_observation(edge, (12.9175, 77.6230), (12.9177, 77.6232))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "distance_exceeded")
        self.assertGreater(res.distance_meters, 30.0)

    @patch("requests.get")
    def test_direction_mismatch_oneway_rejection(self, mock_get):
        # Test 16: OSM edge is one-way North, TomTom polyline is South -> bearing mismatch > 45 deg
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # TomTom segment goes South (lat decreasing)
        mock_resp.json.return_value = make_sample_tomtom_payload(
            coordinates=[
                {"latitude": 12.9180, "longitude": 77.6230},
                {"latitude": 12.9170, "longitude": 77.6230},
            ]
        )
        mock_get.return_value = mock_resp

        # OSM edge goes North (lat increasing)
        edge = {"highway": "primary", "oneway": True}
        res = self.adapter.get_edge_observation(edge, (12.9170, 77.6230), (12.9180, 77.6230))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "direction_mismatch_oneway")

    @patch("requests.get")
    def test_same_direction_alignment_matches(self, mock_get):
        # Unit test: Same-direction alignment matches (East -> East)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_sample_tomtom_payload(
            coordinates=[
                {"latitude": 12.9170, "longitude": 77.6230},
                {"latitude": 12.9170, "longitude": 77.6240},
            ]
        )
        mock_get.return_value = mock_resp

        edge = {"highway": "primary", "oneway": False}
        res = self.adapter.get_edge_observation(edge, (12.9170, 77.6230), (12.9170, 77.6240))
        self.assertTrue(res.matched)
        self.assertIsNotNone(res.observation)
        self.assertLessEqual(res.bearing_diff_degrees, 10.0)

    @patch("requests.get")
    def test_reverse_direction_alignment_twoway_matches(self, mock_get):
        # Unit test: Reverse-direction alignment on two-way road matches via axis_diff
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # TomTom polyline digitized West (lon decreasing)
        mock_resp.json.return_value = make_sample_tomtom_payload(
            coordinates=[
                {"latitude": 12.9170, "longitude": 77.6240},
                {"latitude": 12.9170, "longitude": 77.6230},
            ]
        )
        mock_get.return_value = mock_resp

        # OSM edge goes East (lon increasing), two-way road
        edge = {"highway": "primary", "oneway": False}
        res = self.adapter.get_edge_observation(edge, (12.9170, 77.6230), (12.9170, 77.6240))
        self.assertTrue(res.matched, f"Expected two-way reverse alignment to match, got rejection: {res.rejection_reason}")
        self.assertIsNotNone(res.observation)
        self.assertLessEqual(res.bearing_diff_degrees, 10.0)

    @patch("requests.get")
    def test_reverse_direction_oneway_rejected(self, mock_get):
        # Unit test: Reverse-direction on one-way road must be strictly rejected
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # TomTom polyline goes West
        mock_resp.json.return_value = make_sample_tomtom_payload(
            coordinates=[
                {"latitude": 12.9170, "longitude": 77.6240},
                {"latitude": 12.9170, "longitude": 77.6230},
            ]
        )
        mock_get.return_value = mock_resp

        # OSM edge goes East, one-way road
        edge = {"highway": "primary", "oneway": True}
        res = self.adapter.get_edge_observation(edge, (12.9170, 77.6230), (12.9170, 77.6240))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "direction_mismatch_oneway")

    @patch("requests.get")
    def test_perpendicular_road_rejected(self, mock_get):
        # Unit test: Perpendicular road (North vs East) must be rejected on both one-way and two-way roads
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # TomTom polyline goes East
        mock_resp.json.return_value = make_sample_tomtom_payload(
            coordinates=[
                {"latitude": 12.9170, "longitude": 77.6230},
                {"latitude": 12.9170, "longitude": 77.6240},
            ]
        )
        mock_get.return_value = mock_resp

        # OSM edge goes North, crossing the East-West TomTom road at exactly the same coordinates (dist = 0m)
        edge = {"highway": "primary", "oneway": False}
        res = self.adapter.get_edge_observation(edge, (12.9169, 77.6235), (12.9171, 77.6235))
        self.assertFalse(res.matched)
        self.assertEqual(res.rejection_reason, "bearing_mismatch")


class TestRoutingIntegrationWithTomTom(unittest.TestCase):
    """Tests 18, 19, 20, 21: Full integration tests on graph routing."""

    def setUp(self):
        # Create a simple synthetic test graph:
        # Node 1 -> Node 2 -> Node 3 (corridor A, length 100m each)
        # Node 1 -> Node 4 -> Node 3 (corridor B, length 120m each)
        self.graph = nx.MultiDiGraph()
        self.graph.add_node(1, x=77.6200, y=12.9170)
        self.graph.add_node(2, x=77.6210, y=12.9170)
        self.graph.add_node(3, x=77.6220, y=12.9170)
        self.graph.add_node(4, x=77.6210, y=12.9180)

        # Corridor A: motor road (primary)
        self.graph.add_edge(1, 2, key=0, length=100.0, highway="primary", oneway=False)
        self.graph.add_edge(2, 3, key=0, length=100.0, highway="primary", oneway=False)

        # Corridor B: detour road (secondary)
        self.graph.add_edge(1, 4, key=0, length=120.0, highway="secondary", oneway=False)
        self.graph.add_edge(4, 3, key=0, length=120.0, highway="secondary", oneway=False)

    def test_shortest_route_remains_unchanged(self):
        # Test 20: Shortest path service remains pure physical distance
        provider = MagicMock()
        provider.load.return_value = self.graph.copy()
        service = ShortestPathService(network_provider=provider)

        route = service.calculate(
            Coordinate(latitude=12.9170, longitude=77.6200),
            Coordinate(latitude=12.9170, longitude=77.6220),
        )
        # Corridor A is 200m, Corridor B is 240m. Shortest must choose Corridor A (200m).
        self.assertAlmostEqual(route.distance_meters, 200.0, places=1)

    def test_tomtom_traffic_causes_optimal_detour(self):
        # Provide heavy TomTom congestion on corridor A (edges 1->2 and 2->3)
        mock_tomtom = MagicMock(spec=TomTomTrafficAdapter)
        mock_tomtom.is_available = True

        def mock_get_edge_obs(edge_data, src, tgt):
            # If edge is primary (Corridor A), report heavy congestion (0.80)
            if edge_data.get("highway") == "primary":
                obs = TomTomObservation(
                    current_speed_kmh=10.0,
                    free_flow_speed_kmh=50.0,
                    current_travel_time=150.0,
                    free_flow_travel_time=30.0,
                    confidence=0.90,
                    road_closure=False,
                    frc="FRC1",
                    congestion_index=0.80,
                    coordinates=((12.9170, 77.6200), (12.9170, 77.6220)),
                    fetched_at=1000.0,
                )
                return TomTomMatchResult(matched=True, observation=obs)
            # Corridor B: free flow (0.00)
            obs = TomTomObservation(
                current_speed_kmh=50.0,
                free_flow_speed_kmh=50.0,
                current_travel_time=30.0,
                free_flow_travel_time=30.0,
                confidence=0.90,
                road_closure=False,
                frc="FRC2",
                congestion_index=0.00,
                coordinates=((12.9170, 77.6200), (12.9180, 77.6210)),
                fetched_at=1000.0,
            )
            return TomTomMatchResult(matched=True, observation=obs)

        mock_tomtom.get_edge_observation.side_effect = mock_get_edge_obs

        provider = MagicMock()
        provider.load.return_value = self.graph.copy()

        routing_service = TrafficAwarePathService(
            network_provider=provider,
            traffic_dataset=None,
            vision_adapter=None,
            tomtom_adapter=mock_tomtom,
        )

        candidate, orig, dest, ts, idx = routing_service.calculate(
            Coordinate(latitude=12.9170, longitude=77.6200),
            Coordinate(latitude=12.9170, longitude=77.6220),
            traffic_weight=2.0,
        )

        # Baseline A cost = 200 + 200 * 2.0 * 0.80 = 520.
        # Detour B cost = 240 + 240 * 2.0 * 0.00 = 240.
        # Traffic-aware route should choose Detour B (240m) to avoid heavy congestion!
        self.assertAlmostEqual(candidate.distance_meters, 240.0, places=1)
        self.assertEqual(candidate.traffic_source, "tomtom")
        self.assertEqual(candidate.tomtom_matched_edges_count, 2)

    def test_vision_priority_over_tomtom(self):
        # Test 19: Valid vision observation overrides TomTom on the same edge
        edge_key = TrafficEdgeKey(osm_u=1, osm_v=2, key=0)

        mock_vision = MagicMock(spec=VisionTrafficAdapter)
        # Vision reports mild congestion (0.10)
        mock_vision.get_congestion_index.return_value = 0.10

        # TomTom reports severe congestion (0.90)
        mock_tomtom = MagicMock(spec=TomTomTrafficAdapter)
        mock_tomtom.is_available = True
        mock_obs = TomTomObservation(
            current_speed_kmh=5.0,
            free_flow_speed_kmh=50.0,
            current_travel_time=200.0,
            free_flow_travel_time=20.0,
            confidence=0.90,
            road_closure=False,
            frc="FRC1",
            congestion_index=0.90,
            coordinates=((12.9170, 77.6200), (12.9170, 77.6210)),
            fetched_at=1000.0,
        )
        mock_tomtom.get_edge_observation.return_value = TomTomMatchResult(
            matched=True, observation=mock_obs
        )

        provider = MagicMock()
        provider.load.return_value = self.graph.copy()

        service = TrafficAwarePathService(
            network_provider=provider,
            traffic_dataset=None,
            vision_adapter=mock_vision,
            tomtom_adapter=mock_tomtom,
        )

        working_graph = self.graph.copy()
        service._apply_traffic_weights(working_graph, selected_time_index=0, traffic_weight=2.0)

        edge_data = working_graph.get_edge_data(1, 2)[0]
        # Must use vision!
        self.assertEqual(edge_data["traffic_source"], "vision")
        self.assertEqual(edge_data["traffic_congestion"], 0.10)

    def test_tomtom_fallback_to_simulated_dataset(self):
        # Test 18: If TomTom is disabled or unmatched, edge falls back to simulated dataset
        edge_key = TrafficEdgeKey(osm_u=1, osm_v=2, key=0)

        mock_dataset = MagicMock(spec=TrafficDataset)
        mock_dataset.has_edge.return_value = True
        snap = MagicMock()
        snap.congestion_index = 0.42
        mock_dataset.get_snapshot.return_value = snap

        # TomTom is disabled
        mock_tomtom = MagicMock(spec=TomTomTrafficAdapter)
        mock_tomtom.is_available = False

        provider = MagicMock()
        provider.load.return_value = self.graph.copy()

        service = TrafficAwarePathService(
            network_provider=provider,
            traffic_dataset=mock_dataset,
            vision_adapter=None,
            tomtom_adapter=mock_tomtom,
        )

        working_graph = self.graph.copy()
        service._apply_traffic_weights(working_graph, selected_time_index=0, traffic_weight=2.0)

        edge_data = working_graph.get_edge_data(1, 2)[0]
        self.assertEqual(edge_data["traffic_source"], "simulated")
        self.assertEqual(edge_data["traffic_congestion"], 0.42)

    def test_accessibility_mode_penalizes_stairs(self):
        # Test 21: Accessibility mode penalizes stairs regardless of TomTom traffic
        # Make Corridor A have steps
        self.graph[1][2][0]["highway"] = "steps"
        self.graph[1][2][0]["is_steps"] = True

        provider = MagicMock()
        provider.load.return_value = self.graph.copy()

        disabled_tomtom = TomTomTrafficAdapter(api_key="", enabled=False)
        service = TrafficAwarePathService(
            network_provider=provider,
            traffic_dataset=None,
            vision_adapter=None,
            tomtom_adapter=disabled_tomtom,
        )

        # Standard mode takes Corridor A with steps (200m)
        cand_std, _, _, _, _ = service.calculate(
            Coordinate(12.9170, 77.6200),
            Coordinate(12.9170, 77.6220),
            accessibility_mode="standard",
        )
        self.assertAlmostEqual(cand_std.distance_meters, 200.0, places=1)

        # Step-free mode avoids Corridor A because of stairs penalty, taking Corridor B (240m)
        cand_stepfree, _, _, _, _ = service.calculate(
            Coordinate(12.9170, 77.6200),
            Coordinate(12.9170, 77.6220),
            accessibility_mode="step_free",
        )
        self.assertAlmostEqual(cand_stepfree.distance_meters, 240.0, places=1)
        self.assertEqual(cand_stepfree.steps_count, 0)

    def test_selected_path_bounds_tomtom_queries_to_candidate_edges(self):
        # Create a graph with 30 motor-vehicle edges
        large_graph = nx.MultiDiGraph()
        # Direct path from 0 to 10
        for i in range(10):
            large_graph.add_node(i, x=77.6200 + i * 0.001, y=12.9170)
        for i in range(9):
            large_graph.add_edge(i, i + 1, key=0, length=50.0, highway="primary", oneway=False)

        # Distant background edges that should NOT be queried
        for i in range(10, 30):
            large_graph.add_node(i, x=77.6200 + (i - 10) * 0.001, y=12.9250)
        for i in range(10, 29):
            large_graph.add_edge(i, i + 1, key=0, length=50.0, highway="secondary", oneway=False)

        mock_tomtom = MagicMock(spec=TomTomTrafficAdapter)
        mock_tomtom.is_available = True
        mock_tomtom._get_from_cache.return_value = None
        mock_tomtom.get_edge_observation.return_value = TomTomMatchResult(
            matched=True,
            observation=TomTomObservation(
                current_speed_kmh=30.0,
                free_flow_speed_kmh=40.0,
                current_travel_time=10.0,
                free_flow_travel_time=10.0,
                confidence=0.9,
                road_closure=False,
                frc="FRC2",
                congestion_index=0.25,
                coordinates=((12.9170, 77.6200), (12.9170, 77.6210)),
                fetched_at=100.0,
            )
        )

        provider = MagicMock()
        provider.load.return_value = large_graph

        comp_service = TrafficRouteComparisonService(
            network_provider=provider,
            tomtom_adapter=mock_tomtom,
        )

        res = comp_service.compare(
            Coordinate(12.9170, 77.6200),
            Coordinate(12.9170, 77.6290),
        )

        # Verify that total TomTom edge observations requested is bounded (<= 20) and does not query all 28 edges
        call_count = mock_tomtom.get_edge_observation.call_count
        self.assertLessEqual(call_count, 20)
        self.assertEqual(call_count, 9)  # Exactly the 9 edges of the path!
        self.assertEqual(res.traffic_accessibility_aware.tomtom_matched_edges_count, 9)
        self.assertEqual(res.primary_traffic_source, "tomtom")

    def test_cache_reuse_avoids_duplicate_api_calls(self):
        adapter = TomTomTrafficAdapter(api_key="mock_test_key", enabled=True, ttl_seconds=600.0)
        # Pre-populate cache for an edge midpoint
        cache_key = (round(12.9175, 4), round(77.6235, 4))
        obs = TomTomObservation(
            current_speed_kmh=35.0,
            free_flow_speed_kmh=40.0,
            current_travel_time=10.0,
            free_flow_travel_time=10.0,
            confidence=0.9,
            road_closure=False,
            frc="FRC2",
            congestion_index=0.125,
            coordinates=((12.9170, 77.6230), (12.9180, 77.6240)),
            fetched_at=100.0,
        )
        adapter._put_in_cache(cache_key, obs)

        with patch.object(adapter, "_query_api") as mock_query:
            edge = {"highway": "primary", "oneway": False}
            res = adapter.get_edge_observation(edge, (12.9170, 77.6230), (12.9180, 77.6240))
            self.assertTrue(res.matched)
            mock_query.assert_not_called()  # Must reuse cache without making any HTTP request!
            self.assertEqual(adapter.cache_hits, 1)

    def test_daily_budget_exhaustion_falls_back_gracefully(self):
        adapter = TomTomTrafficAdapter(
            api_key="mock_key",
            enabled=True,
            max_daily_requests=1,
        )
        # Force daily count to max
        adapter._daily_requests_count = 1

        provider = MagicMock()
        provider.load.return_value = self.graph.copy()

        comp_service = TrafficRouteComparisonService(
            network_provider=provider,
            tomtom_adapter=adapter,
        )

        res = comp_service.compare(
            Coordinate(12.9170, 77.6200),
            Coordinate(12.9170, 77.6220),
        )
        # Route comparison completes safely without crashing
        self.assertIsNotNone(res)
        self.assertIn(res.status, ("identical_routes", "no_traffic_coverage_in_area", "traffic_dataset_unavailable"))
        self.assertAlmostEqual(res.shortest_distance.distance_meters, 200.0, places=1)


if __name__ == "__main__":
    unittest.main()
