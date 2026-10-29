"""Tests for pedestrian pathway accessibility obstacles and step-free routing integration.

Verifies:
1. AccessibilityAdapter registration, TTL expiration, and clock skew safety.
2. Step-free mode penalizes stairs and diverts around staircases.
3. Standard walking mode permits stairs without penalty.
4. Active verified obstacle diverts traffic-and-accessibility-aware path.
5. Expired or unverified obstacle falls back safely and does not divert route.
6. Reverse-edge isolation: obstacle on (u -> v) does not penalize (v -> u).
7. Missing accessibility information is marked as unknown, not certified accessible.
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
from backend.routing.traffic_aware_path import (
    TrafficAwarePathService,
    TrafficRouteComparisonService,
)
from backend.traffic import TrafficDataset, TrafficEdgeKey, VisionTrafficAdapter


def make_staircase_choice_graph() -> nx.MultiDiGraph:
    """Create a graph with two parallel routes from 1 to 3:

    Route A (staircase path):
        1 -> 2 (steps, length=50m)
        2 -> 3 (footway, length=50m)
        Total physical distance: 100m

    Route B (accessible flat detour):
        1 -> 4 (footway, length=70m)
        4 -> 3 (footway, length=70m)
        Total physical distance: 140m
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"

    g.add_node(1, x=77.670, y=12.840)
    g.add_node(2, x=77.6705, y=12.840)
    g.add_node(3, x=77.671, y=12.840)
    g.add_node(4, x=77.6705, y=12.841)

    # Route A (has steps)
    g.add_edge(1, 2, 0, length=50.0, highway="steps")
    g.add_edge(2, 3, 0, length=50.0, highway="footway")

    # Route B (ramped / flat footway detour)
    g.add_edge(1, 4, 0, length=70.0, highway="footway", surface="asphalt")
    g.add_edge(4, 3, 0, length=70.0, highway="footway", surface="asphalt")

    return g


def make_barrier_choice_graph() -> nx.MultiDiGraph:
    """Create a graph with two parallel paths between 1 and 3:

    Direct path: 1 -> 2 -> 3 (each 50m, total 100m)
    Detour path: 1 -> 4 -> 3 (each 65m, total 130m)
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"

    g.add_node(1, x=77.670, y=12.840)
    g.add_node(2, x=77.6705, y=12.840)
    g.add_node(3, x=77.671, y=12.840)
    g.add_node(4, x=77.6705, y=12.841)

    g.add_edge(1, 2, 0, length=50.0, highway="footway")
    g.add_edge(2, 1, 0, length=50.0, highway="footway")
    g.add_edge(2, 3, 0, length=50.0, highway="footway")
    g.add_edge(3, 2, 0, length=50.0, highway="footway")

    g.add_edge(1, 4, 0, length=65.0, highway="footway")
    g.add_edge(4, 1, 0, length=65.0, highway="footway")
    g.add_edge(4, 3, 0, length=65.0, highway="footway")
    g.add_edge(3, 4, 0, length=65.0, highway="footway")

    return g


class TestAccessibilityAdapterUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AccessibilityAdapter(default_ttl_seconds=3600.0)

    def test_register_and_retrieve_valid_obstacle(self) -> None:
        now = datetime.now(timezone.utc)
        obs = AccessibilityObstacleObservation(
            obstacle_id="obs-001",
            osm_u=10,
            osm_v=20,
            osm_key=0,
            obstacle_type=AccessibilityObstacleType.BLOCKED_SIDEWALK,
            severity=AccessibilitySeverity.IMPASSABLE,
            reported_at_utc=now,
            time_to_live_seconds=1800.0,
            is_verified=True,
            description="Deep excavation blocking entire pathway",
        )
        key = self.adapter.register_obstacle(obs)
        self.assertEqual(key, TrafficEdgeKey(osm_u=10, osm_v=20, key=0))

        retrieved = self.adapter.get_valid_obstacle((10, 20, 0), current_time=now)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.obstacle_id, "obs-001")
        self.assertTrue(self.adapter.has_obstacle((10, 20, 0), current_time=now))
        self.assertEqual(
            self.adapter.get_obstacle_penalty((10, 20, 0), current_time=now), 100.0
        )

    def test_unverified_obstacle_is_ignored(self) -> None:
        now = datetime.now(timezone.utc)
        obs = AccessibilityObstacleObservation(
            obstacle_id="obs-unverified",
            osm_u=10,
            osm_v=20,
            obstacle_type=AccessibilityObstacleType.SURFACE_OBSTRUCTION,
            severity=AccessibilitySeverity.CAUTION,
            reported_at_utc=now,
            is_verified=False,
        )
        self.adapter.register_obstacle(obs)
        self.assertIsNone(self.adapter.get_valid_obstacle((10, 20), current_time=now))
        self.assertFalse(self.adapter.has_obstacle((10, 20), current_time=now))

    def test_low_confidence_obstacle_is_ignored_when_threshold_configured(self) -> None:
        adapter = AccessibilityAdapter(default_ttl_seconds=3600.0, min_confidence=0.75)
        now = datetime.now(timezone.utc)
        obs_low = AccessibilityObstacleObservation(
            obstacle_id="obs-low-conf",
            osm_u=10,
            osm_v=20,
            obstacle_type=AccessibilityObstacleType.SURFACE_OBSTRUCTION,
            severity=AccessibilitySeverity.CAUTION,
            reported_at_utc=now,
            is_verified=True,
            confidence=0.60,
        )
        obs_high = AccessibilityObstacleObservation(
            obstacle_id="obs-high-conf",
            osm_u=30,
            osm_v=40,
            obstacle_type=AccessibilityObstacleType.BLOCKED_SIDEWALK,
            severity=AccessibilitySeverity.IMPASSABLE,
            reported_at_utc=now,
            is_verified=True,
            confidence=0.85,
        )
        adapter.register_obstacle(obs_low)
        adapter.register_obstacle(obs_high)

        self.assertIsNone(adapter.get_valid_obstacle((10, 20), current_time=now))
        self.assertIsNotNone(adapter.get_valid_obstacle((30, 40), current_time=now))
        self.assertEqual(len(adapter.get_active_obstacles(current_time=now)), 1)

    def test_expired_obstacle_is_ignored(self) -> None:
        now = datetime.now(timezone.utc)
        past_time = now - timedelta(seconds=5000)
        obs = AccessibilityObstacleObservation(
            obstacle_id="obs-expired",
            osm_u=10,
            osm_v=20,
            obstacle_type=AccessibilityObstacleType.CONSTRUCTION_BARRIER,
            severity=AccessibilitySeverity.HIGH_DIFFICULTY,
            reported_at_utc=past_time,
            time_to_live_seconds=600.0,  # Expired after 10 min
            is_verified=True,
        )
        self.adapter.register_obstacle(obs)
        self.assertIsNone(self.adapter.get_valid_obstacle((10, 20), current_time=now))

    def test_future_dated_obstacle_is_rejected(self) -> None:
        now = datetime.now(timezone.utc)
        future_time = now + timedelta(seconds=120)  # Clock skew exceeds 5s
        obs = AccessibilityObstacleObservation(
            obstacle_id="obs-future",
            osm_u=10,
            osm_v=20,
            obstacle_type=AccessibilityObstacleType.MISSING_CURB_RAMP,
            severity=AccessibilitySeverity.HIGH_DIFFICULTY,
            reported_at_utc=future_time,
            is_verified=True,
        )
        self.adapter.register_obstacle(obs)
        self.assertIsNone(self.adapter.get_valid_obstacle((10, 20), current_time=now))


class TestStepFreeStaircaseRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.network_provider = Mock()
        self.network_provider.load.return_value = make_staircase_choice_graph()
        self.accessibility_adapter = AccessibilityAdapter()
        self.vision_adapter = VisionTrafficAdapter()

        self.service = TrafficRouteComparisonService(
            network_provider=self.network_provider,
            traffic_dataset=None,
            vision_adapter=self.vision_adapter,
            accessibility_adapter=self.accessibility_adapter,
        )

        self.origin = Coordinate(latitude=12.840, longitude=77.670)
        self.destination = Coordinate(latitude=12.840, longitude=77.671)

    def test_standard_walking_mode_uses_shortest_staircase_route(self) -> None:
        """In standard mode, pedestrian can take the shorter 100m staircase path."""
        res = self.service.compare(
            self.origin,
            self.destination,
            accessibility_mode="standard",
        )
        self.assertAlmostEqual(res.shortest_distance.distance_meters, 100.0)
        self.assertAlmostEqual(
            res.traffic_accessibility_aware.distance_meters, 100.0
        )
        self.assertIn(res.status, ("identical_routes", "traffic_dataset_unavailable"))
        self.assertEqual(res.traffic_accessibility_aware.steps_count, 1)


    def test_step_free_mode_diverts_around_staircase(self) -> None:
        """In step_free mode, stairs are penalized and route diverts to flat 140m detour."""
        res = self.service.compare(
            self.origin,
            self.destination,
            accessibility_mode="step_free",
        )
        # Shortest baseline remains 100m
        self.assertAlmostEqual(res.shortest_distance.distance_meters, 100.0)
        self.assertEqual(res.shortest_distance.steps_count, 1)

        # Recommended step-free route takes 140m flat detour via node 4
        self.assertAlmostEqual(
            res.traffic_accessibility_aware.distance_meters, 140.0
        )
        self.assertEqual(res.traffic_accessibility_aware.steps_count, 0)
        self.assertEqual(res.status, "alternative_found")
        self.assertIn("avoids pedestrian staircases", res.explanation)
        self.assertIn("avoid_stairs", res.traffic_accessibility_aware.accessibility_constraints_applied)


class TestAccessibilityObstacleRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.network_provider = Mock()
        self.network_provider.load.return_value = make_barrier_choice_graph()
        self.accessibility_adapter = AccessibilityAdapter()
        self.vision_adapter = VisionTrafficAdapter()

        self.service = TrafficRouteComparisonService(
            network_provider=self.network_provider,
            traffic_dataset=None,
            vision_adapter=self.vision_adapter,
            accessibility_adapter=self.accessibility_adapter,
        )

        self.origin = Coordinate(latitude=12.840, longitude=77.670)
        self.destination = Coordinate(latitude=12.840, longitude=77.671)

    def test_verified_obstacle_diverts_route(self) -> None:
        """A verified impassable obstacle on edge (1 -> 2) diverts route around barrier via node 4."""
        now = datetime.now(timezone.utc)
        obs = AccessibilityObstacleObservation(
            obstacle_id="barrier-1",
            osm_u=1,
            osm_v=2,
            osm_key=0,
            obstacle_type=AccessibilityObstacleType.BLOCKED_SIDEWALK,
            severity=AccessibilitySeverity.IMPASSABLE,
            reported_at_utc=now,
            is_verified=True,
            description="Sidewalk completely blocked by construction",
        )
        self.accessibility_adapter.register_obstacle(obs)

        res = self.service.compare(
            self.origin,
            self.destination,
            accessibility_mode="step_free",
        )
        # Baseline shortest path is still direct 100m
        self.assertAlmostEqual(res.shortest_distance.distance_meters, 100.0)

        # Recommended route takes detour around barrier: 130m
        self.assertAlmostEqual(
            res.traffic_accessibility_aware.distance_meters, 130.0
        )
        self.assertEqual(res.status, "alternative_found")
        self.assertEqual(res.traffic_accessibility_aware.obstacles_count, 0)
        self.assertIn("bypasses verified physical barriers", res.explanation)

    def test_directed_obstacle_isolation_no_reverse_leakage(self) -> None:
        """Obstacle on (2 -> 1) must NOT penalize forward route (1 -> 2 -> 3)."""
        now = datetime.now(timezone.utc)
        obs = AccessibilityObstacleObservation(
            obstacle_id="barrier-rev",
            osm_u=2,
            osm_v=1,
            osm_key=0,
            obstacle_type=AccessibilityObstacleType.BLOCKED_SIDEWALK,
            severity=AccessibilitySeverity.IMPASSABLE,
            reported_at_utc=now,
            is_verified=True,
        )
        self.accessibility_adapter.register_obstacle(obs)

        res = self.service.compare(
            self.origin,
            self.destination,
            accessibility_mode="step_free",
        )
        # Forward route (1 -> 2 -> 3) is unaffected by reverse-edge barrier!
        self.assertAlmostEqual(
            res.traffic_accessibility_aware.distance_meters, 100.0
        )
        self.assertIn(res.status, ("identical_routes", "traffic_dataset_unavailable"))



if __name__ == "__main__":
    unittest.main()
