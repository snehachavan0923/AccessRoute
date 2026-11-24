import unittest
from unittest.mock import MagicMock
import networkx as nx
import numpy as np

from backend.pedestrian_network import PedestrianNetworkProvider
from backend.routing.contracts import Coordinate
from backend.routing.traffic_aware_path import (
    TrafficRouteComparisonService,
    get_candidate_simple_paths,
)
from backend.traffic.loader import (
    TrafficDataset,
    TrafficDatasetInfo,
    TrafficEdgeKey,
)
from backend.traffic.tomtom_adapter import (
    TomTomMatchResult,
    TomTomObservation,
    TomTomTrafficAdapter,
)


class ControlledTrafficScenariosTest(unittest.TestCase):
    """Controlled deterministic traffic simulation tests across Scenarios A through F."""

    def setUp(self) -> None:
        self.orig = Coordinate(latitude=12.840, longitude=77.670)
        self.dest = Coordinate(latitude=12.840, longitude=77.680)

    def _build_graph(self, include_stairs: bool = False) -> nx.MultiDiGraph:
        """Constructs synthetic controlled graph with motor corridor, footway bypass, and optional stairs."""
        g = nx.MultiDiGraph(crs="epsg:4326")
        g.add_node(1, x=77.670, y=12.840)
        g.add_node(2, x=77.675, y=12.840)
        g.add_node(3, x=77.680, y=12.840)
        g.add_node(4, x=77.672, y=12.842)
        g.add_node(5, x=77.678, y=12.842)

        # Primary motor corridor: 1 -> 2 -> 3 (200m)
        g.add_edge(1, 2, 0, length=100.0, highway="tertiary", name="Main St 1")
        g.add_edge(2, 3, 0, length=100.0, highway="tertiary", name="Main St 2")
        g.add_edge(2, 1, 0, length=100.0, highway="tertiary", name="Main St 1")
        g.add_edge(3, 2, 0, length=100.0, highway="tertiary", name="Main St 2")

        # Pedestrian bypass corridor: 1 -> 4 -> 5 -> 3 (225m, +25m detour)
        g.add_edge(1, 4, 0, length=75.0, highway="footway", name="Footpath 1")
        g.add_edge(4, 5, 0, length=75.0, highway="footway", name="Footpath 2")
        g.add_edge(5, 3, 0, length=75.0, highway="footway", name="Footpath 3")
        g.add_edge(4, 1, 0, length=75.0, highway="footway", name="Footpath 1")
        g.add_edge(5, 4, 0, length=75.0, highway="footway", name="Footpath 2")
        g.add_edge(3, 5, 0, length=75.0, highway="footway", name="Footpath 3")

        if include_stairs:
            g.add_node(6, x=77.675, y=12.838)
            # Staircase corridor: 1 -> 6 -> 3 (190m physical, shortest overall)
            g.add_edge(1, 6, 0, length=95.0, highway="steps", stairs="yes", name="Steps 1")
            g.add_edge(6, 3, 0, length=95.0, highway="footway", name="Footpath 4")
            g.add_edge(6, 1, 0, length=95.0, highway="steps", stairs="yes", name="Steps 1")
            g.add_edge(3, 6, 0, length=95.0, highway="footway", name="Footpath 4")

        return g

    def _make_obs(
        self,
        congestion: float,
        speed: float = 50.0,
        free_flow: float = 50.0,
        road_closure: bool = False,
    ) -> TomTomObservation:
        return TomTomObservation(
            current_speed_kmh=speed,
            free_flow_speed_kmh=free_flow,
            current_travel_time=100.0,
            free_flow_travel_time=100.0 * (1.0 - congestion) if free_flow > 0 else 100.0,
            confidence=0.9,
            road_closure=road_closure,
            frc="FRC3",
            congestion_index=congestion,
            coordinates=((12.840, 77.670), (12.840, 77.675)),
            fetched_at=1000.0,
        )

    def test_candidate_simple_paths_on_multigraph(self) -> None:
        """Verifies get_candidate_simple_paths correctly handles MultiDiGraph without raising NetworkXNotImplemented."""
        g = self._build_graph()
        # Add a parallel edge with longer length
        g.add_edge(1, 2, 1, length=120.0, highway="tertiary")
        paths = get_candidate_simple_paths(g, 1, 3, max_candidates=3)
        self.assertGreaterEqual(len(paths), 2)
        # First path should be shortest motor corridor [1, 2, 3]
        self.assertEqual(paths[0], [1, 2, 3])
        # Second path should be bypass [1, 4, 5, 3]
        self.assertEqual(paths[1], [1, 4, 5, 3])

    def test_scenario_a_no_congestion(self) -> None:
        """Scenario A: All observed road segments have 0% congestion. Shortest route remains recommended."""
        graph = self._build_graph()
        provider = MagicMock(spec=PedestrianNetworkProvider)
        provider.load.return_value = graph

        tt = MagicMock(spec=TomTomTrafficAdapter)
        tt.is_available = True
        tt._get_from_cache.return_value = None
        tt.get_edge_observation.return_value = TomTomMatchResult(
            matched=True, observation=self._make_obs(0.0)
        )

        service = TrafficRouteComparisonService(network_provider=provider, tomtom_adapter=tt)
        result = service.compare(self.orig, self.dest)

        self.assertEqual(result.status, "identical_routes")
        self.assertAlmostEqual(result.shortest_distance.distance_meters, 200.0)
        self.assertAlmostEqual(result.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertAlmostEqual(result.shortest_distance.cost, 200.0)
        self.assertEqual(result.shortest_distance.tomtom_matched_edges_count, 2)
        self.assertEqual(result.traffic_accessibility_aware.tomtom_matched_edges_count, 2)
        self.assertAlmostEqual(result.shortest_distance.average_congestion, 0.0)

    def test_scenario_b_moderate_congestion(self) -> None:
        """Scenario B: 30% congestion on shortest motor corridor. Cost increases, detour (+25m) selected."""
        graph = self._build_graph()
        provider = MagicMock(spec=PedestrianNetworkProvider)
        provider.load.return_value = graph

        tt = MagicMock(spec=TomTomTrafficAdapter)
        tt.is_available = True
        tt._get_from_cache.return_value = None
        tt.get_edge_observation.return_value = TomTomMatchResult(
            matched=True, observation=self._make_obs(0.30, speed=35.0)
        )

        service = TrafficRouteComparisonService(network_provider=provider, tomtom_adapter=tt)
        result = service.compare(self.orig, self.dest)

        self.assertEqual(result.status, "alternative_found")
        self.assertAlmostEqual(result.shortest_distance.distance_meters, 200.0)
        # Shortest cost = 200 + 200 * 2.0 * 0.30 = 320.0
        self.assertAlmostEqual(result.shortest_distance.cost, 320.0)
        # Alternative bypass = 225m cost
        self.assertAlmostEqual(result.traffic_accessibility_aware.distance_meters, 225.0)
        self.assertAlmostEqual(result.traffic_accessibility_aware.cost, 225.0)
        self.assertAlmostEqual(
            result.traffic_accessibility_aware.distance_meters - result.shortest_distance.distance_meters,
            25.0,
        )

    def test_scenario_c_heavy_congestion(self) -> None:
        """Scenario C: 80% heavy congestion on shortest motor corridor. Significant penalty, alternative selected."""
        graph = self._build_graph()
        provider = MagicMock(spec=PedestrianNetworkProvider)
        provider.load.return_value = graph

        tt = MagicMock(spec=TomTomTrafficAdapter)
        tt.is_available = True
        tt._get_from_cache.return_value = None
        tt.get_edge_observation.return_value = TomTomMatchResult(
            matched=True, observation=self._make_obs(0.80, speed=10.0)
        )

        service = TrafficRouteComparisonService(network_provider=provider, tomtom_adapter=tt)
        result = service.compare(self.orig, self.dest)

        self.assertEqual(result.status, "alternative_found")
        self.assertAlmostEqual(result.shortest_distance.distance_meters, 200.0)
        # Shortest cost = 200 + 200 * 2.0 * 0.80 = 520.0
        self.assertAlmostEqual(result.shortest_distance.cost, 520.0)
        self.assertAlmostEqual(result.traffic_accessibility_aware.distance_meters, 225.0)
        self.assertAlmostEqual(result.traffic_accessibility_aware.cost, 225.0)

    def test_scenario_d_no_traffic_coverage(self) -> None:
        """Scenario D: No TomTom and no dataset coverage. Status is no_traffic_coverage_in_area, congestion Unknown."""
        graph = self._build_graph()
        provider = MagicMock(spec=PedestrianNetworkProvider)
        provider.load.return_value = graph

        tt = MagicMock(spec=TomTomTrafficAdapter)
        tt.is_available = False

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=None,
            tomtom_adapter=tt,
        )
        result = service.compare(self.orig, self.dest)

        self.assertEqual(result.status, "no_traffic_coverage_in_area")
        self.assertAlmostEqual(result.shortest_distance.distance_meters, 200.0)
        self.assertAlmostEqual(result.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertEqual(result.primary_traffic_source, "none")
        self.assertIsNone(result.shortest_distance.average_congestion)
        self.assertEqual(result.shortest_distance.matched_traffic_edges_count, 0)
        self.assertEqual(result.shortest_distance.unmatched_edges_count, 2)

    def test_scenario_e_mixed_sources(self) -> None:
        """Scenario E: Mixed sources (Edge 1->2 has TomTom, Edge 2->3 has dataset). Verify source priority & metrics."""
        graph = self._build_graph()
        provider = MagicMock(spec=PedestrianNetworkProvider)
        provider.load.return_value = graph

        tt = MagicMock(spec=TomTomTrafficAdapter)
        tt.is_available = True
        tt._get_from_cache.return_value = None

        def match_fn(data, src, dst):
            if src[1] == 77.670:  # Edge 1->2
                return TomTomMatchResult(matched=True, observation=self._make_obs(0.40))
            return TomTomMatchResult(matched=False)

        tt.get_edge_observation.side_effect = match_fn

        # Mini dataset for Edge 2->3
        ek = TrafficEdgeKey(osm_u=2, osm_v=3, key=0)
        ds_info = TrafficDatasetInfo(
            total_nodes=2,
            total_edges=1,
            total_timestamps=1,
            start_time="2026-01-01",
            end_time="2026-01-01",
        )
        dataset = TrafficDataset(
            info=ds_info,
            timestamps=["2026-01-01 00:00:00"],
            edge_key_to_id={ek: 0},
            id_to_edge_key={0: ek},
            node_ids={2, 3},
            speed_matrix=np.array([[25.0]]),
            vehicle_matrix=np.array([[10.0]]),
            congestion_matrix=np.array([[0.50]]),
        )

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=dataset,
            tomtom_adapter=tt,
        )
        result = service.compare(self.orig, self.dest)

        self.assertEqual(result.primary_traffic_source, "mixed")
        self.assertEqual(result.shortest_distance.tomtom_matched_edges_count, 1)
        self.assertEqual(result.shortest_distance.simulated_matched_edges_count, 1)
        self.assertEqual(result.shortest_distance.unmatched_edges_count, 0)
        self.assertEqual(result.status, "alternative_found")

    def test_scenario_f_accessibility_constraints(self) -> None:
        """Scenario F: Shortest physical route has steps (190m). Step-free mode avoids steps; standard mode uses it."""
        graph = self._build_graph(include_stairs=True)
        provider = MagicMock(spec=PedestrianNetworkProvider)
        provider.load.return_value = graph

        tt = MagicMock(spec=TomTomTrafficAdapter)
        tt.is_available = False

        service = TrafficRouteComparisonService(
            network_provider=provider,
            traffic_dataset=None,
            tomtom_adapter=tt,
        )

        # Standard walking mode
        res_std = service.compare(self.orig, self.dest, accessibility_mode="standard")
        self.assertAlmostEqual(res_std.shortest_distance.distance_meters, 190.0)
        self.assertAlmostEqual(res_std.traffic_accessibility_aware.distance_meters, 190.0)
        self.assertEqual(res_std.traffic_accessibility_aware.steps_count, 1)

        # Step-free mode
        res_step_free = service.compare(self.orig, self.dest, accessibility_mode="step_free")
        self.assertEqual(res_step_free.status, "alternative_found")
        self.assertAlmostEqual(res_step_free.shortest_distance.distance_meters, 190.0)
        self.assertEqual(res_step_free.shortest_distance.steps_count, 1)
        # Avoids steps by taking motor corridor (200m, +10m detour) with 0 steps
        self.assertAlmostEqual(res_step_free.traffic_accessibility_aware.distance_meters, 200.0)
        self.assertEqual(res_step_free.traffic_accessibility_aware.steps_count, 0)
        self.assertAlmostEqual(
            res_step_free.traffic_accessibility_aware.distance_meters - res_step_free.shortest_distance.distance_meters,
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
