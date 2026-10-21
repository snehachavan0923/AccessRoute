from __future__ import annotations

import concurrent.futures
import time
import unittest
from unittest.mock import Mock, patch

import networkx as nx
from networkx import MultiDiGraph

from backend.pedestrian_network import (
    CachedPedestrianNetworkProvider,
    NetworkBounds,
    NetworkDownloadError,
    PedestrianGraphCache,
    get_default_network_provider,
)
from backend.routing.contracts import Coordinate
from backend.routing.shortest_path import (
    NetworkAreaSettings,
    NetworkAreaTooLargeError,
    ShortestPathService,
)
from backend.routing.traffic_aware_path import (
    TrafficAwarePathService,
    TrafficRouteComparisonService,
)


def create_synthetic_grid_graph() -> MultiDiGraph:
    """Create a synthetic walkable MultiDiGraph with physical coordinates."""
    G = MultiDiGraph(crs="epsg:4326")
    # Add nodes in Electronic City coordinate range
    G.add_node(101, x=77.6658, y=12.8385)
    G.add_node(102, x=77.6710, y=12.8425)
    G.add_node(103, x=77.6750, y=12.8450)

    # Edge 101 -> 102
    G.add_edge(101, 102, key=0, length=450.0, highway="footway", osmid=1001)
    # Edge 102 -> 103
    G.add_edge(102, 103, key=0, length=520.0, highway="footway", osmid=1002)
    # Reverse edges for walkability
    G.add_edge(102, 101, key=0, length=450.0, highway="footway", osmid=1001)
    G.add_edge(103, 102, key=0, length=520.0, highway="footway", osmid=1002)
    return G


class GraphCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = NetworkBounds(north=12.85, south=12.83, east=77.68, west=77.66)
        self.sub_bounds = NetworkBounds(north=12.845, south=12.835, east=77.675, west=77.665)
        self.disjoint_bounds = NetworkBounds(north=12.92, south=12.90, east=77.50, west=77.48)
        self.graph = create_synthetic_grid_graph()

    def test_cache_miss_then_hit(self) -> None:
        cache = PedestrianGraphCache(max_size=5, ttl_seconds=60.0)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(self.bounds))
        self.assertEqual(cache.misses, 1)

        cache.put(self.bounds, self.graph)
        self.assertEqual(len(cache), 1)

        cached_graph = cache.get(self.bounds)
        self.assertIsNotNone(cached_graph)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cached_graph.number_of_nodes(), self.graph.number_of_nodes())

    def test_spatial_containment_reuses_enclosing_graph(self) -> None:
        cache = PedestrianGraphCache(max_size=5, ttl_seconds=60.0)
        cache.put(self.bounds, self.graph)

        # sub_bounds is strictly inside self.bounds
        self.assertTrue(self.bounds.contains(self.sub_bounds))
        retrieved = cache.get(self.sub_bounds)
        self.assertIsNotNone(retrieved)
        self.assertEqual(cache.hits, 1)

    def test_disjoint_bounds_do_not_share_graph(self) -> None:
        cache = PedestrianGraphCache(max_size=5, ttl_seconds=60.0)
        cache.put(self.bounds, self.graph)

        # disjoint_bounds is in Kengeri, completely separate from Electronic City
        self.assertFalse(self.bounds.contains(self.disjoint_bounds))
        self.assertIsNone(cache.get(self.disjoint_bounds))
        self.assertEqual(cache.misses, 1)

    def test_graph_immutability_copies_prevent_cache_corruption(self) -> None:
        cache = PedestrianGraphCache(max_size=5, ttl_seconds=60.0)
        cache.put(self.bounds, self.graph)

        retrieved = cache.get(self.bounds)
        self.assertIsNotNone(retrieved)
        # Mutate retrieved graph
        retrieved.add_node(999, x=0.0, y=0.0)

        # Retrieve again and ensure original in cache was unaffected
        retrieved_again = cache.get(self.bounds)
        self.assertNotIn(999, retrieved_again.nodes)

    def test_failed_or_empty_downloads_do_not_poison_cache(self) -> None:
        cache = PedestrianGraphCache(max_size=5, ttl_seconds=60.0)
        empty_graph = MultiDiGraph()
        cache.put(self.bounds, empty_graph)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(self.bounds))

        # None graph
        cache.put(self.bounds, None)
        self.assertEqual(len(cache), 0)

    def test_cache_ttl_expiration(self) -> None:
        cache = PedestrianGraphCache(max_size=5, ttl_seconds=0.05)
        cache.put(self.bounds, self.graph)
        self.assertEqual(len(cache), 1)

        time.sleep(0.08)
        self.assertIsNone(cache.get(self.bounds))
        self.assertEqual(len(cache), 0)

    def test_cache_lru_eviction(self) -> None:
        cache = PedestrianGraphCache(max_size=2, ttl_seconds=60.0)
        b1 = NetworkBounds(north=12.1, south=12.0, east=77.1, west=77.0)
        b2 = NetworkBounds(north=12.3, south=12.2, east=77.3, west=77.2)
        b3 = NetworkBounds(north=12.5, south=12.4, east=77.5, west=77.4)

        cache.put(b1, self.graph)
        cache.put(b2, self.graph)
        self.assertEqual(len(cache), 2)

        # Access b1 to make b2 the LRU
        self.assertIsNotNone(cache.get(b1))

        # Put b3, evicting b2
        cache.put(b3, self.graph)
        self.assertEqual(len(cache), 2)
        self.assertIsNotNone(cache.get(b1))
        self.assertIsNotNone(cache.get(b3))
        self.assertIsNone(cache.get(b2))

    def test_cached_provider_delegates_only_once_on_repeated_requests(self) -> None:
        mock_provider = Mock()
        mock_provider.load.return_value = self.graph
        cache = PedestrianGraphCache()
        cached_provider = CachedPedestrianNetworkProvider(
            provider=mock_provider, cache=cache
        )

        # First load triggers underlying provider
        g1 = cached_provider.load(self.bounds)
        self.assertEqual(mock_provider.load.call_count, 1)

        # Second load reuses cache
        g2 = cached_provider.load(self.bounds)
        self.assertEqual(mock_provider.load.call_count, 1)
        self.assertEqual(g1.number_of_nodes(), g2.number_of_nodes())

    def test_single_flight_concurrent_requests_share_single_download(self) -> None:
        call_count = 0

        def slow_download(bounds: NetworkBounds) -> MultiDiGraph:
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            return create_synthetic_grid_graph()

        mock_provider = Mock()
        mock_provider.load.side_effect = slow_download
        cached_provider = CachedPedestrianNetworkProvider(
            provider=mock_provider, cache=PedestrianGraphCache()
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cached_provider.load, self.bounds) for _ in range(4)
            ]
            results = [f.result() for f in futures]

        self.assertEqual(call_count, 1)
        for res in results:
            self.assertEqual(res.number_of_nodes(), self.graph.number_of_nodes())


class RouteComparisonPerformanceTests(unittest.TestCase):
    def test_comparison_service_uses_single_graph_download(self) -> None:
        mock_provider = Mock()
        mock_provider.load.return_value = create_synthetic_grid_graph()

        service = TrafficRouteComparisonService(
            network_provider=mock_provider,
            settings=NetworkAreaSettings(buffer_meters=100.0, max_diagonal_meters=5000.0),
        )

        origin = Coordinate(latitude=12.8385, longitude=77.6658)
        dest = Coordinate(latitude=12.8450, longitude=77.6750)

        comparison = service.compare(origin, dest)

        # Exactly 1 download performed for both routes
        self.assertEqual(mock_provider.load.call_count, 1)
        self.assertIsNotNone(comparison.shortest_distance)
        self.assertIsNotNone(comparison.traffic_accessibility_aware)
        self.assertGreater(comparison.shortest_distance.distance_meters, 0)

    def test_partial_result_returned_when_traffic_path_fails(self) -> None:
        mock_provider = Mock()
        graph = create_synthetic_grid_graph()
        mock_provider.load.return_value = graph

        service = TrafficRouteComparisonService(
            network_provider=mock_provider,
            settings=NetworkAreaSettings(buffer_meters=100.0, max_diagonal_meters=5000.0),
        )

        # Mock traffic service to raise an unexpected error during traffic cost assignment
        with patch.object(
            service._traffic_service,
            "_apply_traffic_weights",
            side_effect=RuntimeError("Simulated traffic processing failure"),
        ):
            origin = Coordinate(latitude=12.8385, longitude=77.6658)
            dest = Coordinate(latitude=12.8450, longitude=77.6750)

            comparison = service.compare(origin, dest)
            self.assertEqual(comparison.status, "partial_shortest_only")
            self.assertIsNotNone(comparison.shortest_distance)
            self.assertGreater(comparison.shortest_distance.distance_meters, 0)


if __name__ == "__main__":
    unittest.main()
