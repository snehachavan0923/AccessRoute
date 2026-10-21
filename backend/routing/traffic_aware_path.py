from __future__ import annotations

import math
from math import hypot
from typing import Sequence

import networkx as nx

from backend.pedestrian_network import (
    NetworkBounds,
    NetworkDownloadError,
    OSMnxPedestrianNetworkProvider,
    PedestrianNetworkProvider,
)
from backend.routing.contracts import (
    Coordinate,
    RouteCandidate,
    RouteComparison,
    ShortestRoute,
)
from backend.routing.shortest_path import (
    NetworkAreaSettings,
    NetworkAreaTooLargeError,
    RouteNetworkUnavailableError,
    RouteNotFoundError,
    calculate_network_bounds,
    nearest_network_node,
)
from backend.traffic import (
    TrafficDataset,
    TrafficDatasetError,
    TrafficEdgeKey,
)

COST_FORMULA_DOCUMENTATION = (
    "Cost(e) = Length(e) * (1.0 + traffic_weight * Congestion(e)), "
    "where Length(e) is physical segment length in meters, "
    "Congestion(e) in [0.0, 1.0] is simulated vehicle congestion index (0.0 for unmatched edges), "
    "and traffic_weight >= 0 is a dimensionless penalty weight. Cost is strictly nonnegative."
)

SIMULATED_TRAFFIC_DISCLAIMER = (
    "Traffic data is synthetically simulated for Electronic City, Bengaluru, and does not "
    "represent real-time sensor measurements or certified pedestrian infrastructure accessibility."
)


def extract_path_geometry_and_metrics(
    graph: nx.MultiDiGraph,
    node_path: list[int],
    edge_selection_attribute: str = "traffic_cost",
) -> tuple[tuple[Coordinate, ...], float, float, int, int, float]:
    """Extract coordinates, physical distance, total cost, and traffic coverage from a node path.

    Returns:
        coordinates: tuple of Coordinates along the path
        physical_distance: sum of physical segment lengths in meters
        total_cost: sum of edge routing costs
        edges_count: total number of edges traversed
        matched_edges_count: number of edges with verified traffic data
        average_congestion: average congestion across matched edges
    """
    coordinates: list[Coordinate] = []
    physical_distance = 0.0
    total_cost = 0.0
    edges_count = 0
    matched_edges_count = 0
    congestion_sum = 0.0

    for source, target in zip(node_path, node_path[1:]):
        source_node = graph.nodes[source]
        target_node = graph.nodes[target]
        source_point = (float(source_node["x"]), float(source_node["y"]))
        target_point = (float(target_node["x"]), float(target_node["y"]))

        edges = graph.get_edge_data(source, target)
        if not edges:
            raise RouteNetworkUnavailableError(
                "The pedestrian route geometry is unavailable."
            )

        # Select the edge using the routing criteria (length for shortest path, traffic_cost for traffic-aware)
        best_edge = min(
            edges.values(),
            key=lambda data: data.get(edge_selection_attribute, float("inf")),
        )

        edge_length = float(best_edge.get("length", 1.0))
        edge_cost = float(best_edge.get("traffic_cost", edge_length))
        physical_distance += edge_length
        total_cost += edge_cost
        edges_count += 1


        if best_edge.get("traffic_matched", False):
            matched_edges_count += 1
            congestion_sum += float(best_edge.get("traffic_congestion", 0.0))

        geometry = best_edge.get("geometry")
        edge_coordinates = (
            list(geometry.coords)
            if geometry is not None
            else [source_point, target_point]
        )

        forward_distance = hypot(
            edge_coordinates[0][0] - source_point[0],
            edge_coordinates[0][1] - source_point[1],
        ) + hypot(
            edge_coordinates[-1][0] - target_point[0],
            edge_coordinates[-1][1] - target_point[1],
        )
        reverse_distance = hypot(
            edge_coordinates[-1][0] - source_point[0],
            edge_coordinates[-1][1] - source_point[1],
        ) + hypot(
            edge_coordinates[0][0] - target_point[0],
            edge_coordinates[0][1] - target_point[1],
        )
        if reverse_distance < forward_distance:
            edge_coordinates.reverse()

        if coordinates:
            edge_coordinates = edge_coordinates[1:]
        coordinates.extend(
            Coordinate(latitude=float(latitude), longitude=float(longitude))
            for longitude, latitude, *_ in edge_coordinates
        )

    if not node_path:
        raise RouteNetworkUnavailableError("The pedestrian route is empty.")
    if len(node_path) == 1:
        node = graph.nodes[node_path[0]]
        coordinates.append(
            Coordinate(latitude=float(node["y"]), longitude=float(node["x"]))
        )

    avg_congestion = (
        congestion_sum / matched_edges_count if matched_edges_count > 0 else 0.0
    )

    return (
        tuple(coordinates),
        physical_distance,
        total_cost,
        edges_count,
        matched_edges_count,
        avg_congestion,
    )


class TrafficAwarePathService:
    """Calculates traffic-aware pedestrian routes combining walking distance with traffic impedance."""

    def __init__(
        self,
        network_provider: PedestrianNetworkProvider | None = None,
        traffic_dataset: TrafficDataset | None = None,
        settings: NetworkAreaSettings | None = None,
    ) -> None:
        self._network_provider = network_provider or OSMnxPedestrianNetworkProvider()
        self._settings = settings or NetworkAreaSettings.from_environment()
        self._dataset: TrafficDataset | None = traffic_dataset
        self._dataset_error: str | None = None

        if self._dataset is None:
            try:
                self._dataset = TrafficDataset.from_environment()
            except TrafficDatasetError as exc:
                self._dataset_error = str(exc)

    @property
    def dataset(self) -> TrafficDataset | None:
        return self._dataset

    @property
    def dataset_error(self) -> str | None:
        return self._dataset_error

    def resolve_time(
        self,
        time_index: int | None = None,
        timestamp: str | None = None,
    ) -> tuple[int, str]:
        """Resolve timestamp/time_index against dataset, defaulting to index 0."""
        if self._dataset is None:
            return (0, timestamp or "2026-01-01 00:00:00")

        if time_index is not None:
            if time_index < 0 or time_index >= self._dataset.total_timestamps:
                raise ValueError(
                    f"Invalid time index {time_index}. Dataset contains "
                    f"{self._dataset.total_timestamps} timestamps (0..{self._dataset.total_timestamps - 1})."
                )
            return (time_index, self._dataset.timestamps[time_index])

        if timestamp is not None and timestamp.strip():
            idx = self._dataset.get_time_index(timestamp)
            return (idx, self._dataset.timestamps[idx])

        return (0, self._dataset.timestamps[0])

    def calculate(
        self,
        origin: Coordinate,
        destination: Coordinate,
        timestamp: str | None = None,
        time_index: int | None = None,
        traffic_weight: float = 2.0,
    ) -> tuple[RouteCandidate, Coordinate, Coordinate, str, int]:
        """Calculate traffic-aware route.

        Returns:
            (candidate, snapped_origin, snapped_destination, selected_timestamp, selected_time_index)
        """
        if traffic_weight < 0:
            raise ValueError("traffic_weight must be non-negative.")

        selected_time_index, selected_timestamp = self.resolve_time(
            time_index=time_index,
            timestamp=timestamp,
        )

        bounds = calculate_network_bounds(origin, destination, self._settings)
        try:
            graph = self._network_provider.load(bounds)
        except NetworkDownloadError as error:
            raise RouteNetworkUnavailableError(str(error)) from error

        if graph.number_of_nodes() == 0:
            raise RouteNetworkUnavailableError(
                "No walkable pedestrian network was found for the selected area."
            )

        # Work on a copy to avoid mutating external graph instances
        working_graph = graph.copy()

        # Annotate edges with traffic cost
        self._apply_traffic_weights(
            working_graph,
            selected_time_index=selected_time_index,
            traffic_weight=traffic_weight,
        )

        try:
            origin_node = nearest_network_node(working_graph, origin)
            destination_node = nearest_network_node(working_graph, destination)
            node_path = nx.shortest_path(
                working_graph,
                source=origin_node,
                target=destination_node,
                weight="traffic_cost",
                method="dijkstra",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
            raise RouteNotFoundError(
                "No pedestrian path connects the selected locations."
            ) from error
        except (ValueError, KeyError, TypeError) as error:
            raise RouteNetworkUnavailableError(
                "The pedestrian network data could not be processed."
            ) from error

        (
            coordinates,
            dist_meters,
            cost,
            edges_count,
            matched_count,
            avg_congestion,
        ) = extract_path_geometry_and_metrics(
            working_graph,
            node_path,
            edge_selection_attribute="traffic_cost",
        )

        coverage_pct = (
            (matched_count / edges_count * 100.0) if edges_count > 0 else 0.0
        )

        candidate = RouteCandidate(
            coordinates=coordinates,
            distance_meters=float(dist_meters),
            cost=float(cost),
            edges_count=edges_count,
            matched_traffic_edges_count=matched_count,
            traffic_coverage_percent=round(coverage_pct, 1),
            average_congestion=round(avg_congestion, 4) if matched_count > 0 else None,
        )

        origin_node_data = working_graph.nodes[origin_node]
        destination_node_data = working_graph.nodes[destination_node]
        snapped_origin = Coordinate(
            latitude=float(origin_node_data["y"]),
            longitude=float(origin_node_data["x"]),
        )
        snapped_destination = Coordinate(
            latitude=float(destination_node_data["y"]),
            longitude=float(destination_node_data["x"]),
        )

        return (
            candidate,
            snapped_origin,
            snapped_destination,
            selected_timestamp,
            selected_time_index,
        )

    def _apply_traffic_weights(
        self,
        graph: nx.MultiDiGraph,
        selected_time_index: int,
        traffic_weight: float,
    ) -> None:
        """Assign nonnegative traffic cost to every edge in the graph."""
        for u, v, key, data in graph.edges(keys=True, data=True):
            length = max(0.1, float(data.get("length", 1.0)))
            matched = False
            congestion = 0.0

            traffic_key = TrafficEdgeKey(osm_u=int(u), osm_v=int(v), key=int(key))
            if self._dataset is not None and self._dataset.has_edge(traffic_key):
                try:
                    snap = self._dataset.get_snapshot(traffic_key, selected_time_index)
                    raw_congestion = float(snap.congestion_index)
                    if math.isfinite(raw_congestion) and raw_congestion >= 0.0:
                        congestion = raw_congestion
                        matched = True
                except TrafficDatasetError:
                    congestion = 0.0
                    matched = False

            # Cost formula: Cost(e) = Length(e) * (1.0 + traffic_weight * Congestion(e))
            # Guaranteed nonnegative because length > 0, traffic_weight >= 0, congestion >= 0
            cost = length * (1.0 + traffic_weight * congestion)

            data["traffic_cost"] = float(cost)
            data["traffic_matched"] = matched
            data["traffic_congestion"] = float(congestion)


class TrafficRouteComparisonService:
    """Service to compute and compare shortest walking routes with traffic-aware alternatives."""

    def __init__(
        self,
        network_provider: PedestrianNetworkProvider | None = None,
        traffic_dataset: TrafficDataset | None = None,
        settings: NetworkAreaSettings | None = None,
    ) -> None:
        self._network_provider = network_provider or OSMnxPedestrianNetworkProvider()
        self._settings = settings or NetworkAreaSettings.from_environment()
        self._traffic_service = TrafficAwarePathService(
            network_provider=self._network_provider,
            traffic_dataset=traffic_dataset,
            settings=self._settings,
        )

    def compare(
        self,
        origin: Coordinate,
        destination: Coordinate,
        timestamp: str | None = None,
        time_index: int | None = None,
        traffic_weight: float = 2.0,
    ) -> RouteComparison:
        """Calculate and compare shortest distance and traffic-aware routes."""
        if traffic_weight < 0:
            raise ValueError("traffic_weight must be non-negative.")

        selected_time_index, selected_timestamp = self._traffic_service.resolve_time(
            time_index=time_index,
            timestamp=timestamp,
        )

        bounds = calculate_network_bounds(origin, destination, self._settings)
        try:
            graph = self._network_provider.load(bounds)
        except NetworkDownloadError as error:
            raise RouteNetworkUnavailableError(str(error)) from error

        if graph.number_of_nodes() == 0:
            raise RouteNetworkUnavailableError(
                "No walkable pedestrian network was found for the selected area."
            )

        working_graph = graph.copy()
        self._traffic_service._apply_traffic_weights(
            working_graph,
            selected_time_index=selected_time_index,
            traffic_weight=traffic_weight,
        )

        try:
            origin_node = nearest_network_node(working_graph, origin)
            destination_node = nearest_network_node(working_graph, destination)

            # 1. Shortest distance path (weight="length")
            shortest_node_path = nx.shortest_path(
                working_graph,
                source=origin_node,
                target=destination_node,
                weight="length",
                method="dijkstra",
            )

            # 2. Traffic-aware path (weight="traffic_cost")
            traffic_node_path = nx.shortest_path(
                working_graph,
                source=origin_node,
                target=destination_node,
                weight="traffic_cost",
                method="dijkstra",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
            raise RouteNotFoundError(
                "No pedestrian path connects the selected locations."
            ) from error
        except (ValueError, KeyError, TypeError) as error:
            raise RouteNetworkUnavailableError(
                "The pedestrian network data could not be processed."
            ) from error

        # Extract shortest route metrics under the same traffic conditions
        (
            shortest_coords,
            shortest_dist,
            shortest_cost,
            shortest_edges,
            shortest_matched,
            shortest_avg_cong,
        ) = extract_path_geometry_and_metrics(
            working_graph,
            shortest_node_path,
            edge_selection_attribute="length",
        )
        shortest_cov_pct = (
            (shortest_matched / shortest_edges * 100.0) if shortest_edges > 0 else 0.0
        )
        shortest_candidate = RouteCandidate(
            coordinates=shortest_coords,
            distance_meters=float(shortest_dist),
            cost=float(shortest_cost),
            edges_count=shortest_edges,
            matched_traffic_edges_count=shortest_matched,
            traffic_coverage_percent=round(shortest_cov_pct, 1),
            average_congestion=(
                round(shortest_avg_cong, 4) if shortest_matched > 0 else None
            ),
        )

        # Extract traffic-aware route metrics
        (
            traffic_coords,
            traffic_dist,
            traffic_cost,
            traffic_edges,
            traffic_matched,
            traffic_avg_cong,
        ) = extract_path_geometry_and_metrics(
            working_graph,
            traffic_node_path,
            edge_selection_attribute="traffic_cost",
        )
        traffic_cov_pct = (
            (traffic_matched / traffic_edges * 100.0) if traffic_edges > 0 else 0.0
        )
        traffic_candidate = RouteCandidate(
            coordinates=traffic_coords,
            distance_meters=float(traffic_dist),
            cost=float(traffic_cost),
            edges_count=traffic_edges,
            matched_traffic_edges_count=traffic_matched,
            traffic_coverage_percent=round(traffic_cov_pct, 1),
            average_congestion=(
                round(traffic_avg_cong, 4) if traffic_matched > 0 else None
            ),
        )

        origin_node_data = working_graph.nodes[origin_node]
        destination_node_data = working_graph.nodes[destination_node]
        snapped_origin = Coordinate(
            latitude=float(origin_node_data["y"]),
            longitude=float(origin_node_data["x"]),
        )
        snapped_destination = Coordinate(
            latitude=float(destination_node_data["y"]),
            longitude=float(destination_node_data["x"]),
        )

        # Determine status
        if self._traffic_service.dataset is None:
            status = "traffic_dataset_unavailable"
        elif traffic_matched == 0 and shortest_matched == 0:
            status = "no_traffic_coverage_in_area"
        elif shortest_node_path == traffic_node_path:
            status = "identical_routes"
        else:
            status = "alternative_found"

        return RouteComparison(
            shortest_distance=shortest_candidate,
            traffic_accessibility_aware=traffic_candidate,
            snapped_origin=snapped_origin,
            snapped_destination=snapped_destination,
            selected_timestamp=selected_timestamp,
            time_index=selected_time_index,
            cost_formula=COST_FORMULA_DOCUMENTATION,
            status=status,
            disclaimer=SIMULATED_TRAFFIC_DISCLAIMER,
        )
