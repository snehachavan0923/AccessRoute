from __future__ import annotations

from dataclasses import dataclass
import math
from math import hypot
from typing import Sequence

import networkx as nx

from backend.accessibility import (
    AccessibilityAdapter,
    get_default_accessibility_adapter,
)
from backend.pedestrian_network import (
    NetworkBounds,
    NetworkDownloadError,
    OSMnxPedestrianNetworkProvider,
    PedestrianNetworkProvider,
    get_default_network_provider,
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
    VisionTrafficAdapter,
    get_default_vision_adapter,
)

COST_FORMULA_DOCUMENTATION = (
    "Cost(e) = Length(e) + (Length(e) * traffic_weight * Congestion(e)) + AccessibilityPenalty(e), "
    "where Length(e) is physical segment length in meters, "
    "Congestion(e) in [0.0, 1.0] is derived from validated vision observations or simulated dataset fallback, "
    "traffic_weight >= 0 is a dimensionless penalty weight, "
    "and AccessibilityPenalty(e) is a barrier penalty for stairs and active obstacles (applied in step-free mode). "
    "Cost is strictly nonnegative."
)

SIMULATED_TRAFFIC_DISCLAIMER = (
    "Traffic data is synthetically simulated for Electronic City, Bengaluru, and does not "
    "represent real-time sensor measurements or certified pedestrian infrastructure accessibility."
)


@dataclass(frozen=True)
class PathGeometryMetrics:
    """Internal container for metrics extracted along a traversed graph node path."""

    coordinates: tuple[Coordinate, ...]
    physical_distance: float
    total_cost: float
    edges_count: int
    matched_edges_count: int
    average_congestion: float
    vision_matched_count: int = 0
    simulated_matched_count: int = 0
    steps_count: int = 0
    unpaved_count: int = 0
    obstacles_count: int = 0
    traffic_cost_penalty: float = 0.0
    accessibility_cost_penalty: float = 0.0
    traffic_source: str = "none"

    def __iter__(self):
        """Enable backwards-compatible unpacking of the first 6 elements."""
        return iter(
            (
                self.coordinates,
                self.physical_distance,
                self.total_cost,
                self.edges_count,
                self.matched_edges_count,
                self.average_congestion,
            )
        )


def extract_path_geometry_and_metrics(
    graph: nx.MultiDiGraph,
    node_path: list[int],
    edge_selection_attribute: str = "traffic_cost",
) -> PathGeometryMetrics:
    """Extract coordinates, physical distance, total cost, and traffic coverage from a node path.

    Returns:
        PathGeometryMetrics container, also unpackable as a 6-tuple for backward compatibility.
    """
    coordinates: list[Coordinate] = []
    physical_distance = 0.0
    total_cost = 0.0
    edges_count = 0
    matched_edges_count = 0
    congestion_sum = 0.0
    vision_matched_count = 0
    simulated_matched_count = 0
    steps_count = 0
    unpaved_count = 0
    obstacles_count = 0
    traffic_penalty_sum = 0.0
    accessibility_penalty_sum = 0.0

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
            if best_edge.get("traffic_source") == "vision":
                vision_matched_count += 1
            elif best_edge.get("traffic_source") == "simulated":
                simulated_matched_count += 1

        if best_edge.get("is_steps", False):
            steps_count += 1
        if best_edge.get("is_unpaved", False):
            unpaved_count += 1
        if best_edge.get("has_obstacle", False):
            obstacles_count += 1

        traffic_penalty_sum += float(best_edge.get("traffic_penalty", 0.0))
        accessibility_penalty_sum += float(best_edge.get("accessibility_penalty", 0.0))

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

    if vision_matched_count > 0 and simulated_matched_count == 0:
        overall_traffic_source = "vision"
    elif vision_matched_count > 0 and simulated_matched_count > 0:
        overall_traffic_source = "mixed"
    elif simulated_matched_count > 0:
        overall_traffic_source = "simulated"
    else:
        overall_traffic_source = "none"

    return PathGeometryMetrics(
        coordinates=tuple(coordinates),
        physical_distance=physical_distance,
        total_cost=total_cost,
        edges_count=edges_count,
        matched_edges_count=matched_edges_count,
        average_congestion=avg_congestion,
        vision_matched_count=vision_matched_count,
        simulated_matched_count=simulated_matched_count,
        steps_count=steps_count,
        unpaved_count=unpaved_count,
        obstacles_count=obstacles_count,
        traffic_cost_penalty=traffic_penalty_sum,
        accessibility_cost_penalty=accessibility_penalty_sum,
        traffic_source=overall_traffic_source,
    )


def metrics_to_candidate(
    metrics: PathGeometryMetrics,
    applied_constraints: tuple[str, ...] = (),
) -> RouteCandidate:
    """Convert extracted path metrics to a public RouteCandidate."""
    coverage_pct = (
        (metrics.matched_edges_count / metrics.edges_count * 100.0)
        if metrics.edges_count > 0
        else 0.0
    )
    return RouteCandidate(
        coordinates=metrics.coordinates,
        distance_meters=float(metrics.physical_distance),
        cost=float(metrics.total_cost),
        edges_count=metrics.edges_count,
        matched_traffic_edges_count=metrics.matched_edges_count,
        traffic_coverage_percent=round(coverage_pct, 1),
        average_congestion=(
            round(metrics.average_congestion, 4)
            if metrics.matched_edges_count > 0
            else None
        ),
        traffic_source=metrics.traffic_source,
        vision_matched_edges_count=metrics.vision_matched_count,
        simulated_matched_edges_count=metrics.simulated_matched_count,
        steps_count=metrics.steps_count,
        unpaved_edges_count=metrics.unpaved_count,
        obstacles_count=metrics.obstacles_count,
        traffic_cost_penalty=round(metrics.traffic_cost_penalty, 2),
        accessibility_cost_penalty=round(metrics.accessibility_cost_penalty, 2),
        accessibility_constraints_applied=applied_constraints,
    )


class TrafficAwarePathService:
    """Calculates traffic-aware pedestrian routes combining walking distance with traffic impedance."""

    def __init__(
        self,
        network_provider: PedestrianNetworkProvider | None = None,
        traffic_dataset: TrafficDataset | None = None,
        settings: NetworkAreaSettings | None = None,
        vision_adapter: VisionTrafficAdapter | None = None,
        accessibility_adapter: AccessibilityAdapter | None = None,
    ) -> None:
        self._network_provider = network_provider or get_default_network_provider()
        self._settings = settings or NetworkAreaSettings.from_environment()
        self._dataset: TrafficDataset | None = traffic_dataset
        self._dataset_error: str | None = None
        self._vision_adapter: VisionTrafficAdapter = (
            vision_adapter
            if vision_adapter is not None
            else get_default_vision_adapter()
        )
        self._accessibility_adapter: AccessibilityAdapter = (
            accessibility_adapter
            if accessibility_adapter is not None
            else get_default_accessibility_adapter()
        )

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

    @property
    def vision_adapter(self) -> VisionTrafficAdapter:
        return self._vision_adapter

    @property
    def accessibility_adapter(self) -> AccessibilityAdapter:
        return self._accessibility_adapter

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
        accessibility_mode: str = "standard",
    ) -> tuple[RouteCandidate, Coordinate, Coordinate, str, int]:
        """Calculate traffic-aware route.

        Returns:
            (candidate, snapped_origin, snapped_destination, selected_timestamp, selected_time_index)
        """
        if traffic_weight < 0:
            raise ValueError("traffic_weight must be non-negative.")
        if accessibility_mode not in ("standard", "step_free"):
            raise ValueError(
                f"Unsupported accessibility_mode '{accessibility_mode}'. Must be 'standard' or 'step_free'."
            )

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

        # Annotate edges with traffic and accessibility costs
        self._apply_traffic_weights(
            working_graph,
            selected_time_index=selected_time_index,
            traffic_weight=traffic_weight,
            accessibility_mode=accessibility_mode,
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

        applied_constraints = (
            ("avoid_stairs", "penalize_active_obstacles")
            if accessibility_mode == "step_free"
            else ()
        )

        metrics = extract_path_geometry_and_metrics(
            working_graph,
            node_path,
            edge_selection_attribute="traffic_cost",
        )
        candidate = metrics_to_candidate(metrics, applied_constraints=applied_constraints)

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
        accessibility_mode: str = "standard",
    ) -> None:
        """Assign nonnegative routing cost to every edge in the graph combining traffic and accessibility."""
        for u, v, key, data in graph.edges(keys=True, data=True):
            length = max(0.1, float(data.get("length", 1.0)))
            matched = False
            congestion = 0.0
            traffic_source = "none"

            traffic_key = TrafficEdgeKey(osm_u=int(u), osm_v=int(v), key=int(key))

            # 1. Consult VisionTrafficAdapter for a validated vision observation
            vision_congestion = None
            if self._vision_adapter is not None:
                vision_congestion = self._vision_adapter.get_congestion_index(traffic_key)

            if (
                vision_congestion is not None
                and math.isfinite(vision_congestion)
                and vision_congestion >= 0.0
            ):
                congestion = float(vision_congestion)
                matched = True
                traffic_source = "vision"
            elif self._dataset is not None and self._dataset.has_edge(traffic_key):
                # 2. Transparent fallback to simulated traffic dataset
                try:
                    snap = self._dataset.get_snapshot(traffic_key, selected_time_index)
                    raw_congestion = float(snap.congestion_index)
                    if math.isfinite(raw_congestion) and raw_congestion >= 0.0:
                        congestion = raw_congestion
                        matched = True
                        traffic_source = "simulated"
                except TrafficDatasetError:
                    congestion = 0.0
                    matched = False
                    traffic_source = "none"

            # 3. Accessibility constraints and penalties
            highway = str(data.get("highway", "")).lower()
            surface = str(data.get("surface", "")).lower()
            is_steps = (highway == "steps" or str(data.get("stairs", "")).lower() == "yes")
            is_unpaved = surface in ("unpaved", "gravel", "ground", "dirt", "grass", "sand", "earth")

            obstacle = None
            if self._accessibility_adapter is not None:
                obstacle = self._accessibility_adapter.get_valid_obstacle(traffic_key)

            accessibility_penalty = 0.0
            if accessibility_mode == "step_free":
                # Heavily penalize stairs for wheelchair / stroller / mobility users
                if is_steps:
                    accessibility_penalty += length * 500.0
                # Active verified obstacle penalty
                if obstacle is not None:
                    accessibility_penalty += length * obstacle.penalty_multiplier
            else:
                # Standard walking mode: penalize only impassable or high-difficulty obstacles
                if obstacle is not None:
                    if obstacle.severity.value == "impassable":
                        accessibility_penalty += length * 20.0
                    elif obstacle.severity.value == "high_difficulty":
                        accessibility_penalty += length * 2.0

            # Traffic penalty: Length * weight * congestion
            traffic_penalty = length * traffic_weight * congestion

            # Total cost: strictly nonnegative
            cost = length + traffic_penalty + accessibility_penalty

            data["traffic_cost"] = float(cost)
            data["traffic_matched"] = matched
            data["traffic_source"] = traffic_source
            data["traffic_congestion"] = float(congestion)
            data["traffic_penalty"] = float(traffic_penalty)
            data["is_steps"] = is_steps
            data["is_unpaved"] = is_unpaved
            data["has_obstacle"] = (obstacle is not None)
            data["obstacle_type"] = obstacle.obstacle_type.value if obstacle else None
            data["accessibility_penalty"] = float(accessibility_penalty)


class TrafficRouteComparisonService:
    """Service to compute and compare shortest walking routes with traffic-aware alternatives."""

    def __init__(
        self,
        network_provider: PedestrianNetworkProvider | None = None,
        traffic_dataset: TrafficDataset | None = None,
        settings: NetworkAreaSettings | None = None,
        vision_adapter: VisionTrafficAdapter | None = None,
        accessibility_adapter: AccessibilityAdapter | None = None,
    ) -> None:
        self._network_provider = network_provider or get_default_network_provider()
        self._settings = settings or NetworkAreaSettings.from_environment()
        self._vision_adapter: VisionTrafficAdapter = (
            vision_adapter
            if vision_adapter is not None
            else get_default_vision_adapter()
        )
        self._accessibility_adapter: AccessibilityAdapter = (
            accessibility_adapter
            if accessibility_adapter is not None
            else get_default_accessibility_adapter()
        )
        self._traffic_service = TrafficAwarePathService(
            network_provider=self._network_provider,
            traffic_dataset=traffic_dataset,
            settings=self._settings,
            vision_adapter=self._vision_adapter,
            accessibility_adapter=self._accessibility_adapter,
        )

    @property
    def vision_adapter(self) -> VisionTrafficAdapter:
        return self._vision_adapter

    @property
    def accessibility_adapter(self) -> AccessibilityAdapter:
        return self._accessibility_adapter

    def compare(
        self,
        origin: Coordinate,
        destination: Coordinate,
        timestamp: str | None = None,
        time_index: int | None = None,
        traffic_weight: float = 2.0,
        accessibility_mode: str = "standard",
    ) -> RouteComparison:
        """Calculate and compare shortest distance and traffic-aware routes."""
        if traffic_weight < 0:
            raise ValueError("traffic_weight must be non-negative.")
        if accessibility_mode not in ("standard", "step_free"):
            raise ValueError(
                f"Unsupported accessibility_mode '{accessibility_mode}'. Must be 'standard' or 'step_free'."
            )

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
        traffic_weighting_failed = False
        try:
            self._traffic_service._apply_traffic_weights(
                working_graph,
                selected_time_index=selected_time_index,
                traffic_weight=traffic_weight,
                accessibility_mode=accessibility_mode,
            )
        except Exception:
            traffic_weighting_failed = True

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
        except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
            raise RouteNotFoundError(
                "No pedestrian path connects the selected locations."
            ) from error
        except (ValueError, KeyError, TypeError) as error:
            raise RouteNetworkUnavailableError(
                "The pedestrian network data could not be processed."
            ) from error

        # 2. Traffic-aware path (weight="traffic_cost") with graceful partial fallback
        traffic_routing_failed = traffic_weighting_failed
        if not traffic_weighting_failed:
            try:
                traffic_node_path = nx.shortest_path(
                    working_graph,
                    source=origin_node,
                    target=destination_node,
                    weight="traffic_cost",
                    method="dijkstra",
                )
            except Exception:
                traffic_routing_failed = True
                traffic_node_path = shortest_node_path
        else:
            traffic_node_path = shortest_node_path

        applied_constraints = (
            ("avoid_stairs", "penalize_active_obstacles")
            if accessibility_mode == "step_free"
            else ()
        )

        # Extract shortest route metrics under the same traffic conditions
        shortest_metrics = extract_path_geometry_and_metrics(
            working_graph,
            shortest_node_path,
            edge_selection_attribute="length",
        )
        shortest_candidate = metrics_to_candidate(
            shortest_metrics, applied_constraints=()
        )

        # Extract traffic-aware route metrics
        traffic_metrics = extract_path_geometry_and_metrics(
            working_graph,
            traffic_node_path,
            edge_selection_attribute="traffic_cost",
        )
        traffic_candidate = metrics_to_candidate(
            traffic_metrics, applied_constraints=applied_constraints
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
        if traffic_routing_failed:
            status = "partial_shortest_only"
        elif shortest_node_path != traffic_node_path:
            status = "alternative_found"
        elif (
            self._traffic_service.dataset is None
            and traffic_candidate.matched_traffic_edges_count == 0
            and shortest_candidate.matched_traffic_edges_count == 0
            and accessibility_mode == "standard"
            and not shortest_candidate.obstacles_count
        ):
            status = "traffic_dataset_unavailable"
        elif (
            traffic_candidate.matched_traffic_edges_count == 0
            and shortest_candidate.matched_traffic_edges_count == 0
            and accessibility_mode == "standard"
            and not shortest_candidate.obstacles_count
        ):
            status = "no_traffic_coverage_in_area"
        elif shortest_node_path == traffic_node_path:
            status = "identical_routes"
        else:
            status = "alternative_found"


        # Determine overall primary traffic source across evaluated routes
        has_vision = (
            traffic_candidate.vision_matched_edges_count > 0
            or shortest_candidate.vision_matched_edges_count > 0
        )
        has_simulated = (
            traffic_candidate.simulated_matched_edges_count > 0
            or shortest_candidate.simulated_matched_edges_count > 0
        )

        if has_vision and has_simulated:
            primary_traffic_source = "mixed"
        elif has_vision:
            primary_traffic_source = "vision"
        elif has_simulated:
            primary_traffic_source = "simulated"
        else:
            primary_traffic_source = "none"

        vision_active = bool(
            self._vision_adapter
            and len(self._vision_adapter.get_active_observations()) > 0
        )

        # Build clear explainability explanation
        explanation_parts = []
        if shortest_node_path == traffic_node_path:
            explanation_parts.append(
                "The shortest walking route is also recommended under current traffic and accessibility conditions."
            )
        else:
            detour_meters = round(
                traffic_candidate.distance_meters - shortest_candidate.distance_meters, 1
            )
            reasons = []
            if (
                shortest_candidate.steps_count > 0
                and traffic_candidate.steps_count == 0
                and accessibility_mode == "step_free"
            ):
                reasons.append("avoids pedestrian staircases")
            if (
                shortest_candidate.obstacles_count > 0
                and traffic_candidate.obstacles_count == 0
            ):
                reasons.append("bypasses verified physical barriers")
            if (
                shortest_candidate.average_congestion or 0.0
            ) > (traffic_candidate.average_congestion or 0.0):
                reasons.append("avoids vehicle traffic congestion")
            if not reasons:
                reasons.append("optimizes total pedestrian travel cost")
            explanation_parts.append(
                f"An alternative pedestrian route was selected (detour: +{detour_meters}m) because it {' and '.join(reasons)}."
            )

        # Source attribution
        if (
            traffic_candidate.vision_matched_edges_count > 0
            or shortest_candidate.vision_matched_edges_count > 0
        ):
            v_count = max(
                traffic_candidate.vision_matched_edges_count,
                shortest_candidate.vision_matched_edges_count,
            )
            explanation_parts.append(
                f"Validated computer vision traffic observation active on {v_count} corridor segment(s)."
            )
        elif (
            traffic_candidate.simulated_matched_edges_count > 0
            or shortest_candidate.simulated_matched_edges_count > 0
        ):
            explanation_parts.append(
                "Traffic impedance calculated from simulated Electronic City traffic dataset fallback."
            )
        else:
            explanation_parts.append("No traffic observations matched this path corridor.")

        if accessibility_mode == "step_free":
            explanation_parts.append(
                "Step-free mobility constraints applied (stairs and verified obstacles penalized)."
            )

        explanation = " ".join(explanation_parts)

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
            primary_traffic_source=primary_traffic_source,
            vision_observation_active=vision_active,
            accessibility_mode=accessibility_mode,
            explanation=explanation,
        )
