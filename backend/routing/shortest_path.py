from dataclasses import dataclass
import math
from math import cos, hypot, isfinite, radians, sin
from os import environ

import networkx as nx

from backend.pedestrian_network import (
    NetworkBounds,
    NetworkDownloadError,
    OSMnxPedestrianNetworkProvider,
    PedestrianNetworkProvider,
    get_default_network_provider,
    is_inaccessible_elevated_motor_edge,
)
from backend.routing.contracts import Coordinate, ShortestRoute

METERS_PER_DEGREE_LATITUDE = 111_320


class NetworkAreaTooLargeError(ValueError):
    pass


class RouteNetworkUnavailableError(RuntimeError):
    pass


class RouteNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class NetworkAreaSettings:
    buffer_meters: float = 1_000
    min_buffer_meters: float = 350.0
    adaptive: bool = True
    max_diagonal_meters: float = 20_000

    def __post_init__(self) -> None:
        if (
            not isfinite(self.buffer_meters)
            or self.buffer_meters <= 0
            or not isfinite(self.min_buffer_meters)
            or self.min_buffer_meters <= 0
            or not isfinite(self.max_diagonal_meters)
            or self.max_diagonal_meters <= 0
        ):
            raise ValueError("Network area settings must be finite and greater than zero.")

    @classmethod
    def from_environment(cls) -> "NetworkAreaSettings":
        return cls(
            buffer_meters=float(environ.get("ROUTING_NETWORK_BUFFER_METERS", "1000")),
            min_buffer_meters=float(environ.get("ROUTING_MIN_NETWORK_BUFFER_METERS", "350")),
            adaptive=environ.get("ROUTING_ADAPTIVE_BUFFER", "true").lower() in ("true", "1", "yes"),
            max_diagonal_meters=float(
                environ.get("ROUTING_MAX_NETWORK_DIAGONAL_METERS", "20000")
            ),
        )


def calculate_network_bounds(
    origin: Coordinate,
    destination: Coordinate,
    settings: NetworkAreaSettings,
) -> NetworkBounds:
    latitudes = (origin.latitude, destination.latitude)
    longitudes = (origin.longitude, destination.longitude)

    if settings.adaptive:
        # Scale buffer with walking distance to avoid downloading massive areas for short walks
        avg_lat = radians((origin.latitude + destination.latitude) / 2.0)
        meters_per_deg_lon = METERS_PER_DEGREE_LATITUDE * max(cos(avg_lat), 0.01)
        delta_y = (destination.latitude - origin.latitude) * METERS_PER_DEGREE_LATITUDE
        delta_x = (destination.longitude - origin.longitude) * meters_per_deg_lon
        straight_line_dist = hypot(delta_x, delta_y)

        effective_buffer = max(
            settings.min_buffer_meters,
            min(settings.buffer_meters, settings.min_buffer_meters + straight_line_dist * 0.25),
        )
    else:
        effective_buffer = settings.buffer_meters

    latitude_padding = effective_buffer / METERS_PER_DEGREE_LATITUDE
    north = min(90.0, max(latitudes) + latitude_padding)
    south = max(-90.0, min(latitudes) - latitude_padding)

    longitude_reference = max(abs(south), abs(north))
    meters_per_degree_longitude = METERS_PER_DEGREE_LATITUDE * max(
        cos(radians(longitude_reference)),
        0.01,
    )
    longitude_padding = effective_buffer / meters_per_degree_longitude
    west = min(longitudes) - longitude_padding
    east = max(longitudes) + longitude_padding

    if west < -180 or east > 180:
        raise NetworkAreaTooLargeError(
            "The requested locations are too close to the international date line "
            "for a safe network download."
        )

    width_meters = (east - west) * meters_per_degree_longitude
    height_meters = (north - south) * METERS_PER_DEGREE_LATITUDE
    if hypot(width_meters, height_meters) > settings.max_diagonal_meters:
        raise NetworkAreaTooLargeError(
            "The requested locations are too far apart for a bounded pedestrian "
            "network download."
        )

    return NetworkBounds(north=north, south=south, east=east, west=west)


WALKABLE_HIGHWAYS = frozenset({
    "footway",
    "pedestrian",
    "path",
    "steps",
    "living_street",
    "residential",
    "service",
    "cycleway",
    "track",
    "unclassified",
    "secondary",
    "tertiary",
    "secondary_link",
    "tertiary_link",
})

PEDESTRIAN_EXPLICIT_ACCESS_VALUES = frozenset({
    "yes",
    "designated",
    "permissive",
    "both",
    "left",
    "right",
    "separate",
})


def is_node_pedestrian_preferred(graph: nx.MultiDiGraph, node: int) -> bool:
    """Evaluates whether a network node has appropriate pedestrian access.

    Returns True if the node is connected to walkable ground-level ways or
    pedestrian-accessible bridges (footbridges, sidewalks), and False if it is
    isolated or exclusively connected to inaccessible elevated motor-road spans.
    """
    out_edges = graph.edges(node, data=True)
    in_edges = graph.in_edges(node, data=True)
    all_edges = list(out_edges) + list(in_edges)
    if not all_edges:
        return False

    has_elevated_inaccessible = False
    has_walkable = False

    for _, _, d in all_edges:
        if is_inaccessible_elevated_motor_edge(d):
            has_elevated_inaccessible = True
            continue

        hw = d.get("highway")
        hws = hw if isinstance(hw, list) else [hw] if hw else []
        foot = str(d.get("foot", "")).lower()
        sidewalk = str(d.get("sidewalk", "")).lower()
        pedestrian = str(d.get("pedestrian", "")).lower()

        has_explicit_foot = (
            foot in PEDESTRIAN_EXPLICIT_ACCESS_VALUES
            or sidewalk in PEDESTRIAN_EXPLICIT_ACCESS_VALUES
            or pedestrian in PEDESTRIAN_EXPLICIT_ACCESS_VALUES
        )
        is_dedicated_foot = any(h in {"footway", "pedestrian", "path", "steps"} for h in hws)

        if has_explicit_foot or is_dedicated_foot or any(h in WALKABLE_HIGHWAYS for h in hws):
            has_walkable = True

    if has_walkable and not has_elevated_inaccessible:
        return True
    if has_walkable:
        return True
    if not has_elevated_inaccessible:
        return True
    return False


def nearest_network_node(
    graph: nx.MultiDiGraph,
    coordinate: Coordinate,
    search_radius_meters: float = 100.0,
    target_component: set[int] | None = None,
) -> int:
    """Finds the nearest network node to the given coordinate.

    Prefers ground-level pedestrian nodes over inaccessible elevated flyovers
    within search_radius_meters. Also prioritizes nodes belonging to major
    connected components (or a specified target_component) to avoid snapping
    onto isolated dead-end campus spurs or disconnected clusters. Preserves
    explicitly pedestrian-accessible bridges, footbridges, and mapped pedestrian
    connections.
    """
    if graph.number_of_nodes() == 0:
        raise RouteNetworkUnavailableError("The pedestrian network is empty.")

    latitude_radians = radians(coordinate.latitude)
    longitude_radians = radians(coordinate.longitude)

    def haversine_distance_m(node_data: dict[str, float]) -> float:
        node_lat = radians(float(node_data["y"]))
        node_lon = radians(float(node_data["x"]))
        lat_delta = node_lat - latitude_radians
        lon_delta = node_lon - longitude_radians
        a = (
            sin(lat_delta / 2) ** 2
            + cos(latitude_radians) * cos(node_lat) * sin(lon_delta / 2) ** 2
        )
        return 2 * 6_371_000 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    node_distances: list[tuple[float, int, dict[str, float]]] = []
    for node, data in graph.nodes(data=True):
        dist = haversine_distance_m(data)
        node_distances.append((dist, node, data))

    node_distances.sort(key=lambda item: item[0])

    # Analyze connected components to detect major subnetworks
    undir = graph.to_undirected(as_view=True)
    components = list(nx.connected_components(undir))
    largest_size = max((len(c) for c in components), default=0)
    major_threshold = max(20, int(largest_size * 0.05)) if largest_size >= 20 else 1
    major_nodes: set[int] = (
        set().union(*[c for c in components if len(c) >= major_threshold])
        if components
        else set()
    )

    preferred_component_nodes = target_component if target_component is not None else major_nodes
    extended_radius_meters = max(search_radius_meters, 350.0)

    # Buckets for candidate selection
    p1_preferred_in_component: list[tuple[float, int]] = []
    p2_connected_in_component: list[tuple[float, int]] = []
    p3_preferred_ext_component: list[tuple[float, int]] = []
    p4_connected_ext_component: list[tuple[float, int]] = []
    p5_preferred_in_radius: list[tuple[float, int]] = []
    p6_connected_in_radius: list[tuple[float, int]] = []

    for dist, node, _ in node_distances:
        if graph.degree(node) == 0:
            continue
        pref = is_node_pedestrian_preferred(graph, node)
        in_comp = node in preferred_component_nodes

        if dist <= search_radius_meters:
            p6_connected_in_radius.append((dist, node))
            if pref:
                p5_preferred_in_radius.append((dist, node))
            if in_comp:
                p2_connected_in_component.append((dist, node))
                if pref:
                    p1_preferred_in_component.append((dist, node))

        if dist <= extended_radius_meters:
            if in_comp:
                p4_connected_ext_component.append((dist, node))
                if pref:
                    p3_preferred_ext_component.append((dist, node))
        else:
            break

    if p1_preferred_in_component:
        return p1_preferred_in_component[0][1]

    if p2_connected_in_component:
        return p2_connected_in_component[0][1]

    if p3_preferred_ext_component:
        return p3_preferred_ext_component[0][1]

    if p4_connected_ext_component:
        return p4_connected_ext_component[0][1]

    if p5_preferred_in_radius:
        return p5_preferred_in_radius[0][1]

    if p6_connected_in_radius:
        return p6_connected_in_radius[0][1]

    # If no candidates matched above, search whole graph for component node, then any connected node
    for dist, node, _ in node_distances:
        if graph.degree(node) > 0 and node in preferred_component_nodes:
            return node

    for dist, node, _ in node_distances:
        if graph.degree(node) > 0:
            return node

    # Final fallback to closest node overall
    return node_distances[0][1]


def build_route_coordinates(
    graph: nx.MultiDiGraph,
    node_path: list[int],
) -> tuple[Coordinate, ...]:
    coordinates: list[Coordinate] = []

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
        edge = min(edges.values(), key=lambda data: data.get("length", 1))
        geometry = edge.get("geometry")
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

    return tuple(coordinates)


class ShortestPathService:
    def __init__(
        self,
        network_provider: PedestrianNetworkProvider | None = None,
        settings: NetworkAreaSettings | None = None,
    ) -> None:
        self._network_provider = network_provider or get_default_network_provider()
        self._settings = settings or NetworkAreaSettings.from_environment()

    def calculate(
        self,
        origin: Coordinate,
        destination: Coordinate,
    ) -> ShortestRoute:
        bounds = calculate_network_bounds(origin, destination, self._settings)
        try:
            graph = self._network_provider.load(bounds)
        except NetworkDownloadError as error:
            raise RouteNetworkUnavailableError(str(error)) from error

        if graph.number_of_nodes() == 0:
            raise RouteNetworkUnavailableError(
                "No walkable pedestrian network was found for the selected area."
            )

        try:
            undir = graph.to_undirected(as_view=True)
            origin_node = nearest_network_node(graph, origin)
            orig_comp = nx.node_connected_component(undir, origin_node)
            destination_node = nearest_network_node(
                graph, destination, target_component=orig_comp
            )
            node_path = nx.shortest_path(
                graph,
                source=origin_node,
                target=destination_node,
                weight="length",
                method="dijkstra",
            )
            distance = nx.shortest_path_length(
                graph,
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

        try:
            coordinates = build_route_coordinates(graph, node_path)
        except (KeyError, TypeError, ValueError) as error:
            raise RouteNetworkUnavailableError(
                "The pedestrian route geometry could not be processed."
            ) from error

        origin_node_data = graph.nodes[origin_node]
        destination_node_data = graph.nodes[destination_node]
        return ShortestRoute(
            coordinates=coordinates,
            distance_meters=float(distance),
            snapped_origin=Coordinate(
                latitude=float(origin_node_data["y"]),
                longitude=float(origin_node_data["x"]),
            ),
            snapped_destination=Coordinate(
                latitude=float(destination_node_data["y"]),
                longitude=float(destination_node_data["x"]),
            ),
        )
