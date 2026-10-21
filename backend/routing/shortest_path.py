from dataclasses import dataclass
from math import cos, hypot, isfinite, radians, sin
from os import environ

import networkx as nx

from backend.pedestrian_network import (
    NetworkBounds,
    NetworkDownloadError,
    OSMnxPedestrianNetworkProvider,
    PedestrianNetworkProvider,
    get_default_network_provider,
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
    max_diagonal_meters: float = 20_000

    def __post_init__(self) -> None:
        if (
            not isfinite(self.buffer_meters)
            or self.buffer_meters <= 0
            or not isfinite(self.max_diagonal_meters)
            or self.max_diagonal_meters <= 0
        ):
            raise ValueError("Network area settings must be finite and greater than zero.")

    @classmethod
    def from_environment(cls) -> "NetworkAreaSettings":
        return cls(
            buffer_meters=float(environ.get("ROUTING_NETWORK_BUFFER_METERS", "1000")),
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
    latitude_padding = settings.buffer_meters / METERS_PER_DEGREE_LATITUDE
    north = min(90, max(latitudes) + latitude_padding)
    south = max(-90, min(latitudes) - latitude_padding)

    longitude_reference = max(abs(south), abs(north))
    meters_per_degree_longitude = METERS_PER_DEGREE_LATITUDE * max(
        cos(radians(longitude_reference)),
        0.01,
    )
    longitude_padding = settings.buffer_meters / meters_per_degree_longitude
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


def nearest_network_node(
    graph: nx.MultiDiGraph,
    coordinate: Coordinate,
) -> int:
    latitude_radians = radians(coordinate.latitude)
    longitude_radians = radians(coordinate.longitude)

    def distance_squared(node_data: dict[str, float]) -> float:
        node_latitude = radians(float(node_data["y"]))
        node_longitude = radians(float(node_data["x"]))
        latitude_delta = node_latitude - latitude_radians
        longitude_delta = node_longitude - longitude_radians
        return (
            sin(latitude_delta / 2) ** 2
            + cos(latitude_radians)
            * cos(node_latitude)
            * sin(longitude_delta / 2) ** 2
        )

    nearest_node, _ = min(
        graph.nodes(data=True),
        key=lambda item: distance_squared(item[1]),
    )
    return nearest_node


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
            origin_node = nearest_network_node(graph, origin)
            destination_node = nearest_network_node(graph, destination)
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
