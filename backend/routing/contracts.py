from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Coordinate:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class RouteCandidate:
    coordinates: tuple[Coordinate, ...]
    distance_meters: float


@dataclass(frozen=True)
class RouteComparison:
    shortest_distance: RouteCandidate
    traffic_accessibility_aware: RouteCandidate


@dataclass(frozen=True)
class ShortestRoute:
    coordinates: tuple[Coordinate, ...]
    distance_meters: float
    snapped_origin: Coordinate
    snapped_destination: Coordinate


class RouteComparisonService(Protocol):
    def compare(
        self,
        origin: Coordinate,
        destination: Coordinate,
    ) -> RouteComparison: ...
