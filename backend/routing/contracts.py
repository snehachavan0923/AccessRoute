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
    cost: float = 0.0
    edges_count: int = 0
    matched_traffic_edges_count: int = 0
    traffic_coverage_percent: float = 0.0
    average_congestion: float | None = None


@dataclass(frozen=True)
class RouteComparison:
    shortest_distance: RouteCandidate
    traffic_accessibility_aware: RouteCandidate
    snapped_origin: Coordinate | None = None
    snapped_destination: Coordinate | None = None
    selected_timestamp: str = ""
    time_index: int = 0
    cost_formula: str = ""
    status: str = "success"
    disclaimer: str = (
        "Traffic values are synthetically simulated for Electronic City, Bengaluru, "
        "and do not represent real-time sensor measurements or guaranteed pedestrian safety."
    )


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
        timestamp: str | None = None,
        time_index: int | None = None,
        traffic_weight: float = 2.0,
    ) -> RouteComparison: ...

