from dataclasses import dataclass
from typing import Protocol

import osmnx as ox
from networkx import MultiDiGraph
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from requests import RequestException


@dataclass(frozen=True)
class NetworkBounds:
    north: float
    south: float
    east: float
    west: float


class PedestrianNetworkProvider(Protocol):
    def load(self, bounds: NetworkBounds) -> MultiDiGraph: ...


class NetworkDownloadError(Exception):
    pass


class OSMnxPedestrianNetworkProvider:
    def load(self, bounds: NetworkBounds) -> MultiDiGraph:
        try:
            return ox.graph_from_bbox(
                (bounds.west, bounds.south, bounds.east, bounds.north),
                network_type="walk",
            )
        except (
            InsufficientResponseError,
            ResponseStatusCodeError,
            RequestException,
            TimeoutError,
        ) as error:
            raise NetworkDownloadError(
                "OpenStreetMap pedestrian network data is unavailable."
            ) from error
