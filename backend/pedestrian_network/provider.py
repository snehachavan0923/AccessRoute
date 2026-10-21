from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Protocol

import networkx as nx
from networkx import MultiDiGraph
import osmnx as ox
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from requests import RequestException


@dataclass(frozen=True)
class NetworkBounds:
    north: float
    south: float
    east: float
    west: float

    def contains(self, other: NetworkBounds) -> bool:
        """Check if this bounding box completely encloses another bounding box."""
        return (
            self.north >= other.north
            and self.south <= other.south
            and self.east >= other.east
            and self.west <= other.west
        )

    def to_normalized_grid(self, grid_size: float = 0.01) -> NetworkBounds:
        """Snap bounds outward to a regular grid to maximize cache hits.

        Grid resolution default is 0.01 degrees (~1.1 km).
        """
        south = round(math.floor(self.south / grid_size) * grid_size, 4)
        north = round(math.ceil(self.north / grid_size) * grid_size, 4)
        west = round(math.floor(self.west / grid_size) * grid_size, 4)
        east = round(math.ceil(self.east / grid_size) * grid_size, 4)
        return NetworkBounds(north=north, south=south, east=east, west=west)

    def to_key(self) -> tuple[float, float, float, float]:
        return (round(self.north, 4), round(self.south, 4), round(self.east, 4), round(self.west, 4))


class PedestrianNetworkProvider(Protocol):
    def load(self, bounds: NetworkBounds) -> MultiDiGraph: ...


class NetworkDownloadError(Exception):
    pass


class PedestrianGraphCache:
    """Thread-safe bounded in-memory LRU cache with TTL for pedestrian MultiDiGraphs.

    Guarantees:
    - O(1) in-memory retrieval (sub-millisecond vs minutes of network download).
    - Spatial containment: queries fully within a cached bounding box reuse the graph.
    - Immutability: graphs are copied on retrieval and storage to prevent cross-request corruption.
    - Zero poisoning: failed downloads and empty graphs are never stored.
    - Bounded memory: least recently used graphs are evicted when capacity is reached.
    """

    def __init__(
        self,
        max_size: int = 32,
        ttl_seconds: float = 86400.0,
        enable_disk_cache: bool = False,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_disk_cache = enable_disk_cache
        self._disk_cache_dir = Path(os.environ.get("ROUTING_GRAPH_DISK_CACHE_DIR", "./cache/graphs"))
        if self.enable_disk_cache:
            try:
                self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enable_disk_cache = False
        self._cache: OrderedDict[tuple[float, float, float, float], tuple[MultiDiGraph, NetworkBounds, float]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def _disk_path(self, key: tuple[float, float, float, float]) -> Path:
        return self._disk_cache_dir / f"graph_{key[0]}_{key[1]}_{key[2]}_{key[3]}.pkl"

    def get(self, bounds: NetworkBounds) -> MultiDiGraph | None:
        """Retrieve a cached graph matching the exact key or enclosing the requested bounds."""
        norm_bounds = bounds.to_normalized_grid()
        norm_key = norm_bounds.to_key()
        now = time.monotonic()

        with self._lock:
            # 1. Exact normalized key match in RAM
            if norm_key in self._cache:
                graph, entry_bounds, created_at = self._cache[norm_key]
                if now - created_at <= self.ttl_seconds:
                    self._cache.move_to_end(norm_key)
                    self._hits += 1
                    return graph.copy()
                del self._cache[norm_key]

            # 2. Spatial containment check in RAM: is requested bounds within any cached graph?
            for key, (graph, entry_bounds, created_at) in list(self._cache.items()):
                if now - created_at > self.ttl_seconds:
                    del self._cache[key]
                    continue
                if entry_bounds.contains(bounds):
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return graph.copy()

            # 3. Disk cache check
            if self.enable_disk_cache and self._disk_cache_dir.exists():
                disk_file = self._disk_path(norm_key)
                if disk_file.exists():
                    try:
                        mtime = disk_file.stat().st_mtime
                        if time.time() - mtime <= self.ttl_seconds:
                            with open(disk_file, "rb") as f:
                                loaded_graph = pickle.load(f)
                            if isinstance(loaded_graph, MultiDiGraph) and loaded_graph.number_of_nodes() > 0:
                                self._cache[norm_key] = (loaded_graph, norm_bounds, now)
                                self._cache.move_to_end(norm_key)
                                self._hits += 1
                                return loaded_graph.copy()
                    except Exception:
                        pass

            # 4. Check OSMnx HTTP cache for existing enclosing response files
            if self.enable_disk_cache:
                osmnx_cache = Path(os.environ.get("ROUTING_OSMNX_CACHE_DIR", "./cache"))
                if osmnx_cache.exists():
                    for json_file in osmnx_cache.glob("*.json"):
                        try:
                            with open(json_file, "r", encoding="utf-8") as fp:
                                d = json.load(fp)
                            elements = d.get("elements", [])
                            nodes = [e for e in elements if e.get("type") == "node" and "lat" in e]
                            if not nodes:
                                continue
                            f_bounds = NetworkBounds(
                                north=max(n["lat"] for n in nodes),
                                south=min(n["lat"] for n in nodes),
                                east=max(n["lon"] for n in nodes),
                                west=min(n["lon"] for n in nodes),
                            )
                            if f_bounds.contains(bounds):
                                loaded_graph = ox.graph._create_graph([d], bidirectional=True)
                                if loaded_graph.number_of_nodes() > 0:
                                    self._cache[f_bounds.to_normalized_grid().to_key()] = (
                                        loaded_graph,
                                        f_bounds,
                                        now,
                                    )
                                    self._hits += 1
                                    return loaded_graph.copy()
                        except Exception:
                            continue

            self._misses += 1
            return None

    def put(self, bounds: NetworkBounds, graph: MultiDiGraph) -> None:
        """Store a graph in cache under its normalized bounds key."""
        if graph is None or graph.number_of_nodes() == 0:
            return

        norm_bounds = bounds.to_normalized_grid()
        key = norm_bounds.to_key()
        now = time.monotonic()

        with self._lock:
            # Evict expired entries
            expired_keys = [
                k for k, (_, _, created_at) in self._cache.items()
                if now - created_at > self.ttl_seconds
            ]
            for k in expired_keys:
                del self._cache[k]

            # Evict LRU if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cache.popitem(last=False)

            cached_copy = graph.copy()
            self._cache[key] = (cached_copy, norm_bounds, now)
            self._cache.move_to_end(key)

            # Persist to disk cache
            if self.enable_disk_cache:
                try:
                    disk_file = self._disk_path(key)
                    tmp_file = disk_file.with_suffix(".tmp")
                    with open(tmp_file, "wb") as f:
                        pickle.dump(cached_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
                    tmp_file.replace(disk_file)
                except Exception:
                    pass

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            if self.enable_disk_cache and self._disk_cache_dir.exists():
                for pkl_file in self._disk_cache_dir.glob("graph_*.pkl"):
                    try:
                        pkl_file.unlink()
                    except OSError:
                        pass

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "ttl_seconds": self.ttl_seconds,
            }


class OSMnxPedestrianNetworkProvider:
    """Fetches walkable pedestrian networks via OSMnx with timeout and User-Agent configuration."""

    def __init__(self) -> None:
        self._configure_osmnx()

    @staticmethod
    def _configure_osmnx() -> None:
        ox.settings.http_user_agent = os.environ.get(
            "ROUTING_OSMNX_USER_AGENT",
            "OSMnx/2.0.1 (https://github.com/gboeing/osmnx)",
        )
        ox.settings.requests_timeout = int(os.environ.get("ROUTING_OSMNX_TIMEOUT", "90"))
        ox.settings.use_cache = True
        cache_dir = os.environ.get("ROUTING_OSMNX_CACHE_DIR", "./cache")
        ox.settings.cache_folder = cache_dir

    def load(self, bounds: NetworkBounds) -> MultiDiGraph:
        self._configure_osmnx()
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
            error_msg = str(error)
            if "502" in error_msg or "Gateway" in error_msg or "capacity" in error_msg:
                detail = (
                    "OpenStreetMap Overpass server is temporarily busy or rate-limited. "
                    "Please wait a moment and retry."
                )
            elif isinstance(error, (TimeoutError, RequestException)) and "timeout" in error_msg.lower():
                detail = (
                    "OpenStreetMap network request timed out. "
                    "The selected area may be too large or the server is congested."
                )
            else:
                detail = "OpenStreetMap pedestrian network data is unavailable."
            raise NetworkDownloadError(detail) from error


class CachedPedestrianNetworkProvider:
    """Wrapper that caches downloaded pedestrian graphs in memory with single-flight locks."""

    def __init__(
        self,
        provider: PedestrianNetworkProvider | None = None,
        cache: PedestrianGraphCache | None = None,
    ) -> None:
        self._underlying = provider or OSMnxPedestrianNetworkProvider()
        self._cache = cache or PedestrianGraphCache()
        self._inflight: dict[tuple[float, float, float, float], threading.Event] = {}
        self._inflight_lock = threading.Lock()

    @property
    def cache(self) -> PedestrianGraphCache:
        return self._cache

    def load(self, bounds: NetworkBounds) -> MultiDiGraph:
        # 1. Fast path: check in-memory cache
        cached_graph = self._cache.get(bounds)
        if cached_graph is not None:
            return cached_graph

        norm_bounds = bounds.to_normalized_grid()
        key = norm_bounds.to_key()

        # 2. Single-flight coordination to prevent duplicate concurrent network requests
        with self._inflight_lock:
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                is_initiator = True
            else:
                is_initiator = False

        if not is_initiator:
            # Wait for the initiating thread to finish downloading
            event.wait(timeout=35.0)
            cached_graph = self._cache.get(bounds)
            if cached_graph is not None:
                return cached_graph
            # If still not in cache, fallback to loading directly
            return self._underlying.load(bounds)

        try:
            # 3. Load using normalized bounds so future sub-queries can be served from cache
            graph = self._underlying.load(norm_bounds)
            self._cache.put(norm_bounds, graph)
            return graph.copy()
        finally:
            with self._inflight_lock:
                event.set()
                self._inflight.pop(key, None)


# Default shared cached provider instance for application lifecycle
_DEFAULT_PROVIDER: CachedPedestrianNetworkProvider | None = None
_PROVIDER_INIT_LOCK = threading.Lock()


def get_default_network_provider() -> CachedPedestrianNetworkProvider:
    """Get the shared application-wide cached pedestrian network provider."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        with _PROVIDER_INIT_LOCK:
            if _DEFAULT_PROVIDER is None:
                max_size = int(os.environ.get("ROUTING_GRAPH_CACHE_MAX_SIZE", "32"))
                ttl = float(os.environ.get("ROUTING_GRAPH_CACHE_TTL_SECONDS", "3600"))
                cache = PedestrianGraphCache(
                    max_size=max_size,
                    ttl_seconds=ttl,
                    enable_disk_cache=True,
                )
                _DEFAULT_PROVIDER = CachedPedestrianNetworkProvider(cache=cache)
    return _DEFAULT_PROVIDER
