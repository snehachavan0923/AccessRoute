from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


class TrafficDatasetError(Exception):
    """Raised when traffic dataset files cannot be found, loaded, or validated."""
    pass


@dataclass(frozen=True)
class TrafficEdgeKey:
    """Unique identity for a directed road edge (osm_u, osm_v, key)."""
    osm_u: int
    osm_v: int
    key: int = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.osm_u, self.osm_v, self.key)

    @classmethod
    def from_tuple(cls, edge_tuple: Sequence[int]) -> TrafficEdgeKey:
        if len(edge_tuple) == 2:
            return cls(osm_u=int(edge_tuple[0]), osm_v=int(edge_tuple[1]), key=0)
        if len(edge_tuple) == 3:
            return cls(
                osm_u=int(edge_tuple[0]),
                osm_v=int(edge_tuple[1]),
                key=int(edge_tuple[2]),
            )
        raise ValueError(
            f"Expected tuple of length 2 or 3 for edge key, got {edge_tuple}"
        )


@dataclass(frozen=True)
class TrafficSnapshot:
    """Point-in-time traffic metrics for a single directed road edge."""
    speed_kmh: float
    vehicle_count: float
    congestion_index: float
    timestamp: str | None = None
    time_index: int | None = None

    @property
    def speed(self) -> float:
        return self.speed_kmh

    @property
    def vehicles(self) -> float:
        return self.vehicle_count

    @property
    def congestion(self) -> float:
        return self.congestion_index


@dataclass(frozen=True)
class TrafficDatasetInfo:
    """Summary of dataset dimensions, coverage, and provenance."""
    total_nodes: int
    total_edges: int
    total_timestamps: int
    start_time: str
    end_time: str
    time_step_minutes: int = 5
    is_simulated: bool = True
    area_name: str = "Electronic City, Bengaluru"
    description: str = (
        "Simulated traffic time-series and road-network dataset "
        "for Electronic City, Bengaluru."
    )


class TrafficDataset:
    """Loads and provides query access to the Electronic City traffic dataset.

    Traffic values are stored in 2D NumPy matrices of shape (num_timestamps, num_edges),
    indexed by time row and edge column.
    """

    DEFAULT_ENV_VAR = "TRAFFIC_DATASET_DIR"
    DEFAULT_PATH = Path("~/Downloads/electronic-city-traffic-clean").expanduser()

    def __init__(
        self,
        *,
        info: TrafficDatasetInfo,
        timestamps: list[str],
        edge_key_to_id: dict[TrafficEdgeKey, int],
        id_to_edge_key: dict[int, TrafficEdgeKey],
        node_ids: set[int],
        speed_matrix: np.ndarray,
        vehicle_matrix: np.ndarray,
        congestion_matrix: np.ndarray,
    ) -> None:
        self._info = info
        self._timestamps = tuple(timestamps)
        self._timestamp_to_index: dict[str, int] = {
            ts: idx for idx, ts in enumerate(timestamps)
        }
        self._edge_key_to_id = edge_key_to_id
        self._id_to_edge_key = id_to_edge_key
        self._node_ids = frozenset(node_ids)
        self._edge_keys = frozenset(edge_key_to_id.keys())
        self._speed_matrix = speed_matrix
        self._vehicle_matrix = vehicle_matrix
        self._congestion_matrix = congestion_matrix

    @property
    def info(self) -> TrafficDatasetInfo:
        return self._info

    @property
    def timestamps(self) -> tuple[str, ...]:
        return self._timestamps

    @property
    def edge_keys(self) -> frozenset[TrafficEdgeKey]:
        return self._edge_keys

    @property
    def node_ids(self) -> frozenset[int]:
        return self._node_ids

    @property
    def total_edges(self) -> int:
        return len(self._edge_key_to_id)

    @property
    def total_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def total_timestamps(self) -> int:
        return len(self._timestamps)

    @classmethod
    def from_environment(
        cls,
        env_var: str = DEFAULT_ENV_VAR,
        fallback_path: str | Path | None = DEFAULT_PATH,
    ) -> TrafficDataset:
        """Instantiate TrafficDataset using the directory specified in an environment variable."""
        env_value = os.environ.get(env_var)
        if env_value:
            path = Path(os.path.expanduser(env_value))
            if not path.exists():
                raise TrafficDatasetError(
                    f"Traffic dataset path specified in {env_var} does not exist: {path}"
                )
            return cls.load(path)

        # Also check alternative env var ELECTRONIC_CITY_DATASET_DIR
        alt_value = os.environ.get("ELECTRONIC_CITY_DATASET_DIR")
        if alt_value:
            path = Path(os.path.expanduser(alt_value))
            if not path.exists():
                raise TrafficDatasetError(
                    f"Traffic dataset path specified in ELECTRONIC_CITY_DATASET_DIR does not exist: {path}"
                )
            return cls.load(path)

        if fallback_path is not None:
            resolved_fallback = Path(os.path.expanduser(str(fallback_path)))
            if resolved_fallback.exists():
                return cls.load(resolved_fallback)

        raise TrafficDatasetError(
            f"Traffic dataset directory not configured. "
            f"Please set the {env_var} environment variable to the dataset directory "
            f"(e.g., export {env_var}=~/Downloads/electronic-city-traffic-clean)."
        )

    @classmethod
    def load(cls, dataset_dir: str | Path) -> TrafficDataset:
        """Load and validate dataset files from a directory."""
        dir_path = Path(os.path.expanduser(str(dataset_dir)))
        if not dir_path.exists() or not dir_path.is_dir():
            raise TrafficDatasetError(
                f"Traffic dataset directory does not exist or is not a directory: {dir_path}"
            )

        # Locate files in either nested (graph/, traffic/) or flat structure
        edges_path = cls._find_file(dir_path, "graph/edges_final.csv", "edges_final.csv")
        nodes_path = cls._find_file(dir_path, "graph/nodes_final.csv", "nodes_final.csv")
        timestamps_path = cls._find_file(
            dir_path, "traffic/timestamps.csv", "timestamps.csv"
        )
        speed_path = cls._find_file(
            dir_path, "traffic/speed_matrix.npy", "speed_matrix.npy"
        )
        vehicle_path = cls._find_file(
            dir_path, "traffic/vehicle_matrix.npy", "vehicle_matrix.npy"
        )
        congestion_path = cls._find_file(
            dir_path, "traffic/congestion_matrix.npy", "congestion_matrix.npy"
        )

        # 1. Parse timestamps
        timestamps = cls._load_timestamps(timestamps_path)
        num_timestamps = len(timestamps)
        if num_timestamps == 0:
            raise TrafficDatasetError("Timestamps file is empty or contains no records.")

        # 2. Parse nodes
        node_ids = cls._load_nodes(nodes_path)
        if not node_ids:
            raise TrafficDatasetError("Nodes file is empty or contains no records.")

        # 3. Parse edges & map edge_id to column index
        edge_key_to_id, id_to_edge_key = cls._load_edges(edges_path)
        num_edges = len(edge_key_to_id)
        if num_edges == 0:
            raise TrafficDatasetError("Edges file is empty or contains no records.")

        # Verify edge_id covers range [0, num_edges - 1] without gaps
        expected_ids = set(range(num_edges))
        actual_ids = set(id_to_edge_key.keys())
        if actual_ids != expected_ids:
            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids
            raise TrafficDatasetError(
                f"Invalid edge_id sequence in edges file. "
                f"Expected 0..{num_edges - 1}. Missing: {missing_ids}, Extra: {extra_ids}"
            )

        # 4. Load matrices using mmap_mode="r" to avoid duplicating memory
        try:
            speed_matrix = np.load(str(speed_path), mmap_mode="r")
            vehicle_matrix = np.load(str(vehicle_path), mmap_mode="r")
            congestion_matrix = np.load(str(congestion_path), mmap_mode="r")
        except Exception as exc:
            raise TrafficDatasetError(
                f"Failed to load traffic matrix file: {exc}"
            ) from exc

        # 5. Validate matrix shapes
        expected_shape = (num_timestamps, num_edges)
        for name, mat in [
            ("speed_matrix", speed_matrix),
            ("vehicle_matrix", vehicle_matrix),
            ("congestion_matrix", congestion_matrix),
        ]:
            if mat.ndim != 2:
                raise TrafficDatasetError(
                    f"Expected 2D matrix for {name}, got shape {mat.shape}"
                )
            if mat.shape != expected_shape:
                raise TrafficDatasetError(
                    f"Shape mismatch for {name}: got {mat.shape}, "
                    f"expected {expected_shape} (num_timestamps={num_timestamps}, num_edges={num_edges})"
                )

        info = TrafficDatasetInfo(
            total_nodes=len(node_ids),
            total_edges=num_edges,
            total_timestamps=num_timestamps,
            start_time=timestamps[0],
            end_time=timestamps[-1],
            time_step_minutes=5,
            is_simulated=True,
            area_name="Electronic City, Bengaluru",
        )

        return cls(
            info=info,
            timestamps=timestamps,
            edge_key_to_id=edge_key_to_id,
            id_to_edge_key=id_to_edge_key,
            node_ids=node_ids,
            speed_matrix=speed_matrix,
            vehicle_matrix=vehicle_matrix,
            congestion_matrix=congestion_matrix,
        )

    @staticmethod
    def _find_file(base_dir: Path, *relative_candidates: str) -> Path:
        for rel in relative_candidates:
            candidate = base_dir / rel
            if candidate.exists() and candidate.is_file():
                return candidate
        raise TrafficDatasetError(
            f"Required dataset file not found in {base_dir}. Looked for: {', '.join(relative_candidates)}"
        )

    @staticmethod
    def _load_timestamps(path: Path) -> list[str]:
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return []
                # Determine timestamp column index
                col_idx = 0
                for idx, col_name in enumerate(header):
                    if "timestamp" in col_name.lower():
                        col_idx = idx
                        break
                timestamps = []
                for row in reader:
                    if row and len(row) > col_idx:
                        val = row[col_idx].strip()
                        if val:
                            timestamps.append(val)
                return timestamps
        except Exception as exc:
            if isinstance(exc, TrafficDatasetError):
                raise
            raise TrafficDatasetError(f"Error reading timestamps file {path}: {exc}") from exc

    @staticmethod
    def _load_nodes(path: Path) -> set[int]:
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return set()
                osm_idx = None
                for idx, col in enumerate(header):
                    if col.strip().lower() in ("osm_node_id", "osm_id", "osmid", "node_id"):
                        osm_idx = idx
                        if "osm" in col.strip().lower():
                            break
                if osm_idx is None:
                    osm_idx = 0

                node_ids = set()
                for row in reader:
                    if row and len(row) > osm_idx:
                        val = row[osm_idx].strip()
                        if val:
                            try:
                                node_ids.add(int(val))
                            except ValueError as exc:
                                raise TrafficDatasetError(
                                    f"Invalid node ID '{val}' in {path}: {exc}"
                                ) from exc
                return node_ids
        except Exception as exc:
            if isinstance(exc, TrafficDatasetError):
                raise
            raise TrafficDatasetError(f"Error reading nodes file {path}: {exc}") from exc

    @staticmethod
    def _load_edges(path: Path) -> tuple[dict[TrafficEdgeKey, int], dict[int, TrafficEdgeKey]]:
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return {}, {}
                header_map = {col.strip().lower(): idx for idx, col in enumerate(header)}

                if "edge_id" not in header_map:
                    raise TrafficDatasetError(f"Edges file {path} missing 'edge_id' column.")
                if "osm_u" not in header_map or "osm_v" not in header_map:
                    raise TrafficDatasetError(
                        f"Edges file {path} missing 'osm_u' or 'osm_v' column."
                    )

                edge_id_col = header_map["edge_id"]
                osm_u_col = header_map["osm_u"]
                osm_v_col = header_map["osm_v"]
                key_col = header_map.get("key")

                edge_key_to_id: dict[TrafficEdgeKey, int] = {}
                id_to_edge_key: dict[int, TrafficEdgeKey] = {}

                for row_num, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    try:
                        edge_id = int(row[edge_id_col].strip())
                        osm_u = int(row[osm_u_col].strip())
                        osm_v = int(row[osm_v_col].strip())
                        key = int(row[key_col].strip()) if key_col is not None and row[key_col].strip() else 0
                    except (ValueError, IndexError) as exc:
                        raise TrafficDatasetError(
                            f"Malformed edge data in {path} at line {row_num}: {exc}"
                        ) from exc

                    edge_key = TrafficEdgeKey(osm_u=osm_u, osm_v=osm_v, key=key)

                    if edge_key in edge_key_to_id:
                        raise TrafficDatasetError(
                            f"Duplicate edge key {edge_key} found in {path} at line {row_num}."
                        )
                    if edge_id in id_to_edge_key:
                        raise TrafficDatasetError(
                            f"Duplicate edge_id {edge_id} found in {path} at line {row_num}."
                        )

                    edge_key_to_id[edge_key] = edge_id
                    id_to_edge_key[edge_id] = edge_key

                return edge_key_to_id, id_to_edge_key
        except Exception as exc:
            if isinstance(exc, TrafficDatasetError):
                raise
            raise TrafficDatasetError(f"Error reading edges file {path}: {exc}") from exc

    def _normalize_edge_key(
        self, edge_key: TrafficEdgeKey | Sequence[int]
    ) -> TrafficEdgeKey:
        if isinstance(edge_key, TrafficEdgeKey):
            return edge_key
        if isinstance(edge_key, (tuple, list)):
            return TrafficEdgeKey.from_tuple(edge_key)
        raise TrafficDatasetError(f"Unsupported edge key type: {type(edge_key)}")

    def has_edge(self, edge_key: TrafficEdgeKey | Sequence[int]) -> bool:
        """Check whether a directed edge is present in the dataset."""
        try:
            key = self._normalize_edge_key(edge_key)
            return key in self._edge_key_to_id
        except (ValueError, TrafficDatasetError):
            return False

    def has_node(self, osm_node_id: int) -> bool:
        """Check whether an OSM node ID is present in the dataset."""
        return osm_node_id in self._node_ids

    def get_edge_id(self, edge_key: TrafficEdgeKey | Sequence[int]) -> int:
        """Get the matrix column index (edge_id) for the given edge key."""
        key = self._normalize_edge_key(edge_key)
        if key not in self._edge_key_to_id:
            raise TrafficDatasetError(f"Edge {key} not found in traffic dataset.")
        return self._edge_key_to_id[key]

    def get_time_index(self, timestamp: str | datetime) -> int:
        """Resolve a timestamp string or datetime object to an integer matrix row index."""
        if isinstance(timestamp, datetime):
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(timestamp).strip()

        if ts_str in self._timestamp_to_index:
            return self._timestamp_to_index[ts_str]

        # If not exact match, try parsing ISO format or nearest timestamp
        try:
            target_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            # Find closest timestamp
            best_idx = 0
            best_diff = float("inf")
            for idx, stored_ts in enumerate(self._timestamps):
                try:
                    s_dt = datetime.strptime(stored_ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                diff = abs((target_dt - s_dt).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
                if diff == 0:
                    break
            return best_idx
        except Exception:
            raise TrafficDatasetError(
                f"Timestamp '{timestamp}' could not be resolved in dataset time series."
            )

    def get_snapshot(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        time_index: int,
    ) -> TrafficSnapshot:
        """Retrieve traffic metrics for a specific directed edge and time index."""
        if time_index < 0 or time_index >= len(self._timestamps):
            raise TrafficDatasetError(
                f"Invalid time index {time_index}. Dataset contains {len(self._timestamps)} "
                f"timestamps (index 0 to {len(self._timestamps) - 1})."
            )

        col_idx = self.get_edge_id(edge_key)

        speed = float(self._speed_matrix[time_index, col_idx])
        vehicle = float(self._vehicle_matrix[time_index, col_idx])
        congestion = float(self._congestion_matrix[time_index, col_idx])

        # Validate finite values
        if not (math.isfinite(speed) and math.isfinite(vehicle) and math.isfinite(congestion)):
            raise TrafficDatasetError(
                f"Non-finite traffic value detected at time index {time_index} for edge {edge_key}: "
                f"speed={speed}, vehicle={vehicle}, congestion={congestion}."
            )

        return TrafficSnapshot(
            speed_kmh=speed,
            vehicle_count=vehicle,
            congestion_index=congestion,
            timestamp=self._timestamps[time_index],
            time_index=time_index,
        )

    def get_snapshot_at_timestamp(
        self,
        edge_key: TrafficEdgeKey | Sequence[int],
        timestamp: str | datetime,
    ) -> TrafficSnapshot:
        """Retrieve traffic metrics for a specific directed edge and timestamp."""
        time_index = self.get_time_index(timestamp)
        return self.get_snapshot(edge_key, time_index)
