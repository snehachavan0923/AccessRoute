import csv
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.traffic.loader import (
    TrafficDataset,
    TrafficDatasetError,
    TrafficDatasetInfo,
    TrafficEdgeKey,
    TrafficSnapshot,
)


def create_synthetic_dataset(
    base_dir: Path,
    *,
    timestamps: list[str] | None = None,
    edges: list[dict] | None = None,
    nodes: list[dict] | None = None,
    speed_matrix: np.ndarray | None = None,
    vehicle_matrix: np.ndarray | None = None,
    congestion_matrix: np.ndarray | None = None,
    nested: bool = True,
) -> Path:
    """Helper to create a small synthetic traffic dataset in a directory."""
    if timestamps is None:
        timestamps = [
            "2026-01-01 00:00:00",
            "2026-01-01 00:05:00",
            "2026-01-01 00:10:00",
            "2026-01-01 00:15:00",
        ]

    if nodes is None:
        nodes = [
            {"node_id": "0", "osm_node_id": "101", "x": "77.67", "y": "12.84"},
            {"node_id": "1", "osm_node_id": "102", "x": "77.68", "y": "12.85"},
            {"node_id": "2", "osm_node_id": "103", "x": "77.69", "y": "12.86"},
        ]

    if edges is None:
        # 3 edges:
        # edge 0: 101 -> 102 (key 0)
        # edge 1: 102 -> 101 (key 0) [reverse direction]
        # edge 2: 101 -> 102 (key 1) [parallel edge]
        edges = [
            {
                "edge_id": "0",
                "u": "0",
                "v": "1",
                "osm_u": "101",
                "osm_v": "102",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "1",
                "u": "1",
                "v": "0",
                "osm_u": "102",
                "osm_v": "101",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "2",
                "u": "0",
                "v": "1",
                "osm_u": "101",
                "osm_v": "102",
                "key": "1",
                "length_m": "120.0",
                "highway": "secondary",
                "maxspeed": "40",
            },
        ]

    num_t = len(timestamps)
    num_e = len(edges)

    if speed_matrix is None:
        speed_matrix = np.array(
            [
                [30.0, 40.0, 50.0],
                [32.0, 42.0, 52.0],
                [34.0, 44.0, 54.0],
                [36.0, 46.0, 56.0],
            ],
            dtype=np.float32,
        )

    if vehicle_matrix is None:
        vehicle_matrix = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
                [13.0, 23.0, 33.0],
            ],
            dtype=np.float32,
        )

    if congestion_matrix is None:
        congestion_matrix = np.array(
            [
                [0.10, 0.20, 0.30],
                [0.15, 0.25, 0.35],
                [0.20, 0.30, 0.40],
                [0.25, 0.35, 0.45],
            ],
            dtype=np.float32,
        )

    graph_dir = base_dir / "graph" if nested else base_dir
    traffic_dir = base_dir / "traffic" if nested else base_dir
    graph_dir.mkdir(parents=True, exist_ok=True)
    traffic_dir.mkdir(parents=True, exist_ok=True)

    # Write timestamps.csv
    with open(traffic_dir / "timestamps.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"])
        for ts in timestamps:
            writer.writerow([ts])

    # Write nodes_final.csv
    with open(graph_dir / "nodes_final.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["node_id", "osm_node_id", "x", "y"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nodes)

    # Write edges_final.csv
    with open(graph_dir / "edges_final.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "edge_id",
            "u",
            "v",
            "osm_u",
            "osm_v",
            "key",
            "length_m",
            "highway",
            "maxspeed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)

    # Save numpy matrices
    np.save(str(traffic_dir / "speed_matrix.npy"), speed_matrix)
    np.save(str(traffic_dir / "vehicle_matrix.npy"), vehicle_matrix)
    np.save(str(traffic_dir / "congestion_matrix.npy"), congestion_matrix)

    return base_dir


class TrafficLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        create_synthetic_dataset(self.base_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_successful_loading_valid_fixture(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        self.assertEqual(dataset.total_edges, 3)
        self.assertEqual(dataset.total_nodes, 3)
        self.assertEqual(dataset.total_timestamps, 4)
        self.assertEqual(dataset.info.start_time, "2026-01-01 00:00:00")
        self.assertEqual(dataset.info.end_time, "2026-01-01 00:15:00")
        self.assertTrue(dataset.info.is_simulated)
        self.assertIn("Electronic City", dataset.info.area_name)

    def test_directed_edge_distinction(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        forward_key = TrafficEdgeKey(osm_u=101, osm_v=102, key=0)
        reverse_key = TrafficEdgeKey(osm_u=102, osm_v=101, key=0)

        self.assertTrue(dataset.has_edge(forward_key))
        self.assertTrue(dataset.has_edge(reverse_key))

        snap_forward = dataset.get_snapshot(forward_key, time_index=0)
        snap_reverse = dataset.get_snapshot(reverse_key, time_index=0)

        # Forward edge (col 0): speed=30, veh=10, cong=0.10
        self.assertAlmostEqual(snap_forward.speed_kmh, 30.0)
        self.assertAlmostEqual(snap_forward.vehicle_count, 10.0)
        self.assertAlmostEqual(snap_forward.congestion_index, 0.10)

        # Reverse edge (col 1): speed=40, veh=20, cong=0.20
        self.assertAlmostEqual(snap_reverse.speed_kmh, 40.0)
        self.assertAlmostEqual(snap_reverse.vehicle_count, 20.0)
        self.assertAlmostEqual(snap_reverse.congestion_index, 0.20)

    def test_parallel_edge_distinction(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        edge_key_0 = TrafficEdgeKey(osm_u=101, osm_v=102, key=0)
        edge_key_1 = TrafficEdgeKey(osm_u=101, osm_v=102, key=1)

        self.assertTrue(dataset.has_edge(edge_key_0))
        self.assertTrue(dataset.has_edge(edge_key_1))

        snap_0 = dataset.get_snapshot(edge_key_0, time_index=0)
        snap_1 = dataset.get_snapshot(edge_key_1, time_index=0)

        # Edge key 0 is col 0, Edge key 1 is col 2
        self.assertAlmostEqual(snap_0.speed_kmh, 30.0)
        self.assertAlmostEqual(snap_1.speed_kmh, 50.0)
        self.assertAlmostEqual(snap_0.vehicle_count, 10.0)
        self.assertAlmostEqual(snap_1.vehicle_count, 30.0)

    def test_speed_vehicle_and_congestion_retrieval(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        key = (101, 102, 0)
        snapshot = dataset.get_snapshot(key, time_index=2)

        self.assertAlmostEqual(snapshot.speed_kmh, 34.0)
        self.assertAlmostEqual(snapshot.speed, 34.0)
        self.assertAlmostEqual(snapshot.vehicle_count, 12.0)
        self.assertAlmostEqual(snapshot.vehicles, 12.0)
        self.assertAlmostEqual(snapshot.congestion_index, 0.20)
        self.assertAlmostEqual(snapshot.congestion, 0.20)
        self.assertEqual(snapshot.time_index, 2)
        self.assertEqual(snapshot.timestamp, "2026-01-01 00:10:00")

    def test_timestamp_parsing_and_selection(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        key = TrafficEdgeKey(101, 102, 0)

        # Exact match
        snap_exact = dataset.get_snapshot_at_timestamp(key, "2026-01-01 00:05:00")
        self.assertEqual(snap_exact.time_index, 1)
        self.assertAlmostEqual(snap_exact.speed_kmh, 32.0)

        # DateTime object
        dt = datetime(2026, 1, 1, 0, 15, 0)
        snap_dt = dataset.get_snapshot_at_timestamp(key, dt)
        self.assertEqual(snap_dt.time_index, 3)
        self.assertAlmostEqual(snap_dt.speed_kmh, 36.0)

        # Invalid timestamp raises TrafficDatasetError
        with self.assertRaises(TrafficDatasetError):
            dataset.get_snapshot_at_timestamp(key, "not-a-valid-timestamp-string")

    def test_invalid_time_index_handling(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        key = TrafficEdgeKey(101, 102, 0)

        with self.assertRaises(TrafficDatasetError):
            dataset.get_snapshot(key, time_index=-1)

        with self.assertRaises(TrafficDatasetError):
            dataset.get_snapshot(key, time_index=4)

    def test_unknown_edge_handling(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        unknown_key = TrafficEdgeKey(999, 888, 0)

        self.assertFalse(dataset.has_edge(unknown_key))
        with self.assertRaises(TrafficDatasetError):
            dataset.get_snapshot(unknown_key, time_index=0)

    def test_matrix_shape_validation(self) -> None:
        # Speed matrix shape (3, 3) instead of expected (4, 3)
        with tempfile.TemporaryDirectory() as bad_dir:
            bad_path = Path(bad_dir)
            create_synthetic_dataset(
                bad_path,
                speed_matrix=np.ones((3, 3), dtype=np.float32),
            )
            with self.assertRaises(TrafficDatasetError) as ctx:
                TrafficDataset.load(bad_path)
            self.assertIn("Shape mismatch", str(ctx.exception))

    def test_edge_id_mapping_and_order_validation(self) -> None:
        # Non-contiguous or disordered edges
        disordered_edges = [
            {
                "edge_id": "2",
                "u": "0",
                "v": "1",
                "osm_u": "101",
                "osm_v": "102",
                "key": "1",
                "length_m": "120.0",
                "highway": "secondary",
                "maxspeed": "40",
            },
            {
                "edge_id": "0",
                "u": "0",
                "v": "1",
                "osm_u": "101",
                "osm_v": "102",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "1",
                "u": "1",
                "v": "0",
                "osm_u": "102",
                "osm_v": "101",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
        ]
        with tempfile.TemporaryDirectory() as reorder_dir:
            reorder_path = Path(reorder_dir)
            create_synthetic_dataset(reorder_path, edges=disordered_edges)
            dataset = TrafficDataset.load(reorder_path)
            # Edge (101, 102, 1) was edge_id 2 in row 0, must map to column 2
            snap_col2 = dataset.get_snapshot((101, 102, 1), 0)
            self.assertAlmostEqual(snap_col2.speed_kmh, 50.0)

    def test_duplicate_edge_id_raises_error(self) -> None:
        duplicate_edges = [
            {
                "edge_id": "0",
                "u": "0",
                "v": "1",
                "osm_u": "101",
                "osm_v": "102",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "0",  # Duplicate edge_id
                "u": "1",
                "v": "0",
                "osm_u": "102",
                "osm_v": "101",
                "key": "0",
                "length_m": "100.0",
                "highway": "primary",
                "maxspeed": "50",
            },
            {
                "edge_id": "2",
                "u": "0",
                "v": "1",
                "osm_u": "101",
                "osm_v": "102",
                "key": "1",
                "length_m": "120.0",
                "highway": "secondary",
                "maxspeed": "40",
            },
        ]
        with tempfile.TemporaryDirectory() as dup_dir:
            dup_path = Path(dup_dir)
            create_synthetic_dataset(dup_path, edges=duplicate_edges)
            with self.assertRaises(TrafficDatasetError):
                TrafficDataset.load(dup_path)

    def test_missing_files_handling(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(TrafficDatasetError) as ctx:
                TrafficDataset.load(empty_dir)
            self.assertIn("Required dataset file not found", str(ctx.exception))

    def test_non_finite_values_handling(self) -> None:
        nan_matrix = np.array(
            [
                [np.nan, 40.0, 50.0],
                [32.0, 42.0, 52.0],
                [34.0, 44.0, 54.0],
                [36.0, 46.0, 56.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as nan_dir:
            nan_path = Path(nan_dir)
            create_synthetic_dataset(nan_path, speed_matrix=nan_matrix)
            dataset = TrafficDataset.load(nan_path)
            # Edge 0 has NaN at index 0
            with self.assertRaises(TrafficDatasetError) as ctx:
                dataset.get_snapshot((101, 102, 0), time_index=0)
            self.assertIn("Non-finite", str(ctx.exception))

    def test_coverage_and_dataset_info(self) -> None:
        dataset = TrafficDataset.load(self.base_path)
        self.assertTrue(dataset.has_node(101))
        self.assertTrue(dataset.has_node(102))
        self.assertFalse(dataset.has_node(999999))
        self.assertEqual(len(dataset.node_ids), 3)
        self.assertEqual(len(dataset.edge_keys), 3)

    def test_environment_variable_configuration(self) -> None:
        os.environ["TEST_TRAFFIC_DATASET_DIR"] = str(self.base_path)
        try:
            dataset = TrafficDataset.from_environment(
                env_var="TEST_TRAFFIC_DATASET_DIR"
            )
            self.assertEqual(dataset.total_edges, 3)
        finally:
            del os.environ["TEST_TRAFFIC_DATASET_DIR"]

    def test_flat_directory_structure_supported(self) -> None:
        with tempfile.TemporaryDirectory() as flat_dir:
            flat_path = Path(flat_dir)
            create_synthetic_dataset(flat_path, nested=False)
            dataset = TrafficDataset.load(flat_path)
            self.assertEqual(dataset.total_edges, 3)


if __name__ == "__main__":
    unittest.main()
