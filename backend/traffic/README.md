# Traffic Data Loading Layer

## Overview

The `backend.traffic` package provides a validated, memory-efficient loading and query layer for the Electronic City traffic dataset. It provides point-in-time traffic snapshots (speed, vehicle count, congestion index) indexed by directed OpenStreetMap edges `(osm_u, osm_v, key)` and temporal indices or timestamps.

## Important: Simulated Dataset

> [!NOTE]
> The traffic data is **synthetically simulated** for an academic research study in Electronic City, Bengaluru, India. It reproduces temporal and spatial patterns but is **not** real-time or physical sensor data. The road network coverage is strictly confined to 602 directed road segments in Electronic City.

### Dataset Dimensions

- **Temporal Coverage**: 60 days (January 1, 2026 to March 1, 2026) at 5-minute resolution (17,280 timestamps).
- **Network Coverage**: 262 nodes, 602 directed edges.
- **Traffic Metrics**:
  - `speed_matrix.npy`: Estimated vehicle speed (km/h)
  - `vehicle_matrix.npy`: Estimated vehicle count
  - `congestion_matrix.npy`: Congestion index `[0.0, 0.875]`

## macOS Configuration

The dataset is kept outside the version-controlled repository to prevent bloating Git with ~125 MB of binary matrices.

### Default Location
By default, the loader inspects:
```bash
~/Downloads/electronic-city-traffic-clean
```
If the dataset resides there, no configuration is required.

### Custom Dataset Path (Environment Variable)
To specify a custom dataset location on macOS, export `TRAFFIC_DATASET_DIR`:

#### Option 1: Temporary (Current Terminal Session)
```bash
export TRAFFIC_DATASET_DIR="$HOME/Downloads/electronic-city-traffic-clean"
```

#### Option 2: Persistent (Zsh shell - default on macOS)
Add to your `~/.zshrc`:
```bash
echo 'export TRAFFIC_DATASET_DIR="$HOME/Downloads/electronic-city-traffic-clean"' >> ~/.zshrc
source ~/.zshrc
```

#### Option 3: `.env` file in Project Root
Create or edit `.env` in the project root:
```env
TRAFFIC_DATASET_DIR=/Users/your_username/Downloads/electronic-city-traffic-clean
```

## Python API Usage

```python
from backend.traffic import (
    TrafficDataset,
    TrafficEdgeKey,
    TrafficSnapshot,
    TrafficDatasetError,
)

# Load dataset (uses TRAFFIC_DATASET_DIR env var or default fallback)
dataset = TrafficDataset.from_environment()

# Inspect dataset metadata
print(dataset.info.total_edges)        # 602
print(dataset.info.total_timestamps)   # 17280
print(dataset.info.start_time)         # 2026-01-01 00:00:00
print(dataset.info.end_time)           # 2026-03-01 23:55:00

# Query edge by directed identity (osm_u, osm_v, key)
edge_key = TrafficEdgeKey(osm_u=459516935, osm_v=1724044273, key=0)

if dataset.has_edge(edge_key):
    # Snapshot at index 0
    snapshot = dataset.get_snapshot(edge_key, time_index=0)
    print(f"Speed: {snapshot.speed_kmh} km/h")
    print(f"Vehicle count: {snapshot.vehicle_count}")
    print(f"Congestion index: {snapshot.congestion_index}")

    # Snapshot at ISO timestamp string
    snap_ts = dataset.get_snapshot_at_timestamp(edge_key, "2026-01-01 08:30:00")
    print(f"Congestion at 8:30 AM: {snap_ts.congestion}")
```

## Architecture & Safety Guarantees

1. **Memory-Mapped Loading**: Large 40 MB matrices are opened via `np.load(..., mmap_mode="r")`, allowing instant lookup without duplicate memory allocations.
2. **Direction & Parallel Edge Integrity**: Directed edges `(u, v, key)` and reverse copies `(v, u, key)` have distinct `edge_id` column mappings. Parallel edges with differing `key` values are preserved.
3. **Data Integrity & Non-Fabrication**: Values are never silently fabricated or borrowed from neighbouring edges. Unknown edges or non-finite values (NaN / Inf) raise `TrafficDatasetError`.
4. **Validation on Load**: Validates matrix dimensions `(num_timestamps, num_edges)`, completeness of `edge_id` sets, timestamp files, and node IDs upon initialization.
