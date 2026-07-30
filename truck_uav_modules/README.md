# Truck--UAV Modules

Standalone extraction of the final Xinjiekou and Baijiahu road products plus the existing local vector visibility-graph planner.  It does not contain Phase 2 global K-nearest-neighbour air graphs, old fixed-100-node selection code, experiment figures, temporary results, online data download, scheduling, or reinforcement learning.

## Install and test

```powershell
cd truck_uav_modules
python -m pip install -r requirements.txt
$env:PYTHONPATH = "."
python -m pytest -q
python examples/xinjiekou_minimal.py
```

All paths are package-relative. `configs/regions.yaml` maps region names to packaged final-stage data; `configs/planner.yaml` contains the building clearance, waypoint and timeout parameters.

## Packaged data

Each `data/<region>/` directory contains only runtime inputs:

- `launch_recovery_nodes.geojson`: final launch/recovery candidates, including intersection and supplementary candidates.
- `intersection_launch_nodes.geojson` and `supplementary_launch_nodes.geojson`: provenance subsets.
- `major_road_graph.graphml`: final complete main-road graph with true line geometry, distance and travel-time attributes.
- `road_edges.geojson` and `road_network.graphml`: Xinjiekou compressed real-road paths with original node and edge sequences.
- `buildings.geojson`: Xinjiekou clipped source building footprints and original height field.
- `road_topology_summary.json`: final topology metadata.

The packaged final launch/recovery counts are 333 for Xinjiekou and 129 for Baijiahu. These are final intersection-led node sets, not a fixed-count selection.

## APIs

```python
from ground_network import load_ground_network, get_launch_recovery_nodes, shortest_ground_path
from uav_path_planner import plan_uav_path

network = load_ground_network("xinjiekou")
nodes = get_launch_recovery_nodes("xinjiekou")
road = shortest_ground_path("xinjiekou", nodes.iloc[0].node_id, nodes.iloc[1].node_id)
flight = plan_uav_path("xinjiekou", nodes.iloc[0].geometry, nodes.iloc[1].geometry)
```

`shortest_ground_path` returns the original road-node sequence, edge sequence, `LineString`, distance in metres and travel time in seconds. `plan_uav_path` tests `20, 40, 60, 80, 100, 120` m in order. At each height it filters buildings by source height, applies the configured safety buffer, checks a direct path, and then runs one complete visibility graph inside the fixed start--goal bounding-box buffer. If every local aerial attempt fails, it snaps both endpoints to the packaged main-road graph and checks the real road geometry at each candidate height as a final `road_following` fallback.

The result schema is:

```python
{
    "feasible": True,
    "path_mode": "direct | local_detour | road_following | infeasible",
    "selected_height_m": 60.0,
    "path_geometry": ...,  # Shapely LineString
    "distance_m": 0.0,
    "planning_time_s": 0.0,
    "tested_heights": [20.0, 40.0],
    "failure_reason_by_height": {},
}
```

## Scheduling integration

Keep truck/CAV state as a current ground-graph node ID. Use `shortest_ground_path` for truck travel time and geometry. For a sortie, pass the truck's current node geometry and the task geometry to `plan_uav_path`; store the returned `path_mode`, height, distance and planning time with the action. Reject an action when `feasible` is false. The package deliberately does not make assignment, dispatch or energy-policy decisions.
