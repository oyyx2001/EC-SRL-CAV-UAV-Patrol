# Ground-network and UAV path-planning runtime

This directory contains the runtime modules and packaged map assets used by
the Rainbow DQN environment:

- `ground_network/` loads the regional road graph and launch/recovery nodes.
- `uav_path_planner/` provides the 2-D fallback and validated 3-D voxel A*
  interfaces used by sortie materialization.
- `data/` contains the Xinjiekou and Baijiahu road, building, launch/recovery,
  and voxel-map assets referenced by the scene configurations.
- `build_and_benchmark_3d_voxel_astar.py` contains the shared 3-D A* core used
  by `uav_path_planner.voxel_astar` at runtime.

The standalone map-generation, benchmark, and visualization utilities are not
part of this reviewer-facing release.
