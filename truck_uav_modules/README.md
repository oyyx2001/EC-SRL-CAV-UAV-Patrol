# Ground-network and UAV path-planning runtime

This directory contains the runtime modules and map-reading interfaces used by
the Rainbow DQN environment:

- `ground_network/` loads the regional road graph and launch/recovery nodes.
- `uav_path_planner/` provides the 2-D fallback and validated 3-D voxel A*
  interfaces used by sortie materialization.
- `data/README.md` documents the Xinjiekou and Baijiahu road/building,
  launch/recovery and voxel inputs expected by the unchanged configurations.
  Their redistribution rights are unverified, so data are withheld from the
  current public file set while the author's local copies remain intact.
- `build_and_benchmark_3d_voxel_astar.py` contains the shared 3-D A* core used
  by `uav_path_planner.voxel_astar` at runtime.

`rasterize_building_occupancy.py` supplies the formerly missing offline helpers.
Run `python -m tools.rebuild_voxel_maps` from the repository root to independently
reconstruct the paper's 10 x 10 x 20 m road-cleared maps into a new ignored output
directory. No historical 2-D metadata/mapping CSV or prebuilt array is required
for construction. The A* core's old benchmark entry point still uses 10 m
vertical layers and is not the paper-map reconstruction command. See
`VOXEL_REBUILD.md`; runtime planning calculations remain unchanged.

Author-owned code and the two named original CS-RDQN policies are MIT-licensed;
third-party geographic data and installed libraries are not relicensed by that
grant. Run `python -m tools.check_data_assets` before any original-case rollout.
See the root `LICENSE_SCOPE.md`, `DATA_AVAILABILITY.md` and
`DATA_PROVENANCE.md` for source evidence, access limits and the historical Git
warning. No data download or synthetic replacement is provided.
