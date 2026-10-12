# Withheld geographic inputs

Road networks, building footprints/heights, launch/recovery coordinates and derived voxel maps are **not included in the current public release**. Their source redistribution permissions are unverified; the repository MIT License does not apply to these data.

The author's local files remain at their original paths. Public configs and reading interfaces are unchanged. Restore authorized original-case assets to:

```text
truck_uav_modules/data/xinjiekou/
truck_uav_modules/data/baijiahu/
```

Core frozen evaluation uses each case's `road_network.graphml`, `buildings.geojson`, `launch_recovery_nodes.geojson`, `voxel_map_3d/metadata.json`, `voxel_map_3d/occupancy.npy`, plus the registry described in `data/README.md`. Legacy ground/2-D interfaces also refer to the auxiliary scene filenames in `truck_uav_modules/configs/regions.yaml`; retain those interfaces and provide authorized inputs when using them.

The provenance audit identifies Baidu road-centreline and height-source indicators, not a verified open license. A provider acquisition channel, contract/license version and permission for derivatives are still needed. No alternative download/scraping route or substitute map is supplied.

`python -m tools.check_data_assets` reports missing/mismatched core input files. An integrity match is not a legal authorization. See `DATA_PROVENANCE.md`; previous Git commits may still contain these files and were not rewritten.

Offline reconstruction also reads each scene's `road_edges.geojson`.
`python -m tools.rebuild_voxel_maps` derives its extent and launch mapping from
authorized inputs instead of missing historical intermediate artifacts. Outputs
remain ignored; the command never overwrites this input tree. See
`VOXEL_REBUILD.md` for source-height defaults and exact-build checks.
