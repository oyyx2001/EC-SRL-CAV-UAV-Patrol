# Rebuild and validate the original 3-D maps

## Scope and data-access prerequisite

The code is MIT-licensed. **Original geographic inputs and derived maps are not
licensed for redistribution by this repository.** The author reported on
2026-10-12 that the files were obtained through a WeChat public account; the
account/article, upstream provider, date/version and permission document have
not been supplied. A public post is an acquisition channel, not a license.
See [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

Supply authorized original inputs at the paths in
`truck_uav_modules/configs/regions.yaml`. Per scene the builder reads only:

```text
truck_uav_modules/data/{xinjiekou,baijiahu}/buildings.geojson
truck_uav_modules/data/{xinjiekou,baijiahu}/road_edges.geojson
truck_uav_modules/data/{xinjiekou,baijiahu}/launch_recovery_nodes.geojson
```

It does not download data, depend on files outside the repository, require old
2-D grids, or read a prebuilt occupancy array to construct the new one. Existing
canonical maps are read only for comparison **after** independent construction.
Absent references produce `UNAVAILABLE`, never a claimed match.

## Run from the repository root

Use the pinned environment in `ENVIRONMENT_SETUP.md`. No rasterio or additional
package is needed beyond the existing NumPy, Shapely, GeoPandas and PyYAML stack.

```powershell
python -m tools.rebuild_voxel_maps --scene all --output-dir results/voxel_rebuild --astar-trials 10
```

The output directory must be new, inside the repository, and outside both input
data trees and `.git`. Existing outputs/canonical assets are never overwritten.
Use another `--output-dir` for a later run. Missing input, bad provenance fields,
incorrect building count, invalid height, incompatible CRS or out-of-bounds
launch node fails explicitly. A reference mismatch gives a nonzero exit code;
no original measurements/maps are adjusted to force agreement.

Outputs per scene:

- `occupancy.npy` and `occupancy.npz` (`occupancy_3d`): uint8, order z/row/column.
- `metadata.json`: numerical grid specification and build timing.
- `summary.json`: occupancy/count/height-source diagnostics.
- `launch_recovery_grid_mapping.csv`: mapping derived from launch GeoJSON.

The common `rebuild_report.json` records input/output SHA-256, dependency versions,
reference differences and every validated A* trial. These geographic derivatives
stay in Git-ignored `results/`; they are not public release assets.

## Exact construction rules

1. Read projected metre-based Polygon/MultiPolygon footprints. Repair invalid
   geometry explicitly and report the count; do not silently drop features.
   `height_m` must be finite and positive. Retain source/default-height labels;
   do not invent measured heights.
2. Derive a north-up 10 m horizontal grid from footprint bounds plus a 50 m
   margin on each side. Round rows/columns up to whole cells. Both original
   scenes yield 260 x 260 cells in EPSG:32650.
3. All-touched rasterization uses closed-cell intersection. Store the maximum
   building height touching each cell. A source voxel is blocked iff that
   height is strictly greater than its lower z edge. Source edges are 0,10,...,120 m.
4. Map each launch/recovery point by floor-based affine conversion and free its
   complete vertical column in the 12-layer source map.
5. Aggregate each pair of 10 m layers by maximum, giving six 20 m layers.
   No interpolation, smoothing or resampling of buildings is used.
6. Buffer road-edge centerlines by 10 m (16 segments per quadrant), rasterize
   all-touched, and release road/node cells in layers 1--5, i.e. [20,120) m.
   The ground layer is **not** road-released. The six layers span [0,120) m;
   120 m is an upper grid boundary, not a seventh layer. The retained historical
   `road_node_free_layers_m` list ends in 120, but the actual released layer
   indices are 1--5. Core world-to-voxel conversion rejects an endpoint at z=120;
   operational voxel centers are 30,50,70,90,110 m. No boundary behavior is changed.

The original `build_and_benchmark_3d_voxel_astar.py` retains its historical
10 x 10 x 10 m benchmark defaults. Its missing rasterization helpers and offline
metadata/node-mapping fallback have been restored. Use the new command above
for the **paper's 10 x 10 x 20 m road-cleared maps**, not the historical benchmark
entry point. A* search, neighbor costs, corner-cutting rules and runtime planning
budgets are unchanged. No training code/configuration or weights are changed by
this map-generation update.

## Local validation, 2026-10-12

| Check | Case 1: Xinjiekou | Case 2: Baijiahu |
| --- | ---: | ---: |
| Unique buildings | 10,348 | 6,872 |
| Geometry repairs | 0 | 0 |
| Supplied default heights | 0 | 6 |
| Shape | 6 x 260 x 260 | 6 x 260 x 260 |
| Occupied voxels | 77,427 | 51,719 |
| Differing voxels vs original | **0** | **0** |
| Numerical metadata differences | **0** | **0** |
| Validated 26-neighbor A* paths | 10/10 | 10/10 |

The rebuilt NPY files have the same byte hashes as the original arrays:

- Case 1: `be81c23f2ca62e6ae90d8f28139eacb295ce2bd57d4fa28878a0246ba7b00bda`
- Case 2: `5549ee17c58515ce076b2b69575a7fedc01741c2d86ff08d3064e31b66923b9d`

Metadata comparison excludes only nondeterministic `build_time_s`. Newly saved
JSON formatting/timing and NPZ archive bytes need not match historical files;
this is not a claim of byte-identical regeneration of every container. Original
`tools.check_data_assets` intentionally checks the full historical metadata hash.
Replacing it with a new build's JSON can therefore fail that historical-identity
preflight even when numerical grid fields match. Do not overwrite canonical
metadata or loosen the preflight to hide this distinction.

Baijiahu `buildings.geojson` and the legacy `buildings_local.geojson` have
identical footprints and heights locally, but different attribute records. The
new pipeline follows the configured **buildings.geojson**; it does not require
the duplicate legacy file. Six supplied default heights remain explicitly
identified, not presented as measured values.

The detailed local report is at `results/voxel_rebuild_20261012/rebuild_report.json`.
It is an author-local verification record, not evidence of public data access.

## Material Passport

- Mode: deterministic reproducibility verification (not statistical inference).
- Inputs: author's unchanged original-case building/road/launch files.
- Verification: independently rebuilt arrays, exact voxel and numerical-metadata
  comparison; validated A* paths with fixed diagnostic seed 20260728.
- Original inputs/canonical maps modified: no. Training performed: no.
- Limit: third-party research-use/redistribution permissions remain unverified;
  heights include six supplied defaults; fresh-platform verification is separate.
