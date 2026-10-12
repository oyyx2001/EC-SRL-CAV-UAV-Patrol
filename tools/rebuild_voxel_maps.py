"""Rebuild the original 10 x 10 x 20 m maps into NEW output directories.

Requires locally supplied, authorized building/road/launch GeoJSONs. No downloads,
no training, no canonical-map overwrite and no historical 2-D grid dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import geopandas as gpd
import numpy as np
import yaml

from truck_uav_modules import rasterize_building_occupancy as raster

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("xinjiekou", "baijiahu")
EXPECTED_COUNTS = dict(xinjiekou=10348, baijiahu=6872)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(relative, root=ROOT):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input/output paths must stay inside the repository")
    return path


def resolve_inputs(scene, root=ROOT):
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")
    settings = yaml.safe_load((root / "truck_uav_modules/configs/regions.yaml").read_text())
    data_dir = repo_path(Path("truck_uav_modules/configs") / settings["data_root"], root)
    region = settings["regions"][scene]
    inputs = {name: repo_path((data_dir / scene / region[key]).relative_to(root), root)
              for name, key in [("buildings", "buildings_file"),
                                ("roads", "compressed_road_edges_file"),
                                ("launch_nodes", "launch_nodes_file")]}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing authorized {name} input: {path.relative_to(root)}; "
                                    "see DATA_AVAILABILITY.md. No data are downloaded.")
    return inputs, data_dir / scene / "voxel_map_3d"


def build_map(scene, inputs):
    started = time.perf_counter()
    raw = raster.load_buildings(inputs["buildings"])
    buildings = raster.normalize_building_height(raster.clean_building_geometries(raw))
    if not {"building_id", "height_source"}.issubset(buildings.columns):
        raise ValueError("Original-case inputs require building_id and height_source provenance")
    if len(buildings) != EXPECTED_COUNTS[scene] or buildings.building_id.nunique() != len(buildings):
        raise ValueError(f"{scene}: expected {EXPECTED_COUNTS[scene]} unique buildings")
    grid = raster.derive_grid_spec(buildings)  # Footprint extent + 50 m; no prebuilt-map input.
    nodes = gpd.read_file(inputs["launch_nodes"])
    mapping = raster.map_launch_nodes(nodes, grid, scene)
    roads = gpd.read_file(inputs["roads"])
    if roads.empty or roads.crs != buildings.crs or (~roads.geom_type.isin(["LineString", "MultiLineString"])).any():
        raise ValueError("Roads must be nonempty lines in the building CRS")
    if roads.geometry.isna().any() or roads.geometry.is_empty.any() or not roads.is_valid.all():
        raise ValueError("Invalid road geometry")

    heights = raster.rasterize_max_height(buildings, grid)
    source_edges = np.arange(0.0, 121.0, 10.0)
    source = (heights[None, :, :] > source_edges[:-1, None, None]).astype(np.uint8)
    # Preserve the historical full-column launch/recovery release BEFORE aggregation.
    source[:, mapping.row.to_numpy(), mapping.col.to_numpy()] = 0
    occupancy = source.reshape(6, 2, grid.n_rows, grid.n_cols).max(axis=1)
    road_pairs = raster.intersection_pairs(roads.geometry.to_numpy(), grid, 10.0)
    road_mask = np.zeros(grid.n_rows * grid.n_cols, dtype=bool)
    road_mask[road_pairs[0]] = True
    road_mask = road_mask.reshape(grid.n_rows, grid.n_cols)
    occupancy[1:, road_mask] = 0  # [20,120) m; ground layer is NOT road-released.
    occupancy[1:, mapping.row.to_numpy(), mapping.col.to_numpy()] = 0

    z_edges = np.arange(0.0, 121.0, 20.0)
    metadata = grid.metadata()
    metadata.pop("grid_size_m")
    metadata.update(scene=scene, voxel_size_x_m=10.0, voxel_size_y_m=10.0,
                    voxel_size_z_m=20.0, z_min_m=0.0, z_max_m=120.0,
                    z_edges_m=z_edges.tolist(), z_centers_m=((z_edges[:-1] + z_edges[1:]) / 2).tolist(),
                    n_z=6, min_operational_altitude_m=20.0,
                    horizontal_safety_buffer_m=0.0, vertical_safety_margin_m=0.0,
                    all_touched=True, force_launch_recovery_column_free=True,
                    building_count=len(buildings), build_time_s=time.perf_counter() - started,
                    map_variant="road_free_10m_xy_20m_z", road_buffer_m=10.0,
                    road_node_free_layers_m=[20, 40, 60, 80, 100, 120],
                    road_node_free_operational_layers=[1, 2, 3, 4, 5],
                    road_node_free_operational_height_range_m=[20.0, 120.0],
                    vertical_aggregation="max_of_each_two_10m_source_layers_then_road_node_release")
    summary = dict(shape=list(occupancy.shape), total_voxels=int(occupancy.size),
                   free_voxels=int((occupancy == 0).sum()), occupied_voxels=int(occupancy.sum()),
                   operational_road_cells_free=bool(not occupancy[1:, road_mask].any()),
                   operational_node_cells_free=bool(not occupancy[:, mapping.row, mapping.col].any()),
                   building_count=len(buildings), geometry_repairs=int((~raw.is_valid).sum()),
                   height_source_counts=buildings.height_source.value_counts(dropna=False).to_dict())
    return occupancy, metadata, summary, mapping


def compare_reference(occupancy, metadata, reference):
    array_path, meta_path = reference / "occupancy.npy", reference / "metadata.json"
    if not array_path.is_file() or not meta_path.is_file():
        return dict(status="UNAVAILABLE", reason="Reference map/metadata absent; build does not need them")
    original = np.load(array_path, allow_pickle=False)
    original_meta = json.loads(meta_path.read_text())
    # Exclude only nondeterministic build timing. All numerical/runtime fields must match.
    metadata_keys = sorted((set(original_meta) | set(metadata)) - {"build_time_s"})
    meta_differences = [key for key in metadata_keys if metadata.get(key) != original_meta.get(key)]
    same_shape = original.shape == occupancy.shape
    different = int(np.count_nonzero(original != occupancy)) if same_shape else None
    same_dtype = original.dtype == occupancy.dtype
    return dict(status="MATCH" if same_shape and same_dtype and different == 0 and not meta_differences else "MISMATCH",
                differing_voxels=different, reference_shape=list(original.shape),
                reference_dtype=str(original.dtype), dtype_match=same_dtype,
                metadata_differences=meta_differences, reference_npy_sha256=file_hash(array_path),
                per_layer_differing_voxels=[int(np.count_nonzero(a != b)) for a, b in zip(original, occupancy)] if same_shape else None)


def astar_smoke(occupancy, metadata, trials=10):
    from truck_uav_modules import build_and_benchmark_3d_voxel_astar as core
    rng = np.random.default_rng(20260728)
    offsets = core._neighbor_offsets((10.0, 10.0, 20.0))
    workspace = core.AStarWorkspace.create(occupancy.shape)
    core.validate_voxel_coordinate_conversion(occupancy, metadata, rng)
    labels, members = core.compute_3d_connected_components(occupancy)
    records = []
    for index in range(trials):
        start, goal, *_ = core.sample_reachable_3d_start_goal(members, labels, rng)
        path, cost, expanded, generated = core.plan_3d_path(
            occupancy, start, goal, workspace=workspace, offsets=offsets,
            voxel_size_xyz=(10.0, 10.0, 20.0), heuristic_weight=1.0)
        if path is None:
            raise AssertionError("A* failed for a pair sampled from a connected component")
        core.validate_3d_path(occupancy, path, start, goal, offsets, cost)
        records.append(dict(trial=index, start=list(start), goal=list(goal),
                            cost_m=cost, expanded_nodes=expanded, generated_nodes=generated,
                            path_valid=True))
    return dict(seed=20260728, trials=trials, validated_paths=len(records),
                neighbor_mode=26, prevent_corner_cutting=True, records=records)


def rebuild(scenes, output, root=ROOT, trials=10):
    output = repo_path(output, root)
    # Never write inside input trees, including the canonical voxel directories.
    protected = [root / "data", root / "truck_uav_modules/data", root / ".git"]
    if output == root.resolve() or any(output.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Output must be a NEW directory outside geographic input trees and .git")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output}")
    resolved = [(scene, *resolve_inputs(scene, root)) for scene in scenes]
    reports = []
    for scene, inputs, reference in resolved:
        print(f"Building {scene} from local GeoJSON inputs (no prebuilt-map dependency)...", flush=True)
        occupancy, metadata, summary, mapping = build_map(scene, inputs)
        comparison = compare_reference(occupancy, metadata, reference)
        smoke = astar_smoke(occupancy, metadata, trials)
        destination = output / scene
        destination.mkdir(parents=True)
        np.save(destination / "occupancy.npy", occupancy, allow_pickle=False)
        np.savez_compressed(destination / "occupancy.npz", occupancy_3d=occupancy)
        for name, record in [("metadata.json", metadata), ("summary.json", summary)]:
            (destination / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        mapping.to_csv(destination / "launch_recovery_grid_mapping.csv", index=False)
        report = dict(scene=scene, summary=summary, comparison=comparison, astar_smoke=smoke,
                      input_sha256={p.relative_to(root).as_posix(): file_hash(p) for p in inputs.values()},
                      output_npy_sha256=file_hash(destination / "occupancy.npy"))
        reports.append(report)
        print(f"{scene}: {summary['building_count']} buildings, {summary['occupied_voxels']} occupied voxels; "
              f"reference={comparison['status']}; A* validated={smoke['validated_paths']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rebuild_report.json").write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, geopandas=gpd.__version__,
        reports=reports, data_permission="Unverified; identity is not authorization",
        canonical_maps_modified=False, training_performed=False), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES + ("all",), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/voxel_rebuild"))
    parser.add_argument("--astar-trials", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.astar_trials <= 1000:
        parser.error("--astar-trials must be in [1,1000]")
    scenes = SCENES if args.scene == "all" else (args.scene,)
    try:
        reports = rebuild(scenes, args.output_dir, trials=args.astar_trials)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"{error}\n")
    return int(any(r["comparison"]["status"] == "MISMATCH" for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
