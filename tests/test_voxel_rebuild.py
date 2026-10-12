"""Offline voxel-generation tests with small non-paper geometry fixtures."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from tools.rebuild_voxel_maps import ROOT, compare_reference, rebuild, repo_path, resolve_inputs
from truck_uav_modules import rasterize_building_occupancy as raster


def fixture_buildings():
    return gpd.GeoDataFrame(dict(height_m=[10., 30.]),
                            geometry=[box(1, 1, 9, 9), box(11, 1, 19, 9)], crs="EPSG:32650")


def test_grid_derived_without_historical_2d_inputs():
    grid = raster.derive_grid_spec(fixture_buildings(), margin_m=0)
    assert (grid.n_rows, grid.n_cols) == (1, 2)
    assert grid.transform == (10., 0., 1., 0., -10., 9.)


def test_all_touched_and_center_only_have_explicit_semantics():
    grid = raster.RasterGridSpec(2, 2, 10, (10, 0, 0, 0, -10, 20), 0, 0, 20, 20, "EPSG:32650")
    tiny = gpd.GeoDataFrame(geometry=[box(9, 9, 11, 11)], crs="EPSG:32650")
    assert raster.rasterize_blocking_buildings(tiny, grid).sum() == 4
    assert raster.rasterize_blocking_buildings(tiny, grid, all_touched=False).sum() == 0
    assert raster.rasterize_blocking_buildings(tiny.iloc[:0], grid).sum() == 0


def test_max_height_and_pairwise_vertical_aggregation():
    buildings = raster.normalize_building_height(fixture_buildings())
    grid = raster.RasterGridSpec(1, 2, 10, (10, 0, 0, 0, -10, 10), 0, 0, 20, 10, "EPSG:32650")
    heights = raster.rasterize_max_height(buildings, grid)
    np.testing.assert_array_equal(heights, [[10, 30]])
    source = heights[None] > np.arange(0, 120, 10)[:, None, None]
    aggregated = source.reshape(6, 2, 1, 2).max(axis=1)
    np.testing.assert_array_equal(aggregated, heights[None] > np.arange(0, 120, 20)[:, None, None])
    assert not aggregated[1, 0, 0] and aggregated[1, 0, 1]


@pytest.mark.parametrize("value", [None, np.nan, 0, -1, np.inf, "unknown"])
def test_missing_or_bad_heights_are_not_imputed(value):
    buildings = fixture_buildings()
    buildings["height_m"] = buildings["height_m"].astype(object)
    buildings.loc[0, "height_m"] = value
    with pytest.raises(ValueError, match="finite and positive"):
        raster.normalize_building_height(buildings)


def test_launch_mapping_and_out_of_bounds_rejection():
    grid = raster.RasterGridSpec(1, 2, 10, (10, 0, 0, 0, -10, 10), 0, 0, 20, 10, "EPSG:32650")
    nodes = gpd.GeoDataFrame(dict(node_id=["n1"]), geometry=[Point(15, 5)], crs=grid.crs)
    mapping = raster.map_launch_nodes(nodes, grid, "unit")
    assert mapping.iloc[0][["row", "col"]].tolist() == [0, 1]
    nodes.geometry = [Point(25, 5)]
    with pytest.raises(ValueError, match="outside"):
        raster.map_launch_nodes(nodes, grid, "unit")


def test_reference_compare_does_not_need_reference_for_build(tmp_path):
    array = np.zeros((6, 2, 2), dtype="uint8")
    assert compare_reference(array, {}, tmp_path)["status"] == "UNAVAILABLE"
    np.save(tmp_path / "occupancy.npy", array)
    (tmp_path / "metadata.json").write_text(json.dumps(dict(scene="unit", build_time_s=9)))
    assert compare_reference(array, dict(scene="unit", build_time_s=1), tmp_path)["status"] == "MATCH"
    array[0, 0, 0] = 1
    assert compare_reference(array, dict(scene="unit"), tmp_path)["differing_voxels"] == 1


def test_safe_output_and_inputs(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        repo_path("../escaped", tmp_path)
    with pytest.raises(ValueError, match="outside geographic"):
        rebuild([], "truck_uav_modules/data/new", root=tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        rebuild([], existing, root=tmp_path)


def test_missing_authorized_data_has_actionable_message(tmp_path):
    config = tmp_path / "truck_uav_modules/configs"
    config.mkdir(parents=True)
    (config / "regions.yaml").write_text((ROOT / "truck_uav_modules/configs/regions.yaml").read_text())
    with pytest.raises(FileNotFoundError, match="authorized buildings"):
        resolve_inputs("xinjiekou", tmp_path)


def test_legacy_offline_loader_no_longer_requires_old_grids(tmp_path):
    from truck_uav_modules import build_and_benchmark_3d_voxel_astar as core
    for scene in core.SCENES:
        directory = tmp_path / scene
        directory.mkdir()
        buildings = fixture_buildings()
        buildings.to_file(directory / "buildings.geojson", driver="GeoJSON")
        nodes = gpd.GeoDataFrame(dict(node_id=["n"]), geometry=[Point(5, 5)], crs=buildings.crs)
        nodes.to_file(directory / "launch_recovery_nodes.geojson", driver="GeoJSON")
    metadata = core.load_grid_metadata(tmp_path)
    assert metadata["xinjiekou"]["grid_size_m"] == 10
    mapping = core.load_launch_recovery_mapping(tmp_path, "baijiahu")
    assert len(mapping) == 1 and mapping.in_bounds.all()
    assert not (tmp_path / "processed_grid_layers").exists()
