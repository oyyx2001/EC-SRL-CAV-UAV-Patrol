"""Offline building rasterization using Shapely (no rasterio or external grids).

These helpers are not called by policy evaluation. Heights are never imputed.
All-touched means closed-cell geometric intersection, including cell boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


@dataclass(frozen=True)
class RasterGridSpec:
    n_rows: int
    n_cols: int
    grid_size_m: float
    transform: tuple
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: str

    def metadata(self) -> dict:
        return dict(rows=self.n_rows, cols=self.n_cols, grid_size_m=self.grid_size_m,
                    transform=list(self.transform), crs=self.crs,
                    bounds=dict(min_x=self.min_x, min_y=self.min_y,
                                max_x=self.max_x, max_y=self.max_y))


def load_buildings(path: Path) -> gpd.GeoDataFrame:
    buildings = gpd.read_file(path)
    if buildings.empty or buildings.crs is None or not buildings.crs.is_projected:
        raise ValueError("Buildings need a nonempty projected, metre-based CRS")
    if any(axis.unit_conversion_factor != 1 for axis in buildings.crs.axis_info):
        raise ValueError("Building CRS units must be metres")
    return buildings


def clean_building_geometries(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = buildings.copy()
    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise ValueError("Missing/empty building geometry; no features may be silently dropped")
    result.geometry = shapely.make_valid(result.geometry.to_numpy())
    if (~result.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).any():
        raise ValueError("Buildings must contain polygonal footprints")
    if (result.geometry.area <= 0).any() or not np.isfinite(result.total_bounds).all():
        raise ValueError("Invalid building footprint area or coordinates")
    return result


def normalize_building_height(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "height_m" not in buildings:
        raise ValueError("Missing required height_m; heights are not inferred")
    result = buildings.copy()
    heights = pd.to_numeric(result["height_m"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(heights).all() or (heights <= 0).any():
        raise ValueError("height_m must be finite and positive; no default heights are inserted")
    result["building_height_m"] = heights
    return result


def derive_grid_spec(buildings: gpd.GeoDataFrame, grid_size_m=10.0,
                     margin_m=50.0) -> RasterGridSpec:
    if not math.isfinite(grid_size_m) or grid_size_m <= 0:
        raise ValueError("grid_size_m must be positive")
    if not math.isfinite(margin_m) or margin_m < 0 or buildings.empty:
        raise ValueError("Need nonempty buildings and a nonnegative margin")
    min_x, min_y, max_x, max_y = map(float, buildings.total_bounds)
    min_x -= margin_m
    max_y += margin_m
    cols = math.ceil((max_x + margin_m - min_x) / grid_size_m)
    rows = math.ceil((max_y - (min_y - margin_m)) / grid_size_m)
    if rows <= 0 or cols <= 0 or rows * cols > 10_000_000:
        raise ValueError("Invalid or excessive grid extent")
    return RasterGridSpec(rows, cols, grid_size_m,
                          (grid_size_m, 0.0, min_x, 0.0, -grid_size_m, max_y),
                          min_x, max_y - rows * grid_size_m,
                          min_x + cols * grid_size_m, max_y, str(buildings.crs))


def cell_geometries(grid_spec, all_touched=True):
    values = tuple(grid_spec.transform)
    a, b, x, d, e, y = values[:6]
    if a <= 0 or e >= 0 or b != 0 or d != 0:
        raise ValueError("Only axis-aligned, north-up grids are supported")
    rows, cols = np.indices((grid_spec.n_rows, grid_spec.n_cols))
    if all_touched:
        return shapely.box(x + cols.ravel() * a, y + (rows.ravel() + 1) * e,
                           x + (cols.ravel() + 1) * a, y + rows.ravel() * e)
    return shapely.points(x + (cols.ravel() + .5) * a, y + (rows.ravel() + .5) * e)


def intersection_pairs(geometries, grid_spec, horizontal_buffer_m=0.0,
                       all_touched=True):
    if not math.isfinite(horizontal_buffer_m) or horizontal_buffer_m < 0:
        raise ValueError("Buffer must be finite and nonnegative")
    geometries = np.asarray(geometries, dtype=object)
    if horizontal_buffer_m:
        geometries = shapely.buffer(geometries, horizontal_buffer_m, quad_segs=16)
    return shapely.STRtree(geometries).query(cell_geometries(grid_spec, all_touched),
                                            predicate="intersects")


def rasterize_blocking_buildings(buildings, grid_spec, horizontal_buffer_m=0.0,
                                 all_touched=True) -> np.ndarray:
    grid = np.zeros(grid_spec.n_rows * grid_spec.n_cols, dtype=np.uint8)
    pairs = intersection_pairs(buildings.geometry.to_numpy(), grid_spec,
                               horizontal_buffer_m, all_touched)
    grid[pairs[0]] = 1
    return grid.reshape(grid_spec.n_rows, grid_spec.n_cols)


def rasterize_max_height(buildings, grid_spec) -> np.ndarray:
    pairs = intersection_pairs(buildings.geometry.to_numpy(), grid_spec)
    heights = np.zeros(grid_spec.n_rows * grid_spec.n_cols, dtype=float)
    np.maximum.at(heights, pairs[0], buildings.building_height_m.to_numpy()[pairs[1]])
    return heights.reshape(grid_spec.n_rows, grid_spec.n_cols)


def map_launch_nodes(nodes: gpd.GeoDataFrame, grid_spec, scene: str) -> pd.DataFrame:
    if nodes.empty or nodes.crs != grid_spec.crs or (nodes.geom_type != "Point").any():
        raise ValueError("Launch nodes must be nonempty points in the building CRS")
    a, _, x, _, e, y = tuple(grid_spec.transform)[:6]
    if not np.isfinite(np.array([nodes.geometry.x, nodes.geometry.y])).all():
        raise ValueError("Nonfinite launch-node coordinates")
    cols = np.floor((nodes.geometry.x.to_numpy() - x) / a).astype(int)
    rows = np.floor((nodes.geometry.y.to_numpy() - y) / e).astype(int)
    inside = (rows >= 0) & (rows < grid_spec.n_rows) & (cols >= 0) & (cols < grid_spec.n_cols)
    if not inside.all():
        raise ValueError("Launch nodes lie outside the derived grid")
    return pd.DataFrame(dict(scene=scene, node_id=nodes["node_id"].to_numpy(),
                             row=rows, col=cols, node_x=nodes.geometry.x.to_numpy(),
                             node_y=nodes.geometry.y.to_numpy(), in_bounds=inside))
