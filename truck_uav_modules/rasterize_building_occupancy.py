"""Create multi-height UAV occupancy grids for Xinjiekou and Baijiahu.

Install dependencies with:
    pip install numpy pandas geopandas matplotlib shapely rasterio pyproj

Values in every occupancy grid are uint8: 0 means free-flight and 1 means
occupied/non-flyable.  Original input files are read only and never modified.
"""

# from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import wkt
from shapely.geometry import Point, box

try:
    from rasterio.features import rasterize as rasterize_features
    from rasterio.transform import from_origin, rowcol, xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

    @dataclass(frozen=True)
    class SimpleTransform:
        """Small rasterio.transform.Affine-compatible fallback."""

        a: float
        b: float
        c: float
        d: float
        e: float
        f: float

        def __iter__(self):
            return iter((self.a, self.b, self.c, self.d, self.e, self.f))

    def from_origin(west: float, north: float, xsize: float, ysize: float) -> SimpleTransform:
        return SimpleTransform(xsize, 0.0, west, 0.0, -ysize, north)

    def rowcol(transform: SimpleTransform, x: float, y: float) -> Tuple[int, int]:
        return int(np.floor((transform.f - y) / abs(transform.e))), int(np.floor((x - transform.c) / transform.a))

    def xy(transform: SimpleTransform, row: int, col: int, offset: str = "center") -> Tuple[float, float]:
        offset_x = 0.5 if offset == "center" else 0.0
        offset_y = 0.5 if offset == "center" else 0.0
        return transform.c + (col + offset_x) * transform.a, transform.f - (row + offset_y) * abs(transform.e)

    rasterize_features = None

try:
    from shapely.validation import make_valid
except ImportError:  # Shapely < 2 fallback
    make_valid = None


LOGGER = logging.getLogger("building_occupancy")
SCENES = ("xinjiekou", "baijiahu")
FLIGHT_LAYERS_M = (20, 40, 60, 80, 100, 120)
VECTOR_SUFFIXES = {".geojson", ".json", ".shp", ".gpkg"}
TABULAR_SUFFIXES = {".csv", ".tsv"}
HEIGHT_COLUMN_CANDIDATES = (
    "height_m", "height", "building_height", "building_h", "bldg_height",
    "bld_height", "elevation", "roof_height", "altitude", "source_height_m",
    "source_height",
)
LEVEL_COLUMN_CANDIDATES = (
    "levels", "building_levels", "floors", "floor_count", "storeys", "stories",
)

# Change these values to regenerate grids at a different resolution or safety margin.
CONFIG = {
    "grid_size_m": 30.0,
    "flight_layers_m": FLIGHT_LAYERS_M,
    "map_padding_m": 50.0,
    "horizontal_safety_buffer_m": 0.0,
    "vertical_safety_margin_m": 0.0,
    "all_touched": True,
    "show_grid_lines": False,
    "dpi": 300,
    "default_floor_height_m": 3.0,
}


@dataclass(frozen=True)
class GridSpec:
    """Fixed spatial definition shared by all layers of one scene."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    grid_size_m: float
    n_rows: int
    n_cols: int
    transform: object
    crs: str

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.min_x, self.min_y, self.max_x, self.max_y


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(columns: Iterable[object], names: Sequence[str]) -> Optional[str]:
    normalized = {_normalized_name(column): str(column) for column in columns}
    for name in names:
        if _normalized_name(name) in normalized:
            return normalized[_normalized_name(name)]
    return None


def find_scene_building_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Recursively find scene-specific building/footprint files."""
    candidates = [
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VECTOR_SUFFIXES | TABULAR_SUFFIXES
    ]
    found: Dict[str, List[Path]] = {}
    for scene in SCENES:
        scene_files = []
        for path in candidates:
            path_text = _normalized_name(" ".join(path.parts))
            stem_text = _normalized_name(path.stem)
            if _normalized_name(scene) in path_text and any(
                word in stem_text for word in ("building", "bldg", "footprint")
            ):
                scene_files.append(path)
        found[scene] = sorted(
            scene_files,
            key=lambda path: ("buildingslocal" not in _normalized_name(path.stem), str(path)),
        )
    return found


def _geometry_from_table(table: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry_column = _find_column(table.columns, ("geometry", "geom", "wkt", "the_geom"))
    if geometry_column:
        geometry = table[geometry_column].map(
            lambda value: wkt.loads(value) if isinstance(value, str) and value.strip() else None
        )
        return gpd.GeoDataFrame(table.drop(columns=[geometry_column]), geometry=geometry)
    x_column = _find_column(table.columns, ("x", "lon", "longitude", "easting"))
    y_column = _find_column(table.columns, ("y", "lat", "latitude", "northing"))
    if x_column and y_column:
        geometry = [Point(x, y) for x, y in zip(table[x_column], table[y_column])]
        return gpd.GeoDataFrame(table, geometry=geometry)
    raise ValueError("CSV/TSV has no geometry/WKT or x/y coordinate columns")


def load_buildings(file_path: Path) -> gpd.GeoDataFrame:
    """Load a vector or tabular building file."""
    if file_path.suffix.lower() in VECTOR_SUFFIXES:
        gdf = gpd.read_file(file_path)
    elif file_path.suffix.lower() in TABULAR_SUFFIXES:
        table = pd.read_csv(file_path, sep="\t" if file_path.suffix.lower() == ".tsv" else ",")
        gdf = _geometry_from_table(table)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    if "geometry" not in gdf:
        raise ValueError("The building file has no geometry column")
    return gdf


def detect_height_column(gdf: gpd.GeoDataFrame) -> Tuple[Optional[str], bool]:
    """Return the height column and whether it contains floor counts."""
    height_column = _find_column(gdf.columns, HEIGHT_COLUMN_CANDIDATES)
    if height_column:
        return height_column, False
    level_column = _find_column(gdf.columns, LEVEL_COLUMN_CANDIDATES)
    return level_column, level_column is not None


def clean_building_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove unusable geometries and repair invalid polygons."""
    if "geometry" not in gdf:
        raise ValueError("No geometry column found")
    cleaned = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    initial_invalid = int((~cleaned.geometry.is_valid).sum())
    if initial_invalid:
        LOGGER.info("Repairing %d invalid geometries.", initial_invalid)
        if make_valid is not None:
            cleaned["geometry"] = cleaned.geometry.map(make_valid)
        else:
            cleaned["geometry"] = cleaned.geometry.buffer(0)
    still_invalid = ~cleaned.geometry.is_valid
    if still_invalid.any():
        cleaned.loc[still_invalid, "geometry"] = cleaned.loc[still_invalid, "geometry"].buffer(0)
    cleaned = cleaned.loc[cleaned.geometry.notna() & ~cleaned.geometry.is_empty].copy()
    cleaned = cleaned.loc[cleaned.geometry.geom_type.isin(("Polygon", "MultiPolygon"))].copy()
    LOGGER.info("Geometry cleanup retained %d polygon features.", len(cleaned))
    return cleaned


def normalize_building_height(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create numeric ``building_height_m`` from metres or floor counts."""
    height_column, is_levels = detect_height_column(gdf)
    if height_column is None:
        LOGGER.error("Available fields: %s", ", ".join(map(str, gdf.columns)))
        raise ValueError("No building height or floor-count field was found")
    values = pd.to_numeric(gdf[height_column], errors="coerce")
    if is_levels:
        LOGGER.info(
            "Using floor field '%s' with %.1f m per floor.",
            height_column,
            CONFIG["default_floor_height_m"],
        )
        values = values * CONFIG["default_floor_height_m"]
    else:
        LOGGER.info("Using building height field '%s'.", height_column)
    result = gdf.copy()
    result["building_height_m"] = values
    invalid_height = result["building_height_m"].isna() | (result["building_height_m"] < 0)
    if invalid_height.any():
        LOGGER.warning("Dropping %d records with missing or negative heights.", int(invalid_height.sum()))
        result = result.loc[~invalid_height].copy()
    unusual = result["building_height_m"] > 1000
    if unusual.any():
        LOGGER.warning("%d buildings have heights above 1000 m; please verify units.", int(unusual.sum()))
    return result


def ensure_projected_crs(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, str, str]:
    """Return metric data, original CRS text, and projected CRS text."""
    original_crs = str(gdf.crs) if gdf.crs is not None else "None"
    if gdf.crs is None:
        LOGGER.warning("Original CRS is missing; assuming coordinates are already metres.")
        return gdf, original_crs, original_crs
    if not gdf.crs.is_geographic:
        return gdf, original_crs, str(gdf.crs)
    projected_crs = gdf.estimate_utm_crs()
    if projected_crs is None:
        projected_crs = "EPSG:3857"
    LOGGER.info("Reprojecting from %s to %s.", original_crs, projected_crs)
    return gdf.to_crs(projected_crs), original_crs, str(projected_crs)


def calculate_fixed_grid_spec(gdf: gpd.GeoDataFrame, grid_size_m: float, map_padding_m: float) -> GridSpec:
    """Calculate one fixed grid extent and transform for all scene layers."""
    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be positive")
    min_x, min_y, max_x, max_y = map(float, gdf.total_bounds)
    min_x -= map_padding_m
    min_y -= map_padding_m
    max_x += map_padding_m
    max_y += map_padding_m
    n_cols = int(np.ceil((max_x - min_x) / grid_size_m))
    n_rows = int(np.ceil((max_y - min_y) / grid_size_m))
    # The raster extent is made exact after ceil so every cell is full-sized.
    max_x = min_x + n_cols * grid_size_m
    max_y = min_y + n_rows * grid_size_m
    transform = from_origin(min_x, max_y, grid_size_m, grid_size_m)
    return GridSpec(min_x, min_y, max_x, max_y, grid_size_m, n_rows, n_cols, transform, str(gdf.crs))


def select_blocking_buildings(gdf: gpd.GeoDataFrame, flight_layer_m: float, vertical_margin_m: float) -> gpd.GeoDataFrame:
    """Select buildings that reach the flight layer after vertical margin."""
    return gdf.loc[gdf["building_height_m"] + vertical_margin_m >= flight_layer_m].copy()


def rasterize_blocking_buildings(
    blocking_gdf: gpd.GeoDataFrame,
    grid_spec: GridSpec,
    horizontal_buffer_m: float,
    all_touched: bool,
) -> np.ndarray:
    """Rasterize buffered footprints, using rasterio when it is available."""
    geometries = blocking_gdf.geometry
    if horizontal_buffer_m:
        geometries = geometries.buffer(horizontal_buffer_m)
    valid_geometries = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if HAS_RASTERIO:
        shapes = ((geometry, 1) for geometry in valid_geometries)
        return rasterize_features(
            shapes=shapes,
            out_shape=(grid_spec.n_rows, grid_spec.n_cols),
            transform=grid_spec.transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )

    # Fallback for environments where the optional rasterio wheel is unavailable.
    # It visits only cells in each footprint's bounding box, never every
    # building for every cell in the full scene.
    grid = np.zeros((grid_spec.n_rows, grid_spec.n_cols), dtype=np.uint8)
    for geometry in valid_geometries:
        geom_min_x, geom_min_y, geom_max_x, geom_max_y = geometry.bounds
        min_row, min_col = rowcol(grid_spec.transform, geom_min_x, geom_max_y)
        max_row, max_col = rowcol(grid_spec.transform, geom_max_x, geom_min_y)
        min_row = max(0, min_row - 1)
        min_col = max(0, min_col - 1)
        max_row = min(grid_spec.n_rows - 1, max_row + 1)
        max_col = min(grid_spec.n_cols - 1, max_col + 1)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = box(
                    grid_spec.min_x + col * grid_spec.grid_size_m,
                    grid_spec.max_y - (row + 1) * grid_spec.grid_size_m,
                    grid_spec.min_x + (col + 1) * grid_spec.grid_size_m,
                    grid_spec.max_y - row * grid_spec.grid_size_m,
                )
                if geometry.intersects(cell) if all_touched else geometry.covers(cell.centroid):
                    grid[row, col] = 1
    return grid


def world_to_grid(x: float, y: float, transform, rows: int, cols: int) -> Tuple[int, int]:
    """Convert metric world coordinates to NumPy (row, col), with bounds checking."""
    row, col = rowcol(transform, x, y)
    row, col = int(row), int(col)
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"world coordinate ({x}, {y}) is outside the grid")
    return row, col


def grid_to_world(row: int, col: int, transform) -> Tuple[float, float]:
    """Convert a NumPy cell index to the cell-center metric world coordinate."""
    x_value, y_value = xy(transform, row, col, offset="center")
    return float(x_value), float(y_value)


def validate_coordinate_conversion(grid_spec: GridSpec) -> None:
    """Check cell-center world/grid round trips for representative cells."""
    samples = {(0, 0), (grid_spec.n_rows - 1, grid_spec.n_cols - 1), (grid_spec.n_rows // 2, grid_spec.n_cols // 2)}
    for row, col in samples:
        x_value, y_value = grid_to_world(row, col, grid_spec.transform)
        converted = world_to_grid(x_value, y_value, grid_spec.transform, grid_spec.n_rows, grid_spec.n_cols)
        if converted != (row, col):
            raise AssertionError(f"Coordinate round trip failed: {(row, col)} -> {converted}")
    LOGGER.info("Coordinate conversion validation passed for %d cells.", len(samples))


def _transform_values(transform) -> List[float]:
    return [float(value) for value in tuple(transform)]


def save_grid_data(
    grid: np.ndarray,
    grid_spec: GridSpec,
    scene_name: str,
    flight_layer_m: int,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save raw uint8 NPY and metadata-rich compressed NPZ files."""
    stem = f"{scene_name}_occupancy_grid_{flight_layer_m}m"
    npy_path = output_dir / f"{stem}.npy"
    npz_path = output_dir / f"{stem}.npz"
    np.save(npy_path, grid.astype(np.uint8, copy=False))
    np.savez_compressed(
        npz_path,
        occupancy_grid=grid.astype(np.uint8, copy=False),
        scene_name=np.array(scene_name),
        flight_layer_m=np.array(flight_layer_m),
        grid_size_m=np.array(grid_spec.grid_size_m),
        n_rows=np.array(grid_spec.n_rows),
        n_cols=np.array(grid_spec.n_cols),
        min_x=np.array(grid_spec.min_x),
        min_y=np.array(grid_spec.min_y),
        max_x=np.array(grid_spec.max_x),
        max_y=np.array(grid_spec.max_y),
        transform=np.array(_transform_values(grid_spec.transform)),
        crs=np.array(grid_spec.crs),
        all_touched=np.array(CONFIG["all_touched"]),
        horizontal_safety_buffer_m=np.array(CONFIG["horizontal_safety_buffer_m"]),
        vertical_safety_margin_m=np.array(CONFIG["vertical_safety_margin_m"]),
    )
    return npy_path, npz_path


def save_scene_metadata(scene_name: str, grid_spec: GridSpec, output_dir: Path) -> Path:
    """Save one JSON metadata file shared by all height layers of a scene."""
    metadata = {
        "scene": scene_name,
        "crs": grid_spec.crs,
        "grid_size_m": grid_spec.grid_size_m,
        "rows": grid_spec.n_rows,
        "cols": grid_spec.n_cols,
        "bounds": dict(zip(("min_x", "min_y", "max_x", "max_y"), grid_spec.bounds)),
        "map_padding_m": CONFIG["map_padding_m"],
        "all_touched": CONFIG["all_touched"],
        "horizontal_safety_buffer_m": CONFIG["horizontal_safety_buffer_m"],
        "vertical_safety_margin_m": CONFIG["vertical_safety_margin_m"],
        "layers_m": list(CONFIG["flight_layers_m"]),
        "transform": _transform_values(grid_spec.transform),
    }
    path = output_dir / f"{scene_name}_grid_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def plot_occupancy_grid(
    grid: np.ndarray,
    buildings: gpd.GeoDataFrame,
    blocking_buildings: gpd.GeoDataFrame,
    scene_name: str,
    flight_layer_m: int,
    grid_spec: GridSpec,
    output_path: Path,
) -> None:
    """Render one clean 2-D occupancy map with blocking footprint outlines."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    cmap = ListedColormap(["#f7f7f7", "#666b73"])
    ax.imshow(
        grid,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="none",
        origin="upper",
        extent=(grid_spec.min_x, grid_spec.max_x, grid_spec.min_y, grid_spec.max_y),
    )
    buildings.boundary.plot(ax=ax, color="#c4c8cd", linewidth=0.12, alpha=0.55)
    if not blocking_buildings.empty:
        buffered = blocking_buildings.geometry.buffer(CONFIG["horizontal_safety_buffer_m"])
        gpd.GeoSeries(buffered, crs=blocking_buildings.crs).boundary.plot(
            ax=ax, color="#b33a32", linewidth=0.35, alpha=0.85
        )
    if CONFIG["show_grid_lines"]:
        ax.set_xticks(np.arange(grid_spec.min_x, grid_spec.max_x + grid_spec.grid_size_m, grid_spec.grid_size_m), minor=True)
        ax.set_yticks(np.arange(grid_spec.min_y, grid_spec.max_y + grid_spec.grid_size_m, grid_spec.grid_size_m), minor=True)
        ax.grid(which="minor", color="#b8bcc2", linewidth=0.2, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{scene_name.title()} UAV flight occupancy grid at {flight_layer_m} m", fontsize=13)
    ax.legend(
        handles=[
            Patch(facecolor="#f7f7f7", edgecolor="#999999", label="Free-flight cell"),
            Patch(facecolor="#666b73", edgecolor="#444444", label="Blocked cell"),
            Line2D([0], [0], color="#b33a32", linewidth=1.2, label="Blocking building footprint"),
        ],
        loc="upper right",
        framealpha=0.95,
        fontsize=8.5,
    )
    total_cells = int(grid.size)
    blocked_cells = int(grid.sum())
    free_cells = total_cells - blocked_cells
    ratio = blocked_cells / total_cells if total_cells else 0.0
    text = (
        f"Grid size: {grid_spec.grid_size_m:g} m\n"
        f"Rows x Columns: {grid_spec.n_rows} x {grid_spec.n_cols}\n"
        f"Total cells: {total_cells:,}\n"
        f"Blocked cells: {blocked_cells:,}\n"
        f"Blocked ratio: {ratio:.2%}\n"
        f"Blocking buildings: {len(blocking_buildings):,}"
    )
    ax.text(0.01, 0.01, text, transform=ax.transAxes, va="bottom", fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#b8bec5", "alpha": 0.92, "pad": 4})
    fig.savefig(output_path, dpi=CONFIG["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)


def export_summary_csv(records: List[dict], output_path: Path) -> None:
    """Write per-layer occupancy statistics."""
    columns = [
        "scene", "flight_layer_m", "grid_size_m", "rows", "cols", "total_cells",
        "free_cells", "blocked_cells", "blocked_ratio", "total_buildings",
        "blocking_buildings", "crs", "all_touched", "horizontal_safety_buffer_m",
        "vertical_safety_margin_m",
    ]
    pd.DataFrame(records, columns=columns).to_csv(output_path, index=False)


def process_scene(scene_name: str, building_file: Path, config: dict, figure_dir: Path, grid_output_dir: Path) -> List[dict]:
    """Load one scene once, then create six fixed-shape occupancy layers."""
    LOGGER.info("[%s] Loading %s", scene_name.title(), building_file)
    raw = load_buildings(building_file)
    original_crs = str(raw.crs) if raw.crs is not None else "None"
    buildings = normalize_building_height(clean_building_geometries(raw))
    buildings, _, projected_crs = ensure_projected_crs(buildings)
    grid_spec = calculate_fixed_grid_spec(buildings, config["grid_size_m"], config["map_padding_m"])
    validate_coordinate_conversion(grid_spec)
    LOGGER.info("[%s] Original CRS: %s", scene_name.title(), original_crs)
    LOGGER.info("[%s] Projected CRS: %s", scene_name.title(), projected_crs)
    LOGGER.info("[%s] Bounds: %s", scene_name.title(), grid_spec.bounds)
    LOGGER.info("[%s] Grid size: %.1f m; rows=%d; columns=%d; total=%d", scene_name.title(), config["grid_size_m"], grid_spec.n_rows, grid_spec.n_cols, grid_spec.n_rows * grid_spec.n_cols)
    save_scene_metadata(scene_name, grid_spec, grid_output_dir)
    records = []
    for flight_layer_m in config["flight_layers_m"]:
        blocking = select_blocking_buildings(buildings, flight_layer_m, config["vertical_safety_margin_m"])
        grid = rasterize_blocking_buildings(blocking, grid_spec, config["horizontal_safety_buffer_m"], config["all_touched"])
        grid_path, _ = save_grid_data(grid, grid_spec, scene_name, flight_layer_m, grid_output_dir)
        figure_path = figure_dir / f"{scene_name}_occupancy_grid_{flight_layer_m}m.png"
        plot_occupancy_grid(grid, buildings, blocking, scene_name, flight_layer_m, grid_spec, figure_path)
        blocked_cells = int(grid.sum())
        total_cells = int(grid.size)
        free_cells = total_cells - blocked_cells
        blocked_ratio = blocked_cells / total_cells if total_cells else 0.0
        records.append({
            "scene": scene_name,
            "flight_layer_m": flight_layer_m,
            "grid_size_m": config["grid_size_m"],
            "rows": grid_spec.n_rows,
            "cols": grid_spec.n_cols,
            "total_cells": total_cells,
            "free_cells": free_cells,
            "blocked_cells": blocked_cells,
            "blocked_ratio": blocked_ratio,
            "total_buildings": len(buildings),
            "blocking_buildings": len(blocking),
            "crs": projected_crs,
            "all_touched": config["all_touched"],
            "horizontal_safety_buffer_m": config["horizontal_safety_buffer_m"],
            "vertical_safety_margin_m": config["vertical_safety_margin_m"],
        })
        print(f"[{scene_name.title()}][{flight_layer_m}m]")
        print(f"Total buildings: {len(buildings)}")
        print(f"Blocking buildings: {len(blocking)}")
        print(f"Free cells: {free_cells}")
        print(f"Blocked cells: {blocked_cells}")
        print(f"Blocked ratio: {blocked_ratio:.4%}")
        print(f"Saved grid: {grid_path}")
        print(f"Saved figure: {figure_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--grid-output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or project_root / "truck_uav_modules" / "data"
    figure_dir = args.figure_dir or project_root / "truck_uav_modules" / "figure"
    grid_output_dir = args.grid_output_dir or data_dir / "processed_grid_layers"
    figure_dir.mkdir(parents=True, exist_ok=True)
    grid_output_dir.mkdir(parents=True, exist_ok=True)

    scene_files = find_scene_building_files(data_dir)
    records: List[dict] = []
    for scene_name in SCENES:
        files = scene_files.get(scene_name, [])
        if not files:
            LOGGER.error("[%s] No building data file found under %s", scene_name.title(), data_dir)
            continue
        try:
            records.extend(process_scene(scene_name, files[0], CONFIG, figure_dir, grid_output_dir))
        except Exception as exc:
            LOGGER.exception("[%s] Failed: %s", scene_name.title(), exc)
    summary_path = grid_output_dir / "occupancy_grid_summary.csv"
    export_summary_csv(records, summary_path)
    LOGGER.info("Wrote %d layer records to %s", len(records), summary_path)


if __name__ == "__main__":
    main()
