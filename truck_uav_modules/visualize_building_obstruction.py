"""Visualize building obstruction at multiple UAV flight layers.

Dependencies:
    pip install geopandas shapely matplotlib pandas

The script scans ``truck_uav_modules/data`` for scene-specific building
files, normalizes their geometry and height information, and writes six
figures plus one CSV summary for each scene.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from shapely import wkt
from shapely.geometry import Point


LOGGER = logging.getLogger("building_obstruction")
SCENES = ("xinjiekou", "baijiahu")
FLIGHT_LAYERS_M = (20, 40, 60, 80, 100, 120)
VECTOR_SUFFIXES = {".geojson", ".json", ".shp", ".gpkg", ".parquet", ".feather"}
TABULAR_SUFFIXES = {".csv", ".tsv"}
HEIGHT_FIELDS = (
    "height_m", "height", "building_height", "bldg_height", "source_height_m",
    "source_height", "buildingheight", "bldgheight",
)
FLOOR_FIELDS = ("floors", "floor", "levels", "level", "building_levels", "num_floors")


def _normalized_name(value: object) -> str:
    """Normalize a column/path name for case- and separator-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_scene_building_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Find likely building data files for each requested scene."""
    candidates = [
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VECTOR_SUFFIXES | TABULAR_SUFFIXES
    ]
    result: Dict[str, List[Path]] = {}
    for scene in SCENES:
        scene_candidates = []
        for path in candidates:
            haystack = _normalized_name(" ".join(path.parts))
            if _normalized_name(scene) not in haystack:
                continue
            filename = _normalized_name(path.stem)
            if "building" in filename or "bldg" in filename:
                scene_candidates.append(path)
        # Prefer the packaged local footprint product when multiple files exist.
        result[scene] = sorted(
            scene_candidates,
            key=lambda p: ("buildingslocal" not in _normalized_name(p.stem), str(p)),
        )
    return result


def _find_column(columns: Iterable[object], names: Sequence[str]) -> Optional[str]:
    by_normalized = {_normalized_name(column): str(column) for column in columns}
    for name in names:
        if _normalized_name(name) in by_normalized:
            return by_normalized[_normalized_name(name)]
    return None


def _geometry_from_table(table: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build geometry from a CSV/TSV geometry-like column when possible."""
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
    raise ValueError("tabular file has no geometry/WKT column or x/y coordinate columns")


def load_buildings(file_path: Path) -> gpd.GeoDataFrame:
    """Load a building file and retain its source CRS when available."""
    suffix = file_path.suffix.lower()
    if suffix in VECTOR_SUFFIXES:
        gdf = gpd.read_file(file_path)
    elif suffix in TABULAR_SUFFIXES:
        table = pd.read_csv(file_path, sep="\t" if suffix == ".tsv" else ",")
        gdf = _geometry_from_table(table)
    else:
        raise ValueError(f"unsupported building file format: {file_path.suffix}")
    if "geometry" not in gdf:
        raise ValueError("file does not contain a geometry column")
    return gdf


def detect_height_column(gdf: gpd.GeoDataFrame) -> Tuple[Optional[str], bool]:
    """Return (column, is_floor_count) using common height/level field names."""
    height_column = _find_column(gdf.columns, HEIGHT_FIELDS)
    if height_column:
        return height_column, False
    floor_column = _find_column(gdf.columns, FLOOR_FIELDS)
    return floor_column, floor_column is not None


def normalize_building_height(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clean geometry and create a numeric ``height_m`` column."""
    if gdf.crs is None:
        LOGGER.warning("Input CRS is missing; coordinates will be treated as planar metres.")

    cleaned = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    invalid_count = int((~cleaned.geometry.is_valid).sum())
    if invalid_count:
        LOGGER.info("Repairing %d invalid geometries with buffer(0).", invalid_count)
        cleaned["geometry"] = cleaned.geometry.buffer(0)
    cleaned = cleaned.loc[cleaned.geometry.notna() & ~cleaned.geometry.is_empty].copy()
    cleaned = cleaned.loc[cleaned.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    height_column, is_floor_count = detect_height_column(cleaned)
    if height_column is None:
        raise ValueError(
            "no building height field found; expected height/height_m/building_height "
            "or floors/levels"
        )
    values = pd.to_numeric(cleaned[height_column], errors="coerce")
    if is_floor_count:
        LOGGER.info("Using floor field '%s' and converting floors to height with 3 m/floor.", height_column)
        values = values * 3.0
    else:
        LOGGER.info("Using building height field '%s'.", height_column)
    cleaned["height_m"] = values
    before_drop = len(cleaned)
    cleaned = cleaned.loc[cleaned["height_m"].notna() & (cleaned["height_m"] >= 0)].copy()
    dropped = before_drop - len(cleaned)
    if dropped:
        LOGGER.warning("Dropped %d buildings with missing or negative height values.", dropped)
    return cleaned


def _project_for_plot(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project geographic input to a local metric CRS for plotting."""
    if gdf.crs is None or not gdf.crs.is_geographic:
        return gdf
    try:
        metric_crs = gdf.estimate_utm_crs()
    except Exception as exc:
        LOGGER.warning("Could not estimate UTM CRS (%s); using EPSG:3857.", exc)
        metric_crs = "EPSG:3857"
    if metric_crs is None:
        metric_crs = "EPSG:3857"
    LOGGER.info("Reprojecting geographic input from %s to %s.", gdf.crs, metric_crs)
    return gdf.to_crs(metric_crs)


def plot_single_scene_height_layer(
    gdf: gpd.GeoDataFrame,
    scene_name: str,
    layer_height: int,
    output_path: Path,
) -> Tuple[int, int, float]:
    """Plot one scene at one flight layer and return total/blocking/ratio."""
    total = len(gdf)
    blocking = gdf["height_m"] >= layer_height
    blocking_count = int(blocking.sum())
    ratio = blocking_count / total if total else 0.0

    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    gdf.loc[~blocking].plot(ax=ax, color="#d9dde2", edgecolor="#9aa3ad", linewidth=0.25)
    gdf.loc[blocking].plot(ax=ax, color="#d95f4f", edgecolor="#8f2d25", linewidth=0.3)
    ax.set_title(
        f"{scene_name.title()} building obstruction footprints at {layer_height} m flight layer",
        fontsize=13,
        pad=12,
    )
    ax.set_axis_off()
    ax.legend(
        handles=[
            Patch(facecolor="#d9dde2", edgecolor="#9aa3ad", label="Buildings below flight altitude"),
            Patch(facecolor="#d95f4f", edgecolor="#8f2d25", label="Buildings blocking straight flight at this altitude"),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )
    ax.text(
        0.01,
        0.02,
        f"Total: {total:,}   Blocking: {blocking_count:,}   Ratio: {ratio:.1%}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#b8bec5", "alpha": 0.9, "pad": 4},
    )
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return total, blocking_count, ratio


def export_summary_csv(summary_records: List[dict], output_csv_path: Path) -> None:
    """Write the per-scene/per-layer summary table."""
    columns = ["scene", "layer_m", "total_buildings", "blocking_buildings", "blocking_ratio"]
    pd.DataFrame(summary_records, columns=columns).to_csv(output_csv_path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the packaged data directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the figure output directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or project_root / "truck_uav_modules" / "data"
    output_dir = args.output_dir or project_root / "truck_uav_modules" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_files = find_scene_building_files(data_dir)
    summary_records: List[dict] = []
    for scene in SCENES:
        files = scene_files.get(scene, [])
        if not files:
            LOGGER.error("[%s] No building data file found below %s.", scene.title(), data_dir)
            continue
        file_path = files[0]
        LOGGER.info("[%s] Loading buildings from %s", scene.title(), file_path)
        try:
            buildings = _project_for_plot(normalize_building_height(load_buildings(file_path)))
        except Exception as exc:
            LOGGER.error("[%s] Could not load building data: %s", scene.title(), exc)
            continue
        for layer_height in FLIGHT_LAYERS_M:
            output_path = output_dir / f"{scene}_building_obstruction_{layer_height}m.png"
            total, blocking, ratio = plot_single_scene_height_layer(
                buildings, scene, layer_height, output_path
            )
            summary_records.append(
                {
                    "scene": scene,
                    "layer_m": layer_height,
                    "total_buildings": total,
                    "blocking_buildings": blocking,
                    "blocking_ratio": ratio,
                }
            )
            print(f"[{scene.title()}][{layer_height}m] total={total}, blocking={blocking}, ratio={ratio:.4f}")

    export_summary_csv(summary_records, output_dir / "building_obstruction_summary.csv")
    LOGGER.info("Wrote %d summary records to %s", len(summary_records), output_dir)


if __name__ == "__main__":
    main()
