"""Create publication-ready 2-D and 3-D maps for the two formal case studies."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon


SCENE_LABELS = {"baijiahu": "Baijiahu", "xinjiekou": "Xinjiekou"}
SCENE_ORDER = ("baijiahu", "xinjiekou")
CMAP = matplotlib.colormaps["RdYlGn_r"]
HEIGHT_MIN_M = 0.0
HEIGHT_MAX_M = 260.0
HEIGHT_NORM = colors.Normalize(vmin=HEIGHT_MIN_M, vmax=HEIGHT_MAX_M, clip=True)
ROAD_COLOR = "#44484c"
NODE_COLOR = "#b2182b"
BOUNDARY_COLOR = "#6d7277"
BACKGROUND_COLOR = "#ffffff"
GROUND_COLOR = "#f5f5f3"


@dataclass(frozen=True)
class SceneData:
    key: str
    label: str
    buildings: gpd.GeoDataFrame
    roads: gpd.GeoDataFrame
    nodes: gpd.GeoDataFrame
    bounds: tuple[float, float, float, float]
    sources: dict[str, str]

    @property
    def width_m(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height_m(self) -> float:
        return self.bounds[3] - self.bounds[1]


def configure_style() -> None:
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    if "Times New Roman" not in available:
        raise RuntimeError("Times New Roman is not installed")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BACKGROUND_COLOR,
            "savefig.edgecolor": "none",
        }
    )


def load_scene(scene: str) -> SceneData:
    scene_dir = PROJECT_ROOT / "truck_uav_modules" / "data" / scene
    source_paths = {
        "buildings": scene_dir / "buildings.geojson",
        "roads": scene_dir / "road_edges.geojson",
        "launch_recovery_nodes": scene_dir / "launch_recovery_nodes.geojson",
        "voxel_metadata": scene_dir / "voxel_map_3d" / "metadata.json",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing formal {scene} assets: {missing}")

    buildings = gpd.read_file(source_paths["buildings"])
    roads = gpd.read_file(source_paths["roads"])
    nodes = gpd.read_file(source_paths["launch_recovery_nodes"])
    metadata = json.loads(source_paths["voxel_metadata"].read_text(encoding="utf-8"))
    height = np.asarray(buildings["height_m"], dtype=float)
    if not np.isfinite(height).all() or (height < 0.0).any():
        raise ValueError(f"{scene} has missing or invalid building heights")
    expected_crs = "EPSG:32650"
    if any(str(frame.crs) != expected_crs for frame in (buildings, roads, nodes)):
        raise ValueError(f"{scene} data must use {expected_crs}")

    raw_bounds = metadata["bounds"]
    bounds = tuple(float(raw_bounds[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
    return SceneData(
        key=scene,
        label=SCENE_LABELS[scene],
        buildings=buildings,
        roads=roads,
        nodes=nodes,
        bounds=bounds,
        sources={name: str(path.relative_to(PROJECT_ROOT)) for name, path in source_paths.items()},
    )


def iter_polygons(geometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def iter_lines(geometry) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def projected_line_segments(data: SceneData) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for geometry in data.roads.geometry:
        for line in iter_lines(geometry):
            segments.append(np.asarray(line.coords, dtype=float))
    return segments


def draw_scale_bar_2d(ax, data: SceneData, length_m: float = 500.0) -> None:
    min_x, min_y, _, _ = data.bounds
    x0 = min_x + data.width_m * 0.055
    y0 = min_y + data.height_m * 0.055
    cap = data.height_m * 0.008
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=1.6, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0], [y0 - cap, y0 + cap], color="black", linewidth=1.1, zorder=8)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - cap, y0 + cap], color="black", linewidth=1.1, zorder=8)
    ax.text(x0 + length_m / 2.0, y0 + cap * 1.7, f"{int(length_m)} m", ha="center", va="bottom", fontsize=11)


def draw_north_arrow_2d(ax, data: SceneData) -> None:
    min_x, min_y, max_x, max_y = data.bounds
    x = max_x - data.width_m * 0.055
    base_y = max_y - data.height_m * 0.15
    tip_y = max_y - data.height_m * 0.045
    ax.annotate(
        "",
        xy=(x, tip_y),
        xytext=(x, base_y),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.1},
        zorder=9,
    )
    ax.text(
        x,
        tip_y + data.height_m * 0.012,
        "N",
        ha="center",
        va="bottom",
        fontsize=12,
        zorder=9,
    )


def add_colorbar(fig, ax, *, pad: float = 0.018, shrink: float = 0.74) -> None:
    scalar = ScalarMappable(norm=HEIGHT_NORM, cmap=CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.032, pad=pad, shrink=shrink)
    colorbar.set_label("Building height (m)", labelpad=6)
    colorbar.set_ticks([0, 50, 100, 150, 200, 250])
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.tick_params(length=2.5, width=0.55)


def plot_2d(data: SceneData, *, figsize: tuple[float, float] = (8.4, 8.0)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    facecolors = CMAP(HEIGHT_NORM(np.asarray(data.buildings["height_m"], dtype=float)))
    data.buildings.plot(
        ax=ax,
        facecolor=facecolors,
        edgecolor="#707070",
        linewidth=0.10,
        alpha=0.88,
        zorder=1,
    )
    ax.add_collection(
        LineCollection(projected_line_segments(data), colors=ROAD_COLOR, linewidths=0.90, alpha=0.96, zorder=4)
    )
    min_x, min_y, max_x, max_y = data.bounds

    data.nodes.plot(
        ax=ax,
        color=NODE_COLOR,
        edgecolor="white",
        linewidth=0.28,
        markersize=9.0,
        alpha=0.98,
        zorder=6,
    )
    ax.add_patch(
        Rectangle(
            (min_x, min_y),
            data.width_m,
            data.height_m,
            facecolor="none",
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.65,
            zorder=7,
        )
    )
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"{data.label} case study - 2D urban map", pad=8)
    draw_scale_bar_2d(ax, data)
    draw_north_arrow_2d(ax, data)
    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=ROAD_COLOR, linewidth=1.8, label="Road network"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=NODE_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=7.0,
                label=f"Launch/recovery nodes (n={len(data.nodes)})",
            ),
        ],
        loc="upper left",
        frameon=True,
        borderaxespad=0.8,
        handlelength=1.7,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#808080")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(20)
    add_colorbar(fig, ax, shrink=0.78)
    fig.subplots_adjust(left=0.025, right=0.91, bottom=0.025, top=0.95)
    return fig


def build_3d_faces(data: SceneData) -> tuple[list[list[tuple[float, float, float]]], list[np.ndarray], list[list[tuple[float, float, float]]], list[np.ndarray]]:
    min_x, min_y, _, _ = data.bounds
    tops: list[list[tuple[float, float, float]]] = []
    top_colors: list[np.ndarray] = []
    sides: list[list[tuple[float, float, float]]] = []
    side_colors: list[np.ndarray] = []
    for geometry, height_value in zip(data.buildings.geometry, data.buildings["height_m"]):
        height = float(height_value)
        base_color = np.asarray(CMAP(HEIGHT_NORM(height)))
        darker = base_color.copy()
        darker[:3] *= 0.76
        for polygon in iter_polygons(geometry.simplify(0.8, preserve_topology=True)):
            coordinates = [(float(x) - min_x, float(y) - min_y) for x, y in polygon.exterior.coords]
            if len(coordinates) < 4:
                continue
            tops.append([(x, y, height) for x, y in coordinates[:-1]])
            top_colors.append(base_color)
            for (x0, y0), (x1, y1) in zip(coordinates[:-1], coordinates[1:]):
                sides.append([(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height), (x0, y0, height)])
                side_colors.append(darker)
    return tops, top_colors, sides, side_colors


def style_3d_axes(ax, data: SceneData) -> None:
    ax.set_xlim(0.0, data.width_m)
    ax.set_ylim(0.0, data.height_m)
    ax.set_zlim(0.0, HEIGHT_MAX_M)
    ax.set_box_aspect((1.0, 1.0, 0.34))
    ax.view_init(elev=30.0, azim=-58.0)
    ax.set_xlabel("Easting offset (km)", labelpad=5)
    ax.set_ylabel("Northing offset (km)", labelpad=5)
    xy_ticks = np.arange(0.0, 2501.0, 500.0)
    ax.set_xticks(xy_ticks, [f"{value / 1000.0:.1f}" for value in xy_ticks])
    ax.set_yticks(xy_ticks, [f"{value / 1000.0:.1f}" for value in xy_ticks])
    ax.set_zticks([0, 50, 100, 150, 200, 250])
    ax.tick_params(pad=1.5, width=0.55, length=2.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.55, 0.55, 0.55, 0.65))
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.35)
        axis._axinfo["grid"]["linewidth"] = 0.35


def plot_3d(data: SceneData, *, figsize: tuple[float, float] = (8.8, 8.0)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(BACKGROUND_COLOR)

    tops, top_colors, sides, side_colors = build_3d_faces(data)
    building_colors = np.asarray(side_colors + top_colors, dtype=float)
    building_colors[:, 3] = 0.58
    building_collection = Poly3DCollection(
        sides + tops,
        facecolors=building_colors,
        edgecolors=(0.18, 0.18, 0.18, 0.55),
        linewidths=0.16,
        zorder=2,
    )
    # Hybrid-vector PDF: complex building faces are rasterized at save DPI;
    # axes, labels, and colorbar remain vector objects.
    building_collection.set_rasterized(True)
    ax.add_collection3d(building_collection)

    style_3d_axes(ax, data)
    ax.set_title(f"{data.label} case study - 3D urban map", pad=8)
    add_colorbar(fig, ax, pad=0.012, shrink=0.67)
    fig.subplots_adjust(left=0.01, right=0.92, bottom=0.02, top=0.95)
    return fig


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> dict[str, str]:
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "case_study_maps",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 600:
        raise ValueError("Publication PNG output must use at least 600 dpi")
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = {scene: load_scene(scene) for scene in SCENE_ORDER}
    started = time.perf_counter()
    outputs: dict[str, dict[str, str]] = {}

    for scene in SCENE_ORDER:
        data = scenes[scene]
        fig_2d = plot_2d(data)
        outputs[f"{scene}_2d"] = save_figure(fig_2d, args.output_dir / f"{scene}_2d_case_map", args.dpi)
        plt.close(fig_2d)

        fig_3d = plot_3d(data)
        outputs[f"{scene}_3d"] = save_figure(fig_3d, args.output_dir / f"{scene}_3d_case_map", args.dpi)
        plt.close(fig_3d)

    metadata = {
        "style": "Nature-inspired publication map",
        "font_family": "Times New Roman",
        "dpi": args.dpi,
        "building_height_unit": "m",
        "building_height_color_range_m": [HEIGHT_MIN_M, HEIGHT_MAX_M],
        "colormap": "RdYlGn_r (green-low, yellow-medium, red-high)",
        "scene_statistics": {
            scene: {
                "building_count": len(data.buildings),
                "road_count": len(data.roads),
                "launch_recovery_node_count": len(data.nodes),
                "height_min_m": float(data.buildings["height_m"].min()),
                "height_max_m": float(data.buildings["height_m"].max()),
                "missing_height_count": int(data.buildings["height_m"].isna().sum()),
                "bounds_epsg32650": list(data.bounds),
                "sources": data.sources,
            }
            for scene, data in scenes.items()
        },
        "outputs": outputs,
        "three_dimensional_content": "Buildings only; no road network or launch/recovery nodes.",
        "three_dimensional_building_face_alpha": 0.58,
        "three_dimensional_building_outline": "dark gray, 0.16 pt",
        "three_dimensional_pdf_note": "Building faces are rasterized at 600 dpi; axes, labels, and colorbars remain vector.",
        "elapsed_s": time.perf_counter() - started,
    }
    metadata_path = args.output_dir / "case_study_maps_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
