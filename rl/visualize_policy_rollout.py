from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import pandas as pd
import torch
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon, shape

from dynamic_patrol_marl.env.drone import Drone
from dynamic_patrol_marl.rl.dqn_networks import make_q_network, masked_q_values
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config, project_root


UAV_COLORS = ("#1565c0", "#00897b", "#f9a825", "#7b1fa2", "#6d4c41", "#00838f")
CAV_COLOR = "#d32f2f"
BUILDING_COLOR = "#777777"
ROAD_COLOR = "#b8bdc3"


def _load_torch(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def find_latest_model(root: Path) -> Path:
    models = [
        path
        for path in (root / "results").rglob("*.pt")
        if "checkpoints" not in {part.lower() for part in path.parts}
    ]
    if not models:
        raise FileNotFoundError("No trained .pt model was found below results/")
    return max(models, key=lambda path: path.stat().st_mtime)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def reset_env(env: DynamicPatrolEnv, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    output = env.reset(seed=int(seed))
    if isinstance(output, tuple):
        obs, info = output
        mask = info.get("action_mask", env.action_masks())
    else:
        obs = output
        mask = env.action_masks()
    return np.asarray(obs, dtype=np.float32), np.asarray(mask, dtype=bool)


def step_env(env: DynamicPatrolEnv, action: int) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict[str, Any]]:
    output = env.step(int(action))
    if len(output) == 5:
        obs, reward, terminated, truncated, info = output
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = output
    info = dict(info)
    return (
        np.asarray(obs, dtype=np.float32),
        float(reward),
        bool(done),
        np.asarray(info.get("action_mask", env.action_masks()), dtype=bool),
        info,
    )


@torch.no_grad()
def greedy_action(
    network: torch.nn.Module,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> Tuple[int, np.ndarray]:
    obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
    q_values = network(obs_tensor, mask_tensor)
    masked = masked_q_values(q_values, mask_tensor)
    return int(masked.argmax(dim=1).item()), q_values[0].detach().cpu().numpy()


def _plan_record(env: DynamicPatrolEnv, action: int) -> Dict[str, Any]:
    cache = getattr(env, "joint_plan_cache", None)
    if cache is None or action >= len(cache.plan_slots):
        return {}
    plan = cache.plan_slots[action]
    if plan is None:
        return {}
    recovery_node = None if plan.sortie is None else int(plan.sortie.recovery_node)
    return {
        "plan_id": str(plan.plan_id),
        "plan_type": str(plan.plan_type),
        "ground_action": str(plan.ground_candidate.action_type),
        "cav_target_node": plan.cav_target_node,
        "uav_id": plan.uav_id,
        "task_ids": ";".join(str(value) for value in plan.task_ids),
        "recovery_node": recovery_node,
    }


class SortieTrace:
    """Capture realized 3-D paths before recovery clears Drone.active_path_xyz."""

    def __init__(self, network: Any):
        self.network = network
        self.rows: List[Dict[str, Any]] = []
        self._original = Drone.assign_sortie

    def install(self) -> None:
        trace = self
        original = self._original

        def traced_assign(drone: Drone, sortie: Any, now: float) -> None:
            original(drone, sortie, now)
            points = [tuple(float(value) for value in point) for point in drone.active_path_xyz]
            launch_xy = trace.network.node_xy(int(sortie.launch_node))
            recovery_xy = trace.network.node_xy(int(sortie.recovery_node))
            if points:
                if points[0][2] > 0.0:
                    points.insert(0, (float(launch_xy[0]), float(launch_xy[1]), 0.0))
                if points[-1][2] > 0.0:
                    points.append((float(recovery_xy[0]), float(recovery_xy[1]), 0.0))
            else:
                altitude = float(trace.network.cfg.get("uav_flight", {}).get("cruise_height_m", 60.0))
                points = [
                    (float(launch_xy[0]), float(launch_xy[1]), 0.0),
                    (float(launch_xy[0]), float(launch_xy[1]), altitude),
                    (float(recovery_xy[0]), float(recovery_xy[1]), altitude),
                    (float(recovery_xy[0]), float(recovery_xy[1]), 0.0),
                ]
            trace.rows.append(
                {
                    "sortie_index": len(trace.rows),
                    "drone_id": str(drone.id),
                    "launch_time_s": float(now),
                    "recovery_eta_s": float(getattr(sortie, "recovery_eta", now + sortie.drone_time_s)),
                    "launch_node": int(sortie.launch_node),
                    "recovery_node": int(sortie.recovery_node),
                    "task_ids": [int(value) for value in sortie.task_ids],
                    "actual_distance_m": float(getattr(sortie, "actual_distance_m", 0.0)),
                    "actual_flight_time_s": float(getattr(sortie, "actual_flight_time_s", sortie.drone_time_s)),
                    "path_xyz": points,
                }
            )

        Drone.assign_sortie = traced_assign

    def restore(self) -> None:
        Drone.assign_sortie = self._original


def run_rollout(
    cfg: Dict[str, Any],
    checkpoint: Dict[str, Any],
    seed: int,
    device: torch.device,
    max_decisions: int | None,
) -> Tuple[DynamicPatrolEnv, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cfg.setdefault("training", {})["seed_mode"] = "fixed"
    scenario = str(checkpoint.get("train_scenario", cfg.get("rl", {}).get("eval_scenario", "xinjiekou_eval")))
    env = DynamicPatrolEnv(cfg, scenario=scenario)
    obs, mask = reset_env(env, seed)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    candidate_dim = int(checkpoint.get("candidate_dim", cfg.get("rl", {}).get("hyperparams", {}).get("candidate_dim", 38)))
    expected = (int(checkpoint.get("obs_dim", -1)), int(checkpoint.get("action_dim", -1)))
    if expected != (obs_dim, action_dim):
        raise ValueError(f"Model/environment dimension mismatch: model={expected}, environment={(obs_dim, action_dim)}")
    if str(checkpoint.get("decision_mode", env.decision_mode)) != str(env.decision_mode):
        raise ValueError("Model and environment use different decision modes")

    algo = str(checkpoint.get("algo", "rainbow_dqn"))
    q_network = make_q_network(algo, obs_dim, action_dim, candidate_dim).to(device)
    q_network.load_state_dict(checkpoint["model_state_dict"])
    q_network.eval()

    assert env.sim is not None
    sortie_trace = SortieTrace(env.sim.network)
    sortie_trace.install()
    decisions: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = False
    try:
        while not done and (max_decisions is None or len(decisions) < max_decisions):
            decision_time = float(env.sim.now)
            valid_action_count = int(np.count_nonzero(mask))
            action, q_values = greedy_action(q_network, obs, mask, device)
            plan = _plan_record(env, action)
            valid_q = q_values[mask]
            obs, reward, done, mask, info = step_env(env, action)
            total_reward += reward
            decisions.append(
                {
                    "decision_index": len(decisions),
                    "decision_time_s": decision_time,
                    "next_decision_time_s": float(info.get("time", env.sim.now)),
                    "delta_t_s": float(info.get("delta_t", env.sim.now - decision_time)),
                    "action": int(action),
                    "valid_action_count": valid_action_count,
                    "selected_q": float(q_values[action]),
                    "valid_q_min": float(np.min(valid_q)) if len(valid_q) else math.nan,
                    "valid_q_max": float(np.max(valid_q)) if len(valid_q) else math.nan,
                    "reward": float(reward),
                    "cumulative_reward": float(total_reward),
                    "trigger_events": ";".join(str(value) for value in info.get("trigger_events", [])),
                    **plan,
                }
            )
    finally:
        sortie_trace.restore()

    summary = dict(env.episode_summary())
    summary.update(
        {
            "test_seed": int(seed),
            "episode_reward": float(total_reward),
            "decision_count": len(decisions),
            "sortie_count": len(sortie_trace.rows),
            "rollout_complete": bool(done),
            "device": str(device),
            "algo": algo,
        }
    )
    return env, decisions, sortie_trace.rows, summary


def load_buildings(path: Path) -> List[Tuple[Polygon, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    unique: Dict[bytes, Tuple[Polygon, float]] = {}
    for feature in data.get("features", []):
        geometry = shape(feature.get("geometry"))
        height = float(feature.get("properties", {}).get("height_m", 0.0) or 0.0)
        polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else (geometry,)
        for polygon in polygons:
            if isinstance(polygon, Polygon) and not polygon.is_empty and height > 0.0:
                unique.setdefault(polygon.wkb, (polygon, height))
    return list(unique.values())


def edge_coordinates(network: Any, u: int, v: int) -> List[Tuple[float, float]]:
    geometry = network.G[int(u)][int(v)].get("geometry")
    if geometry:
        line = wkt.loads(geometry) if isinstance(geometry, str) else geometry
        coords = [(float(x), float(y)) for x, y, *_ in line.coords]
        ux, uy = network.node_xy(int(u))
        if coords and math.hypot(coords[-1][0] - ux, coords[-1][1] - uy) < math.hypot(coords[0][0] - ux, coords[0][1] - uy):
            coords.reverse()
        return coords
    return [network.node_xy(int(u)), network.node_xy(int(v))]


def _origin(buildings: Sequence[Tuple[Polygon, float]], network: Any) -> Tuple[float, float]:
    if buildings:
        return min(item[0].bounds[0] for item in buildings), min(item[0].bounds[1] for item in buildings)
    points = [network.node_xy(node) for node in network.nodes]
    return min(x for x, _ in points), min(y for _, y in points)


def _xy(points: Iterable[Sequence[float]], origin: Tuple[float, float]) -> List[Tuple[float, float]]:
    ox, oy = origin
    return [(float(point[0]) - ox, float(point[1]) - oy) for point in points]


def _road_segments(network: Any, origin: Tuple[float, float]) -> List[List[Tuple[float, float]]]:
    seen: set[bytes | Tuple[Tuple[float, float], ...]] = set()
    segments: List[List[Tuple[float, float]]] = []
    for u, v in network.G.edges():
        coords = edge_coordinates(network, int(u), int(v))
        key = tuple((round(x, 3), round(y, 3)) for x, y in coords)
        reverse = tuple(reversed(key))
        canonical = min(key, reverse)
        if canonical not in seen:
            seen.add(canonical)
            segments.append(_xy(coords, origin))
    return segments


def _vehicle_segments(env: DynamicPatrolEnv, origin: Tuple[float, float]) -> List[Dict[str, Any]]:
    assert env.sim is not None
    return [
        {
            "arrival_time_s": float(time_s),
            "u": int(u),
            "v": int(v),
            "coordinates": _xy(edge_coordinates(env.sim.network, int(u), int(v)), origin),
        }
        for time_s, u, v in env.sim.vehicle.route_edge_log
    ]


def _building_patches(buildings: Sequence[Tuple[Polygon, float]], origin: Tuple[float, float]) -> List[PolygonPatch]:
    return [PolygonPatch(_xy(polygon.exterior.coords, origin), closed=True) for polygon, _height in buildings]


def plot_2d(
    env: DynamicPatrolEnv,
    buildings: Sequence[Tuple[Polygon, float]],
    sorties: Sequence[Dict[str, Any]],
    origin: Tuple[float, float],
    output: Path,
) -> None:
    assert env.sim is not None
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.add_collection(PatchCollection(_building_patches(buildings, origin), facecolor=BUILDING_COLOR, edgecolor=BUILDING_COLOR, linewidth=0.15, alpha=0.18, zorder=1))
    ax.add_collection(LineCollection(_road_segments(env.sim.network, origin), colors=ROAD_COLOR, linewidths=0.45, alpha=0.65, zorder=2))
    vehicle = _vehicle_segments(env, origin)
    ax.add_collection(LineCollection([row["coordinates"] for row in vehicle], colors=CAV_COLOR, linewidths=2.2, alpha=0.95, zorder=5))

    drone_ids = [drone.id for drone in env.sim.drones]
    color_by_drone = {drone_id: UAV_COLORS[index % len(UAV_COLORS)] for index, drone_id in enumerate(drone_ids)}
    for drone_id in drone_ids:
        paths = [_xy(row["path_xyz"], origin) for row in sorties if row["drone_id"] == drone_id]
        ax.add_collection(LineCollection(paths, colors=color_by_drone[drone_id], linewidths=1.5, alpha=0.9, zorder=6))

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Easting offset from {origin[0]:.1f} m")
    ax.set_ylabel(f"Northing offset from {origin[1]:.1f} m")
    ax.set_title("Trained-policy rollout: CAV and UAV trajectories (2D)")
    handles = [
        Line2D([0], [0], color=BUILDING_COLOR, lw=6, alpha=0.25, label="Buildings"),
        Line2D([0], [0], color=CAV_COLOR, lw=2.4, label="CAV"),
    ]
    handles.extend(Line2D([0], [0], color=color_by_drone[drone_id], lw=2.0, label=drone_id) for drone_id in drone_ids)
    ax.legend(handles=handles, loc="upper right", frameon=True)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_uav_only_2d(
    sorties: Sequence[Dict[str, Any]],
    origin: Tuple[float, float],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    drone_ids = sorted({str(row["drone_id"]) for row in sorties})
    color_by_drone = {
        drone_id: UAV_COLORS[index % len(UAV_COLORS)]
        for index, drone_id in enumerate(drone_ids)
    }
    for drone_id in drone_ids:
        paths = [
            _xy(row["path_xyz"], origin)
            for row in sorties
            if str(row["drone_id"]) == drone_id and len(row["path_xyz"]) >= 2
        ]
        ax.add_collection(
            LineCollection(
                paths,
                colors=color_by_drone[drone_id],
                linewidths=1.55,
                alpha=0.88,
                zorder=2,
            )
        )

    ax.autoscale()
    ax.margins(0.035)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Easting offset from {origin[0]:.1f} m")
    ax.set_ylabel(f"Northing offset from {origin[1]:.1f} m")
    ax.set_title("Trained-policy UAV trajectories only (2D)")
    ax.legend(
        handles=[
            Line2D([0], [0], color=color_by_drone[drone_id], lw=2.2, label=drone_id)
            for drone_id in drone_ids
        ],
        loc="upper right",
        frameon=True,
    )
    ax.grid(color="#d9dde2", linewidth=0.45, alpha=0.55)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _building_faces(
    buildings: Sequence[Tuple[Polygon, float]],
    origin: Tuple[float, float],
) -> Tuple[List[List[Tuple[float, float, float]]], List[List[Tuple[float, float, float]]]]:
    ox, oy = origin
    tops: List[List[Tuple[float, float, float]]] = []
    walls: List[List[Tuple[float, float, float]]] = []
    for polygon, height in buildings:
        ring = [(float(x) - ox, float(y) - oy) for x, y in polygon.exterior.coords]
        tops.append([(x, y, float(height)) for x, y in ring])
        for first, second in zip(ring[:-1], ring[1:]):
            walls.append([(first[0], first[1], 0.0), (second[0], second[1], 0.0), (second[0], second[1], float(height)), (first[0], first[1], float(height))])
    return tops, walls


def plot_3d(
    env: DynamicPatrolEnv,
    buildings: Sequence[Tuple[Polygon, float]],
    sorties: Sequence[Dict[str, Any]],
    origin: Tuple[float, float],
    output: Path,
) -> None:
    assert env.sim is not None
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    tops, walls = _building_faces(buildings, origin)
    ax.add_collection3d(Poly3DCollection(tops, facecolors=BUILDING_COLOR, edgecolors=BUILDING_COLOR, linewidths=0.12, alpha=0.20))
    ax.add_collection3d(Poly3DCollection(walls, facecolors=BUILDING_COLOR, edgecolors=BUILDING_COLOR, linewidths=0.08, alpha=0.07))
    roads_3d = [[(x, y, 0.0) for x, y in segment] for segment in _road_segments(env.sim.network, origin)]
    ax.add_collection3d(Line3DCollection(roads_3d, colors=ROAD_COLOR, linewidths=0.35, alpha=0.55))
    vehicle = _vehicle_segments(env, origin)
    ax.add_collection3d(Line3DCollection([[(x, y, 1.0) for x, y in row["coordinates"]] for row in vehicle], colors=CAV_COLOR, linewidths=2.4, alpha=0.95))

    drone_ids = [drone.id for drone in env.sim.drones]
    color_by_drone = {drone_id: UAV_COLORS[index % len(UAV_COLORS)] for index, drone_id in enumerate(drone_ids)}
    for row in sorties:
        points = [(x - origin[0], y - origin[1], z) for x, y, z in row["path_xyz"]]
        if len(points) >= 2:
            ax.plot(*zip(*points), color=color_by_drone[row["drone_id"]], linewidth=1.7, alpha=0.95)

    bounds = [polygon.bounds for polygon, _height in buildings]
    if bounds:
        ax.set_xlim(min(value[0] for value in bounds) - origin[0], max(value[2] for value in bounds) - origin[0])
        ax.set_ylim(min(value[1] for value in bounds) - origin[1], max(value[3] for value in bounds) - origin[1])
    max_z = max([height for _polygon, height in buildings] + [120.0])
    ax.set_zlim(0.0, max_z * 1.03)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.view_init(elev=33, azim=-58)
    ax.set_xlabel("Easting offset (m)")
    ax.set_ylabel("Northing offset (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("Trained-policy rollout with building heights (3D)")
    handles = [
        Line2D([0], [0], color=BUILDING_COLOR, lw=6, alpha=0.3, label="Buildings"),
        Line2D([0], [0], color=CAV_COLOR, lw=2.4, label="CAV"),
    ]
    handles.extend(Line2D([0], [0], color=color_by_drone[drone_id], lw=2.0, label=drone_id) for drone_id in drone_ids)
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _path_prefix(points: Sequence[Sequence[float]], fraction: float) -> List[Tuple[float, float]]:
    xy = [(float(point[0]), float(point[1])) for point in points]
    if len(xy) < 2 or fraction >= 1.0:
        return xy
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy[:-1], xy[1:])]
    target = max(0.0, fraction) * sum(lengths)
    result = [xy[0]]
    traversed = 0.0
    for first, second, length in zip(xy[:-1], xy[1:], lengths):
        if traversed + length <= target + 1e-9:
            result.append(second)
            traversed += length
            continue
        ratio = 0.0 if length <= 1e-9 else (target - traversed) / length
        result.append((first[0] + ratio * (second[0] - first[0]), first[1] + ratio * (second[1] - first[1])))
        break
    return result


def animate_2d(
    env: DynamicPatrolEnv,
    buildings: Sequence[Tuple[Polygon, float]],
    sorties: Sequence[Dict[str, Any]],
    decisions: Sequence[Dict[str, Any]],
    origin: Tuple[float, float],
    output: Path,
    max_frames: int,
    fps: int,
) -> None:
    assert env.sim is not None
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(PatchCollection(_building_patches(buildings, origin), facecolor=BUILDING_COLOR, edgecolor=BUILDING_COLOR, linewidth=0.1, alpha=0.16))
    ax.add_collection(LineCollection(_road_segments(env.sim.network, origin), colors=ROAD_COLOR, linewidths=0.35, alpha=0.55))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting offset (m)")
    ax.set_ylabel("Northing offset (m)")
    ax.set_title("Real-time dynamic planning replay")

    vehicle = _vehicle_segments(env, origin)
    drone_ids = [drone.id for drone in env.sim.drones]
    color_by_drone = {drone_id: UAV_COLORS[index % len(UAV_COLORS)] for index, drone_id in enumerate(drone_ids)}
    vehicle_collection = LineCollection([], colors=CAV_COLOR, linewidths=2.2, zorder=5)
    ax.add_collection(vehicle_collection)
    drone_collections: Dict[str, LineCollection] = {}
    drone_markers: Dict[str, Line2D] = {}
    for drone_id in drone_ids:
        collection = LineCollection([], colors=color_by_drone[drone_id], linewidths=1.5, zorder=6)
        ax.add_collection(collection)
        drone_collections[drone_id] = collection
        marker, = ax.plot([], [], marker="o", markersize=5, color=color_by_drone[drone_id], linestyle="None", zorder=8)
        drone_markers[drone_id] = marker
    cav_marker, = ax.plot([], [], marker="s", markersize=6, color=CAV_COLOR, linestyle="None", zorder=8)
    time_text = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    handles = [Line2D([0], [0], color=CAV_COLOR, lw=2.3, label="CAV")]
    handles.extend(Line2D([0], [0], color=color_by_drone[value], lw=2.0, label=value) for value in drone_ids)
    ax.legend(handles=handles, loc="upper right")

    end_time = max([float(row.get("next_decision_time_s", 0.0)) for row in decisions] + [1.0])
    frame_count = min(max_frames, max(2, len(decisions)))
    frame_times = np.linspace(0.0, end_time, frame_count)
    route_log = [(float(t), int(node)) for t, node in env.sim.vehicle.route_log]

    def update(frame_index: int):
        current_time = float(frame_times[frame_index])
        completed_vehicle = [row["coordinates"] for row in vehicle if row["arrival_time_s"] <= current_time + 1e-9]
        vehicle_collection.set_segments(completed_vehicle)
        known_nodes = [node for time_s, node in route_log if time_s <= current_time + 1e-9]
        if known_nodes:
            x, y = _xy([env.sim.network.node_xy(known_nodes[-1])], origin)[0]
            cav_marker.set_data([x], [y])
        for drone_id in drone_ids:
            paths: List[List[Tuple[float, float]]] = []
            marker_point: Tuple[float, float] | None = None
            for row in sorties:
                if row["drone_id"] != drone_id or float(row["launch_time_s"]) > current_time:
                    continue
                duration = max(1e-9, float(row["recovery_eta_s"]) - float(row["launch_time_s"]))
                fraction = min(1.0, (current_time - float(row["launch_time_s"])) / duration)
                prefix = _path_prefix(row["path_xyz"], fraction)
                relative = _xy(prefix, origin)
                if len(relative) >= 2:
                    paths.append(relative)
                    marker_point = relative[-1]
            drone_collections[drone_id].set_segments(paths)
            if marker_point is None:
                drone_markers[drone_id].set_data([], [])
            else:
                drone_markers[drone_id].set_data([marker_point[0]], [marker_point[1]])
        decision = max((row for row in decisions if float(row["decision_time_s"]) <= current_time), key=lambda row: float(row["decision_time_s"]), default=None)
        plan_label = "" if decision is None else f" | {decision.get('plan_type', '')} | action {decision['action']}"
        time_text.set_text(f"t = {current_time:,.0f} s{plan_label}")
        return [vehicle_collection, cav_marker, time_text, *drone_collections.values(), *drone_markers.values()]

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / max(1, fps), blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


def write_outputs(
    env: DynamicPatrolEnv,
    decisions: Sequence[Dict[str, Any]],
    sorties: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    model_path: Path,
    config_path: Path,
    buildings: Sequence[Tuple[Polygon, float]],
    origin: Tuple[float, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_csv(output_dir / "policy_decisions.csv", index=False)
    vehicle = _vehicle_segments(env, origin)
    pd.DataFrame(
        {
            "arrival_time_s": row["arrival_time_s"],
            "u": row["u"],
            "v": row["v"],
            "coordinates_xy": json.dumps(row["coordinates"]),
        }
        for row in vehicle
    ).to_csv(output_dir / "cav_trajectory.csv", index=False)
    (output_dir / "uav_sorties_3d.json").write_text(json.dumps(sorties, ensure_ascii=True), encoding="utf-8")
    manifest = {
        **summary,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "building_count_unique": len(buildings),
        "coordinate_origin_epsg32650": [float(origin[0]), float(origin[1])],
    }
    (output_dir / "rollout_summary.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a trained policy and plot Xinjiekou CAV/UAV trajectories.")
    parser.add_argument("--config", default="configs/default_training.yaml")
    parser.add_argument("--model", default=None, help="Trained .pt file; default is the newest final model under results/.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-decisions", type=int, default=None, help="Optional short diagnostic rollout limit.")
    parser.add_argument("--animation-frames", type=int, default=120)
    parser.add_argument("--animation-fps", type=int, default=10)
    parser.add_argument("--skip-animation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = resolve_path(root, args.config)
    model_path = find_latest_model(root) if args.model is None else resolve_path(root, args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else root / "results" / "policy_visualization" / f"seed_{args.seed}"

    cfg = load_config(config_path)
    checkpoint = _load_torch(model_path, device)
    env, decisions, sorties, summary = run_rollout(cfg, checkpoint, args.seed, device, args.max_decisions)
    assert env.sim is not None
    buildings_path = resolve_path(root, cfg["network"]["buildings_file"])
    buildings = load_buildings(buildings_path)
    origin = _origin(buildings, env.sim.network)
    write_outputs(env, decisions, sorties, summary, model_path, config_path, buildings, origin, output_dir)
    plot_2d(env, buildings, sorties, origin, output_dir / "policy_rollout_2d.png")
    plot_uav_only_2d(sorties, origin, output_dir / "uav_trajectories_only_2d.png")
    plot_3d(env, buildings, sorties, origin, output_dir / "policy_rollout_3d.png")
    if not args.skip_animation:
        animate_2d(env, buildings, sorties, decisions, origin, output_dir / "dynamic_planning.gif", args.animation_frames, args.animation_fps)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, default=float))


if __name__ == "__main__":
    main()
