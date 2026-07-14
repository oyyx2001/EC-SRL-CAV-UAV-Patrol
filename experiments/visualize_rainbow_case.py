from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from dynamic_patrol_marl.experiments.evaluate_masked_rl_models import (
    DEFAULT_MODELS,
    _load_masked_dqn,
    _predict_dqn,
    _reset_env,
    _step_env,
)
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import ensure_dirs, load_config, project_root
from dynamic_patrol_marl.utils.visualization import (
    _coerce_segments,
    _network_bounds,
    _paper_display_labels,
    _paper_hidden_arcs,
    _paper_network_pos,
    _plot_arrow_segment,
)


TASK_COLORS = {
    "routine": "#9bd88f",
    "congestion": "#f39c34",
    "incident": "#d73027",
}
DRONE_COLORS = ["#2e8b57", "#1f77b4", "#f39c12"]
CAV_COLOR = "#ff2a2a"
TASK_LABELS = {
    "routine": "Routine patrol",
    "congestion": "Congestion inspection",
    "incident": "Incident confirmation",
}


def run_rainbow_episode(
    cfg: Dict[str, Any],
    seed: int,
    model_path: Path,
    scenario: str,
    device: str,
) -> tuple[Any, List[Dict[str, Any]]]:
    env = DynamicPatrolEnv(cfg, scenario=scenario)
    loaded = _load_masked_dqn(model_path, "rainbow_dqn", device, action_masking=True)
    obs, info = _reset_env(env, int(seed))
    done = False
    action_log: List[Dict[str, Any]] = []
    episode_reward = 0.0

    while not done:
        action_mask = np.asarray(info["action_mask"], dtype=bool)
        action = _predict_dqn(loaded, obs, action_mask)
        candidates = list(env.candidates)
        selected = candidates[action - 1] if action > 0 and action - 1 < len(candidates) else None
        action_log.append({
            "decision_step": len(action_log),
            "time": float(env.sim.now),
            "cav_node": int(env.sim.vehicle.position),
            "action": int(action),
            "valid_actions": int(action_mask.sum()),
            "num_candidates": len(candidates),
            "selected_drone": "" if selected is None else selected.drone_id,
            "selected_tasks": "" if selected is None else ";".join(map(str, selected.task_ids)),
            "selected_recovery_node": "" if selected is None else int(selected.recovery_node),
        })
        obs, reward, done, info = _step_env(env, action)
        episode_reward += reward
        action_log[-1]["reward"] = float(reward)

    action_log.append({"episode_reward": float(episode_reward)})
    return env.sim, action_log


def _format_clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _window_tasks(sim, window_start: float, now: float) -> List[Any]:
    tasks = []
    for task in sim.tasks:
        release = float(task.release_time)
        start = float(task.start_time) if task.start_time is not None else None
        completion = float(task.completion_time) if task.completion_time is not None else None
        if release > now:
            continue
        completed_in_window = completion is not None and window_start - 1e-9 <= completion <= now + 1e-9
        executing_now = start is not None and start <= now + 1e-9 and (completion is None or completion > now + 1e-9)
        if completed_in_window or executing_now:
            tasks.append(task)
    return sorted(tasks, key=lambda t: (float(t.release_time), int(t.id)))


def _vehicle_edges_in_window(sim, window_start: float, now: float) -> List[Tuple[int, int]]:
    edges = []
    for arrival, u, v in getattr(sim.vehicle, "route_edge_log", []):
        if window_start - 1e-9 <= float(arrival) <= now + 1e-9:
            edges.append((int(u), int(v)))
    return edges


def _sortie_segments_in_window(sim, window_start: float, now: float) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for idx, row in enumerate(sim.sortie_log, start=1):
        launch_time = float(row["time"])
        current = launch_time
        for order, seg in enumerate(_coerce_segments(row.get("segments", [])), start=1):
            if seg.get("from") is None or seg.get("to") is None:
                continue
            duration = float(seg.get("duration", 0.0))
            end = current + duration
            overlaps = current <= now + 1e-9 and end >= window_start - 1e-9
            if overlaps:
                segments.append({
                    "idx": idx,
                    "order": order,
                    "drone_id": str(row.get("drone_id", "")),
                    "from": int(seg["from"]),
                    "to": int(seg["to"]),
                    "task_id": seg.get("task_id"),
                    "type": str(seg.get("type", "fly")),
                    "start": current,
                    "end": end,
                    "active": current <= now + 1e-9 and end > now + 1e-9,
                })
            current = end
    return segments


def _drone_color_map(drone_ids: Sequence[str]) -> Dict[str, str]:
    ids = sorted({str(x) for x in drone_ids if str(x)})
    return {drone_id: DRONE_COLORS[i % len(DRONE_COLORS)] for i, drone_id in enumerate(ids)}


def _draw_plain_segment(ax, pos: Dict[int, Tuple[float, float]], u: int, v: int, *, color: str, linewidth: float, alpha: float, linestyle: str = "-", zorder: int = 7) -> None:
    if u not in pos or v not in pos:
        return
    x0, y0 = pos[int(u)]
    x1, y1 = pos[int(v)]
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, zorder=zorder)


def _plot_arrow_coordinates(
    ax,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    *,
    color: str,
    linewidth: float,
    linestyle: str,
    alpha: float,
    mutation_scale: float,
    zorder: int,
) -> None:
    """Draw a directional arrow for the visible fraction of a timed UAV segment."""
    if math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]) <= 1e-9:
        return
    ax.add_patch(FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        mutation_scale=mutation_scale,
        shrinkA=4.5,
        shrinkB=4.5,
        zorder=zorder,
    ))


def _annotate_task_release(ax, pos: Dict[int, Tuple[float, float]], task, offset_index: int) -> None:
    points = []
    for u, v in task.target_arcs:
        if int(u) in pos and int(v) in pos:
            points.append(((pos[int(u)][0] + pos[int(v)][0]) / 2.0, (pos[int(u)][1] + pos[int(v)][1]) / 2.0))
    if not points:
        return
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    dx = 0.0
    dy = 0.16 + 0.055 * (offset_index % 3)
    ax.text(
        x + dx,
        y + dy,
        f"T{task.id} r={_format_clock(float(task.release_time))}",
        color=TASK_COLORS.get(task.type, "#555555"),
        fontsize=7.4,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        zorder=12,
    )


def _active_drone_positions(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active = []
    for seg in segments:
        if seg.get("active"):
            active.append(seg)
    return active


def _task_lookup(sim) -> Dict[int, Any]:
    return {int(t.id): t for t in sim.tasks}


def _tasks_released_until(sim, end_time: float) -> List[Any]:
    return sorted(
        [task for task in sim.tasks if float(task.release_time) <= float(end_time) + 1e-9],
        key=lambda t: (float(t.release_time), int(t.id)),
    )


def _vehicle_edges_until(sim, end_time: float) -> List[Tuple[int, int]]:
    return [
        (int(u), int(v))
        for arrival, u, v in getattr(sim.vehicle, "route_edge_log", [])
        if float(arrival) <= float(end_time) + 1e-9
    ]


def _sortie_segments_until(sim, end_time: float) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for idx, row in enumerate(sim.sortie_log, start=1):
        launch_time = float(row["time"])
        current = launch_time
        for order, seg in enumerate(_coerce_segments(row.get("segments", [])), start=1):
            if seg.get("from") is None or seg.get("to") is None:
                continue
            duration = float(seg.get("duration", 0.0))
            finish = current + duration
            if current <= float(end_time) + 1e-9:
                segments.append({
                    "idx": idx,
                    "order": order,
                    "drone_id": str(row.get("drone_id", "")),
                    "launch_node": int(row["launch_node"]),
                    "recovery_node": int(row["recovery_node"]),
                    "from": int(seg["from"]),
                    "to": int(seg["to"]),
                    "task_id": seg.get("task_id"),
                    "type": str(seg.get("type", "fly")),
                    "start": current,
                    "end": finish,
                })
            current = finish
    return segments


def _sorties_until(sim, end_time: float) -> List[Dict[str, Any]]:
    sorties = []
    for idx, row in enumerate(sim.sortie_log, start=1):
        launch_time = float(row["time"])
        if launch_time <= float(end_time) + 1e-9:
            sorties.append({
                "idx": idx,
                "drone_id": str(row.get("drone_id", "")),
                "launch_node": int(row["launch_node"]),
                "recovery_node": int(row["recovery_node"]),
                "time": launch_time,
            })
    return sorties


def _annotate_node_marker(
    ax,
    pos: Dict[int, Tuple[float, float]],
    node: int,
    text: str,
    color: str,
    label_counts: Dict[int, int],
) -> None:
    if int(node) not in pos:
        return
    count = label_counts[int(node)]
    label_counts[int(node)] += 1
    x, y = pos[int(node)]
    offsets = [(0.0, 0.24), (0.24, 0.0), (0.0, -0.24), (-0.24, 0.0), (0.2, 0.2), (-0.2, 0.2)]
    dx, dy = offsets[count % len(offsets)]
    ax.text(
        x + dx,
        y + dy,
        text,
        color=color,
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.76, "pad": 0.7, "linewidth": 0.45},
        zorder=15,
    )


def _draw_congestion_service_order(ax, pos: Dict[int, Tuple[float, float]], segments: Sequence[Dict[str, Any]], task_by_id: Dict[int, Any]) -> None:
    service_segments = [
        seg for seg in segments
        if seg.get("type") == "service"
        and seg.get("task_id") is not None
        and int(seg["task_id"]) in task_by_id
        and task_by_id[int(seg["task_id"])].type == "congestion"
    ]
    service_segments.sort(key=lambda s: (int(s["task_id"]), float(s["start"])))
    per_task_count: Dict[int, int] = {}
    for seg in service_segments:
        tid = int(seg["task_id"])
        per_task_count[tid] = per_task_count.get(tid, 0) + 1
        u, v = int(seg["from"]), int(seg["to"])
        if u not in pos or v not in pos:
            continue
        mx = (pos[u][0] + pos[v][0]) / 2.0
        my = (pos[u][1] + pos[v][1]) / 2.0
        ax.text(
            mx,
            my - 0.13,
            str(per_task_count[tid]),
            color="#f39c34",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "#f39c34", "alpha": 0.7, "pad": 0.6, "linewidth": 0.5},
            zorder=13,
        )


def _draw_base_network(ax, sim, pos: Dict[int, Tuple[float, float]]) -> None:
    hidden = _paper_hidden_arcs(sim.network)
    for u, v in sorted(sim.network.G.edges()):
        if (int(u), int(v)) in hidden:
            continue
        x0, y0 = pos[int(u)]
        x1, y1 = pos[int(v)]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#d4d4d4",
            linewidth=0.8,
            alpha=0.38,
            zorder=1,
        )

    center = int(sim.network.patrol_center)
    regular_nodes = [int(n) for n in sim.network.nodes if int(n) != center]
    ax.scatter(
        [pos[n][0] for n in regular_nodes],
        [pos[n][1] for n in regular_nodes],
        s=165,
        marker="o",
        facecolors="white",
        edgecolors="#202020",
        linewidths=1.05,
        zorder=4,
    )
    ax.scatter(
        [pos[center][0]],
        [pos[center][1]],
        s=185,
        marker="s",
        facecolors="#2f80ed",
        edgecolors="#2f80ed",
        linewidths=1.0,
        zorder=5,
    )

    labels = _paper_display_labels(sim.network)
    for node, label in labels.items():
        x, y = pos[int(node)]
        ax.text(
            x,
            y,
            str(label),
            color="white" if int(node) == center else "#111111",
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )


def _sortie_records(sim) -> List[Dict[str, Any]]:
    """Rebuild launch-to-recovery records from sortie and event logs."""
    recoveries: Dict[str, List[float]] = defaultdict(list)
    for event in sim.event_log:
        if event.get("event") == "recover":
            recoveries[str(event.get("agent", ""))].append(float(event["time"]))
    for times in recoveries.values():
        times.sort()

    recovery_cursor: Dict[str, int] = defaultdict(int)
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(sim.sortie_log, start=1):
        drone_id = str(row.get("drone_id", ""))
        launch = float(row["time"])
        current = launch
        segments: List[Dict[str, Any]] = []
        for order, segment in enumerate(_coerce_segments(row.get("segments", [])), start=1):
            if segment.get("from") is None or segment.get("to") is None:
                continue
            duration = max(0.0, float(segment.get("duration", 0.0)))
            finish = current + duration
            segments.append({
                "order": order,
                "type": str(segment.get("type", "fly")),
                "from": int(segment["from"]),
                "to": int(segment["to"]),
                "task_id": segment.get("task_id"),
                "arc_index": segment.get("arc_index"),
                "arc_count": segment.get("arc_count"),
                "service_final": segment.get("service_final", True),
                "start": current,
                "end": finish,
            })
            current = finish

        cursor = recovery_cursor[drone_id]
        drone_recoveries = recoveries.get(drone_id, [])
        while cursor < len(drone_recoveries) and drone_recoveries[cursor] < current - 1e-6:
            cursor += 1
        completed = cursor < len(drone_recoveries)
        if completed:
            recovered = float(drone_recoveries[cursor])
            recovery_cursor[drone_id] = cursor + 1
        else:
            recovered = float(sim.horizon)
            recovery_cursor[drone_id] = cursor

        task_ids = row.get("task_ids", [])
        if isinstance(task_ids, str):
            task_ids = [int(value) for value in task_ids.replace(";", ",").split(",") if value.strip()]
        records.append({
            "idx": idx,
            "drone_id": drone_id,
            "launch": launch,
            "flight_end": current,
            "recover": recovered,
            "completed": completed,
            "wait": max(0.0, recovered - current) if completed else 0.0,
            "launch_node": int(row["launch_node"]),
            "recovery_node": int(row["recovery_node"]),
            "task_ids": [int(task_id) for task_id in task_ids],
            "segments": segments,
        })
    return records


def _group_sortie_cycles(
    records: Sequence[Dict[str, Any]],
    horizon: float,
    merge_gap_s: float = 0.0,
) -> List[Dict[str, Any]]:
    """Group simultaneous launches into complete launch-to-recovery sorties."""
    cycles: List[Dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (float(item["launch"]), int(item["idx"]))):
        if not cycles or float(record["launch"]) > float(cycles[-1]["start"]) + float(merge_gap_s) + 1e-9:
            cycles.append({"start": float(record["launch"]), "end": float(record["recover"]), "sorties": [record]})
        else:
            cycles[-1]["sorties"].append(record)
            cycles[-1]["end"] = max(float(cycles[-1]["end"]), float(record["recover"]))
    for idx, cycle in enumerate(cycles, start=1):
        cycle["idx"] = idx
    return cycles


def _draw_undirected_storyboard_network(ax, sim, pos: Dict[int, Tuple[float, float]]) -> None:
    hidden = _paper_hidden_arcs(sim.network)
    links = set()
    for u, v in sim.network.G.edges():
        u, v = int(u), int(v)
        if (u, v) in hidden or (v, u) in hidden:
            continue
        links.add(tuple(sorted((u, v))))
    for u, v in sorted(links):
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color="#71808a",
            linewidth=1.05,
            alpha=0.88,
            zorder=1,
        )

    center = int(sim.network.patrol_center)
    regular = [int(node) for node in sim.network.nodes if int(node) != center]
    ax.scatter(
        [pos[node][0] for node in regular],
        [pos[node][1] for node in regular],
        s=66,
        facecolors="#f8fafb",
        edgecolors="#34444f",
        linewidths=0.85,
        zorder=2,
    )
    ax.scatter(*pos[center], s=104, marker="s", c="#25343c", edgecolors="white", linewidths=0.85, zorder=3)


def _vehicle_edges_between(sim, start: float, end: float) -> List[Tuple[int, int]]:
    return [
        (int(u), int(v))
        for arrival, u, v in getattr(sim.vehicle, "route_edge_log", [])
        if float(start) - 1e-9 < float(arrival) <= float(end) + 1e-9
    ]


def _vehicle_node_at(sim, time_s: float) -> int:
    """Return the last CAV intersection reached at or before ``time_s``."""
    node = int(sim.network.patrol_center)
    for arrival, arrived_node in getattr(sim.vehicle, "route_log", []):
        if float(arrival) <= float(time_s) + 1e-9:
            node = int(arrived_node)
        else:
            break
    return node


def _sortie_label_offsets(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Spread launch/recovery labels around nodes shared by several sorties."""
    labels_by_node: Dict[int, List[Tuple[str, int, str]]] = defaultdict(list)
    for record in records:
        labels_by_node[int(record["launch_node"])].append(("launch", int(record["idx"]), "top"))
        labels_by_node[int(record["recovery_node"])].append(("recovery", int(record["idx"]), "bottom"))

    offsets: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for entries in labels_by_node.values():
        total = len(entries)
        for slot, (kind, sortie_idx, vertical) in enumerate(entries):
            if total == 1:
                offsets[(kind, sortie_idx)] = (0.0, 44.0 if vertical == "top" else -48.0)
                continue
            angle = math.pi / 2.0 + 2.0 * math.pi * slot / total
            radius = 48.0 + 8.0 * (slot % 2)
            offsets[(kind, sortie_idx)] = (radius * math.cos(angle), radius * math.sin(angle))
    return offsets


def _draw_storyboard_cycle(ax, sim, cycle: Dict[str, Any], pos: Dict[int, Tuple[float, float]], drone_colors: Dict[str, str], panel_label: str) -> None:
    _draw_undirected_storyboard_network(ax, sim, pos)
    start, end = float(cycle["start"]), float(cycle["end"])
    task_by_id = _task_lookup(sim)
    cycle_task_ids = sorted({
        int(task_id)
        for record in cycle["sorties"]
        for task_id in record.get("task_ids", [])
        if int(task_id) in task_by_id
    })
    for task_id in cycle_task_ids:
        task = task_by_id[task_id]
        task_color = TASK_COLORS.get(task.type, "#888888")
        for u, v in task.target_arcs:
            _draw_plain_segment(ax, pos, int(u), int(v), color=task_color, linewidth=14.4, alpha=0.34, zorder=4)

    vehicle_edges = _vehicle_edges_between(sim, start, end)
    for u, v in vehicle_edges:
        _plot_arrow_segment(
            ax,
            pos,
            u,
            v,
            color=CAV_COLOR,
            linewidth=4.8,
            alpha=0.92,
            mutation_scale=9.5,
            shrink=5.5,
            zorder=6,
        )

    cav_start = _vehicle_node_at(sim, start)
    cav_end = _vehicle_node_at(sim, end)
    ax.scatter(*pos[cav_start], s=648, marker="o", c=CAV_COLOR, edgecolors="white", linewidths=1.2, zorder=13)
    ax.scatter(*pos[cav_end], s=648, marker="X", c=CAV_COLOR, edgecolors="white", linewidths=1.2, zorder=13)

    label_offsets = _sortie_label_offsets(cycle["sorties"])
    for record in cycle["sorties"]:
        color = drone_colors.get(record["drone_id"], "#0072bd")
        for segment in record["segments"]:
            if float(segment["start"]) >= float(sim.horizon) - 1e-9:
                continue
            duration = max(1e-9, float(segment["end"]) - float(segment["start"]))
            visible_end = min(float(sim.horizon), float(segment["end"]))
            end_fraction = min(1.0, max(0.0, (visible_end - float(segment["start"])) / duration))
            x0, y0 = pos[segment["from"]]
            x1, y1 = pos[segment["to"]]
            arrow_start = (x0, y0)
            arrow_end = (x0 + end_fraction * (x1 - x0), y0 + end_fraction * (y1 - y0))
            _plot_arrow_coordinates(
                ax,
                arrow_start,
                arrow_end,
                color=color,
                linewidth=1.85,
                linestyle="--",
                alpha=0.90,
                mutation_scale=8.5,
                zorder=9,
            )
            if visible_end < float(segment["end"]) - 1e-9:
                ax.scatter(*arrow_end, s=630, marker="*", c=color, edgecolors="white", linewidths=1.0, zorder=14)
                ax.text(arrow_end[0], arrow_end[1] + 0.14, "at 60:00", color=color, fontsize=18.0, fontweight="bold", ha="center", zorder=14)
        launch_node, recovery_node = record["launch_node"], record["recovery_node"]
        ax.scatter(*pos[launch_node], s=828, marker="^", c=color, edgecolors="white", linewidths=1.25, zorder=13)
        launch_dx, launch_dy = label_offsets[("launch", int(record["idx"]))]
        ax.annotate(
            f"L{record['idx']}",
            xy=pos[launch_node],
            xytext=(launch_dx, launch_dy),
            textcoords="offset points",
            color=color,
            fontsize=25.0,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.88, "pad": 0.65, "linewidth": 0.7},
            zorder=14,
        )
        if record["completed"]:
            ax.scatter(*pos[recovery_node], s=828, marker="v", c=color, edgecolors="white", linewidths=1.25, zorder=13)
            recovery_dx, recovery_dy = label_offsets[("recovery", int(record["idx"]))]
            ax.annotate(
                f"R{record['idx']}",
                xy=pos[recovery_node],
                xytext=(recovery_dx, recovery_dy),
                textcoords="offset points",
                color=color,
                fontsize=25.0,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.88, "pad": 0.65, "linewidth": 0.7},
                zorder=14,
            )

    min_x, max_x, min_y, max_y, span = _network_bounds(pos)
    ax.set_xlim(min_x - 0.035 * span, max_x + 0.035 * span)
    ax.set_ylim(min_y - 0.075 * span, max_y + 0.04 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.5,
        -0.040,
        f"({panel_label}) Batch {cycle['idx']} | {_format_clock(start)}-{_format_clock(end)} | "
        f"{len(cycle['sorties'])} sortie(s)",
        transform=ax.transAxes,
        fontsize=60.0,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
    )


def plot_sortie_storyboards(
    sim,
    out_dir: str | Path,
    prefix: str,
    cycles_per_page: int = 4,
    max_cycles: int | None = None,
    merge_gap_s: float = 0.0,
    exclude_panels: str = "",
) -> tuple[List[Path], pd.DataFrame]:
    """Create multi-panel pages where each panel is one launch-to-recovery cycle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _sortie_records(sim)
    cycles = _group_sortie_cycles(records, sim.horizon, merge_gap_s=merge_gap_s)
    if max_cycles is not None:
        cycles = cycles[: max(0, int(max_cycles))]
    excluded_labels = {
        char.lower()
        for char in str(exclude_panels)
        if char.isalpha()
    }
    if excluded_labels:
        cycles = [
            cycle
            for cycle in cycles
            if chr(ord("a") + int(cycle["idx"]) - 1) not in excluded_labels
        ]
    if not cycles:
        return [], pd.DataFrame()

    cycles_per_page = max(1, int(cycles_per_page))
    max_page_panels = min(cycles_per_page, len(cycles))
    ncols = 3 if max_page_panels <= 9 else (4 if max_page_panels >= 8 else (2 if max_page_panels > 1 else 1))
    nrows = int(math.ceil(max_page_panels / ncols))
    pos = _paper_network_pos(sim.network)
    drone_colors = _drone_color_map([record["drone_id"] for record in records])
    pages: List[Path] = []
    manifest_rows: List[Dict[str, Any]] = []

    for page_start in range(0, len(cycles), cycles_per_page):
        page_cycles = cycles[page_start : page_start + cycles_per_page]
        page_number = page_start // cycles_per_page + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(45.0, 12.5 * nrows), squeeze=False)
        flat_axes = list(axes.flat)
        for panel_idx, (ax, cycle) in enumerate(zip(flat_axes, page_cycles)):
            panel_label = chr(ord("a") + page_start + panel_idx)
            _draw_storyboard_cycle(ax, sim, cycle, pos, drone_colors, panel_label)
            vehicle_edges = _vehicle_edges_between(sim, float(cycle["start"]), float(cycle["end"]))
            for record in cycle["sorties"]:
                manifest_rows.append({
                    "page": page_number,
                    "panel": panel_label,
                    "cycle": int(cycle["idx"]),
                    "cycle_start_s": float(cycle["start"]),
                    "cycle_end_s": float(cycle["end"]),
                    "sortie": int(record["idx"]),
                    "drone_id": record["drone_id"],
                    "launch_s": float(record["launch"]),
                    "flight_end_s": float(record["flight_end"]),
                    "recover_s": float(record["recover"]),
                    "recovered": bool(record["completed"]),
                    "recovery_wait_s": float(record["wait"]),
                    "launch_node": int(record["launch_node"]),
                    "recovery_node": int(record["recovery_node"]),
                    "task_ids": ";".join(map(str, record["task_ids"])),
                    "vehicle_edges_in_cycle": len(vehicle_edges),
                })
        for ax in flat_axes[len(page_cycles) :]:
            ax.axis("off")

        handles = [
            Line2D([0], [0], color="#71808a", lw=1.05, alpha=0.88, label="Undirected road link"),
            Line2D([0], [0], color=TASK_COLORS["routine"], lw=14.4, alpha=0.34, label="Routine patrol"),
            Line2D([0], [0], color=TASK_COLORS["congestion"], lw=14.4, alpha=0.34, label="Congestion inspection"),
            Line2D([0], [0], color=TASK_COLORS["incident"], lw=14.4, alpha=0.34, label="Incident confirmation"),
            Line2D([0], [0], color=CAV_COLOR, lw=4.8, marker=">", markersize=27, label="CAV route direction"),
        ]
        for drone_id, color in drone_colors.items():
            handles.append(Line2D([0], [0], color=color, lw=1.8, linestyle="--", label=f"{drone_id} flight"))
        handles.extend([
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markersize=21, label="Launch L#"),
            Line2D([0], [0], marker="v", color="none", markerfacecolor="#777777", markersize=21, label="Recovery R#"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CAV_COLOR, markersize=21, label="CAV sortie start"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor=CAV_COLOR, markersize=21, label="CAV sortie end"),
        ])
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=40.0, bbox_to_anchor=(0.5, 0.008))
        completed_tasks = sum(task.status == "done" for task in sim.tasks)
        unfinished_tasks = len(sim.tasks) - completed_tasks
        fig.suptitle(
            f"Rainbow DQN cooperative CAV-UAV episode, seed={sim.seed} - sortie-based storyboard "
            f"| completed tasks={completed_tasks}, unfinished tasks={unfinished_tasks}",
            fontsize=32,
            fontweight="bold",
            y=0.992,
        )
        fig.subplots_adjust(left=0.001, right=0.999, top=0.94, bottom=0.120, hspace=0.105, wspace=-0.010)
        page_path = out_dir / f"{prefix}_storyboard_page_{page_number:02d}.png"
        fig.savefig(page_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        pages.append(page_path)

    return pages, pd.DataFrame(manifest_rows)


def plot_rainbow_case_snapshot(sim, out_path: str | Path, now: float, title: str, window_s: float = 200.0) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_start = max(0.0, float(now) - float(window_s))
    pos = _paper_network_pos(sim.network)
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    _draw_base_network(ax, sim, pos)

    window_tasks = _window_tasks(sim, window_start, float(now))
    for task_idx, task in enumerate(window_tasks):
        color = TASK_COLORS.get(task.type, "#888888")
        for u, v in task.target_arcs:
            _draw_plain_segment(
                ax,
                pos,
                int(u),
                int(v),
                color=color,
                linewidth=4.0 if task.type == "incident" else 3.2,
                alpha=0.92,
                zorder=7,
            )
        _annotate_task_release(ax, pos, task, task_idx)

    for u, v in _vehicle_edges_in_window(sim, window_start, float(now)):
        _plot_arrow_segment(
            ax,
            pos,
            u,
            v,
            color="#1f4e79",
            linewidth=3.2,
            alpha=0.95,
            mutation_scale=13.0,
            shrink=13.0,
            zorder=8,
        )

    segments = _sortie_segments_in_window(sim, window_start, float(now))
    drone_colors = _drone_color_map([seg["drone_id"] for seg in segments])
    for seg in segments:
        active = bool(seg.get("active"))
        _plot_arrow_segment(
            ax,
            pos,
            seg["from"],
            seg["to"],
            color="#0072bd",
            linewidth=2.9 if active else 2.0,
            linestyle="--",
            alpha=0.95 if active else 0.58,
            mutation_scale=12.0,
            shrink=14.0,
            zorder=10 if active else 9,
        )
    _draw_congestion_service_order(ax, pos, segments, _task_lookup(sim))

    for seg in _active_drone_positions(segments):
        u, v = int(seg["from"]), int(seg["to"])
        if u in pos and v in pos:
            x = (pos[u][0] + pos[v][0]) / 2.0
            y = (pos[u][1] + pos[v][1]) / 2.0
            color = drone_colors.get(seg["drone_id"], "#0072bd")
            ax.scatter(x, y, s=85, marker="^", c=color, edgecolors="white", linewidths=0.65, zorder=14)
            ax.text(x, y + 0.16, seg["drone_id"], color=color, fontsize=7.5, fontweight="bold", ha="center", zorder=14)

    min_x, max_x, min_y, max_y, span = _network_bounds(pos)
    ax.set_xlim(min_x - 0.06 * span, max_x + 0.24 * span)
    ax.set_ylim(min_y - 0.07 * span, max_y + 0.08 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=8)

    handles = [
        Line2D([0], [0], color=TASK_COLORS["routine"], lw=3.0, label="Routine patrol"),
        Line2D([0], [0], color=TASK_COLORS["congestion"], lw=3.0, label="Congestion inspection"),
        Line2D([0], [0], color=TASK_COLORS["incident"], lw=3.0, label="Incident confirmation"),
        Line2D([0], [0], color="#1f4e79", lw=3.2, label="CAV route"),
        Line2D([0], [0], color="#0072bd", lw=2.3, linestyle="--", label="UAV path"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#2f80ed", markeredgecolor="#2f80ed", markersize=10, label="Patrol center"),
    ]
    for drone_id, color in drone_colors.items():
        handles.append(Line2D([0], [0], marker="^", color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=f"{drone_id} position"))
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.015, 0.52),
        frameon=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        prop={"family": "Times New Roman", "size": 10.5},
        borderpad=0.9,
        labelspacing=0.85,
        handlelength=2.2,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rainbow_first_horizon(sim, out_path: str | Path, end_time: float, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = _paper_network_pos(sim.network)
    fig, ax = plt.subplots(figsize=(13.8, 6.9))
    _draw_base_network(ax, sim, pos)

    released_tasks = _tasks_released_until(sim, end_time)
    for task_idx, task in enumerate(released_tasks):
        color = TASK_COLORS.get(task.type, "#888888")
        for u, v in task.target_arcs:
            _draw_plain_segment(
                ax,
                pos,
                int(u),
                int(v),
                color=color,
                linewidth=4.0 if task.type == "incident" else 3.1,
                alpha=0.88,
                zorder=7,
            )
        _annotate_task_release(ax, pos, task, task_idx)

    for u, v in _vehicle_edges_until(sim, end_time):
        _plot_arrow_segment(
            ax,
            pos,
            u,
            v,
            color="#1f4e79",
            linewidth=3.25,
            alpha=0.95,
            mutation_scale=13.0,
            shrink=13.0,
            zorder=9,
        )

    segments = _sortie_segments_until(sim, end_time)
    drone_colors = _drone_color_map([seg["drone_id"] for seg in segments])
    for seg in segments:
        color = drone_colors.get(seg["drone_id"], "#0072bd")
        _plot_arrow_segment(
            ax,
            pos,
            seg["from"],
            seg["to"],
            color=color,
            linewidth=2.35,
            linestyle="--",
            alpha=0.76,
            mutation_scale=12.0,
            shrink=14.0,
            zorder=10,
        )
    _draw_congestion_service_order(ax, pos, segments, _task_lookup(sim))

    label_counts: Dict[int, int] = defaultdict(int)
    for sortie in _sorties_until(sim, end_time):
        color = drone_colors.get(sortie["drone_id"], "#0072bd")
        _annotate_node_marker(
            ax,
            pos,
            sortie["launch_node"],
            f"L{sortie['idx']}",
            color,
            label_counts,
        )
        _annotate_node_marker(
            ax,
            pos,
            sortie["recovery_node"],
            f"R{sortie['idx']}",
            color,
            label_counts,
        )

    min_x, max_x, min_y, max_y, span = _network_bounds(pos)
    ax.set_xlim(min_x - 0.06 * span, max_x + 0.27 * span)
    ax.set_ylim(min_y - 0.08 * span, max_y + 0.08 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=8)

    handles = [
        Line2D([0], [0], color=TASK_COLORS["routine"], lw=3.0, label="Routine patrol"),
        Line2D([0], [0], color=TASK_COLORS["congestion"], lw=3.0, label="Congestion inspection"),
        Line2D([0], [0], color=TASK_COLORS["incident"], lw=3.0, label="Incident confirmation"),
        Line2D([0], [0], color="#1f4e79", lw=3.2, label="CAV route"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#2f80ed", markeredgecolor="#2f80ed", markersize=10, label="Patrol center"),
    ]
    for drone_id, color in drone_colors.items():
        handles.append(Line2D([0], [0], color=color, lw=2.35, linestyle="--", label=f"{drone_id} route"))
    handles.extend([
        Line2D([0], [0], marker="$L$", color="#333333", lw=0, markersize=10, label="Launch node"),
        Line2D([0], [0], marker="$R$", color="#333333", lw=0, markersize=10, label="Recovery node"),
    ])
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.015, 0.52),
        frameon=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        prop={"family": "Times New Roman", "size": 10.2},
        borderpad=0.9,
        labelspacing=0.82,
        handlelength=2.2,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _snapshot_times(sim, interval_s: float) -> List[float]:
    horizon = float(sim.horizon)
    step = max(1.0, float(interval_s))
    times = list(np.arange(step, horizon + 1e-9, step))
    if not times or times[-1] < horizon - 1e-9:
        times.append(horizon)
    return [float(t) for t in times]


def _task_summary(sim) -> Dict[str, int]:
    return {
        "routine": sum(t.type == "routine" for t in sim.tasks),
        "congestion": sum(t.type == "congestion" for t in sim.tasks),
        "incident": sum(t.type == "incident" for t in sim.tasks),
        "done": sum(t.status == "done" for t in sim.tasks),
        "expired": sum(t.status == "expired" for t in sim.tasks),
        "sorties": len(sim.sortie_log),
        "vehicle_edges": len(getattr(sim.vehicle, "route_edge_log", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a representative Rainbow DQN case on the 60-node network.")
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--model", default=str(DEFAULT_MODELS["mask_rainbow_dqn"].path))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="case")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--snapshot-interval-s", type=float, default=200.0)
    parser.add_argument("--window-s", type=float, default=200.0)
    parser.add_argument("--first-horizon-s", type=float, default=None)
    parser.add_argument("--storyboard", action="store_true", help="Plot launch-to-recovery coordination cycles as multi-panel pages.")
    parser.add_argument("--cycles-per-page", type=int, default=12)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--merge-gap-s", type=float, default=0.0, help="Group launches occurring within this many seconds of the first launch in a dispatch batch.")
    parser.add_argument("--exclude-panels", default="", help="Comma-separated storyboard panel labels to omit, for example h,i,k.")
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--out-dir", default="results/case_visuals/rainbow_60node")
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    summaries = []
    storyboard_manifests = []
    storyboard_page_count = 0
    for seed in range(int(args.seed), int(args.seed) + int(args.seed_count)):
        sim, action_log = run_rainbow_episode(cfg, seed, model_path, args.scenario, args.device)
        prefix = f"rainbow_case_seed_{seed}"
        sim.export_logs(out_dir, prefix=prefix)
        pd.DataFrame(action_log).to_csv(out_dir / f"{prefix}_actions.csv", index=False)
        summary = {"seed": seed, **_task_summary(sim)}
        summaries.append(summary)

        if args.storyboard:
            pages, manifest = plot_sortie_storyboards(
                sim,
                out_dir,
                prefix,
                cycles_per_page=args.cycles_per_page,
                max_cycles=args.max_cycles,
                merge_gap_s=args.merge_gap_s,
                exclude_panels=args.exclude_panels,
            )
            storyboard_page_count += len(pages)
            if not manifest.empty:
                manifest.insert(0, "seed", seed)
                storyboard_manifests.append(manifest)
        elif args.first_horizon_s is not None:
            end_time = float(args.first_horizon_s)
            plot_rainbow_first_horizon(
                sim,
                out_dir / f"{prefix}_first_{int(round(end_time)):04d}s.png",
                end_time,
                title=f"Rainbow DQN case, seed={seed}, first {end_time:.0f} s",
            )
        else:
            times = _snapshot_times(sim, args.snapshot_interval_s)
            for i, now in enumerate(times, start=1):
                plot_rainbow_case_snapshot(
                    sim,
                    out_dir / f"{prefix}_window_{i:02d}_t{int(round(now)):04d}.png",
                    now,
                    title=(
                        f"Rainbow DQN case, seed={seed}, "
                        f"t={now / 60.0:.1f} min, window=[{max(0.0, now - args.window_s):.0f}, {now:.0f}] s"
                    ),
                    window_s=args.window_s,
                )

    pd.DataFrame(summaries).to_csv(out_dir / "rainbow_case_summary.csv", index=False)
    if storyboard_manifests:
        pd.concat(storyboard_manifests, ignore_index=True).to_csv(
            out_dir / "rainbow_case_storyboard_manifest.csv",
            index=False,
        )

    print(pd.DataFrame(summaries).to_string(index=False))
    if args.storyboard:
        print(f"Saved {storyboard_page_count} launch/recovery storyboard pages.")
    elif args.first_horizon_s is not None:
        print(f"Saved {len(summaries)} first-{args.first_horizon_s:.0f}s figures.")
    else:
        print(f"Saved window figures at {args.snapshot_interval_s:.0f}s intervals.")
    print(f"Saved case figures: {out_dir}")


if __name__ == "__main__":
    main()
