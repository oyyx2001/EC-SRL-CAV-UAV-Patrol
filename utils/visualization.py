from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import ast
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
import networkx as nx
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]


class _HandlerDirectedArc(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        arrow = FancyArrowPatch(
            (xdescent, y),
            (xdescent + width, y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=orig_handle.get_linewidth(),
            color=orig_handle.get_edgecolor(),
            shrinkA=0.0,
            shrinkB=0.0,
            transform=trans,
        )
        return [arrow]


def _network_bounds(pos: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float, float, float]:
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    return min_x, max_x, min_y, max_y, span


def _paper_network_pos(network) -> Dict[int, Tuple[float, float]]:
    if all("grid_row" in network.G.nodes[n] and "grid_col" in network.G.nodes[n] for n in network.nodes):
        return {
            int(n): (
                float(network.G.nodes[n]["grid_col"]),
                float(network.G.nodes[n]["grid_row"]) * 0.86,
            )
            for n in network.nodes
        }
    return {int(n): network.node_xy(n) for n in network.nodes}


def _paper_display_labels(network) -> Dict[int, int]:
    center = int(network.patrol_center)
    regular_nodes = [int(n) for n in network.nodes if int(n) != center]
    labels = {int(n): idx for idx, n in enumerate(regular_nodes, start=2)}
    labels[center] = 1
    return labels


def _paper_hidden_arcs(network) -> set[Tuple[int, int]]:
    labels = _paper_display_labels(network)
    hidden_display_pairs = {frozenset((1, 32)), frozenset((32, 34))}
    hidden = set()
    for u, v in network.G.edges():
        if frozenset((labels.get(int(u)), labels.get(int(v)))) in hidden_display_pairs:
            hidden.add((int(u), int(v)))
    return hidden


def _expand_highlighted_bidirectional_arcs(network, arcs: set[Tuple[int, int]]) -> set[Tuple[int, int]]:
    expanded = set(arcs)
    for u, v in arcs:
        if network.G.has_edge(v, u):
            expanded.add((int(v), int(u)))
    return expanded


def _draw_paper_network_edges(
    ax,
    network,
    pos: Dict[int, Tuple[float, float]],
    highlighted_arcs: set[Tuple[int, int]],
    hidden_arcs: set[Tuple[int, int]],
) -> None:
    for u, v in sorted(network.G.edges()):
        if u not in pos or v not in pos:
            continue
        if (int(u), int(v)) in hidden_arcs:
            continue
        is_highlighted = (int(u), int(v)) in highlighted_arcs
        color = "#2f80ed" if is_highlighted else "#adadad"
        linewidth = 2.35 if is_highlighted else 1.65
        alpha = 0.96 if is_highlighted else 0.82
        arrow = FancyArrowPatch(
            pos[int(u)],
            pos[int(v)],
            arrowstyle="-|>",
            mutation_scale=10.5 if is_highlighted else 9.0,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            shrinkA=11.5,
            shrinkB=11.5,
            zorder=1,
        )
        ax.add_patch(arrow)


def plot_network(network, out_path: str | Path, task_arcs: Sequence[Tuple[int, int]] | None = None, title: str = "(a) Road-network representation") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = _paper_network_pos(network)
    highlighted_arcs = _expand_highlighted_bidirectional_arcs(
        network,
        {(int(u), int(v)) for u, v in (task_arcs or [])},
    )
    hidden_arcs = _paper_hidden_arcs(network)
    fig, ax = plt.subplots(figsize=(12.8, 6.2))

    _draw_paper_network_edges(ax, network, pos, highlighted_arcs, hidden_arcs)

    center = int(network.patrol_center)
    regular_nodes = [int(n) for n in network.nodes if int(n) != center]
    regular_collection = nx.draw_networkx_nodes(
        network.G,
        pos,
        ax=ax,
        nodelist=regular_nodes,
        node_shape="o",
        node_size=330,
        node_color="white",
        edgecolors="#202020",
        linewidths=1.55,
    )
    regular_collection.set_zorder(3)
    center_collection = nx.draw_networkx_nodes(
        network.G,
        pos,
        ax=ax,
        nodelist=[center],
        node_shape="s",
        node_size=350,
        node_color="#2f80ed",
        edgecolors="#2f80ed",
        linewidths=1.25,
    )
    center_collection.set_zorder(4)

    display_labels = _paper_display_labels(network)
    labels = {int(n): str(display_labels[int(n)]) for n in regular_nodes}
    label_artists = nx.draw_networkx_labels(
        network.G,
        pos,
        labels=labels,
        ax=ax,
        font_size=10.2,
        font_weight="bold",
        font_color="#111111",
    )
    for artist in label_artists.values():
        artist.set_zorder(5)
    cx, cy = pos[center]
    ax.text(cx, cy, "1", color="white", fontsize=10.8, fontweight="bold", ha="center", va="center", zorder=5)

    min_x, max_x, min_y, max_y, span = _network_bounds(pos)
    pad_x = 0.045 * span
    pad_y = 0.06 * span
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=8)

    arc_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#a7a7a7", linewidth=1.8)
    task_arc_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#2f80ed", linewidth=2.4)
    node_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#202020",
        markeredgewidth=1.45,
        markersize=13,
        label="Node (ID)",
    )
    center_handle = Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor="#2f80ed",
        markeredgecolor="#2f80ed",
        markersize=14,
        label="Patrol center\n(CAV initial location)",
    )
    handles = [arc_handle]
    labels_for_legend = ["Road arc\n(directed)"]
    if highlighted_arcs:
        handles.append(task_arc_handle)
        labels_for_legend.append("Task-prone arc")
    handles.extend([node_handle, center_handle])
    labels_for_legend.extend(["Node (ID)", "Patrol center\n(CAV initial location)"])
    ax.legend(
        handles,
        labels_for_legend,
        loc="center left",
        bbox_to_anchor=(1.015, 0.55),
        frameon=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        prop={"family": "Times New Roman", "size": 12},
        borderpad=1.0,
        labelspacing=1.25,
        handlelength=2.0,
        handler_map={FancyArrowPatch: _HandlerDirectedArc()},
    )

    fig.tight_layout()
    dpi = int(network.cfg.get("visualization", {}).get("dpi", 300))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_simple_25_node_network(out_path: str | Path, title: str = "(b) Simplified road-network representation") -> None:
    """Draw a standalone 25-node undirected road-network sketch."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    center = 0
    pos = {
        0: (3.2, 2.0),
        1: (3.2, 2.85),
        2: (4.25, 2.85),
        3: (4.25, 2.0),
        4: (3.2, 1.15),
        5: (2.15, 2.85),
        6: (2.15, 3.65),
        7: (3.2, 3.65),
        8: (4.25, 3.65),
        9: (5.3, 3.65),
        10: (6.35, 2.85),
        11: (6.35, 2.0),
        12: (5.3, 2.0),
        13: (5.3, 0.3),
        14: (4.25, 0.3),
        15: (3.2, 0.3),
        16: (2.15, 0.3),
        17: (1.1, 0.3),
        18: (1.1, 1.15),
        19: (0.05, 1.15),
        20: (0.05, 2.0),
        21: (0.05, 2.85),
        22: (1.1, 2.85),
        23: (1.1, 3.65),
    }
    display_labels = {node: node + 1 for node in pos}
    edges = {
        tuple(sorted(pair))
        for pair in [
            (0, 1),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 4),
            (2, 3),
            (2, 8),
            (3, 4),
            (3, 12),
            (3, 14),
            (4, 15),
            (4, 3),
            (5, 6),
            (5, 22),
            (5, 18),
            (6, 7),
            (6, 5),
            (7, 8),
            (7, 5),
            (8, 9),
            (8, 2),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (16, 18),
            (17, 18),
            (17, 19),
            (18, 21),
            (18, 22),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 5),
            (23, 6),
        ]
    }

    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    for u, v in sorted(edges):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#adadad", linewidth=1.85, alpha=0.86, zorder=1)

    regular_nodes = [n for n in sorted(pos) if n != center]
    ax.scatter(
        [pos[n][0] for n in regular_nodes],
        [pos[n][1] for n in regular_nodes],
        s=360,
        marker="o",
        facecolors="white",
        edgecolors="#202020",
        linewidths=1.6,
        zorder=3,
    )
    ax.scatter(
        [pos[center][0]],
        [pos[center][1]],
        s=390,
        marker="s",
        facecolors="#2f80ed",
        edgecolors="#2f80ed",
        linewidths=1.25,
        zorder=4,
    )

    for node, label in display_labels.items():
        x, y = pos[node]
        ax.text(
            x,
            y,
            str(label),
            color="white" if node == center else "#111111",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )

    min_x, max_x, min_y, max_y, span = _network_bounds(pos)
    ax.set_xlim(min_x - 0.28 * span, max_x + 0.34 * span)
    ax.set_ylim(min_y - 0.12 * span, max_y + 0.12 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=8)

    road_handle = Line2D([0], [0], color="#adadad", lw=2.0, label="Road link")
    node_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#202020",
        markeredgewidth=1.45,
        markersize=13,
        label="Node (ID)",
    )
    center_handle = Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor="#2f80ed",
        markeredgecolor="#2f80ed",
        markersize=14,
        label="Patrol center\n(CAV initial location)",
    )
    ax.legend(
        [road_handle, node_handle, center_handle],
        ["Road link", "Node (ID)", "Patrol center\n(CAV initial location)"],
        loc="center left",
        bbox_to_anchor=(1.015, 0.55),
        frameon=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        prop={"family": "Times New Roman", "size": 12},
        borderpad=1.0,
        labelspacing=1.25,
        handlelength=2.0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_task_stats(tasks_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    counts = tasks_df["type"].value_counts().sort_index()
    counts.plot(kind="bar")
    plt.ylabel("count")
    plt.title("Generated dynamic task types")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for typ, sub in tasks_df.groupby("type"):
        plt.hist(sub["release_time"], bins=12, alpha=0.5, label=str(typ))
    plt.xlabel("release time (s)")
    plt.ylabel("count")
    plt.title("Task release distribution")
    plt.legend()
    plt.tight_layout()
    hist_path = out_path.with_name(out_path.stem + "_release_hist.png")
    plt.savefig(hist_path, dpi=160)
    plt.close()


def _plot_arrow_segment(
    ax,
    pos: Dict[int, Tuple[float, float]],
    u: int,
    v: int,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    mutation_scale: float = 14.0,
    shrink: float = 9.0,
    zorder: int = 4,
) -> None:
    if u not in pos or v not in pos or u == v:
        return
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    if math.hypot(x1 - x0, y1 - y0) <= 1e-9:
        return
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        mutation_scale=mutation_scale,
        shrinkA=shrink,
        shrinkB=shrink,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _label_base_offset(pos: Dict[int, Tuple[float, float]]) -> float:
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return 0.022 * span


def _label_delta(index: int, base: float) -> Tuple[float, float]:
    offsets = [
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.45, 0.0),
        (0.0, 1.45),
        (-1.45, 0.0),
        (0.0, -1.45),
    ]
    ox, oy = offsets[index % len(offsets)]
    scale = 1.0 + 0.35 * (index // len(offsets))
    return ox * base * scale, oy * base * scale


def _annotate_node(
    ax,
    pos: Dict[int, Tuple[float, float]],
    node: int,
    text: str,
    color: str,
    label_counts,
    *,
    font_size: int = 7,
) -> None:
    if node not in pos:
        return
    base = _label_base_offset(pos)
    idx = label_counts[int(node)]
    label_counts[int(node)] += 1
    dx, dy = _label_delta(idx, base)
    x, y = pos[node]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        color=color,
        fontsize=font_size,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.8},
        zorder=9,
    )


def _annotate_point(
    ax,
    xy: Tuple[float, float],
    text: str,
    color: str,
    offset_index: int,
    base: float,
    *,
    font_size: int = 7,
) -> None:
    dx, dy = _label_delta(offset_index, base)
    ax.annotate(
        text,
        xy=xy,
        xytext=(xy[0] + dx, xy[1] + dy),
        color=color,
        fontsize=font_size,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.88, "pad": 1.5, "linewidth": 0.6},
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.85},
        zorder=9,
    )


def _coerce_segments(segments) -> List[Dict]:
    if not segments:
        return []
    if isinstance(segments, str):
        try:
            parsed = ast.literal_eval(segments)
        except (SyntaxError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []
    return list(segments)


def _sortie_nodes(row: Dict) -> List[int]:
    pts = [int(row["launch_node"])]
    for seg in _coerce_segments(row.get("segments", [])):
        if seg.get("type") in {"fly", "service"} and seg.get("to") is not None:
            pts.append(int(seg["to"]))
    if "recovery_node" in row and (not pts or pts[-1] != int(row["recovery_node"])):
        pts.append(int(row["recovery_node"]))
    return pts


def _tab10_hex(index: int) -> str:
    color = plt.get_cmap("tab10")(index % 10)
    return "#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in color[:3]))


def _drone_color_map(drone_ids: Sequence[str]) -> Dict[str, str]:
    unique_ids = sorted({str(drone_id) for drone_id in drone_ids})
    return {drone_id: _tab10_hex(idx) for idx, drone_id in enumerate(unique_ids)}


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _interpolate_xy(pos: Dict[int, Tuple[float, float]], u: int, v: int, ratio: float) -> Tuple[float, float]:
    ratio = min(1.0, max(0.0, float(ratio)))
    x0, y0 = pos[int(u)]
    x1, y1 = pos[int(v)]
    return x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio


def _task_state_at(task, now: float) -> str:
    if now < float(task.release_time):
        return "future"
    if task.completion_time is not None and now >= float(task.completion_time):
        return "done"
    if now >= float(task.deadline) and task.status == "expired":
        return "expired"
    if task.start_time is not None and now >= float(task.start_time):
        return "executing"
    return "available"


def _vehicle_motion_segments(sim) -> List[Dict]:
    segments: List[Dict] = []
    for arrival, u, v in getattr(sim.vehicle, "route_edge_log", []):
        u = int(u)
        v = int(v)
        duration = float(sim.network.G[u][v]["vehicle_time_s"]) if sim.network.G.has_edge(u, v) else 0.0
        segments.append({"start": max(0.0, float(arrival) - duration), "end": float(arrival), "from": u, "to": v})
    return sorted(segments, key=lambda x: (x["start"], x["end"]))


def _point_on_vehicle(sim, pos: Dict[int, Tuple[float, float]], now: float, segments: Sequence[Dict]) -> Tuple[float, float, int]:
    initial = int(sim.vehicle.route_log[0][1]) if sim.vehicle.route_log else int(sim.network.patrol_center)
    node = initial
    for seg in segments:
        if now < seg["start"]:
            break
        if seg["start"] <= now <= seg["end"] and seg["end"] > seg["start"]:
            ratio = (now - seg["start"]) / (seg["end"] - seg["start"])
            x, y = _interpolate_xy(pos, seg["from"], seg["to"], ratio)
            return x, y, int(seg["to"])
        node = int(seg["to"])
    x, y = pos[node]
    return x, y, node


def _recover_time_for_sortie(sim, row: Dict, sortie_end: float) -> float:
    drone_id = str(row.get("drone_id", ""))
    recovery_node = int(row.get("recovery_node"))
    candidates = [
        float(e["time"])
        for e in sim.event_log
        if e.get("event") == "recover"
        and str(e.get("agent")) == drone_id
        and int(e.get("node", recovery_node)) == recovery_node
        and float(e.get("time", 0.0)) >= sortie_end - 1e-9
    ]
    return min(candidates) if candidates else sortie_end


def _sortie_schedules(sim) -> List[Dict]:
    schedules: List[Dict] = []
    for idx, row in enumerate(sim.sortie_log, start=1):
        launch_time = float(row["time"])
        current = launch_time
        segments = []
        for seg in _coerce_segments(row.get("segments", [])):
            if seg.get("from") is None or seg.get("to") is None:
                continue
            duration = float(seg.get("duration", 0.0))
            segments.append({
                "start": current,
                "end": current + duration,
                "from": int(seg["from"]),
                "to": int(seg["to"]),
                "type": str(seg.get("type", "fly")),
            })
            current += duration
        schedules.append({
            "idx": idx,
            "drone_id": str(row.get("drone_id", f"drone{idx}")),
            "launch": int(row["launch_node"]),
            "recovery": int(row["recovery_node"]),
            "start": launch_time,
            "end": current,
            "recover": _recover_time_for_sortie(sim, row, current),
            "task_ids": list(row.get("task_ids", [])) if not isinstance(row.get("task_ids"), str) else row.get("task_ids"),
            "segments": segments,
        })
    return schedules


def _point_on_sortie(pos: Dict[int, Tuple[float, float]], schedule: Dict, now: float) -> Tuple[Optional[float], Optional[float], str]:
    if now < schedule["start"] or now > schedule["recover"]:
        return None, None, "idle"
    for seg in schedule["segments"]:
        if seg["start"] <= now <= seg["end"] and seg["end"] > seg["start"]:
            ratio = (now - seg["start"]) / (seg["end"] - seg["start"])
            x, y = _interpolate_xy(pos, seg["from"], seg["to"], ratio)
            return x, y, seg["type"]
    node = schedule["recovery"] if now >= schedule["end"] else schedule["launch"]
    x, y = pos[node]
    mode = "waiting recovery" if now >= schedule["end"] else "ready"
    return x, y, mode


def _draw_task_snapshot(ax, sim, pos: Dict[int, Tuple[float, float]], now: float) -> Dict[str, int]:
    type_colors = {
        "routine": "#1f78b4",
        "incident": "#e6550d",
        "congestion": "#6a3d9a",
    }
    state_counts = {"future": 0, "available": 0, "executing": 0, "done": 0, "expired": 0}
    state_alpha = {"future": 0.18, "available": 0.82, "executing": 1.0, "done": 0.28, "expired": 0.75}
    state_width = {"future": 1.0, "available": 2.6, "executing": 4.2, "done": 1.7, "expired": 2.2}
    for task in sim.tasks:
        state = _task_state_at(task, now)
        state_counts[state] += 1
        color = type_colors.get(task.type, "#525252")
        linestyle = "--" if state == "expired" else "-"
        for u, v in task.target_arcs:
            _plot_arrow_segment(
                ax,
                pos,
                int(u),
                int(v),
                color=color,
                linewidth=state_width[state],
                linestyle=linestyle,
                alpha=state_alpha[state],
                mutation_scale=12 if state == "future" else 15,
                shrink=12,
                zorder=4 if state in {"future", "done"} else 6,
            )
        if state in {"available", "executing", "expired"}:
            u, v = task.target_arcs[0]
            cx = (pos[int(u)][0] + pos[int(v)][0]) / 2.0
            cy = (pos[int(u)][1] + pos[int(v)][1]) / 2.0
            ax.text(
                cx,
                cy,
                f"T{task.id}",
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                zorder=9,
            )
    return state_counts


def _recent_events(sim, now: float, window_s: float = 180.0, limit: int = 5) -> List[str]:
    rows = []
    for e in sim.event_log:
        t = float(e.get("time", 0.0))
        if t <= now and now - t <= window_s:
            event = str(e.get("event", "event"))
            agent = str(e.get("agent", ""))
            node = e.get("node", "")
            tasks = e.get("tasks", "")
            suffix = f" T{tasks}" if tasks not in {"", None} else ""
            rows.append(f"{_format_time(t)} {event} {agent}@{node}{suffix}")
    return rows[-limit:]


def plot_episode_animation(
    sim,
    out_path: str | Path,
    title: str = "Episode animation",
    frames: int = 60,
    fps: int = 5,
) -> None:
    """Save a compact GIF animation for one simulated episode."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    network = sim.network
    pos = {n: network.node_xy(n) for n in network.nodes}
    vehicle_segments = _vehicle_motion_segments(sim)
    sortie_schedules = _sortie_schedules(sim)
    drone_colors = _drone_color_map([s["drone_id"] for s in sortie_schedules])
    horizon = float(sim.horizon)
    frames = max(20, int(frames))
    fps = max(1, int(fps))
    frame_times = [horizon * i / (frames - 1) for i in range(frames)]

    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(11.5, 7.0),
        gridspec_kw={"width_ratios": [4.6, 1.35]},
    )
    def draw(now: float):
        ax.clear()
        info_ax.clear()
        nx.draw_networkx_edges(
            network.G,
            pos,
            ax=ax,
            edge_color="#d7dbe0",
            arrows=True,
            arrowsize=6,
            width=0.7,
            alpha=0.5,
        )
        nx.draw_networkx_nodes(network.G, pos, ax=ax, node_size=36, node_color="#2b83ba", edgecolors="white", linewidths=0.35)
        nx.draw_networkx_nodes(
            network.G,
            pos,
            ax=ax,
            nodelist=[network.patrol_center],
            node_shape="s",
            node_size=145,
            node_color="#2b83ba",
            edgecolors="white",
            linewidths=0.7,
        )
        nx.draw_networkx_labels(network.G, pos, ax=ax, font_size=6, font_color="#425466")
        task_counts = _draw_task_snapshot(ax, sim, pos, now)

        for seg in vehicle_segments:
            if seg["end"] <= now:
                _plot_arrow_segment(ax, pos, seg["from"], seg["to"], color="#d73027", linewidth=2.2, alpha=0.35, mutation_scale=11, shrink=12, zorder=5)
            elif seg["start"] <= now <= seg["end"]:
                _plot_arrow_segment(ax, pos, seg["from"], seg["to"], color="#d73027", linewidth=3.0, alpha=0.88, mutation_scale=15, shrink=12, zorder=7)
                break
        cav_x, cav_y, cav_node = _point_on_vehicle(sim, pos, now, vehicle_segments)
        ax.scatter(cav_x, cav_y, s=115, marker="s", c="#d73027", edgecolors="white", linewidths=0.8, zorder=10)
        ax.text(cav_x, cav_y, "CAV", color="white", fontsize=6, fontweight="bold", ha="center", va="center", zorder=11)

        active_uavs = []
        for schedule in sortie_schedules:
            color = drone_colors.get(schedule["drone_id"], _tab10_hex(0))
            for seg in schedule["segments"]:
                if seg["end"] <= now:
                    _plot_arrow_segment(
                        ax,
                        pos,
                        seg["from"],
                        seg["to"],
                        color=color,
                        linewidth=1.4,
                        linestyle="--" if seg["type"] == "fly" else ":",
                        alpha=0.22,
                        mutation_scale=9,
                        shrink=13,
                        zorder=4,
                    )
                elif seg["start"] <= now <= seg["end"]:
                    _plot_arrow_segment(
                        ax,
                        pos,
                        seg["from"],
                        seg["to"],
                        color=color,
                        linewidth=2.4,
                        linestyle="--" if seg["type"] == "fly" else ":",
                        alpha=0.92,
                        mutation_scale=14,
                        shrink=13,
                        zorder=8,
                    )
                    break
            x, y, mode = _point_on_sortie(pos, schedule, now)
            if x is not None and y is not None:
                ax.scatter(x, y, s=95, marker="^", c=color, edgecolors="black", linewidths=0.5, zorder=11)
                ax.text(x, y, f"U{schedule['idx']}", color="white", fontsize=6, fontweight="bold", ha="center", va="center", zorder=12)
                active_uavs.append(f"U{schedule['idx']} {schedule['drone_id']}: {mode}")

        ax.set_title(f"{title} | t={_format_time(now)}")
        ax.axis("equal")
        ax.axis("off")

        info_ax.axis("off")
        info_ax.set_xlim(0.0, 1.0)
        info_ax.set_ylim(0.0, 1.0)
        progress = now / max(horizon, 1.0)
        info_ax.text(0.0, 0.97, "Episode clock", fontsize=10, fontweight="bold", va="top")
        info_ax.text(0.0, 0.91, f"{_format_time(now)} / {_format_time(horizon)}", fontsize=9, va="top")
        info_ax.add_line(Line2D([0.0, 0.95], [0.86, 0.86], color="#d7dbe0", lw=5))
        info_ax.add_line(Line2D([0.0, 0.95 * progress], [0.86, 0.86], color="#2b83ba", lw=5))
        y = 0.78
        info_ax.text(0.0, y, "Tasks", fontsize=10, fontweight="bold", va="top")
        y -= 0.06
        for key in ["future", "available", "executing", "done", "expired"]:
            info_ax.text(0.02, y, f"{key}: {task_counts[key]}", fontsize=8, va="top")
            y -= 0.045
        y -= 0.02
        info_ax.text(0.0, y, "Agents", fontsize=10, fontweight="bold", va="top")
        y -= 0.06
        info_ax.text(0.02, y, f"CAV node target: {cav_node}", fontsize=8, va="top")
        y -= 0.045
        for line in active_uavs[:5]:
            info_ax.text(0.02, y, line, fontsize=8, va="top")
            y -= 0.045
        y -= 0.02
        info_ax.text(0.0, y, "Recent events", fontsize=10, fontweight="bold", va="top")
        y -= 0.06
        for line in _recent_events(sim, now):
            info_ax.text(0.02, y, line, fontsize=7.2, va="top")
            y -= 0.047

    def update(frame_idx: int):
        draw(frame_times[frame_idx])
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_times), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def plot_episode_tasks(sim, out_path: str | Path, title: str = "Generated tasks") -> None:
    """Plot generated task locations without CAV/UAV trajectories."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    network = sim.network
    pos = {n: network.node_xy(n) for n in network.nodes}
    fig, ax = plt.subplots(figsize=(10, 7.2))

    nx.draw_networkx_edges(
        network.G,
        pos,
        ax=ax,
        edge_color="#d7dbe0",
        arrows=True,
        arrowsize=7,
        width=0.8,
        alpha=0.55,
    )
    nx.draw_networkx_nodes(network.G, pos, ax=ax, node_size=42, node_color="#2b83ba", edgecolors="white", linewidths=0.4)
    nx.draw_networkx_nodes(
        network.G,
        pos,
        ax=ax,
        nodelist=[network.patrol_center],
        node_shape="s",
        node_size=180,
        node_color="#2b83ba",
        edgecolors="white",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(network.G, pos, ax=ax, font_size=7, font_color="#425466")

    type_colors = {
        "routine": "#1f78b4",
        "incident": "#e6550d",
        "congestion": "#6a3d9a",
    }
    status_styles = {
        "done": "-",
        "expired": "--",
        "timeout": "--",
    }
    base_offset = _label_base_offset(pos) * 1.4
    for idx, task in enumerate(sim.tasks):
        color = type_colors.get(task.type, "#525252")
        linestyle = status_styles.get(task.status, "-.")
        midpoint_xs: List[float] = []
        midpoint_ys: List[float] = []
        for u, v in task.target_arcs:
            if u not in pos or v not in pos:
                continue
            _plot_arrow_segment(
                ax,
                pos,
                int(u),
                int(v),
                color=color,
                linewidth=3.1,
                linestyle=linestyle,
                alpha=0.92,
                mutation_scale=16,
                shrink=10,
                zorder=6,
            )
            midpoint_xs.append((pos[u][0] + pos[v][0]) / 2.0)
            midpoint_ys.append((pos[u][1] + pos[v][1]) / 2.0)
        if not midpoint_xs:
            continue
        center = (sum(midpoint_xs) / len(midpoint_xs), sum(midpoint_ys) / len(midpoint_ys))
        label = (
            f"T{task.id} {task.type}\n"
            f"r={task.release_time:.0f}s d={task.deadline:.0f}s\n"
            f"p={task.priority:.1f} {task.status}"
        )
        _annotate_point(ax, center, label, color, idx, base_offset, font_size=7)

    handles = [
        Line2D([0], [0], color="#d7dbe0", lw=1.2, label="road network"),
        Line2D([0], [0], color=type_colors["routine"], lw=3.1, label="routine task arc"),
        Line2D([0], [0], color=type_colors["incident"], lw=3.1, label="incident task arc"),
        Line2D([0], [0], color=type_colors["congestion"], lw=3.1, label="congestion task arc"),
        Line2D([0], [0], color="#525252", lw=2.2, linestyle="-", label="completed/active"),
        Line2D([0], [0], color="#525252", lw=2.2, linestyle="--", label="expired"),
    ]
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)
    fig.tight_layout()
    dpi = int(sim.cfg.get("visualization", {}).get("dpi", 160))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_episode(sim, out_path: str | Path, title: str = "Episode trajectory") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    network = sim.network
    pos = {n: network.node_xy(n) for n in network.nodes}
    fig, ax = plt.subplots(figsize=(10, 7.2))
    nx.draw_networkx_edges(
        network.G,
        pos,
        ax=ax,
        edge_color="#d7dbe0",
        arrows=True,
        arrowsize=7,
        width=0.8,
        alpha=0.65,
    )
    nx.draw_networkx_nodes(network.G, pos, ax=ax, node_size=46, node_color="#2b83ba", edgecolors="white", linewidths=0.4)
    nx.draw_networkx_nodes(
        network.G,
        pos,
        ax=ax,
        nodelist=[network.patrol_center],
        node_shape="s",
        node_size=190,
        node_color="#2b83ba",
        edgecolors="white",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(network.G, pos, ax=ax, font_size=7, font_color="#425466")

    # Completed task arcs.
    done_arcs = []
    expired_arcs = []
    for t in sim.tasks:
        if t.status == "done":
            done_arcs.extend(t.target_arcs)
        elif t.status == "expired":
            expired_arcs.extend(t.target_arcs)
    if done_arcs:
        nx.draw_networkx_edges(
            network.G,
            pos,
            ax=ax,
            edgelist=done_arcs,
            edge_color="#2ca25f",
            width=3.0,
            arrows=True,
            arrowsize=13,
            alpha=0.72,
        )
    if expired_arcs:
        nx.draw_networkx_edges(
            network.G,
            pos,
            ax=ax,
            edgelist=expired_arcs,
            edge_color="#7a7a7a",
            width=2.0,
            style="dashed",
            arrows=True,
            arrowsize=12,
            alpha=0.75,
        )

    label_counts = defaultdict(int)

    # CAV route. Prefer strict directed edge log; fall back to node-to-node arrows
    # only for backward compatibility with old simulation outputs.
    route_edges = [(int(u), int(v)) for _, u, v in getattr(sim.vehicle, "route_edge_log", [])]
    route_nodes = [int(node) for _, node in sim.vehicle.route_log]
    if route_edges:
        for u, v in route_edges:
            _plot_arrow_segment(ax, pos, u, v, color="#d73027", linewidth=3.0, mutation_scale=17, shrink=10, zorder=6)
    else:
        if len(route_nodes) >= 2:
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                _plot_arrow_segment(ax, pos, u, v, color="#d73027", linewidth=3.0, mutation_scale=17, shrink=10, zorder=6)
    if route_nodes:
        xs = [pos[n][0] for n in route_nodes if n in pos]
        ys = [pos[n][1] for n in route_nodes if n in pos]
        if xs and ys:
            ax.scatter(xs, ys, s=30, c="#d73027", edgecolors="white", linewidths=0.5, zorder=7)
        _annotate_node(ax, pos, route_nodes[0], "CAV start", "#b2182b", label_counts, font_size=7)
        _annotate_node(ax, pos, route_nodes[-1], "CAV end", "#b2182b", label_counts, font_size=7)

    # UAV sorties: dashed directed arrows plus launch/recovery markers.
    drone_colors = _drone_color_map([row.get("drone_id", "") for row in sim.sortie_log])
    for idx, row in enumerate(sim.sortie_log, start=1):
        drone_id = str(row.get("drone_id", ""))
        color = drone_colors.get(drone_id, _tab10_hex(0))
        segments = _coerce_segments(row.get("segments", []))
        for seg in segments:
            if seg.get("type") not in {"fly", "service"}:
                continue
            if seg.get("from") is None or seg.get("to") is None:
                continue
            _plot_arrow_segment(
                ax,
                pos,
                int(seg["from"]),
                int(seg["to"]),
                color=color,
                linewidth=1.8 if seg.get("type") == "fly" else 2.2,
                linestyle="--" if seg.get("type") == "fly" else ":",
                alpha=0.95,
                mutation_scale=13,
                shrink=11,
                zorder=5,
            )
        launch = int(row["launch_node"])
        recovery = int(row["recovery_node"])
        if launch in pos:
            ax.scatter(*pos[launch], s=88, marker="^", c=color, edgecolors="black", linewidths=0.6, zorder=8)
            _annotate_node(ax, pos, launch, f"L{idx}", color, label_counts)
        if recovery in pos:
            ax.scatter(*pos[recovery], s=88, marker="v", c=color, edgecolors="black", linewidths=0.6, zorder=8)
            _annotate_node(ax, pos, recovery, f"R{idx}", color, label_counts)
        pts = [p for p in _sortie_nodes(row) if p in pos]
        if len(pts) >= 2:
            mid = pts[len(pts) // 2]
            _annotate_node(ax, pos, mid, f"U{idx}", color, label_counts)

    handles = [
        Line2D([0], [0], color="#d7dbe0", lw=1.2, label="road network"),
        Line2D([0], [0], color="#d73027", lw=3.0, marker="o", markersize=5, label="CAV route"),
        Line2D([0], [0], color="#1f78b4", lw=1.8, linestyle="--", label="UAV fly direction"),
        Line2D([0], [0], color="#1f78b4", lw=2.2, linestyle=":", label="UAV service direction"),
        Line2D([0], [0], color="#2ca25f", lw=3.0, label="completed task arc"),
        Line2D([0], [0], color="#7a7a7a", lw=2.0, linestyle="--", label="expired task arc"),
        Line2D([0], [0], marker="^", color="black", markerfacecolor="white", lw=0, markersize=8, label="UAV launch L#"),
        Line2D([0], [0], marker="v", color="black", markerfacecolor="white", lw=0, markersize=8, label="UAV recovery R#"),
    ]
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)
    fig.tight_layout()
    dpi = int(sim.cfg.get("visualization", {}).get("dpi", 160))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plt.close()


def plot_baseline_results(results_csv: str | Path, out_dir: str | Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    metrics = ["completion_rate", "timeout_rate", "avg_response_time_s", "weighted_aoi", "total_energy"]
    paths: List[Path] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        grouped = df.groupby("method")[metric].mean().sort_values()
        grouped.plot(kind="bar")
        plt.ylabel(metric)
        plt.title(f"Baseline comparison: {metric}")
        plt.tight_layout()
        p = out_dir / f"comparison_{metric}.png"
        plt.savefig(p, dpi=160)
        plt.close()
        paths.append(p)
    return paths
