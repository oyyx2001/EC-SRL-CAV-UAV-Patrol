from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import math
import json
import pickle
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class ArcRef:
    u: int
    v: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.u, self.v)


class RoadNetwork:
    """Directed urban road graph.

    Vehicle travel uses directed shortest paths. Drone transfer uses Euclidean time,
    but when a drone performs an arc inspection task, service time is modeled in the
    Task object and the target arc is logged as covered.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.G = nx.DiGraph()
        self._patrol_center_cfg = cfg["network"].get("patrol_center", 0)
        self.patrol_center = 0
        self.vehicle_time: Dict[int, Dict[int, float]] = {}
        self.vehicle_paths: Dict[int, Dict[int, List[int]]] = {}
        self.drone_time_matrix: Dict[int, Dict[int, float]] = {}
        self._build()
        self._resolve_patrol_center()
        self._compute_shortest_paths()

    @property
    def nodes(self) -> List[int]:
        return list(self.G.nodes())

    @property
    def arcs(self) -> List[Tuple[int, int]]:
        return list(self.G.edges())

    def _build(self) -> None:
        net_cfg = self.cfg["network"]
        kind = net_cfg.get("type", "synthetic_grid")
        if kind not in {"synthetic_grid", "irregular_city", "imported_osm"}:
            raise ValueError(f"Supported network types: synthetic_grid, irregular_city, imported_osm; got {kind}")
        rng = np.random.default_rng(int(self.cfg["project"].get("seed", 42)) + int(net_cfg.get("risk_seed_shift", 100)))
        if kind == "irregular_city":
            self._build_irregular_city(net_cfg, rng)
        elif kind == "imported_osm":
            self._build_imported_osm(net_cfg, rng)
        else:
            self._build_jittered_grid(net_cfg, rng)
        self._assign_risk(rng)

    def _resolve_import_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        root = Path(__file__).resolve().parents[1]
        candidate = root / path
        if candidate.exists():
            return candidate
        package_root = Path(__file__).resolve().parents[1]
        candidate = package_root / path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Imported road network file not found: {raw_path}")

    def _build_imported_osm(self, net_cfg: Dict, rng: np.random.Generator) -> None:
        data_path = self._resolve_import_path(net_cfg["path"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data.get("nodes", []):
            node = int(row["id"])
            attrs = {
                "x": float(row["x"]),
                "y": float(row["y"]),
            }
            for key in ("lon", "lat", "osm_id", "highway", "name"):
                if key in row:
                    attrs[key] = row[key]
            self.G.add_node(node, **attrs)

        default_speed = float(net_cfg.get("vehicle_speed_mps", 8.3))
        for row in data.get("edges", []):
            u = int(row["u"])
            v = int(row["v"])
            if u == v or u not in self.G or v not in self.G:
                continue
            length = float(row.get("length_m", self._distance(u, v)))
            speed = float(row.get("vehicle_speed_mps", default_speed))
            attrs = {
                "length_m": length,
                "vehicle_speed_mps": max(speed, 1.0),
                "vehicle_time_s": float(row.get("vehicle_time_s", length / max(speed, 1.0))),
                "congestion": float(row.get("congestion", rng.uniform(0.0, 0.7))),
                "risk": float(row.get("risk", 0.0)),
                "drone_only": bool(row.get("drone_only", False)),
                "source": row.get("source", "osm"),
            }
            for key in ("osm_id", "name", "highway", "oneway", "path_osm_ids", "path_node_ids"):
                if key in row:
                    attrs[key] = row[key]
            self.G.add_edge(u, v, **attrs)

        if len(self.G) == 0 or self.G.number_of_edges() == 0:
            raise ValueError(f"Imported road network is empty: {data_path}")

        if not nx.is_weakly_connected(self.G):
            components = sorted(nx.weakly_connected_components(self.G), key=len, reverse=True)
            keep = set(components[0])
            self.G.remove_nodes_from([n for n in list(self.G.nodes) if n not in keep])

        if bool(net_cfg.get("force_bidirectional", True)) and not nx.is_strongly_connected(self.G):
            for u, v, attrs in list(self.G.edges(data=True)):
                if not self.G.has_edge(v, u):
                    reverse_attrs = dict(attrs)
                    reverse_attrs["source"] = "osm_reverse"
                    self.G.add_edge(v, u, **reverse_attrs)

    def _build_jittered_grid(self, net_cfg: Dict, rng: np.random.Generator) -> None:
        rows = int(net_cfg.get("rows", 5))
        cols = int(net_cfg.get("cols", 6))
        spacing = float(net_cfg.get("spacing_m", 400))
        jitter = min(float(net_cfg.get("coordinate_jitter_m", 0.0)), spacing * 0.35)
        for r in range(rows):
            for c in range(cols):
                node = r * cols + c
                x = c * spacing + (float(rng.uniform(-jitter, jitter)) if jitter > 0 else 0.0)
                y = r * spacing + (float(rng.uniform(-jitter, jitter)) if jitter > 0 else 0.0)
                self.G.add_node(node, x=x, y=y)
        # Main grid roads. Include both directions most of the time; remove some directions to make it directed.
        extra_prob = float(net_cfg.get("directed_extra_prob", 0.15))
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if c + 1 < cols:
                    v = r * cols + c + 1
                    self._add_road_pair(u, v, rng, extra_prob)
                if r + 1 < rows:
                    v = (r + 1) * cols + c
                    self._add_road_pair(u, v, rng, extra_prob)
        # Some diagonal/shortcut arcs make the network less grid-like.
        for r in range(rows - 1):
            for c in range(cols - 1):
                if rng.random() < extra_prob:
                    u = r * cols + c
                    v = (r + 1) * cols + c + 1
                    self._add_arc(u, v, rng)
                if rng.random() < extra_prob:
                    u = r * cols + c + 1
                    v = (r + 1) * cols + c
                    self._add_arc(u, v, rng)
        # A few longer local shortcuts emulate arterial roads and make the
        # synthetic city less lattice-like while preserving strong connectivity.
        shortcut_prob = float(net_cfg.get("arterial_shortcut_prob", 0.0))
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if c + 2 < cols and rng.random() < shortcut_prob:
                    self._add_arc(u, r * cols + c + 2, rng)
                if r + 2 < rows and rng.random() < shortcut_prob:
                    self._add_arc(u, (r + 2) * cols + c, rng)
                if r + 2 < rows and c + 1 < cols and rng.random() < shortcut_prob * 0.5:
                    self._add_arc(u, (r + 2) * cols + c + 1, rng)
                if r + 2 < rows and c - 1 >= 0 and rng.random() < shortcut_prob * 0.5:
                    self._add_arc(u, (r + 2) * cols + c - 1, rng)
        # Ensure strongly connected enough by adding reverse edges if needed.
        if not nx.is_strongly_connected(self.G):
            # The grid pair construction usually makes it connected. If not, add missing reverse arcs for existing edges.
            for u, v in list(self.G.edges()):
                if not self.G.has_edge(v, u):
                    self._add_arc(v, u, rng)

    def _build_irregular_city(self, net_cfg: Dict, rng: np.random.Generator) -> None:
        n_nodes = int(net_cfg.get("num_nodes", 60))
        width = float(net_cfg.get("width_m", 3600.0))
        height = float(net_cfg.get("height_m", 2100.0))
        rows = int(net_cfg.get("city_rows", 7))
        cols = int(net_cfg.get("city_cols", math.ceil(n_nodes / max(rows, 1))))
        if rows * cols < n_nodes:
            cols = int(math.ceil(n_nodes / max(rows, 1)))

        xs = self._irregular_axis_positions(cols, width, rng, margin_frac=0.04)
        ys = self._irregular_axis_positions(rows, height, rng, margin_frac=0.06)
        missing = self._select_missing_grid_points(rows, cols, rows * cols - n_nodes, rng)

        row_offsets = rng.normal(0.0, width * 0.018, size=rows)
        col_offsets = rng.normal(0.0, height * 0.018, size=cols)
        node_at: Dict[Tuple[int, int], int] = {}
        node_id = 0
        for r in range(rows):
            for c in range(cols):
                if (r, c) in missing:
                    continue
                bend_x = math.sin((r + 1) * 0.9) * width * 0.012
                bend_y = math.cos((c + 1) * 0.8) * height * 0.012
                x = float(np.clip(xs[c] + row_offsets[r] + bend_x + rng.normal(0.0, width * 0.008), 0.0, width))
                y = float(np.clip(ys[r] + col_offsets[c] + bend_y + rng.normal(0.0, height * 0.008), 0.0, height))
                self.G.add_node(node_id, x=x, y=y, grid_row=r, grid_col=c)
                node_at[(r, c)] = node_id
                node_id += 1

        self._add_city_block_roads(node_at, rows, cols, net_cfg, rng)
        self._ensure_weak_connectivity(rng)
        self._ensure_strong_connectivity(rng)

    def _irregular_axis_positions(
        self,
        count: int,
        span: float,
        rng: np.random.Generator,
        margin_frac: float,
    ) -> List[float]:
        if count <= 1:
            return [span / 2.0]
        margin = span * margin_frac
        base = span - 2.0 * margin
        gaps = rng.uniform(0.75, 1.35, size=count - 1)
        gaps = gaps / gaps.sum() * base
        positions = [margin]
        for gap in gaps:
            positions.append(positions[-1] + float(gap))
        return positions

    def _select_missing_grid_points(
        self,
        rows: int,
        cols: int,
        remove_count: int,
        rng: np.random.Generator,
    ) -> set[Tuple[int, int]]:
        if remove_count <= 0:
            return set()
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0
        candidates = [(r, c) for r in range(rows) for c in range(cols)]
        weights = []
        for r, c in candidates:
            center_distance = math.hypot(r - center_r, c - center_c)
            edge_bonus = 1.4 if r in {0, rows - 1} or c in {0, cols - 1} else 1.0
            weights.append((0.6 + center_distance) * edge_bonus)
        probs = np.asarray(weights, dtype=float)
        probs = probs / probs.sum()
        chosen = rng.choice(len(candidates), size=min(remove_count, len(candidates)), replace=False, p=probs)
        return {candidates[int(i)] for i in chosen}

    def _add_city_block_roads(
        self,
        node_at: Dict[Tuple[int, int], int],
        rows: int,
        cols: int,
        net_cfg: Dict,
        rng: np.random.Generator,
    ) -> None:
        keep_prob = float(net_cfg.get("block_connection_prob", 0.90))
        avenue_keep_prob = float(net_cfg.get("avenue_connection_prob", 0.98))
        diagonal_prob = float(net_cfg.get("diagonal_connection_prob", 0.06))
        arterial_skip_prob = float(net_cfg.get("arterial_skip_connection_prob", 0.18))
        bidirectional_prob = float(net_cfg.get("bidirectional_prob", 0.82))
        main_rows = {rows // 2}
        main_cols = {cols // 2, max(0, cols // 2 - 2), min(cols - 1, cols // 2 + 2)}

        for r in range(rows):
            row_nodes = [(c, node_at[(r, c)]) for c in range(cols) if (r, c) in node_at]
            for (c1, u), (c2, v) in zip(row_nodes, row_nodes[1:]):
                prob = avenue_keep_prob if r in main_rows or c2 - c1 > 1 else keep_prob
                if rng.random() < prob:
                    self._add_city_connection(u, v, rng, bidirectional_prob)

        for c in range(cols):
            col_nodes = [(r, node_at[(r, c)]) for r in range(rows) if (r, c) in node_at]
            for (r1, u), (r2, v) in zip(col_nodes, col_nodes[1:]):
                prob = avenue_keep_prob if c in main_cols or r2 - r1 > 1 else keep_prob
                if rng.random() < prob:
                    self._add_city_connection(u, v, rng, bidirectional_prob)

        for r in range(rows - 1):
            for c in range(cols - 1):
                if rng.random() < diagonal_prob and (r, c) in node_at and (r + 1, c + 1) in node_at:
                    self._add_city_connection(node_at[(r, c)], node_at[(r + 1, c + 1)], rng, bidirectional_prob * 0.8)
                if rng.random() < diagonal_prob and (r + 1, c) in node_at and (r, c + 1) in node_at:
                    self._add_city_connection(node_at[(r + 1, c)], node_at[(r, c + 1)], rng, bidirectional_prob * 0.8)

        for r in main_rows:
            for c in range(cols - 2):
                if rng.random() < arterial_skip_prob and (r, c) in node_at and (r, c + 2) in node_at:
                    self._add_city_connection(node_at[(r, c)], node_at[(r, c + 2)], rng, bidirectional_prob)
        for c in main_cols:
            for r in range(rows - 2):
                if rng.random() < arterial_skip_prob and (r, c) in node_at and (r + 2, c) in node_at:
                    self._add_city_connection(node_at[(r, c)], node_at[(r + 2, c)], rng, bidirectional_prob)

    def _add_sparse_city_roads(self, net_cfg: Dict, rng: np.random.Generator) -> None:
        nodes = list(self.G.nodes)
        min_degree = int(net_cfg.get("min_undirected_degree", 2))
        k_nearest = int(net_cfg.get("k_nearest", 5))
        connection_prob = float(net_cfg.get("near_connection_prob", 0.42))
        long_prob = float(net_cfg.get("long_connection_prob", 0.04))
        bidirectional_prob = float(net_cfg.get("bidirectional_prob", 0.72))
        for u in nodes:
            ranked = sorted((v for v in nodes if v != u), key=lambda v: self._distance(u, v))
            chosen = set(ranked[:min_degree])
            for v in ranked[min_degree:k_nearest]:
                if rng.random() < connection_prob:
                    chosen.add(v)
            for v in ranked[k_nearest:]:
                if rng.random() < long_prob / max(1, len(nodes) - k_nearest):
                    chosen.add(v)
            for v in chosen:
                self._add_city_connection(u, int(v), rng, bidirectional_prob)

    def _add_city_connection(self, u: int, v: int, rng: np.random.Generator, bidirectional_prob: float) -> None:
        if u == v:
            return
        if rng.random() < 0.5:
            self._add_arc(u, v, rng)
            if rng.random() < bidirectional_prob:
                self._add_arc(v, u, rng)
        else:
            self._add_arc(v, u, rng)
            if rng.random() < bidirectional_prob:
                self._add_arc(u, v, rng)

    def _ensure_weak_connectivity(self, rng: np.random.Generator) -> None:
        undirected = self.G.to_undirected()
        while not nx.is_connected(undirected):
            comps = [list(c) for c in nx.connected_components(undirected)]
            base = comps[0]
            other = min(comps[1:], key=len)
            u, v = min(
                ((a, b) for a in base for b in other),
                key=lambda pair: self._distance(pair[0], pair[1]),
            )
            self._add_arc(int(u), int(v), rng)
            self._add_arc(int(v), int(u), rng)
            undirected = self.G.to_undirected()

    def _ensure_strong_connectivity(self, rng: np.random.Generator) -> None:
        if nx.is_strongly_connected(self.G):
            return
        undirected_mst = nx.minimum_spanning_tree(self.G.to_undirected(), weight="length_m")
        for u, v in undirected_mst.edges():
            self._add_arc(int(u), int(v), rng)
            self._add_arc(int(v), int(u), rng)
        if not nx.is_strongly_connected(self.G):
            for u, v in list(self.G.edges()):
                if not self.G.has_edge(v, u):
                    self._add_arc(int(v), int(u), rng)

    def _resolve_patrol_center(self) -> None:
        raw = self._patrol_center_cfg
        if isinstance(raw, str) and raw.lower() in {"center", "middle", "centroid"}:
            xs = [self.G.nodes[n]["x"] for n in self.G.nodes]
            ys = [self.G.nodes[n]["y"] for n in self.G.nodes]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            self.patrol_center = int(min(self.G.nodes, key=lambda n: math.hypot(self.G.nodes[n]["x"] - cx, self.G.nodes[n]["y"] - cy)))
            return
        center = int(raw)
        if center not in self.G:
            raise ValueError(f"Configured patrol_center is not in the network: {center}")
        self.patrol_center = center

    def _distance(self, u: int, v: int) -> float:
        ux, uy = self.node_xy(u)
        vx, vy = self.node_xy(v)
        return float(math.hypot(ux - vx, uy - vy))

    def _add_arc(self, u: int, v: int, rng: np.random.Generator) -> None:
        if u == v or self.G.has_edge(u, v):
            return
        length = self._distance(u, v) * float(rng.uniform(0.95, 1.18))
        speed = float(self.cfg["network"].get("vehicle_speed_mps", 8.3)) * float(rng.uniform(0.65, 1.15))
        self.G.add_edge(
            u,
            v,
            length_m=length,
            vehicle_speed_mps=max(speed, 1.0),
            vehicle_time_s=length / max(speed, 1.0),
            congestion=float(rng.uniform(0.0, 0.7)),
            risk=0.0,
            drone_only=False,
        )

    def _add_road_pair(self, u: int, v: int, rng: np.random.Generator, asym_prob: float) -> None:
        self._add_arc(u, v, rng)
        self._add_arc(v, u, rng)
        # Randomly make one direction less attractive to emulate traffic restrictions/congestion.
        if rng.random() < asym_prob:
            a, b = (u, v) if rng.random() < 0.5 else (v, u)
            if self.G.has_edge(a, b):
                self.G[a][b]["vehicle_time_s"] *= float(rng.uniform(1.4, 2.2))

    def _assign_risk(self, rng: np.random.Generator) -> None:
        high_frac = float(self.cfg["network"].get("high_risk_fraction", 0.25))
        edges = list(self.G.edges())
        risk_values = rng.beta(1.6, 4.0, size=len(edges))
        high_count = max(1, int(len(edges) * high_frac))
        high_idx = np.argsort(risk_values)[-high_count:]
        for i, (u, v) in enumerate(edges):
            risk = float(risk_values[i])
            if i in high_idx:
                risk = min(1.0, risk + float(rng.uniform(0.35, 0.65)))
            self.G[u][v]["risk"] = risk
            self.G[u][v]["drone_only"] = bool(risk > 0.82 and rng.random() < 0.35)

    def _compute_shortest_paths(self) -> None:
        self.vehicle_time = dict(nx.all_pairs_dijkstra_path_length(self.G, weight="vehicle_time_s"))
        self.vehicle_paths = dict(nx.all_pairs_dijkstra_path(self.G, weight="vehicle_time_s"))
        speed = float(self.cfg["network"].get("drone_speed_mps", 14.0))
        self.drone_time_matrix = {i: {} for i in self.nodes}
        for i in self.nodes:
            for j in self.nodes:
                self.drone_time_matrix[i][j] = 0.0 if i == j else self.euclidean_distance(i, j) / speed

    def node_xy(self, node: int) -> Tuple[float, float]:
        data = self.G.nodes[int(node)]
        return float(data["x"]), float(data["y"])

    def euclidean_distance(self, i: int, j: int) -> float:
        return self._distance(int(i), int(j))

    def get_vehicle_time(self, i: int, j: int) -> float:
        return float(self.vehicle_time.get(int(i), {}).get(int(j), math.inf))

    def get_vehicle_path(self, i: int, j: int) -> List[int]:
        return list(self.vehicle_paths.get(int(i), {}).get(int(j), []))

    def get_drone_time(self, i: int, j: int) -> float:
        return float(self.drone_time_matrix[int(i)][int(j)])

    def arc_length(self, arc: Tuple[int, int]) -> float:
        u, v = arc
        if self.G.has_edge(u, v):
            return float(self.G[u][v]["length_m"])
        return self.euclidean_distance(u, v)

    def arc_risk(self, arc: Tuple[int, int]) -> float:
        u, v = arc
        if self.G.has_edge(u, v):
            return float(self.G[u][v].get("risk", 0.0))
        return 0.0

    def high_risk_arcs(self, n: int | None = None) -> List[Tuple[int, int]]:
        edges = sorted(self.G.edges(), key=lambda e: self.G[e[0]][e[1]].get("risk", 0.0), reverse=True)
        return edges[:n] if n else edges

    def nearest_nodes_to_arc(self, arc: Tuple[int, int], k: int = 4) -> List[int]:
        u, v = arc
        ux, uy = self.node_xy(u)
        vx, vy = self.node_xy(v)
        cx, cy = (ux + vx) / 2.0, (uy + vy) / 2.0
        ranked = sorted(self.nodes, key=lambda node: math.hypot(self.node_xy(node)[0] - cx, self.node_xy(node)[1] - cy))
        return ranked[:k]

    def save_matrices(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "shortest_path_vehicle.pkl", "wb") as f:
            pickle.dump({"time": self.vehicle_time, "paths": self.vehicle_paths}, f)
        with open(out / "drone_travel_time.pkl", "wb") as f:
            pickle.dump(self.drone_time_matrix, f)

    def validate(self, sample_pairs: int = 100) -> Dict[str, float]:
        rng = np.random.default_rng(int(self.cfg["project"].get("seed", 42)))
        nodes = self.nodes
        unreachable = 0
        for _ in range(sample_pairs):
            i, j = rng.choice(nodes, 2, replace=True)
            if not math.isfinite(self.get_vehicle_time(int(i), int(j))):
                unreachable += 1
        diag_ok = all(self.get_vehicle_time(i, i) == 0.0 and self.get_drone_time(i, i) == 0.0 for i in nodes)
        return {"num_nodes": len(nodes), "num_arcs": len(self.arcs), "unreachable_samples": unreachable, "diag_ok": float(diag_ok)}


def build_network(cfg: Dict) -> RoadNetwork:
    return RoadNetwork(cfg)
