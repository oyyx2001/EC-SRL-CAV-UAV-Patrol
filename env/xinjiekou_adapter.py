"""Small, auditable adapter from packaged Xinjiekou assets to core IDs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import math
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from .road_network import RoadNetwork
from .task_generator import Task


DIRECT_EDGE_MAPPING_VERSION = "xinjiekou_direct_compressed_edge_v1"


def generate_edge_task_registry(
    network: RoadNetwork,
    edge_file: str,
    *,
    mapping_version: str = DIRECT_EDGE_MAPPING_VERSION,
) -> Tuple[pd.DataFrame, str]:
    """Build the deterministic 601-edge logical task registry.

    The packaged compressed-edge ID is authoritative.  Stable sorting is used
    only to serialize the existing IDs deterministically; it never renumbers
    them from a NetworkX iteration order.
    """

    edges = gpd.read_file(network._resolve_import_path(edge_file))
    rows: List[Dict[str, Any]] = []
    for _, edge in edges.sort_values("compressed_edge_id", kind="mergesort").iterrows():
        edge_id = str(edge["compressed_edge_id"])
        u_external, v_external = str(edge["source"]), str(edge["target"])
        u_internal = network.to_internal_node(u_external)
        v_internal = network.to_internal_node(v_external)
        length = float(edge.get("length_m", edge.get("network_distance_m", math.nan)))
        geometry = edge.geometry
        eligible = bool(u_internal != v_internal and math.isfinite(length) and length > 0.0)
        rows.append(
            {
                "edge_id": edge_id,
                "u_external_id": u_external,
                "v_external_id": v_external,
                "u_internal_id": int(u_internal),
                "v_internal_id": int(v_internal),
                "edge_key": 0,
                "length_m": length,
                "road_name": str(edge.get("road_name", "")),
                "geometry_available": bool(geometry is not None and not geometry.is_empty),
                "geometry_wkt": "" if geometry is None else geometry.wkt,
                "eligible_routine": eligible,
                "eligible_incident": eligible,
                "eligible_congestion": eligible,
                "eligible_road_obstruction_inspection": eligible,
                "eligible_critical_period_patrol": eligible,
                "eligible_incident_confirmation": eligible,
                "eligible_congestion_anomaly_scan": eligible,
                "mapping_version": str(mapping_version),
            }
        )
    registry = pd.DataFrame(rows).sort_values("edge_id", kind="mergesort").reset_index(drop=True)
    canonical = registry.to_csv(index=False, lineterminator="\n", float_format="%.9f")
    return registry, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_edge_task_registry(network: RoadNetwork, registry: pd.DataFrame) -> Dict[str, Any]:
    normalized_pairs = registry.apply(
        lambda row: tuple(sorted((str(row["u_external_id"]), str(row["v_external_id"])))), axis=1
    )
    valid_internal_arcs = all(
        network.G.has_edge(int(row.u_internal_id), int(row.v_internal_id))
        and network.G.has_edge(int(row.v_internal_id), int(row.u_internal_id))
        for row in registry.itertuples()
    )
    duplicate_endpoint_pairs = int(len(registry) - normalized_pairs.nunique())
    return {
        "source_undirected_edges": int(getattr(network, "source_edge_count", -1)),
        "registry_edges": int(len(registry)),
        "unique_edge_ids": int(registry["edge_id"].nunique()),
        "unique_undirected_pairs": int(normalized_pairs.nunique()),
        # A MultiGraph may legitimately contain two different physical roads
        # with the same endpoint pair.  ``edge_id`` (not the endpoint pair) is
        # therefore the logical task identity.  Keep this count for audit
        # rather than treating a parallel source edge as a duplicate task.
        "duplicate_endpoint_pair_count": duplicate_endpoint_pairs,
        "valid_internal_arc_pairs": bool(valid_internal_arcs),
        "positive_finite_lengths": bool(
            np.isfinite(pd.to_numeric(registry["length_m"], errors="coerce")).all()
            and (pd.to_numeric(registry["length_m"], errors="coerce") > 0.0).all()
        ),
    }


def load_edge_task_registry(network: RoadNetwork, registry_path: str) -> pd.DataFrame:
    registry = pd.read_csv(network._resolve_import_path(registry_path))
    registry["edge_id"] = registry["edge_id"].astype(str)
    return registry.sort_values("edge_id", kind="mergesort").reset_index(drop=True)


def build_direct_edge_task_stream(
    network: RoadNetwork,
    registry_path: str,
    task_cfg: Dict[str, Any],
    *,
    release_horizon_s: float,
    seed: int,
) -> List[Task]:
    """Create reproducible physical-road inspection tasks from the registry.

    ``compressed_edge_*`` is the stable ID of one stored physical road segment
    in the source ``MultiGraph``.  It is consequently also exposed as the
    ``inspection_segment_id`` used by the core simulator.  The directed CAV
    graph is only an execution representation and never duplicates a task.
    """

    registry = load_edge_task_registry(network, registry_path)
    registry_checks = validate_edge_task_registry(network, registry)
    if (
        registry_checks["registry_edges"] != registry_checks["source_undirected_edges"]
        or registry_checks["unique_edge_ids"] != registry_checks["registry_edges"]
        or not registry_checks["valid_internal_arc_pairs"]
        or not registry_checks["positive_finite_lengths"]
    ):
        raise ValueError(f"Invalid direct Xinjiekou edge-task registry: {registry_checks}")
    edge_selection = dict(task_cfg.get("edge_selection", {}) or {})
    eligible = registry[
        registry["eligible_routine"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    if bool(edge_selection.get("exclude_self_loops", True)):
        eligible = eligible[eligible["u_internal_id"] != eligible["v_internal_id"]]
    if bool(edge_selection.get("exclude_zero_length", True)):
        eligible = eligible[pd.to_numeric(eligible["length_m"], errors="coerce") > 0.0]
    if bool(edge_selection.get("eligible_only", True)) and eligible.empty:
        raise ValueError("No eligible direct Xinjiekou edge tasks remain after configured filters")
    generation_mode = str(task_cfg.get("task_generation_mode", "legacy_fixed"))
    selection = dict(task_cfg.get("periodic_patrol", task_cfg.get("routine_patrol", {})) or {})
    mode = str(selection.get("selection_mode", "seeded_sample"))
    dynamic_periodic = str(selection.get("release_mode", selection.get("task_generation_mode", ""))).lower() in {"periodic_dynamic", "dynamic_periodic"} or mode in {"least_recently_inspected", "periodic_dynamic"}
    if dynamic_periodic:
        selected_ids = []
    elif mode in {"fixed_edge_set", "fixed_segments"}:
        selected_ids = [str(value) for value in selection.get("inspection_segment_ids", selection.get("edge_ids", []))]
    elif mode == "seeded_sample":
        count = min(int(selection.get("number_of_edges", 20)), len(eligible))
        rng_selection = np.random.default_rng(int(selection.get("selection_seed", 2026)))
        selected_ids = sorted(rng_selection.choice(eligible["edge_id"].to_numpy(), size=count, replace=False).tolist())
    else:
        raise ValueError(f"Unknown Xinjiekou routine edge selection mode: {mode}")
    by_id = {str(row.edge_id): row for row in registry.itertuples()}
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"Formal edge task IDs missing from registry: {missing}")

    tasks: List[Task] = []
    next_id = 0

    service_model = dict(task_cfg.get("service_time_model", {}) or {})
    spatial_cfg = dict(task_cfg.get("spatial_sampling", {}) or {})

    def service_time_for(row: Any, typ: str) -> Tuple[float, bool]:
        """Observation time only; approach and recovery remain sortie segments."""
        if str(service_model.get("mode", "legacy_fixed")) != "segment_length_based":
            legacy = dict(task_cfg.get("dynamic_tasks", {}).get(typ, {}) or {})
            return float(legacy.get("service_duration_s", 30.0)), False
        spec = dict(service_model.get(typ, {}) or {})
        if not spec and typ == "routine":
            spec = dict(service_model.get("periodic_patrol", {}) or {})
        minimum = float(spec.get("min_service_time_s", 20.0))
        maximum = float(spec.get("max_service_time_s", max(20.0, minimum)))
        if maximum < minimum:
            raise ValueError(f"Invalid service-time bounds for {typ!r}: {minimum} > {maximum}")
        speed = float(spec.get("observation_speed_mps", 0.0))
        extra = float(spec.get("additional_observation_time_s", 0.0))
        length = float(getattr(row, "length_m", math.nan))
        fallback = not (math.isfinite(length) and length > 0.0 and math.isfinite(speed) and speed > 0.0)
        if fallback:
            # The midpoint is the deterministic type-level median proxy when
            # neither registry length nor geometry can provide a valid length.
            value = float(spec.get("fallback_service_time_s", (minimum + maximum) / 2.0))
        else:
            value = length / speed + extra
        return float(np.clip(value, minimum, maximum)), fallback

    def weighted_row(pool: pd.DataFrame) -> Any:
        """Use only documented geometry/graph proxies; no traffic flow is invented."""
        if pool.empty:
            raise ValueError("No eligible direct Xinjiekou edge tasks remain")
        weights = np.ones(len(pool), dtype=float)
        if bool(spatial_cfg.get("use_road_importance_weight", False)):
            lengths = pd.to_numeric(pool["length_m"], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(float)
            endpoint_degree = np.array([
                max(1.0, float(network.G.degree[int(row.u_internal_id)] + network.G.degree[int(row.v_internal_id)]))
                for row in pool.itertuples()
            ])
            weights *= lengths * np.sqrt(endpoint_degree)
        if bool(spatial_cfg.get("use_hotspot_weight", False)):
            # The packaged data has no measured traffic/hotspot field.  The
            # endpoint-degree proxy makes central junction-adjacent segments
            # more likely without claiming observed demand data exists.
            weights *= np.array([
                1.0 + 0.05 * (network.G.degree[int(row.u_internal_id)] + network.G.degree[int(row.v_internal_id)])
                for row in pool.itertuples()
            ])
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()
        return pool.iloc[int(rng.choice(len(pool), p=weights))]

    def append_instance(
        row: Any,
        typ: str,
        release: float,
        deadline_s: float,
        priority: float,
        service: Optional[float],
        label: str,
        *,
        cycle_index: Optional[int] = None,
        affected_direction: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> None:
        nonlocal next_id
        instance_id = f"{typ}__{row.edge_id}__{label}"
        computed_service, fallback = service_time_for(row, typ)
        actual_service = computed_service if service is None else float(service)
        metadata = {
            "task_instance_id": instance_id,
            "task_id": instance_id,
            "task_type": typ,
            "edge_task": True,
            "edge_id": str(row.edge_id),
            "inspection_segment_id": str(row.edge_id),
            "original_compressed_edge_id": str(row.edge_id),
            "edge_u": int(row.u_internal_id),
            "edge_v": int(row.v_internal_id),
            "edge_u_external_id": str(row.u_external_id),
            "edge_v_external_id": str(row.v_external_id),
            "edge_key": int(row.edge_key),
            "edge_length_m": float(row.length_m),
            "edge_geometry_wkt": str(row.geometry_wkt),
            "mapping_source": "direct_compressed_edge",
            "mapping_version": str(row.mapping_version),
            "task_generation_mode": generation_mode,
            "affected_direction": affected_direction,
            "event_id": event_id,
            "service_time_fallback": bool(fallback),
        }
        if cycle_index is not None:
            metadata.update(
                periodic_task=True,
                cycle_index=int(cycle_index),
                last_inspection_time=None,
                age_since_last_inspection_s=None,
                periodic_completion_count=0,
                periodic_expired_count=0,
                current_cycle_status="unreleased",
            )
        tasks.append(
            Task(
                id=next_id,
                type=typ,
                target_arcs=[(int(row.u_internal_id), int(row.v_internal_id))],
                release_time=float(release),
                deadline=float(release + deadline_s),
                priority=float(priority),
                service_time=actual_service,
                metadata=metadata,
            )
        )
        next_id += 1

    interval = float(selection.get("interval_s", 600.0))
    release = float(selection.get("initial_release_time_s", 0.0))
    cycle = 0
    routine_type = str(selection.get("task_type", "routine"))
    deadline_window = float(selection.get("deadline_window_s", selection.get("deadline_s", selection.get("service_window_s", interval))))
    routine_service = None if str(service_model.get("mode", "legacy_fixed")) == "segment_length_based" else float(selection.get("service_duration_s", 30.0))
    if dynamic_periodic:
        base_seed = int(selection.get("base_seed", selection.get("selection_seed", seed)))
        routine_rng = np.random.default_rng(np.random.SeedSequence([base_seed, int(seed), 0x5254]))
        eligible = eligible.drop_duplicates(subset=["edge_id"], keep="first").reset_index(drop=True)
        last_release: Dict[str, float] = {}
        active_until: Dict[str, float] = {}
        while release < float(release_horizon_s):
            min_revisit = float(selection.get("min_revisit_interval_s", 0.0))
            candidates = eligible[eligible["edge_id"].map(lambda edge: active_until.get(str(edge), -math.inf) <= release and release - last_release.get(str(edge), -math.inf) >= min_revisit)]
            if not candidates.empty:
                ages = candidates["edge_id"].map(lambda edge: last_release.get(str(edge), -math.inf)).to_numpy(float)
                oldest = np.nanmin(ages)
                oldest_rows = candidates.iloc[np.flatnonzero(ages == oldest)]
                if bool(selection.get("random_tie_break", True)) and len(oldest_rows) > 1:
                    row = oldest_rows.iloc[int(routine_rng.integers(0, len(oldest_rows)))]
                else:
                    row = oldest_rows.sort_values("edge_id", kind="mergesort").iloc[0]
                edge_id = str(row.edge_id)
                append_instance(row, routine_type, release, deadline_window, float(selection.get("priority", 2.0)), routine_service, f"cycle_{cycle:04d}", cycle_index=cycle)
                last_release[edge_id] = release
                active_until[edge_id] = release + deadline_window
            cycle += 1
            release += interval
    else:
        while release < float(release_horizon_s):
            for edge_id in selected_ids:
                append_instance(by_id[edge_id], routine_type, release, deadline_window, float(selection.get("priority", 2.0)), routine_service, f"cycle_{cycle:04d}", cycle_index=cycle)
            release += interval
            cycle += 1

    rng = np.random.default_rng(int(seed))
    dynamic_cfg = dict(task_cfg.get("dynamic_tasks", {}) or {})
    defaults = {
        "incident": (420.0, 10.0, 45.0),
        "congestion": (720.0, 6.0, 60.0),
        "incident_confirmation": (300.0, 10.0, 60.0),
        "congestion_anomaly_scan": (600.0, 6.0, 75.0),
        "road_obstruction_inspection": (300.0, 8.0, 35.0),
        "critical_period_patrol": (360.0, 4.0, 40.0),
    }
    # Preserve the legacy event ordering, then add any explicitly configured
    # Xinjiekou task types in stable lexical order.  A task type is never
    # activated merely because the registry supports it: it must be present
    # in the experiment configuration.
    configured_types = [typ for typ in ("incident", "congestion") if typ in dynamic_cfg]
    configured_types.extend(sorted(set(dynamic_cfg) - set(configured_types)))
    for typ in configured_types:
        spec = dict(dynamic_cfg.get(typ, {}) or {})
        eligible_column = f"eligible_{typ}"
        if eligible_column not in registry.columns:
            raise ValueError(f"No registry eligibility column for Xinjiekou task type {typ!r}")
        pool = registry[registry[eligible_column].astype(str).str.lower().isin({"true", "1"})]
        if pool.empty:
            raise ValueError(f"No eligible direct Xinjiekou edge tasks for task type {typ!r}")
        default_deadline, default_priority, default_service = defaults.get(typ, (420.0, 5.0, 30.0))
        release_mode = str(spec.get("arrival_process", spec.get("release_mode", "poisson"))).lower()
        if release_mode == "poisson":
            rate = float(spec.get("rate_per_hour", 0.0))
            count = int(rng.poisson(max(0.0, rate * float(release_horizon_s) / 3600.0)))
            releases = sorted(rng.uniform(0.0, float(release_horizon_s) * 0.92, size=count).tolist())
        elif release_mode == "piecewise_poisson":
            releases = []
            for block in spec.get("time_blocks", []):
                start = max(0.0, float(block["start_s"]))
                end = min(float(release_horizon_s), float(block["end_s"]))
                if end <= start:
                    continue
                count = int(rng.poisson(max(0.0, float(block.get("rate_per_hour", 0.0)) * (end - start) / 3600.0)))
                releases.extend(rng.uniform(start, end, size=count).tolist())
            releases.sort()
        elif release_mode in {"fixed_times", "scheduled"}:
            releases = sorted(
                float(value) for value in spec.get("release_times_s", [])
                if 0.0 <= float(value) < float(release_horizon_s)
            )
        else:
            raise ValueError(f"Unknown release_mode {release_mode!r} for Xinjiekou task type {typ!r}")
        # Optional fixed additions let a light-pilot workload be increased
        # without changing the stochastic arrival process or routine-patrol
        # one-task-per-slot rule.  These use the same eligible pool and
        # weighting as ordinary generated events.
        additional_times = [
            float(value) for value in spec.get("additional_release_times_s", [])
            if 0.0 <= float(value) < float(release_horizon_s)
        ]
        releases.extend(additional_times)
        releases.sort()
        scheduled = list(spec.get("scheduled_tasks", []) or [])
        if scheduled:
            releases = []
            for index, item in enumerate(scheduled):
                release = float(item["release_time_s"])
                if 0.0 <= release < float(release_horizon_s):
                    edge_id = str(item["inspection_segment_id"])
                    if edge_id not in by_id:
                        raise ValueError(f"Scheduled task references unknown inspection segment {edge_id!r}")
                    releases.append((release, by_id[edge_id], item))
        else:
            releases = [(release, weighted_row(pool), {}) for release in releases]
        for event_index, (event_release, row, scheduled_item) in enumerate(releases):
            direction = scheduled_item.get("affected_direction")
            if direction is None and bool(spec.get("sample_affected_direction", typ != "critical_period_patrol")):
                direction = "u_to_v" if int(rng.integers(0, 2)) == 0 else "v_to_u"
            append_instance(
                row, typ, event_release, float(spec.get("deadline_window_s", spec.get("deadline_s", default_deadline))),
                float(spec.get("priority", default_priority)),
                None if str(service_model.get("mode", "legacy_fixed")) == "segment_length_based" else float(spec.get("service_duration_s", default_service)),
                f"event_{event_index:04d}", affected_direction=direction,
                event_id=str(scheduled_item.get("event_id", f"{typ}_{event_index:04d}")),
            )
    return sorted(tasks, key=lambda task: (task.release_time, task.id))


def direct_edge_task_manifest(tasks: List[Task], scenario_seed: int) -> pd.DataFrame:
    """Return a portable, lossless scenario list for fair algorithm comparison."""
    records: List[Dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: (item.release_time, item.id)):
        metadata = dict(task.metadata)
        records.append({
            "scenario_seed": int(scenario_seed),
            "task_id": int(task.id),
            "task_instance_id": str(metadata.get("task_instance_id", task.id)),
            "task_type": str(task.type),
            "inspection_segment_id": str(metadata.get("inspection_segment_id", metadata.get("edge_id", ""))),
            "original_compressed_edge_id": str(metadata.get("original_compressed_edge_id", metadata.get("edge_id", ""))),
            "release_time": float(task.release_time),
            "deadline": float(task.deadline),
            "priority": float(task.priority),
            "affected_direction": metadata.get("affected_direction"),
            "service_time": float(task.service_time),
            "event_id": metadata.get("event_id"),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(records)


def write_direct_edge_task_manifest(tasks: List[Task], scenario_seed: int, path: Union[str, Path]) -> pd.DataFrame:
    manifest = direct_edge_task_manifest(tasks, scenario_seed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return manifest


def load_direct_edge_task_manifest(network: RoadNetwork, registry_path: str, path: Union[str, Path]) -> List[Task]:
    """Load a previously frozen direct-edge scenario without resampling events."""
    # ``round_trip`` preserves event timestamps exactly across the CSV freeze /
    # reload boundary, which is required when comparing algorithms on a fixed
    # stochastic scenario.
    manifest = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    required = {
        "task_id", "task_type", "inspection_segment_id", "release_time", "deadline",
        "priority", "service_time", "metadata_json",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Task manifest is missing required columns: {missing}")
    registry = load_edge_task_registry(network, registry_path)
    by_id = {str(row.edge_id): row for row in registry.itertuples()}
    tasks: List[Task] = []
    for row in manifest.itertuples(index=False):
        segment = str(getattr(row, "inspection_segment_id"))
        source = by_id.get(segment)
        if source is None:
            raise ValueError(f"Task manifest references unknown inspection segment {segment!r}")
        metadata = json.loads(str(getattr(row, "metadata_json")))
        if str(metadata.get("inspection_segment_id", segment)) != segment:
            raise ValueError(f"Task manifest inspection segment mismatch for task {getattr(row, 'task_id')}")
        tasks.append(Task(
            id=int(getattr(row, "task_id")), type=str(getattr(row, "task_type")),
            target_arcs=[(int(source.u_internal_id), int(source.v_internal_id))],
            release_time=float(getattr(row, "release_time")), deadline=float(getattr(row, "deadline")),
            priority=float(getattr(row, "priority")), service_time=float(getattr(row, "service_time")),
            metadata=metadata,
        ))
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Task manifest contains duplicate task_id values")
    return sorted(tasks, key=lambda task: (task.release_time, task.id))


def map_launch_recovery_nodes(network: RoadNetwork, geojson_file: str, max_distance_m: float) -> List[Dict[str, Any]]:
    points = gpd.read_file(network._resolve_import_path(geojson_file))
    rows: List[Dict[str, Any]] = []
    for _, item in points.iterrows():
        external = str(item.get("node_id", item.get("representative_node_id", "")))
        x, y = float(item.geometry.x), float(item.geometry.y)
        if external in network.external_to_internal:
            internal = network.to_internal_node(external)
            matched = external
        else:
            internal = min(network.nodes, key=lambda n: math.hypot(network.node_xy(n)[0] - x, network.node_xy(n)[1] - y))
            matched = network.to_external_node(internal)
        mx, my = network.node_xy(internal)
        distance = math.hypot(mx - x, my - y)
        if distance > float(max_distance_m):
            raise ValueError(f"Launch/recovery point {external!r} is {distance:.3f}m from nearest graph node")
        rows.append({"source_node_id": external, "source_x": x, "source_y": y,
                     "external_node_id": matched, "internal_node_id": internal,
                     "match_distance_m": distance, "launch_allowed": bool(item.get("launch_allowed", True)),
                     "recovery_allowed": bool(item.get("recovery_allowed", True))})
    return rows


def convert_task_model(network: RoadNetwork, task_cfg: Dict[str, Any], edge_file: str, overrides: Dict[str, str] | None = None) -> Tuple[List[Task], List[Dict[str, Any]]]:
    edges = gpd.read_file(network._resolve_import_path(edge_file))
    by_id = {str(row["compressed_edge_id"]): row for _, row in edges.iterrows()}
    tasks: List[Task] = []
    unmapped: List[Dict[str, Any]] = []
    for index, raw_id in enumerate(task_cfg.get("periodic_patrol", {}).get("inspection_segment_ids", [])):
        mapped_id = (overrides or {}).get(str(raw_id), str(raw_id))
        row = by_id.get(mapped_id)
        if row is None:
            unmapped.append({"task_source_id": raw_id, "reason": "edge_not_found"}); continue
        try:
            u, v = network.to_internal_node(str(row["source"])), network.to_internal_node(str(row["target"]))
        except KeyError as exc:
            unmapped.append({"task_source_id": raw_id, "reason": str(exc)}); continue
        model = task_cfg["periodic_patrol"]
        tasks.append(Task(id=index, type="routine", target_arcs=[(u, v)],
                          release_time=float(model.get("initial_release_time_s", 0.0)),
                          deadline=float(model.get("service_window_s", 600.0)),
                          priority=float(model.get("priority", 2.0)),
                          service_time=float(model.get("service_duration_s", 30.0)),
                          metadata={"source_task_id": str(raw_id), "source_edge_id": mapped_id,
                                    "external_start": str(row["source"]), "external_end": str(row["target"]) }))
    return tasks, unmapped


def build_periodic_task_stream(
    network: RoadNetwork,
    task_model_file: str,
    edge_file: str,
    *,
    release_horizon_s: float,
    overrides: Dict[str, str] | None = None,
    mapping_source: str = "formal",
) -> Tuple[List[Task], List[Dict[str, Any]]]:
    """Expand the packaged periodic model into core-internal task instances.

    The caller deliberately supplies the mapping source.  A smoke override is
    therefore visible in every generated task rather than silently becoming a
    formal business-road mapping.
    """

    path = network._resolve_import_path(task_model_file)
    with open(path, "r", encoding="utf-8") as handle:
        model = yaml.safe_load(handle) or {}
    templates, unmapped = convert_task_model(network, model, edge_file, overrides)
    periodic = dict(model.get("periodic_patrol", {}) or {})
    interval = float(periodic.get("interval_s", 0.0))
    initial = float(periodic.get("initial_release_time_s", 0.0))
    service_window = float(periodic.get("service_window_s", 600.0))
    if interval <= 0.0:
        release_times = [initial]
    else:
        release_times = []
        release = initial
        while release < float(release_horizon_s):
            release_times.append(release)
            release += interval

    tasks: List[Task] = []
    task_id = 0
    for cycle, release in enumerate(release_times):
        for template in templates:
            metadata = dict(template.metadata)
            metadata.update(
                mapping_source=str(mapping_source),
                periodic_cycle=int(cycle),
                task_model_file=str(task_model_file),
            )
            tasks.append(
                Task(
                    id=task_id,
                    type=template.type,
                    target_arcs=list(template.target_arcs),
                    release_time=float(release),
                    deadline=float(release + service_window),
                    priority=float(template.priority),
                    service_time=float(template.service_time),
                    executor=str(template.executor),
                    risk=float(template.risk),
                    metadata=metadata,
                )
            )
            task_id += 1
    return tasks, unmapped
