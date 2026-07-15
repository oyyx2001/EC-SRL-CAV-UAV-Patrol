# Stage 3: Active CAV Routing and Rendezvous Coordination

## 1. Executive status

Stage 3A–3C has been implemented as an independent, deterministic ground controller on top of the existing Dynamic Recovery Insertion and Event-Driven SMDP implementation.

Implementation and compatibility acceptance passed:

- Stage 1 regression: 10/10 tests passed.
- Stage 2 regression: 19/19 tests passed.
- Stage 3 requirements and regression: 25/25 passed; 54/54 tests pass across Stages 1–3.
- Six deterministic micro scenarios: 6/6 passed.
- Formal C0–C3 paired comparison: all 80 episodes across 20 seeds completed.
- Spatial, road-detour, and 5-UAV supplements: all 280 episodes completed.
- C0 passive mode reproduces the current Stage 2 event-driven baseline exactly for the paired regression seed.

The formal result passes the predefined Stage-4 gate. Relative to C0, C3 reduces mean UAV recovery wait by 51.6%, increases completion by 4.82 percentage points, and reduces CAV distance by 16.9%, with no invalid edge, negative battery, or route reversal. UAV return distance increases by 64.8%, so Stage 4 must explicitly represent this trade-off.

## 2. Research boundary

Stage 3 changes only CAV ground-route control. It does not:

- add CAV actions to the 25-dimensional UAV/Rainbow action space;
- change UAV sortie candidate ordering or the maximum tasks per sortie;
- change the Rainbow architecture or add a neural network;
- retrain Rainbow;
- change the reward function or continuous-time SMDP discount;
- change the Stage 1 insertion cost;
- change task arrival intensity; or
- construct a joint CAV–UAV action.

The main experiment fixes the UAV policy to the deterministic existing candidate-score heuristic. C0–C3 differ only in `cav_control.mode` and `cav_control.action_set`.

## 3. Pre-change code map

| Function | File | Class/function | Existing logic | Stage 3 integration |
|---|---|---|---|---|
| CAV state | `env/vehicle.py` | `Vehicle` | Node, directed path, current edge time, target, route logs, schedule version | Added explicit event-bounded wait, arbitrary directed route commit, and commitment history |
| CAV position/current edge/route | `env/vehicle.py` | `position`, `path_nodes`, `path_index`, `edge_remaining_time` | Strict edge-by-edge motion | Preserved; action changes are rejected while `at_intersection` is false |
| Route generation | `env/vehicle.py` | `set_target()` | Directed shortest path to one recovery target | Added validated `set_route()` for waypoint + committed recovery routes |
| Shortest path/travel time | `env/road_network.py` | `get_vehicle_path()`, `get_vehicle_time()` | Precomputed directed road shortest paths | Reused by every ground candidate; no second distance/time model |
| Road distance | `env/road_network.py` | `arc_length()` | Physical directed-edge length | Reused for CAV route, detour, task coverage, and metric conservation |
| CAV arrival event | `env/events.py`, `rl/env_wrapper.py` | `CAV_NODE_ARRIVAL`, `_schedule_dynamic_events()` | Versioned event-heap marker for the next physical edge arrival | Added `CAV_WAIT_COMPLETE`; route changes invalidate old arrivals through `schedule_version` |
| Recovery queue | `env/simulator.py` | `recovery_queue` | FIFO or dynamic insertion; CAV follows queue head | Active variants defer automatic head commit until the ground decision |
| Dynamic insertion | `env/recovery_scheduler.py` | `RecoveryInsertionScheduler` | Side-effect-free insertion and battery/queue prediction | Added `simulate_ground_plan()` using the same recovery timing, hover, and battery rules |
| Locked/flexible status | `env/recovery_scheduler.py` | `locked_prefix_length()`, `_runtime_status()` | Protects committed, returning, and recovering UAVs | Ground updates are allowed only outside this exact locked prefix |
| UAV ETA/battery | `env/recovery_scheduler.py` | `make_entry()`, `simulate()` | Queue entry ETA, battery at ETA, reserve, hover power | Ground controller recomputes final-return duration/energy before a flexible update |
| UAV recovery node | `planner/sortie_generator.py` | `Sortie.recovery_node`, final fly segment | Selected when the sortie is generated | Flexible-only update changes recovery node and final fly segment without changing tasks |
| Stage 2 event heap | `env/events.py` | `EventQueue` | Stable priority heap, same-time batch, stale filtering | Reused unchanged; new physical events carry existing entity versions |
| Schedule versions | `env/vehicle.py`, `env/drone.py` | `schedule_version` | Invalidates stale CAV/UAV events | Incremented after committed route/wait or flexible return-plan change |
| Task distribution | `env/task_generator.py` | `_poisson_tasks()` | Fixed-rate Poisson releases with risk-weighted spatial selection | Optional low/medium/high spatial concentration changes support only, not count/rate |
| UAV heuristic | `planner/greedy_baseline.py` | candidate-score/priority policies | Fixed sortie dispatch | Main Stage 3 runner uses the same deterministic candidate-score action for all variants |
| Rainbow entry/checkpoint | `rl/train_masked_dqn.py`, `experiments/evaluate_masked_rl_models.py` | existing DQN interfaces | 25 UAV actions and deterministic evaluation support | Unchanged; frozen-Rainbow Stage 3 supplement was not run |
| Metrics | `env/simulator.py`, `utils/metrics.py` | `metrics()`, `EpisodeMetrics` | Task, energy, recovery, Stage 1 metrics | Added return distance/time, ground actions, detour, stability, slack, and timing |
| Stage 1 outputs | `results/stage1_dynamic_recovery/` and Stage 1 runner paths | existing experiment runner | Recovery insertion raw/summary outputs | Preserved |
| Stage 2 outputs | `results/stage2_event_driven_smdp/` | `run_stage2_event_smdp.py` | Fixed/event raw results, transitions, timelines | Preserved |

The repository was dirty before Stage 3. Existing Stage 2 modifications and untracked files were preserved and used as the C0 baseline; no reset, checkout, or overwrite was performed.

## 4. Actions

### 4.1 `keep_route`

Always feasible. It preserves an existing directed route. If the active variant deferred passive queue-head assignment, committing `keep_route` calls the original `_set_vehicle_next_recovery_target()` and therefore reproduces the old behavior.

### 4.2 `wait`

Available only at a road node. Its end is:

```text
min(next relevant UAV ETA, next valid system event, now + max_wait_s)
```

For a flexible return, waiting may update that return rendezvous to the current CAV node after recomputing UAV return time and battery. Locked or final-return UAVs cannot be updated. `Vehicle.begin_wait()` increments the CAV version and creates a `CAV_WAIT_COMPLETE` boundary.

### 4.3 `move_to(v)`

Commits a shortest directed path from the current node to one bounded candidate node. A flexible rendezvous may be updated to `v`. The controller predicts the subsequent queue schedule but commits only the direct target, so the next node event can reconsider the route.

### 4.4 `reroute(v|H)`

Commits a finite route prefix:

```text
current node -> waypoint v -> next committed recovery node
```

The action does not reorder the locked prefix. An intermediate `move_to` receives an anticipated follow-up switch term, while a `reroute` commits the waypoint-to-recovery continuation once. This makes the route-stability difference explicit.

## 5. Candidate nodes

The implementation does not enumerate every road node as an action. A finite union is built from:

- current recovery-queue nodes;
- up to `max_rendezvous_nodes_per_uav` flexible rendezvous nodes;
- current CAV out-neighbors;
- the next two nodes of the current route; and
- endpoints of up to three highest priority–urgency available tasks.

Nodes are deduplicated, checked for road reachability and prediction horizon, ranked deterministically by source priority/travel time/node ID, and truncated at `max_ground_candidates`. `keep_route` and `wait` are independent of the node cap.

## 6. Candidate data and features

`GroundCandidate` records the requested action, node/waypoint, route, affected recovery IDs, predicted recovery schedule, CAV and UAV time/distance/energy, detour, task coverage, recovery wait/slack, route stability, feasibility reason, score, computation time, and state fingerprint.

Task coverage uses the existing task priority and urgency:

```text
weight_i = priority_i / max(1, deadline_i - now)
D_task(v) = sum(weight_i * directed_road_distance(v, task_i.start)) / sum(weight_i)
```

No learned score was introduced.

## 7. Side-effect-free forward prediction

`evaluate_cav_candidate(..., commit=False)`:

1. copies the recovery queue;
2. predicts any flexible recovery-node update using the existing sortie segments and energy rates;
3. builds a directed CAV route with existing shortest paths;
4. calls the Stage 1 recovery scheduler's `simulate_ground_plan()`;
5. computes recovery wait, CAV wait, hover energy, battery slack, return time/distance, and propagated queue delay; and
6. evaluates constraints and normalized cost.

It does not mutate time, the event heap, CAV/UAV state, queue, task state, RNG state, or any schedule version. Test 15 snapshots all of those simulator-owned fields and verifies equality.

## 8. Feasibility

Non-keep candidates are removed when any of the following applies:

- CAV is inside an edge;
- directed route is unreachable;
- target is beyond the prediction horizon;
- detour time, distance, or ratio exceeds its configured maximum;
- recovery slack is below the minimum;
- predicted UAV queue wait exceeds the maximum;
- Stage 1 recovery/battery simulation is infeasible;
- a locked entry would change;
- a UAV is in final return/recovery;
- the action immediately reverses the last directed edge; or
- route commitment/cooldown/change-count rules reject it.

`keep_route` remains the safe fallback.

## 9. Ground cost

The deterministic cost is:

```text
J(g) = 1.00 * UAV_recovery_wait/H
     + 0.50 * UAV_return_time/H
     + 0.25 * CAV_wait/H
     + 0.30 * extra_CAV_time/H
     + 0.20 * task_distance/D_reference
     + 0.20 * route_switch_term
```

Every term is normalized before addition. Ties prefer keep, fewer route changes, smaller detour, greater recovery slack, then lower candidate ID. A non-keep action must also pass the configured absolute or relative improvement threshold.

## 10. Route commitment and oscillation control

The Vehicle tracks route-change timestamps, target history, required committed nodes, and commitment durations. Default behavior:

- travel at least one node after a route commit;
- 120 s reroute cooldown;
- no immediate directed-edge reverse;
- at most two changes in a 180 s horizon; and
- 5% minimum relative predicted improvement.

The route-oscillation micro scenario and unit tests 12–14 pass. Smoke route reversals were zero.

## 11. Event-Driven SMDP integration

For active variants, the passive route assignment at launch is deferred. The wrapper:

1. completes a same-time system-event batch;
2. updates task/UAV/queue state;
3. permits all bounded same-time UAV dispatch actions;
4. executes one ground decision after the last dispatch; and
5. stores `last_decision_time`, preventing a second ground commit at the timestamp.

If an event such as `CAV_WAIT_COMPLETE` has no valid UAV dispatch action, a pure ground decision is executed immediately after the batch. This avoids CAV deadlock without adding a Rainbow action. Existing `delta_t` and `transition_discount()` are unchanged.

The formal rerun exposed and fixed a same-time boundary defect: an active rendezvous update can make the final UAV return segment zero seconds. The old `dt=0` early return repeatedly rescheduled that event and grew memory without advancing the episode. The simulator now flushes zero-duration physical segments at the current timestamp, the wrapper invokes that flush for same-time batches, and the new test 25 plus the former failing C1 seed 11 regression pass.

## 12. Commit and rollback

`commit_cav_action()` checks the candidate fingerprint against current time, vehicle/version, recovery version, queue, and UAV states. It then:

- updates flexible-only recovery nodes and final return segments;
- increments affected UAV/recovery/CAV versions;
- commits wait or a validated directed route;
- updates predicted schedule fields and logs; and
- lets the existing event queue discard stale physical events.

If any operation fails, vehicle, queue, drones, and recovery version are restored. Test 16 verifies the stale-candidate path; Test 17 verifies stale CAV event removal.

## 13. Modified and new files

### Modified

- `env/events.py`
- `env/recovery_scheduler.py`
- `env/simulator.py`
- `env/task_generator.py`
- `env/vehicle.py`
- `rl/env_wrapper.py`
- `utils/metrics.py`

### Added

- `env/cav_controller.py`
- `tests/test_stage3_cav_active_routing.py`
- `experiments/run_stage3_cav_active_routing.py`
- `configs/scenario_60node_stage3_cav_common.yaml`
- `configs/scenario_60node_stage3_c0_passive.yaml`
- `configs/scenario_60node_stage3_c1_wait.yaml`
- `configs/scenario_60node_stage3_c2_move.yaml`
- `configs/scenario_60node_stage3_c3_reroute.yaml`
- `configs/stage3_spatial_low.yaml`
- `configs/stage3_spatial_medium.yaml`
- `configs/stage3_spatial_high.yaml`
- `configs/stage3_c3_5uav.yaml`
- `STAGE3_CAV_ACTIVE_ROUTING_REPORT.md`

No Rainbow-network file or sortie ranking function was changed.

## 14. Unit tests

Command:

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m unittest `
  tests.test_recovery_insertion `
  tests.test_stage2_event_smdp `
  tests.test_stage3_cav_active_routing -v
```

Result: **54/54 passed** in the final run.

The Stage 3 module contains the original 24 requested cases plus a regression proving that a zero-duration final return segment completes at the same timestamp.

## 15. Micro scenarios

| Scenario | Selected action | Result |
|---|---:|---:|
| Single imminent UAV | wait | Pass |
| Candidate rendezvous | move_to | Pass |
| Two consecutive recoveries | move_to with sequential Stage 1 schedule | Pass |
| Locked recovery | keep_route | Pass |
| Route oscillation | keep_route under cooldown | Pass |
| Detour benefit/cost | feasible reroute within ceiling | Pass |

Full initial state, candidates, feasibility, cost, selected action, predicted event timeline, committed state, expectation, and pass flag are in `results/stage3_cav_active_routing_formal_20seed/micro_scenario_results.csv`.

## 16. Formal 20-seed main experiment

The table reports paired-seed means over seeds 0–19; SDs and medians are in `stage3_aggregate_statistics.csv`.

| Variant | Completion | Timeout | Avg response (s) | Weighted AoI | Recovery wait (s) | Hover energy | UAV return dist. (m) | CAV dist. (m) | Non-keep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.8722 | 0.0657 | 345.8 | 583.41 | 177.0 | 2065.1 | 9903 | 25197 | 0.0000 |
| C1 | 0.9186 | 0.0444 | 317.0 | 579.43 | 95.1 | 1291.3 | 15757 | 21045 | 0.0802 |
| C2 | 0.9222 | 0.0389 | 314.8 | 578.45 | 91.3 | 1225.6 | 16099 | 21048 | 0.0883 |
| C3 | 0.9204 | 0.0391 | 315.3 | 578.99 | 85.7 | 1150.3 | 16325 | 20942 | 0.0910 |

All 80 main episodes completed. Invalid CAV edges, unreachable committed paths, negative-battery episodes, and route reversals were all zero.

## 17. Formal paired effects

The principal C3-versus-C0 paired results are:

| Metric | Relative change | 95% paired-difference CI | Paired t p | Cohen dz |
|---|---:|---:|---:|---:|
| Completion | +5.52% (+4.82 pp) | [0.0264, 0.0700] | 1.84e-4 | 1.03 |
| Timeout | -40.49% | [-0.0403, -0.0129] | 6.73e-4 | -0.91 |
| Mean response | -8.82% | [-55.63, -5.40] s | 1.99e-2 | -0.57 |
| Weighted AoI | -0.76% | [-6.84, -2.00] | 1.15e-3 | -0.85 |
| Mean recovery wait | -51.57% | [-115.56, -67.00] s | 2.14e-7 | -1.76 |
| Hover energy | -44.30% | [-1148.73, -680.90] | 1.19e-7 | -1.83 |
| CAV distance | -16.89% | [-5787.75, -2723.92] m | 1.33e-5 | -1.30 |
| UAV return distance | **+64.84%** | [4464.23, 8379.25] m | 1.50e-6 | 1.54 |

Active rendezvous clearly improves recovery hover and CAV travel, but transfers cost to longer UAV returns. Both sides of this trade-off must be reported. C3 mean ground-decision time is 4.17 ms and route reversal remains zero.

## 18. Active-action frequency and Stage 3C diagnosis

Across 1,633 C3 ground decisions, the controller selected 1,481 keep, 135 wait, 7 move, and 10 reroute actions. Reroute is no longer micro-scenario-only, but remains rare.

Relative to C2, C3 lowers mean recovery wait by another 6.1% and CAV distance by 0.5%, while completion decreases by 0.18 percentage points and UAV return distance increases by 1.4%. Stage 3C is therefore incremental rather than decisive; keep reroute as a distinct Stage-4 ablation.

## 19. Spatial, road-detour, and 5-UAV layers

Each layer uses 20 paired seeds. C3 versus C0 is directionally consistent:

| Layer | Completion change | Recovery-wait change | UAV-return change | CAV-distance change |
|---|---:|---:|---:|---:|
| Spatial Low | +5.20 pp | -57.2% | +75.7% | -18.9% |
| Spatial Medium | +4.82 pp | -51.6% | +64.8% | -16.9% |
| Spatial High | +6.37 pp | -61.7% | +65.2% | -18.2% |
| Road-detour Low | +4.00 pp | -32.7% | +213.5% | -16.7% |
| Road-detour Medium | +4.82 pp | -51.6% | +64.8% | -16.9% |
| Road-detour High | +5.24 pp | -56.5% | +86.2% | -8.3% |
| 5 UAV | -0.02 pp | -24.6% | +30.8% | -11.1% |

With five UAVs, completion is effectively unchanged (0.96408 versus 0.96392) while recovery wait falls from 225.6 s to 170.0 s. No supplement episode has a negative battery, invalid CAV edge, or unreachable committed path. The UAV-return increase in Road-detour Low is especially large and needs a Stage-4 penalty or cap.

## 20. Figures and raw outputs

The final main directory is `results/stage3_cav_active_routing_formal_20seed/`; spatial/road/5-UAV outputs are in `results/stage3_cav_active_routing_supplementary_20seed/`. The older `results/stage3_cav_active_routing/` and Smoke directory are retained as historical partial/smoke records and are not the formal conclusion source. The final directories contain:

- per-variant episode results;
- `paired_comparison.csv` and `paired_statistical_summary.csv`;
- `cav_action_log.csv`;
- `ground_candidate_log.csv`;
- `route_change_log.csv`;
- `recovery_schedule_log.csv`;
- `micro_scenario_results.csv`;
- `scenario_stratified_results.csv`;
- `stage3_summary.json`;
- main-metric, recovery-wait, return-distance, CAV trade-off, action-frequency, detour, route-change, spatial, road-detour, and representative-route figures.

The representative route plot includes the directed road background, executed CAV route, UAV rendezvous markers, and active CAV decision markers.

## 21. Commands

### C0

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --variants C0 --seeds 0:20 --skip-stratified --skip-micro
```

### C1

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --variants C1 --seeds 0:20 --skip-stratified --skip-micro
```

### C2

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --variants C2 --seeds 0:20 --skip-stratified --skip-micro
```

### C3

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --variants C3 --seeds 0:20 --skip-stratified --skip-micro
```

### Micro scenarios

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --micro-only
```

### Three-seed Smoke

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --smoke --skip-stratified
```

### Twenty-seed main + short stratified supplement

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --seeds 0:20 --stratified-seeds 0:3 --include-uav5
```

### Thirty-seed main

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --seeds 0:30 --stratified-seeds 0:10 --include-uav5
```

In the tested Windows Anaconda environment, plotting may require setting `$env:KMP_DUPLICATE_LIB_OK='TRUE'` before the experiment command because SciPy/Matplotlib load two OpenMP runtimes.

## 22. Acceptance and Stage 4 recommendation

### Correctness gate

- C0 reproduces Stage 2: **pass**.
- Locked recovery never changes: **pass**.
- Directed CAV routes remain valid: **pass**.
- No negative battery: **pass**.
- No infinite wait: **pass**.
- No high-frequency route oscillation/reversal: **pass**.
- Dynamic Recovery Insertion remains functional: **pass**.
- Event timeline and SMDP discount remain functional: **pass**.
- Determinism: **pass**.

### Value gate

The formal 20-seed result passes recovery condition A: mean recovery wait, hover energy, and CAV distance fall significantly while completion rises significantly. Spatial, road-detour, and 5-UAV supplements preserve the recovery-wait direction. The significant UAV-return-distance increase is the one persistent major cost.

### Recommendation

**Proceed to joint-action Stage 4.** Keep, wait, and move should be the primary actions, with low-frequency reroute retained as an ablation. Add an explicit UAV return-distance/energy term or constraint, particularly for low road-detour-factor maps. Continue reporting C0–C3 as fixed baselines and require Stage 4 to report recovery wait, completion, CAV distance, and UAV return distance together.
