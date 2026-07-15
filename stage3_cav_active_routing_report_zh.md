# 阶段3：CAV主动路径规划与会合协调

## 1. 执行状态概述

阶段3A–3C已在现有动态回收插入（Dynamic Recovery Insertion）与事件驱动SMDP实现的基础上，作为一个独立且确定性的地面控制器完成开发。

实现与兼容性验收结果如下：

- 阶段1回归测试：10/10项通过。
- 阶段2回归测试：19/19项通过。
- 阶段3需求与回归测试：25/25项通过；阶段1–3合计54/54项通过。
- 六个确定性微场景：6/6项通过。
- C0–C3正式成对实验：20个随机种子、80个主实验回合全部完成。
- 空间分层、道路绕行与5架UAV补充实验：共280个回合全部完成。
- 在成对回归随机种子下，C0被动模式能够精确复现当前阶段2事件驱动基线。

正式结果满足进入阶段4的预设门槛：C3相对C0将平均UAV回收等待降低51.6%，完成率提高4.82个百分点，CAV行驶距离降低16.9%，且未出现无效道路边、负电池或路径反向。代价是UAV返航距离增加64.8%，因此该权衡必须在阶段4奖励与约束中显式处理。

## 2. 研究边界

阶段3仅改变CAV的地面路径控制，不涉及以下内容：

- 不向25维UAV/Rainbow动作空间中增加CAV动作；
- 不改变UAV架次候选排序或每个架次允许执行的最大任务数量；
- 不改变Rainbow网络结构，也不增加新的神经网络；
- 不重新训练Rainbow；
- 不改变奖励函数或连续时间SMDP折扣机制；
- 不改变阶段1的插入代价；
- 不改变任务到达强度；
- 不构造CAV–UAV联合动作。

主实验将UAV策略固定为现有确定性候选评分启发式策略。C0–C3之间仅在`cav_control.mode`和`cav_control.action_set`上存在差异。

## 3. 修改前代码结构映射

| 功能 | 文件 | 类/函数 | 原有逻辑 | 阶段3集成方式 |
|---|---|---|---|---|
| CAV状态 | `env/vehicle.py` | `Vehicle` | 节点、有向路径、当前道路边剩余时间、目标、路径日志、调度版本 | 新增显式的事件边界等待、任意有向路径提交和承诺历史记录 |
| CAV位置/当前边/路径 | `env/vehicle.py` | `position`, `path_nodes`, `path_index`, `edge_remaining_time` | 严格按道路边逐段移动 | 保持不变；当`at_intersection`为false时，拒绝动作切换 |
| 路径生成 | `env/vehicle.py` | `set_target()` | 生成到单个回收目标的有向最短路径 | 新增经过验证的`set_route()`，用于提交“途经点+回收点”路径 |
| 最短路径/行程时间 | `env/road_network.py` | `get_vehicle_path()`, `get_vehicle_time()` | 预计算的有向道路最短路径 | 所有地面候选动作均复用该模型，不建立第二套距离/时间模型 |
| 道路距离 | `env/road_network.py` | `arc_length()` | 有向道路边的物理长度 | 复用于CAV路径、绕行、任务覆盖和指标守恒计算 |
| CAV到达事件 | `env/events.py`, `rl/env_wrapper.py` | `CAV_NODE_ARRIVAL`, `_schedule_dynamic_events()` | 采用版本控制的事件堆标记下一次物理道路边到达 | 新增`CAV_WAIT_COMPLETE`；路径变化通过`schedule_version`使旧到达事件失效 |
| 回收队列 | `env/simulator.py` | `recovery_queue` | FIFO或动态插入；CAV跟随队首 | 主动模式下，将自动提交队首目标的操作推迟到地面决策阶段 |
| 动态插入 | `env/recovery_scheduler.py` | `RecoveryInsertionScheduler` | 无副作用地进行插入、预测电池与队列 | 新增`simulate_ground_plan()`，复用相同的回收时序、悬停和电池计算规则 |
| 锁定/灵活状态 | `env/recovery_scheduler.py` | `locked_prefix_length()`, `_runtime_status()` | 保护已承诺、返航中和回收中的UAV | 地面更新仅允许作用于该锁定前缀之外的条目 |
| UAV ETA/电池 | `env/recovery_scheduler.py` | `make_entry()`, `simulate()` | 计算队列条目ETA、到达时电池、保留电量和悬停功率 | 地面控制器在更新灵活条目前重新计算最终返航时间与能耗 |
| UAV回收节点 | `planner/sortie_generator.py` | `Sortie.recovery_node`及最终飞行段 | 在生成架次时确定 | 仅对灵活条目更新回收节点和最终飞行段，不改变任务集合 |
| 阶段2事件堆 | `env/events.py` | `EventQueue` | 稳定优先队列、同一时刻批处理、失效事件过滤 | 保持原样复用；新增物理事件继续携带现有实体版本号 |
| 调度版本 | `env/vehicle.py`, `env/drone.py` | `schedule_version` | 用于使过期的CAV/UAV事件失效 | 在提交路径/等待动作或修改灵活返航计划后递增 |
| 任务分布 | `env/task_generator.py` | `_poisson_tasks()` | 固定速率泊松到达，并依据风险权重进行空间选取 | 可选低/中/高空间集中度仅改变空间支持，不改变任务数量或到达率 |
| UAV启发式策略 | `planner/greedy_baseline.py` | 候选评分/优先级策略 | 固定架次调度 | 阶段3主实验对所有变体使用相同的确定性候选评分动作 |
| Rainbow入口/检查点 | `rl/train_masked_dqn.py`, `experiments/evaluate_masked_rl_models.py` | 现有DQN接口 | 25个UAV动作并支持确定性评估 | 保持不变；未运行冻结Rainbow的阶段3补充实验 |
| 指标 | `env/simulator.py`, `utils/metrics.py` | `metrics()`, `EpisodeMetrics` | 任务、能耗、回收及阶段1指标 | 新增返航距离/时间、地面动作、绕行、稳定性、裕度和计算时间指标 |
| 阶段1输出 | `results/stage1_dynamic_recovery/`及阶段1实验路径 | 现有实验运行器 | 回收插入原始结果和汇总结果 | 保留 |
| 阶段2输出 | `results/stage2_event_driven_smdp/` | `run_stage2_event_smdp.py` | 固定步长/事件驱动原始结果、转移和时间线 | 保留 |

在阶段3开发之前，代码仓库已经处于非干净状态。现有阶段2修改和未跟踪文件均被保留，并作为C0基线使用；未执行任何reset、checkout或覆盖操作。

## 4. 动作定义

### 4.1 `keep_route`

始终可行。该动作保留当前已有的有向路径。如果主动模式推迟了被动的队首目标分配，则提交`keep_route`时会调用原有的`_set_vehicle_next_recovery_target()`，从而复现旧有行为。

### 4.2 `wait`

仅在CAV位于道路节点时可用。等待结束时间为：

```text
min(下一个相关UAV的ETA, 下一个有效系统事件, 当前时刻 + max_wait_s)
```

对于灵活返航条目，在重新计算UAV返航时间和电池后，等待动作可以将其会合点更新为CAV当前节点。已锁定或处于最终返航阶段的UAV不能被更新。`Vehicle.begin_wait()`会递增CAV版本号，并创建一个`CAV_WAIT_COMPLETE`事件边界。

### 4.3 `move_to(v)`

提交从当前节点到某一受限候选节点的有向最短路径。灵活会合点可以更新为节点`v`。控制器会预测后续队列调度，但仅提交直接目标，因此在下一个节点事件发生时可以重新考虑路径。

### 4.4 `reroute(v|H)`

提交一个有限路径前缀：

```text
当前节点 -> 途经点v -> 下一个已承诺回收节点
```

该动作不会重排锁定前缀。中间`move_to`动作会加入一个预期后续切换项，而`reroute`会一次性提交“途经点到回收点”的后续路径。这样可以显式体现路径稳定性的差异。

## 5. 候选节点

实现中不会将每个道路节点都枚举为动作，而是通过以下来源构造一个有限候选集合：

- 当前回收队列中的节点；
- 每架灵活UAV最多`max_rendezvous_nodes_per_uav`个会合候选节点；
- CAV当前节点的出邻接节点；
- 当前路径中的后续两个节点；
- 优先级–紧迫度最高的最多三个可用任务的端点。

候选节点会进行去重，并检查道路可达性和预测时间范围。随后按照来源优先级、行程时间和节点ID进行确定性排序，并截断至`max_ground_candidates`。`keep_route`和`wait`不受候选节点数量上限影响。

## 6. 候选数据与特征

`GroundCandidate`记录以下信息：请求动作、目标节点/途经点、路径、受影响的回收条目ID、预测回收时序、CAV与UAV时间/距离/能耗、绕行、任务覆盖、回收等待/裕度、路径稳定性、可行性原因、评分、计算时间和状态指纹。

任务覆盖使用现有任务优先级和紧迫度：

```text
weight_i = priority_i / max(1, deadline_i - now)
D_task(v) = sum(weight_i * directed_road_distance(v, task_i.start)) / sum(weight_i)
```

未引入学习型评分函数。

## 7. 无副作用前向预测

`evaluate_cav_candidate(..., commit=False)`执行以下步骤：

1. 复制回收队列；
2. 使用现有架次飞行段和能耗率，预测灵活回收节点更新；
3. 使用现有最短路径构建CAV有向路径；
4. 调用阶段1回收调度器中的`simulate_ground_plan()`；
5. 计算回收等待、CAV等待、悬停能耗、电池裕度、返航时间/距离及传播后的队列延迟；
6. 检查约束并计算归一化代价。

该函数不会修改仿真时间、事件堆、CAV/UAV状态、队列、任务状态、随机数生成器状态或任何调度版本号。测试15对所有由仿真器管理的字段进行快照，并验证前后完全一致。

## 8. 可行性判定

当出现以下任一情况时，除`keep_route`外的候选动作会被移除：

- CAV正位于道路边内部；
- 有向路径不可达；
- 目标超出预测时间范围；
- 绕行时间、距离或比例超过配置的最大值；
- 回收裕度低于最小值；
- 预测UAV队列等待时间超过最大值；
- 阶段1回收/电池仿真不可行；
- 某个锁定条目将发生变化；
- UAV处于最终返航或回收阶段；
- 动作会立即反向经过上一条有向道路边；
- 路径承诺、冷却时间或路径变更次数规则拒绝该动作。

`keep_route`始终作为安全后备动作保留。

## 9. 地面控制代价函数

确定性代价函数为：

```text
J(g) = 1.00 * UAV_recovery_wait/H
     + 0.50 * UAV_return_time/H
     + 0.25 * CAV_wait/H
     + 0.30 * extra_CAV_time/H
     + 0.20 * task_distance/D_reference
     + 0.20 * route_switch_term
```

所有分项均在求和前进行归一化。出现并列时，依次优先选择：保持路径、路径变更次数更少、绕行更小、回收裕度更大、候选ID更小。非保持动作还必须达到配置的绝对改进阈值或相对改进阈值。

## 10. 路径承诺与振荡控制

`Vehicle`记录路径变更时间戳、目标历史、必须承诺的节点以及承诺持续时间。默认行为包括：

- 每次路径提交后至少行驶一个节点；
- 重新规划冷却时间为120 s；
- 禁止立即反向经过上一条有向道路边；
- 在180 s时间窗口内最多允许两次路径变更；
- 预测结果至少需要取得5%的相对改进。

路径振荡微场景以及单元测试12–14均通过。冒烟测试中的路径反向次数为0。

## 11. 事件驱动SMDP集成

对于主动控制变体，架次发射后原有的被动路径分配会被延迟。环境包装器按以下顺序执行：

1. 完成同一时刻的一批系统事件；
2. 更新任务、UAV和队列状态；
3. 允许在同一时刻执行所有受限UAV调度动作；
4. 在最后一次调度后执行一次地面决策；
5. 保存`last_decision_time`，防止同一时间戳再次提交地面动作。

如果`CAV_WAIT_COMPLETE`等事件发生时不存在有效的UAV调度动作，则在该事件批次之后立即执行一次纯地面决策。该机制在不增加Rainbow动作的前提下避免CAV死锁。现有`delta_t`和`transition_discount()`保持不变。

正式续跑发现并修复了一个同时间事件边界缺陷：当主动会合把最终返航点更新为UAV当前节点时，最终飞行段可能为0秒。旧逻辑对`dt=0`直接返回，造成同一事件无限重排和内存增长。现在仿真器会在不推进物理时间的情况下清空零时长飞行段，环境包装器在同时间事件批次显式调用该过程；新增测试25和C1 seed 11回归均通过。

## 12. 提交与回滚

`commit_cav_action()`会将候选动作的状态指纹与当前时刻、车辆/版本、回收版本、队列及UAV状态进行一致性检查。随后执行：

- 更新仅限灵活条目的回收节点和最终返航飞行段；
- 递增受影响UAV、回收模块和CAV的版本号；
- 提交等待动作或经过验证的有向路径；
- 更新预测时序字段和日志；
- 由现有事件队列自动丢弃过期物理事件。

如果任一步骤失败，则恢复车辆、队列、无人机和回收版本。测试16验证过期候选动作路径，测试17验证过期CAV事件的移除机制。

## 13. 修改与新增文件

### 修改文件

- `env/events.py`
- `env/recovery_scheduler.py`
- `env/simulator.py`
- `env/task_generator.py`
- `env/vehicle.py`
- `rl/env_wrapper.py`
- `utils/metrics.py`

### 新增文件

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

未修改任何Rainbow网络文件或架次排序函数。

## 14. 单元测试

命令如下：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m unittest `
  tests.test_recovery_insertion `
  tests.test_stage2_event_smdp `
  tests.test_stage3_cav_active_routing -v
```

最终运行结果：**54/54项通过**。

阶段3测试模块包含25类测试：原有24类C0回归、动作有效性、约束、无副作用、回滚、兼容性、确定性和指标守恒测试，以及新增的“零时长最终返航段在同一时间戳完成”回归测试。

## 15. 微场景

| 场景 | 选定动作 | 结果 |
|---|---:|---:|
| 单架即将到达的UAV | wait | 通过 |
| 候选会合点 | move_to | 通过 |
| 两次连续回收 | move_to，并采用阶段1顺序回收时序 | 通过 |
| 已锁定回收 | keep_route | 通过 |
| 路径振荡 | 冷却规则下选择keep_route | 通过 |
| 绕行收益/成本 | 在约束上限内选择可行reroute | 通过 |

完整初始状态、候选动作、可行性、代价、选定动作、预测事件时间线、提交后状态、预期结果和通过标记均保存在`results/stage3_cav_active_routing_formal_20seed/micro_scenario_results.csv`中。

## 16. 20随机种子正式主实验

下表给出了成对随机种子0–19上的均值；标准差和中位数见`stage3_aggregate_statistics.csv`。

| 变体 | 完成率 | 超时率 | 平均响应（s） | 加权AoI | 回收等待（s） | 悬停能耗 | UAV返航距离（m） | CAV距离（m） | 非保持比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.8722 | 0.0657 | 345.8 | 583.41 | 177.0 | 2065.1 | 9903 | 25197 | 0.0000 |
| C1 | 0.9186 | 0.0444 | 317.0 | 579.43 | 95.1 | 1291.3 | 15757 | 21045 | 0.0802 |
| C2 | 0.9222 | 0.0389 | 314.8 | 578.45 | 91.3 | 1225.6 | 16099 | 21048 | 0.0883 |
| C3 | 0.9204 | 0.0391 | 315.3 | 578.99 | 85.7 | 1150.3 | 16325 | 20942 | 0.0910 |

80个主实验回合全部完成。无效CAV道路边、不可达已提交路径、负电池回合和路径反向次数均为0；锁定回收保护、过期事件失效、阶段1/2回归和确定性检查均通过。

## 17. 正式成对效应

C3相对C0的主要成对结果如下：

| 指标 | 相对变化 | 95%配对差置信区间 | 配对t检验p值 | Cohen dz |
|---|---:|---:|---:|---:|
| 完成率 | +5.52%（+4.82个百分点） | [0.0264, 0.0700] | 1.84×10⁻⁴ | 1.03 |
| 超时率 | -40.49% | [-0.0403, -0.0129] | 6.73×10⁻⁴ | -0.91 |
| 平均响应时间 | -8.82% | [-55.63, -5.40] s | 1.99×10⁻² | -0.57 |
| 加权AoI | -0.76% | [-6.84, -2.00] | 1.15×10⁻³ | -0.85 |
| 平均回收等待 | -51.57% | [-115.56, -67.00] s | 2.14×10⁻⁷ | -1.76 |
| 悬停能耗 | -44.30% | [-1148.73, -680.90] | 1.19×10⁻⁷ | -1.83 |
| CAV距离 | -16.89% | [-5787.75, -2723.92] m | 1.33×10⁻⁵ | -1.30 |
| UAV返航距离 | **+64.84%** | [4464.23, 8379.25] m | 1.50×10⁻⁶ | 1.54 |

正式结果确认了主动会合的价值，也确认了明确的代价转移：更少的回收悬停和CAV行驶来自更长的UAV返航段。论文结论必须同时报告两方面，不能表述为所有距离/能耗指标同步改善。

C3平均地面决策计算时间为4.17 ms，路径反向次数为0。

## 18. 主动动作频率与阶段3C诊断

C3在20个回合、1633次地面决策中选择`keep_route` 1481次、`wait` 135次、`move_to` 7次、`reroute` 10次。`reroute`不再是仅在微场景中触发，但频率仍低。

相对C2，C3平均回收等待进一步下降6.1%（91.3 s降至85.7 s），CAV距离下降0.5%，但完成率降低0.18个百分点、UAV返航距离增加1.4%。因此，阶段3C提供了小幅增益而非决定性优势；阶段4应保留`reroute`并继续做独立消融，不应将其与`wait/move`的主要收益混为一谈。

## 19. 空间分层、道路绕行与5架UAV实验层

每个分层单元均使用20个成对随机种子。C3相对C0的结果方向在所有层级一致：

| 层级 | 完成率变化 | 回收等待变化 | UAV返航距离变化 | CAV距离变化 |
|---|---:|---:|---:|---:|
| Spatial Low | +5.20个百分点 | -57.2% | +75.7% | -18.9% |
| Spatial Medium | +4.82个百分点 | -51.6% | +64.8% | -16.9% |
| Spatial High | +6.37个百分点 | -61.7% | +65.2% | -18.2% |
| Road-detour Low | +4.00个百分点 | -32.7% | +213.5% | -16.7% |
| Road-detour Medium | +4.82个百分点 | -51.6% | +64.8% | -16.9% |
| Road-detour High | +5.24个百分点 | -56.5% | +86.2% | -8.3% |
| 5 UAV | -0.02个百分点 | -24.6% | +30.8% | -11.1% |

5-UAV下完成率基本持平（C0 0.96408，C3 0.96392），回收等待由225.6 s降至170.0 s。全部280个补充回合中无负电池、无效CAV边或不可达提交路径。道路绕行Low层的UAV返航距离增幅尤其大，阶段4需要对该指标设置代价或上限。

## 20. 图表与原始输出

最终主实验目录为`results/stage3_cav_active_routing_formal_20seed/`，空间/道路/5-UAV结果目录为`results/stage3_cav_active_routing_supplementary_20seed/`。旧的`results/stage3_cav_active_routing/`和`results/stage3_cav_active_routing_smoke_latest/`仅保留历史冒烟/部分运行记录，不再作为正式结论来源。最终目录包括：

- 各变体逐回合结果；
- `paired_comparison.csv`和`paired_statistical_summary.csv`；
- `cav_action_log.csv`；
- `ground_candidate_log.csv`；
- `route_change_log.csv`；
- `recovery_schedule_log.csv`；
- `micro_scenario_results.csv`；
- `scenario_stratified_results.csv`；
- `stage3_summary.json`；
- 主指标、回收等待、返航距离、CAV权衡、动作频率、绕行、路径变更、空间分层、道路绕行以及代表性路径图。

代表性路径图包含有向道路背景、CAV实际执行路径、UAV会合点标记和主动CAV决策标记。

## 21. 运行命令

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

### 微场景

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --micro-only
```

### 三随机种子冒烟测试

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --smoke --skip-stratified
```

### 20随机种子主实验+短分层补充实验

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --seeds 0:20 --stratified-seeds 0:3 --include-uav5
```

### 30随机种子主实验

```powershell
python -B -m dynamic_patrol_marl.experiments.run_stage3_cav_active_routing --seeds 0:30 --stratified-seeds 0:10 --include-uav5
```

在已测试的Windows Anaconda环境中，由于SciPy/Matplotlib可能加载两个OpenMP运行时，绘图前可能需要设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

然后再执行实验命令。

## 22. 验收结果与阶段4建议

### 正确性门槛

- C0复现阶段2：**通过**。
- 锁定回收条目从不改变：**通过**。
- CAV有向路径始终有效：**通过**。
- 不出现负电池电量：**通过**。
- 不出现无限等待：**通过**。
- 不出现高频路径振荡或反向：**通过**。
- 动态回收插入保持正常：**通过**。
- 事件时间线和SMDP折扣保持正常：**通过**。
- 确定性：**通过**。

### 价值门槛

20-seed正式结果满足恢复条件A：平均回收等待、悬停能耗和CAV距离显著下降，完成率显著提高；空间分层、道路绕行和5-UAV补充实验也保持相同的回收等待改善方向。UAV返航距离显著增加，是唯一明确且跨层级持续存在的主要代价。

### 建议

**建议进入联合动作阶段4。** 阶段4应以`keep`、`wait`和`move`为主要动作，并保留低频`reroute`用于消融；奖励或约束中必须加入UAV返航距离/能耗项，尤其防止道路绕行Low层出现过大的返航代价。建议继续使用C0–C3作为固定基线，阶段4模型必须同时报告回收等待、完成率、CAV距离与UAV返航距离，避免只优化会合等待而转移成本。
