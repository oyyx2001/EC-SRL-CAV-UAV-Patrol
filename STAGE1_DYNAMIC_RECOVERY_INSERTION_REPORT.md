# 阶段一：动态恢复插入调度报告

生成日期：2026-07-14

## 1. 原代码映射

| 功能 | 文件 | 类/函数 | 修改前逻辑 |
|---|---|---|---|
| Recovery queue 数据结构 | `env/simulator.py` | `Simulator.__init__` | `List[Dict]`，条目仅含 UAV、回收节点、发射时间和任务 ID |
| 新架次加入队列 | `env/simulator.py` | `Simulator.dispatch_sorties` | 固定 `append` 到队尾 |
| CAV 回收路线 | `env/simulator.py`, `env/vehicle.py` | `_set_vehicle_next_recovery_target`, `Vehicle.set_target/step` | CAV 只追踪队首，并沿有向道路最短路逐边行驶 |
| 候选回收可行性 | `planner/sortie_generator.py` | `SortieGenerator.evaluate` | 用队尾锚点和累计队列时间预测，检查截止时间与 UAV 能量 |
| Action mask | `rl/env_wrapper.py` | `DynamicPatrolEnv.action_masks` | 直接读取 `Sortie.feasible` |
| UAV ETA | `planner/sortie_generator.py` | `SortieGenerator.evaluate` | `now + drone_time_s`；原预测把队尾等待能耗计入 `Sortie.energy` |
| 等待、悬停和扣能 | `env/simulator.py` | `_advance_drone`, `_consume`, `_recover_drone` | UAV 到点后进入 `waiting_recovery`，按时间步实际扣除悬停能量 |
| 配置/训练/测试入口 | `configs`, `rl/train_masked_dqn.py`, `experiments/evaluate_masked_rl_models.py` | YAML/CLI | 配置中原先没有 recovery rule |
| 指标与日志 | `env/simulator.py`, `utils/metrics.py` | `metrics`, `export_logs` | 统计任务、总能耗、总等待和 AoI；无插入机制指标 |

## 2. 实现概述

新增 `RecoveryInsertionScheduler`，直接读取现有 `Simulator` 的 CAV、UAV、道路网络和 recovery queue，不建立第二套仿真器。对候选架次执行以下流程：

1. 从现有 sortie 取得 ETA、飞行/服务能耗和到达时电量。
2. 根据显式锁、UAV 最终返航段、等待回收状态和 CAV 已执行路线确定锁定前缀。
3. 对允许的 Python 插入索引 `L..len(queue)` 分别构造临时队列。
4. 使用 `RoadNetwork.get_vehicle_time()` 前向计算 CAV 到达、回收开始/结束、UAV 等待、CAV 等待和悬停能耗。
5. 检查道路可达、重复 UAV、基础架次约束及全队列电量余量。
6. 在可行位置中最小化确定性成本，并按既有任务延迟、位移、距队尾距离和索引稳定破平局。
7. action mask 阶段只评估并缓存到 `Sortie.insertion_evaluation`；动作提交时重新计算状态指纹并提交。

预测悬停能耗只参与可行性和成本计算。实际电量仍只由 `Simulator._consume()` 扣除。

## 3. 队列条目

现有字典条目被扩展为可访问以下字段：`uav_id/drone_id`、`recovery_node`、`eta`、`recovery_duration`、`battery_at_eta`、`battery_reserve`、`sortie_id`、`status`、`locked`、`original_queue_index`、`scheduled_start_time`、`scheduled_finish_time`、`predicted_uav_wait`、`predicted_cav_wait`、`hover_power` 和实际到达时间。

原环境没有独立的 `returning/recovering` UAV 状态，因此队列状态由最终返航段及 `waiting_recovery` 推断。现有 CAV 不支持 reroute，所以只要 CAV 已经驶向队首，队首即被锁定；这是比 `lock_horizon_s` 更严格但与真实执行一致的安全约束。

## 4. 成本与配置

成本函数为：

`J = 1.0 * total_uav_wait + 0.5 * existing_delay + 0.25 * total_cav_wait + 0.10 * route_time_increase`

配置均位于 `recovery_insertion`：

- `lock_horizon_s: 120`
- `preserve_locked_prefix: true`
- `weight_uav_wait: 1.0`
- `weight_existing_delay: 0.5`
- `weight_cav_wait: 0.25`
- `weight_route_time: 0.10`
- `battery_reserve: 0.0`
- `use_battery_feasibility: true`
- `cache_candidate_evaluation: true`
- `log_insertion_details: true`

现有场景的 `recovery_time_s` 均为 0，且原环境允许电量使用到 0。为保持 FIFO 物理行为，阶段一没有虚构新的服务时长，安全余量默认配置为 0；调度器已经支持将两者改为正值。

## 5. 修改文件

- 核心：`env/recovery_scheduler.py`, `env/simulator.py`, `planner/sortie_generator.py`
- 指标：`utils/metrics.py`
- 配置：4 个现有场景 YAML，以及 4 个阶段一 FIFO/Dynamic 覆盖 YAML
- 实验：`experiments/run_stage1_recovery_experiment.py`
- 测试：`tests/test_recovery_insertion.py`
- 运行兼容：`dynamic_patrol_marl/__init__.py`, `utils/config.py`, `env/road_network.py`

兼容入口是必要适配：Git clone 目录名为 `dynamic-patrol`，源码却统一导入 `dynamic_patrol_marl`。该入口让仓库无需改名即可运行，并将 `project_root()` 修正到实际仓库根目录。

## 6. 测试结果

命令：

```powershell
$env:PYTHONPATH=(Get-Location).Path
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m unittest discover -s tests -v
```

结果：`Ran 10 tests ... OK`。

覆盖：FIFO 金样回归、队尾最优、中间最优、队首最优、锁定前缀、电量不可行、全部位置不可行并屏蔽 action、只评估不提交、确定性、预测能耗不重复扣除。

FIFO seed 0 金样与修改前完全一致：20 个完成任务、4 个过期、9 个架次、UAV 等待 400 s、CAV 等待 300 s、总能耗 3680.0885128707264、Weighted-AoI 615.6631987447494，9 次 append 的时间/节点/位置序列也一致。

## 7. Smoke Test（3 个配对 seed）

FIFO 与 Dynamic 均完整结束。负电量、重复回收、不可达回收节点和不可行提交均为 0。Dynamic 在 seed 1 和 seed 2 选择了中间位置。

- Mean UAV recovery wait：下降 18.7%
- Total hover energy：下降 26.5%
- Completion rate：平均下降 0.90 个百分点
- Non-append rate：6.27%

## 8. 启发式机制实验（20 个配对 seed）

策略：`score_based`。两组仅切换 recovery rule。

| 指标 | FIFO 均值 | Dynamic 均值 | 相对变化 |
|---|---:|---:|---:|
| Completion rate | 0.667776 | 0.667323 | -0.068%（-0.045 个百分点） |
| Timeout rate | 0.216267 | 0.209773 | -3.00%（-0.649 个百分点） |
| Average response time (s) | 507.959 | 519.867 | +2.34% |
| High-priority response (s) | 294.479 | 292.431 | -0.70% |
| Weighted-AoI | 604.514 | 604.544 | +0.005% |
| UAV energy/completed task | 182.616 | 177.108 | -3.02% |
| Mean UAV recovery wait (s) | 81.632 | 48.668 | **-40.38%** |
| P95 UAV recovery wait (s) | 301.602 | 183.722 | **-39.08%** |
| Total hover energy | 553.475 | 332.800 | **-39.87%** |
| Mean CAV waiting (s) | 55.082 | 51.821 | -5.92% |
| CAV travel distance (m) | 21786.274 | 21862.098 | +0.35% |
| Recovery contention rate | 17.86% | 16.20% | -1.66 个百分点 |
| Non-append rate | 0% | **9.06%** | +9.06 个百分点 |
| Mean counterfactual gain | 0 | 34.103 | +34.103 |
| Insertion computation (ms) | 0 | **0.073** | +0.073 ms |

回收等待配对差值的 95% CI 为 `[-51.60, -14.33] s`，悬停能耗为 `[-340.49, -100.86]`，均不跨 0。20+20 个 episode 的负电量、重复回收、不可达节点和不可行提交均为 0。

## 9. Rainbow checkpoint 快速诊断（20 个配对 seed）

仓库 checkpoint 的维度为 `obs_dim=219/action_dim=25`，对应 `top_k_per_drone=8`。因此使用专门的 FIFO/Dynamic checkpoint 配置恢复 `top_k=8`；网络、权重、任务和 seed 不变。

| 指标 | FIFO 均值 | Dynamic 均值 | 相对变化 |
|---|---:|---:|---:|
| Completion rate | 0.775041 | 0.809227 | +4.41%（+3.42 个百分点） |
| Timeout rate | 0.138390 | 0.107307 | -22.46%（-3.11 个百分点） |
| Average response time (s) | 453.671 | 425.137 | -6.29% |
| High-priority response (s) | 290.671 | 309.655 | +6.53% |
| Weighted-AoI | 591.456 | 588.174 | -0.55% |
| UAV energy/completed task | 244.867 | 233.380 | -4.69% |
| Mean UAV recovery wait (s) | 170.754 | 126.711 | **-25.79%** |
| Total hover energy | 1536.925 | 1246.050 | **-18.93%** |
| CAV travel distance (m) | 22873.958 | 22799.507 | -0.33% |
| Recovery contention rate | 44.78% | 44.81% | +0.03 个百分点 |
| Non-append rate | 0% | **15.07%** | +15.07 个百分点 |
| Insertion computation (ms) | 0 | **0.167** | +0.167 ms |

该 checkpoint 未针对动态规则重新训练，因此仅作机制诊断。高优先级响应时间变差 6.53%，需要在阶段二重训后重点复核。

## 10. 阶段结论

动态插入达到阶段一验收标准：

- 两组 20-seed 实验的 mean recovery wait 和 hover energy 均下降超过 5%。
- 启发式实验 completion rate 仅下降 0.045 个百分点，timeout rate 下降 0.649 个百分点。
- CAV 距离只增加 0.35%，远低于 5% 上限。
- 无新增不可行回收、负电量、重复条目或不可达节点。
- non-append rate 分别为 9.06% 和 15.07%，机制确实被使用。
- 平均插入计算耗时低于 0.2 ms。

建议进入阶段二，进行 Rainbow + FIFO 与 Rainbow + Dynamic 的同构配对重训。阶段二最小关注点是高优先级响应时间及 ETA 预测误差，不应改任务生成、奖励或候选排序公式。

## 11. 复现命令

先在仓库根目录设置：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$python='D:\ProgramData\Anaconda3\envs\gym\python.exe'
```

单独运行 FIFO/Dynamic 启发式：

```powershell
& $python -B -m dynamic_patrol_marl.experiments.run_baselines --config configs/scenario_60node_stage1_fifo.yaml --seeds 3 --methods score_based
& $python -B -m dynamic_patrol_marl.experiments.run_baselines --config configs/scenario_60node_stage1_dynamic.yaml --seeds 3 --methods score_based
```

运行 20-seed 配对实验：

```powershell
& $python -B -m dynamic_patrol_marl.experiments.run_stage1_recovery_experiment --seeds 20 --method score_based --out-dir results/stage1_dynamic_recovery/paired_20
```

阶段二配对训练命令（已完成 10 步训练冒烟验证，未启动 1,000,000 步完整训练）：

```powershell
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn --config configs/scenario_60node_stage2_rainbow_fifo.yaml --seed 42
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn --config configs/scenario_60node_stage2_rainbow_dynamic.yaml --seed 42
```

完整配对训练说明见 `RAINBOW_FIFO_DYNAMIC_TRAINING.md`。

## 12. 结果文件

- Smoke：`results/stage1_dynamic_recovery/smoke/`
- 20-seed 原始结果：`results/stage1_dynamic_recovery/paired_20/stage1_raw_results.csv`
- 20-seed 汇总：`results/stage1_dynamic_recovery/paired_20/stage1_summary.csv`
- 插入事件日志：`results/stage1_dynamic_recovery/paired_20/stage1_insertion_events.csv`
- Checkpoint 原始/汇总：`results/stage1_dynamic_recovery/rainbow_checkpoint/checkpoint_raw_results.csv`, `checkpoint_summary.csv`

`results/` 按仓库现有 `.gitignore` 不纳入版本控制，但结果已保存在本地工作区。
