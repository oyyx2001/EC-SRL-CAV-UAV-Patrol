# 阶段二：Event-Driven SMDP 实施与初步实验报告

日期：2026-07-14
项目：CAV–UAV 协同动态巡逻
基线：阶段一 Dynamic Recovery Insertion（提交 `2a24d73`）

## 1. 结论摘要

阶段二已经实现为可配置的双时间模式，未删除或替换阶段一代码：

- `time.mode: fixed_interval` 保留原固定 60 s 路径；
- `time.mode: event_driven` 使用确定性优先事件队列推进到下一有效决策机会；
- event-driven transition 保存真实 `delta_t`，折扣为 `gamma_reference ** (delta_t / 60)`；
- Replay、Double-DQN target、PER priority、可选 n-step 聚合均使用 transition-specific discount；
- 当前仓库的 Rainbow 输出为标量 Q，并未实现 Distributional C51。阶段二提供并测试了 SMDP C51 projection，但依照“不得修改网络结构”的边界没有将 C51 强行接入现有标量网络；
- 29 项测试全部通过，其中阶段二 19 项（新增 fixed/event 奖励守恒与同权重校验）、阶段一插入/能耗测试 9 项、固定 60 s 黄金回归 1 项。

20 个共享环境 seed 的确定性候选得分策略对比显示，Event-SMDP 的平均决策数从 88.60 降至 25.55（-71.16%），forced-wait rate 从 0.823 降至 0.221（-73.12%）。完成率、超时率和响应时间均改善，但 UAV 单位完成任务能耗、回收等待和回收拥堵恶化。因此阶段二在“决策效率”和“响应性能”上有明确研究价值。

最新正式实验已将 fixed 60 s 与 Event-SMDP 都重新训练为各 10000 个环境 transition epoch，并在每个 episode 从同一个可复现随机数流中采样新的场景 seed。两组使用完全相同的奖励组成、权重、网络、优化器、PER、基础 seed 与训练 transition 数，仅时间推进和 bootstrap discount 模式不同。20 个独立共享测试 seed 上，Event-SMDP 完成率 0.7939→0.8226、超时率 0.1181→0.0849、平均响应时间 454.4 s→363.3 s、episode reward 736.7→973.8；同时单位完成任务 UAV 能耗增加 10.6%，仍需诊断能耗和恢复拥堵。新权重已覆盖先前交付的模型路径，完整结果位于 `results/stage2_rainbow_5000epoch_comparison/`。

建议进入阶段三，但应把“事件触发导致更积极派遣、回收队列拥堵和能耗增加”列为阶段三首要问题；不能把本报告的启发式环境对比或 80-transition smoke training 当作最终论文结果。

## 2. 修改前代码映射

| 功能 | 文件 | 类/函数 | 原固定 60 s 逻辑 | 阶段二处理 |
|---|---|---|---|---|
| Gym reset/step | `rl/env_wrapper.py` | `DynamicPatrolEnv.reset/step` | step 内推进到 `now + decision_interval` | `step` 按配置分派到 fixed/event 两条路径 |
| 固定决策循环 | `env/simulator.py` | `Simulator.run` | `next_decision += decision_interval` | 保留，确保启发式固定模式回归 |
| 底层时间推进 | `env/simulator.py` | `Simulator.step` | `dt_s` 小步推进 | event 模式按下一个事件精确传入可变 dt |
| CAV 状态 | `env/vehicle.py` | `Vehicle.set_target/step` | 沿最短路逐边更新 | 保留物理逻辑；增加 `schedule_version` |
| UAV 状态 | `env/simulator.py`, `env/drone.py` | `_advance_drone`, `assign_sortie`, `reset_after_recovery` | 飞行/服务/等待/回收 | 保留；增加 UAV `schedule_version` |
| 任务释放/超时 | `env/task_pool.py` | `update_release_and_expiry` | 每物理步扫描 | 事件队列安排 release/timeout 边界，状态仍由原函数更新 |
| 任务完成 | `env/simulator.py` | `_advance_drone` | service segment 完成时 `mark_done` | 保留，并映射为完成类事件 |
| UAV 回收 | `env/simulator.py` | `_drone_arrived_recovery`, `_recover_drone` | 到点等待或立即回收 | 保留；event 模式使用真实事件时间 |
| 阶段一恢复插入 | `env/simulator.py` | `dispatch_sorties → evaluate_insertion/commit_insertion` | 动态插入或 FIFO | 完整保留；新增独立 recovery schedule version |
| 候选架次 | `env/simulator.py` | `get_candidates` | 每个决策点生成 | 每个有效事件批处理后重新生成，主体逻辑不变 |
| Action mask | `rl/env_wrapper.py` | `action_masks` | wait + feasible candidates | 逻辑不变；仅在状态更新后重算 |
| Reward | `rl/env_wrapper.py` | `_reward`, `_event_reward_components` | 阶段一旧配置仍可使用原按决策采样奖励 | 正式 fixed/event 配置均使用同一套区间增量奖励；旧 fixed 路径保持兼容 |
| Weighted-AoI | `env/simulator.py` | `_integrate_aoi` | 左端采样乘 dt | fixed 保留；event 使用线性 AoI 解析积分 |
| UAV 能耗 | `env/simulator.py` | `_consume` | 按实际物理 dt 扣除 | 保留；event reward 只取 transition 能耗增量 |
| Replay | `rl/replay_buffer.py` | `MaskedReplayBuffer`, `PrioritizedMaskedReplayBuffer` | 无时间字段 | 保存 delta、discount、终止类型、决策时间和即时派遣标记 |
| One-step target | `rl/train_masked_dqn.py` | `compute_dqn_loss` | 固定 gamma | 使用 batch `bootstrap_discounts` |
| n-step | 修改前不存在 | — | — | 新增可配置 `NStepSMDPAccumulator`；正式配置为 1-step |
| C51 | 修改前不存在，网络为标量 Q | — | — | 新增并测试独立 SMDP projection；未改变网络结构、未激活 |
| PER priority | `rl/replay_buffer.py`, `rl/train_masked_dqn.py` | `update_priorities` | 基于固定-gamma TD-error | 基于 SMDP target 的 TD-error |
| 训练统计/终止 | `rl/train_masked_dqn.py` | `train_masked_dqn` | 以 transition step 计数 | 增加模拟秒、replay 数、梯度更新和 delta 分组 TD-error |

审计开始时 `git status --short` 为空；阶段一日志和 `results/stage1_dynamic_recovery/` 未被覆盖或删除。

## 3. 事件驱动机制

### 3.1 System Event 与 Decision Epoch

实现支持的事件边界包括：

- `TASK_RELEASE`、`TASK_TIMEOUT`；
- `UAV_TASK_ARRIVAL`、`UAV_SERVICE_COMPLETE`、`UAV_RECOVERY_ARRIVAL`、`UAV_RECOVERED`；
- `CAV_NODE_ARRIVAL`、`CAV_RECOVERY_NODE_ARRIVAL`；
- `UAV_LAUNCH`（动作即时事件）；
- `TASK_SERVICE_COMPLETE`（由原 simulator event log 映射）；
- `EPISODE_END`。

现有配置中 launch/recovery duration 都是 0，因此没有虚构 `LAUNCH_COMPLETE` 或 `RECOVERY_COMPLETE` 延时。任务 service start 仍由原 segment 转换记录，物理状态机没有复制为第二套系统。

每个 system-event 批次处理后重新生成候选与 mask。只有 `mask[1:]` 至少一个非 wait 动作有效时才返回智能体；只有 wait 有效时自动继续到未来事件。Reset 保留首次 Gym 决策语义。

### 3.2 事件队列与确定性

`env/events.py` 使用 `heapq`，排序键为 `(event_time, priority, sequence_id)`。同刻优先级为：episode end、完成类、到达类、timeout、release、launch marker。`sequence_id` 单调递增，同 seed 下顺序稳定。

同刻事件通过 `pop_same_time` 一次弹出，物理状态完成更新后最多检查一次决策。队列统计：

- `stale_event_count`；
- `duplicate_event_count`；
- `invalid_time_event_count`。

20-seed 实验中三项总异常计数均为 0。`EPISODE_END` 是 horizon 内最终队列边界，horizon 后的 release/timeout 不入队。

### 3.3 版本控制

CAV 和 UAV 各有 `schedule_version`。重新指派 UAV、完成回收或 CAV 设置新路径时递增；事件弹出前检查版本。恢复柔性队列另有 `recovery_schedule_version`，不会错误地使已经锁定的 CAV 当前道路边事件失效。

### 3.4 同刻多 UAV 派遣

采用方案 B。当前物理模型 `launch_time_s: 0`，因此允许有限个 `delta_t=0` 连续派遣：

- 每次实际消耗一架 idle UAV；
- `transition_discount=1`；
- 不累计时间率成本；
- 选择 wait 后强制推进到未来事件；
- 上限默认为 UAV 数量；
- transition 标记 `instantaneous_dispatch=true`。

测试确认不会产生无限零时间循环。

## 4. SMDP 奖励与折扣

### 4.1 Reward 分类

| 奖励分量 | 原方式 | 类型 | Event-SMDP 方式 |
|---|---|---|---|
| 完成奖励、priority bonus | 决策间新增完成任务 | 离散事件 | 完成事件只计一次 |
| timeout penalty | 决策间新增过期任务 | 离散事件 | timeout 事件只计一次 |
| illegal penalty | 非法动作 | 即时事件 | 动作时只计一次 |
| wait shaping | 每次 wait 决策 | 时间率 shaping | 乘 `delta_t / 60`；零时长不重复扣 |
| recovery queue penalty | 每次决策采样 | 时间率成本 | 区间首尾队列长度均值乘 `delta_t / 60` |
| vehicle recovery time penalty | 每次决策采样 | 时间率成本 | 区间首尾预测时间均值乘 `delta_t / 60` |
| Weighted-AoI | 累计 AoI 被反复用于 reward | 时间积分 | transition AoI 积分增量除以 60 |
| UAV energy | 累计能耗被反复用于 reward | 物理增量 | 只扣 transition 的真实能耗增量 |
| terminal undone | episode 终点 | 离散终止 | 终止时只计一次 |

所有权重保持配置原值。阶段一 `predicted_hover_energy` 仍只用于插入评估，不进入真实 battery 或 reward。测试验证 reward components 之和严格等于 transition reward。

### 4.2 Weighted-AoI

固定模式继续使用阶段一积分以保持黄金回归。事件模式对没有重置的区间使用：

`age_start * delta_t + 0.5 * delta_t^2`

任务服务完成位于事件边界，积分在边界切分，随后更新 `cover_last_time`。测试覆盖 10→70 的 60 s 积分和 30 s 中间重置。

### 4.3 Transition discount

正式事件配置：

`d_n = 0.99 ** (delta_t / 60)`

测试值：0 s→1，30 s→`sqrt(0.99)`，60 s→0.99，120 s→`0.99^2`。固定模式 replay 使用 `delta_t=60` 和 0.99，以避免把事件 transition 静默当成固定步。

## 5. Replay、TD target、n-step、C51 与 PER

Replay schema version 为 2，保存 observation/action/reward/next observation、terminated、truncated、delta_t、bootstrap discount、双 action mask、decision time、trigger event type 和 instantaneous flag。Event 模式缺少时间字段会直接抛错，不做静默 fallback。Checkpoint 写入 `replay_config`；旧 checkpoint 加载时发出明确 warning。

Double-DQN 的 next action 仍由 online network 在 next mask 内选择，target network 只评价该合法动作：

`target = reward + (1-done) * bootstrap_discount * next_q`

PER 继续使用原 absolute TD-error priority，但该 TD-error 已来自 SMDP target。

可选 n-step 累积使用每个 transition 的折扣乘积，并输出累计 reward、总 delta、bootstrap discount 和末端 mask。当前仓库修改前没有 n-step，正式 B1/B2 均配置 `n_step: 1`，避免将阶段二变成算法结构消融；3-step 数学路径已单测。

当前 Rainbow network 的 head 为标量 Q，不是 C51 atoms。`rl/smdp.py::c51_project_distribution` 已按 `reward + (1-done)*D*z` 实现并测试 terminal/可变 discount，但未在本阶段修改冻结的网络结构。因此报告中的“C51 测试通过”表示 projection 数学正确，不表示当前训练启用了 C51。

## 6. 测试结果

运行命令：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m unittest tests.test_stage2_event_smdp tests.test_recovery_insertion -v
```

结果：28/28 passed，耗时 16.337 s。

- 固定 60 s 黄金回归：passed；seed 0 仍为 completed=20、expired=4、Weighted-AoI=615.6631987447494；
- 事件顺序、同刻 batch、确定性、stale event：passed；
- variable discount、one-step、n-step、C51、terminal：passed；
- only-wait 自动推进、多 UAV 零时长上限：passed；
- 动态插入候选评估、队列插入、CAV 有向边路线、UAV 回收：passed；
- reward conservation、预测/实际能耗不重复：passed。

## 7. 环境级验证与 20-seed 对比

3-seed 首轮验证和 20-seed 配对均使用相同 seed、相同地图/任务/候选/插入权重以及同一确定性 candidate-score action script。两种模式全部推进到 3600 s；没有时间回退、负时间、重复事件或事件队列死锁。

| 指标 | Fixed 60 s Mean | Event-SMDP Mean | 配对变化 | 95% CI（差值） | paired t p |
|---|---:|---:|---:|---:|---:|
| Completion rate | 0.810 | 0.872 | +7.71% | [0.0257, 0.0992] | 0.0021 |
| Timeout rate | 0.117 | 0.0657 | -43.72% | [-0.0733, -0.0287] | 0.00013 |
| Avg response time (s) | 413.23 | 345.81 | -16.32% | [-91.27, -43.57] | 1.07e-5 |
| High-priority response (s) | 285.06 | 256.01 | -10.19% | [-51.01, -7.10] | 0.0122 |
| Weighted-AoI | 589.15 | 583.41 | -0.97% | [-10.17, -1.30] | 0.0139 |
| UAV energy/completed task | 219.62 | 260.76 | +18.73% | [27.96, 54.32] | 2.94e-6 |
| Mean UAV recovery wait (s) | 119.25 | 177.01 | +48.43% | [31.60, 83.91] | 0.00019 |
| CAV wait time (s) | 405.50 | 249.45 | -38.48% | [-293.72, -18.39] | 0.0284 |
| CAV travel distance (m) | 23416 | 25197 | +7.61% | [597, 2965] | 0.0053 |
| Non-append rate | 0.125 | 0.186 | +49.51% | [0.0094, 0.1139] | 0.0231 |
| Decisions/episode | 88.60 | 25.55 | -71.16% | [-64.89, -61.21] | 1.42e-24 |
| Forced-wait rate | 0.823 | 0.221 | -73.12% | [-0.630, -0.574] | 1.10e-20 |
| Mean delta_t (s) | 40.65 | 143.21 | +252.33% | [93.78, 111.35] | 8.11e-16 |
| P95 delta_t (s) | 60.00 | 329.56 | +449.26% | [254.80, 284.31] | 1.94e-19 |

固定模式 mean delta 小于 60 是因为阶段一已存在物理零时长同节点多 UAV 连续派遣；非零固定间隔的 P95 仍为 60 s。

动态插入没有退化为 append-only：Event-SMDP non-append rate 为 0.186，并显著高于 fixed 的 0.125。但更积极的及时派遣提高了 recovery contention（0.390→0.698）、hover energy（1117.7→2065.1）和 UAV recovery wait，这是能耗恶化的主要机制性解释。

## 8. 短训练与旧模型 smoke test

固定与事件模式各运行 seed 41/42/43，每个 80 replay transitions，网络、optimizer、batch、PER 等配置相同。每个 run 有 49 次 gradient update，训练链路、checkpoint schema 和 TD-error 日志均正常。该预算远低于建议的完整预算 10%–20%，用途只是验证训练可运行：

- Fixed 累计模拟时间：10,420 s，完成/部分 episode 共 3；
- Event 累计模拟时间：31,185.8 s，完成/部分 episode 共 11；
- 两者 transition 数相同但模拟时间不相同，不能用最终 eval 值作公平性能结论。

Event 的聚合 mean absolute TD-error 随 delta 增大：0–30 s=33.59，30–60 s=33.45，60–120 s=42.35，>120 s=109.13。长 transition 明显更难学习；阶段三应采用相同累计模拟时间预算做更长训练，并监控长 delta 样本的 PER 占比。

旧 fixed-interval Rainbow checkpoint 在 observation/action dimension 不变的情况下成功完成 event-mode smoke test，并发出 legacy SMDP metadata warning。seed 0 completion=0.290、timeout=0.258；这只说明接口可运行，低结果反映分布偏移，不能作为 Event-SMDP 最终性能。

## 9. 验收判断

实现正确性：

- 固定模式黄金回归：通过；
- 事件不遗漏/不重复、确定性：通过；
- delta、discount、reward scale：通过单测；
- one-step、n-step、C51 projection、PER target：通过；
- 动态恢复插入兼容：通过；
- 无限零时间循环：未出现；
- 20-seed 中 duplicate/invalid/stale 异常总数：0/0/0。

研究价值：满足条件 B（决策数减少 71%、forced-wait 减少 73%，性能没有整体不变而是多数响应指标改善），也满足条件 A 的 completion/timeout/response 改善。但能耗、UAV recovery wait 和 contention 明显恶化，不能宣称全面优于 fixed 60 s。

进入阶段三建议：**是，带条件进入**。阶段三优先事项应是等累计模拟时间的正式 Rainbow 训练、长 delta TD-error 稳定性、以及在不改变奖励权重/候选公式/插入成本函数前提下分析更积极派遣为何造成恢复拥堵。

## 10. 文件与运行方式

新增核心文件：

- `env/events.py`；
- `rl/smdp.py`；
- `configs/scenario_60node_stage2_fixed_dynamic.yaml`；
- `configs/scenario_60node_stage2_event_dynamic.yaml`；
- `tests/test_stage2_event_smdp.py`；
- `experiments/run_stage2_event_smdp.py`；
- 本报告。

修改：`env/drone.py`、`env/vehicle.py`、`env/simulator.py`、`rl/env_wrapper.py`、`rl/replay_buffer.py`、`rl/train_masked_dqn.py`、`experiments/evaluate_masked_rl_models.py`。

固定模式训练：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m dynamic_patrol_marl.rl.train_masked_dqn --config configs/scenario_60node_stage2_fixed_dynamic.yaml
```

事件模式训练：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m dynamic_patrol_marl.rl.train_masked_dqn --config configs/scenario_60node_stage2_event_dynamic.yaml
```

3-seed 环境验证：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m dynamic_patrol_marl.experiments.run_stage2_event_smdp --paired-seeds 0:3 --skip-short-training --out-dir results/stage2_event_driven_smdp_validation
```

短训练（3 seed）：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m dynamic_patrol_marl.experiments.run_stage2_event_smdp --paired-seeds 0:3 --short-train-seeds 41,42,43 --short-train-steps 2000
```

20 个配对 seed：

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m dynamic_patrol_marl.experiments.run_stage2_event_smdp --paired-seeds 0:20 --skip-short-training
```

全部结果位于 `results/stage2_event_driven_smdp/`，包括逐 seed、paired statistics、timeline、transition/reward logs、delta/TD-error、training summary、checkpoint smoke 和论文格式 PNG。
