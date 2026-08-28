# 动态 CAV–UAV 协同巡检：研究背景、研究现状与实验拓展路线

> 更新时间：2026 年 8 月  
> 对应项目：Dynamic CAV-UAV / dynamic-patrol  
> 当前基线：新街口真实路网、1 辆 CAV、3 架 UAV、动态任务、事件驱动 SMDP、带动作掩码的 Rainbow 风格 DQN、三维体素 A* 航路规划

## 1. 研究背景

城市道路巡检、灾后勘察、交通事件响应和基础设施维护具有三个共同特征：任务在空间上分散、到达时间不确定，而且不同任务具有不同优先级、截止时间和服务时长。单独使用地面车辆时，车辆受到道路拓扑与拥堵约束；单独使用 UAV 时，又受到续航、载荷、起降和回收限制。将 CAV/UGV 作为移动补给与回收平台、由 UAV 执行快速空中巡检，可以同时利用地面平台的续航能力和无人机的空间机动性。

这一问题可以追溯到 truck–drone routing 和 Flying Sidekick TSP。Murray 与 Chu 最早系统提出由卡车发射并回收无人机的联合路径与调度模型，并指出该问题是在 TSP/VRP 基础上增加同步、续航与会合约束的 NP-hard 组合优化问题（[Murray & Chu, 2015](https://doi.org/10.1016/j.trc.2015.03.005)）。后续研究逐渐从静态配送扩展到监视巡检、动态需求、随机交通、多无人机协同和灾后响应。

本项目进一步包含以下耦合：

1. CAV 只能沿有向道路网络运动；
2. UAV 需要在三维城市建筑环境中规划可行航路；
3. UAV 必须满足电量、起飞、服务和回收约束；
4. CAV 的路径变化会影响未来 UAV 回收位置和等待时间；
5. 任务动态释放，并具有优先级、截止时间和服务时长；
6. 决策间隔不是固定的，而由任务释放、车辆到达、无人机完成或回收等事件触发。

因此，这不是单纯的 VRP、无人机路径规划或强化学习问题，而是动态随机环境中的空地异构多智能体协同调度问题。

## 2. 国内外研究现状

### 2.1 静态 truck–drone 联合路径优化

早期研究通常假定全部客户或巡检点预先已知，通过 MILP、动态规划、分支定界、遗传算法、禁忌搜索、VNS、ALNS 等方法联合确定卡车路径、无人机 sortie 和会合位置。近年的系统综述表明，研究主题已经从单卡车单无人机扩展到多无人机、时间窗、同时取送、能耗、移动会合和多层网络，但大量工作仍建立在静态或确定性假设上（[2025 systematic review](https://doi.org/10.1016/j.swevo.2024.101825)，[2024 truck–drone survey](https://www.mdpi.com/2504-446X/8/10/550)）。

静态优化的优点是解的可解释性强，并且可以提供最优下界；缺点是计算量随任务和候选会合点快速增长，而且当新任务或交通状态变化时需要重新求解。对本项目而言，静态算法更适合作为小规模最优基准或滚动时域内的局部改进器，而不是唯一的在线决策器。

### 2.2 动态、随机和在线路径规划

现实任务中的需求、交通、天气和服务时间通常是随机的。动态 VRP 研究已经从“周期性批量重规划”转向“事件发生时立即决策”。例如，动态请求 VRP 可以被表述为 route-based MDP，并由价值函数学习与模拟退火组合实现快速在线重规划（[Joe & Lau, ICAPS](https://doi.org/10.1609/icaps.v30i1.6685)）。2025 年的对比研究进一步指出，动态 VRP 中价值型、策略型、线性和神经网络策略的设计选择会显著影响性能，因此算法必须在统一实例和统一计算预算下比较（[CIE 2025](https://doi.org/10.1016/j.cie.2024.110747)）。

面向 truck–drone 系统，近期工作开始加入动态交通、随机任务和灾后多类型需求。例如 VRPD-DT 将时变交通预测嵌入卡车–无人机联合路径搜索（[Imran & Won, 2024](https://arxiv.org/abs/2404.09065)）；动态 truck–drone RL 已被用于同时处理救援物资配送与新出现的侦察任务（[Sun & Zhang, 2024](https://hub.hku.hk/handle/10722/353608)）。

当前仍缺少的内容包括：统一的动态 benchmark、真实城市三维障碍、动态会合失败、任务分布漂移以及决策算法在未见城市上的泛化验证。

### 2.3 事件驱动 SMDP

固定时间步会在“没有任何事件发生”的区间重复产生无意义决策，同时可能错过异步任务与车辆状态改变的真实时间结构。事件驱动建模让决策时刻与请求到达、车辆空闲、任务完成等事件一致，天然形成半马尔可夫决策过程（SMDP）。已有动态车辆调度研究证明，事件驱动 SMDP 能显著压缩决策空间，并可结合离散事件仿真和 Double DQN 处理连续时间调度（[Cordeiro & Pitombeira-Neto, 2023](https://arxiv.org/abs/2307.07508)）。异步调度的后续研究也报告了相对批处理策略更低的延误与取消率（[Artificial Intelligence for Transportation, 2025](https://doi.org/10.1016/j.ait.2025.100038)）。

本项目已经实现事件驱动仿真并在 replay 中保存 `delta_t`，这是明显优势。但当前正式训练的 `discount_mode` 仍为 `fixed_decision`，`n_step=1`。也就是说，不同持续时间的 transition 在价值折扣上仍近似等价，这与严格 SMDP 的时间语义并不完全一致，是最值得深入的算法切入点之一。

### 2.4 深度强化学习与神经组合优化

神经组合优化逐渐从 Pointer Network 转向 attention、Transformer 和 GNN。Attention Model 能在 TSP、VRP、OP 和随机 Prize-Collecting TSP 上学习构造式启发策略，并表现出跨实例泛化能力（[Kool et al., ICLR 2019](https://mlanthology.org/iclr/2019/kool2019iclr-attention/)）。近期动态 VRP 工作进一步将 GNN、Transformer 和记忆结构用于动态需求与交通不确定性，并通过 action masking 保证可行性（[Operations Research Perspectives, 2025](https://doi.org/10.1016/j.orp.2025.100351)）。

对于 truck–drone 问题，已有研究使用 Pointer Network、attention encoder 和 recurrent decoder 学习联合路径（[CIE 2024](https://doi.org/10.1016/j.cie.2024.110659)，[Applied Sciences 2025](https://doi.org/10.3390/app15042007)）。不过，端到端生成完整联合路线通常难以显式满足回收、电量、三维航路和移动会合约束。项目当前采用“候选方案生成 + 可行性掩码 + RL 选择”的混合结构，更容易保证安全约束，也适合与经典优化器组合。

项目使用的 Rainbow 思路来源于将 Double DQN、Dueling、Prioritized Replay、Noisy Network、n-step 和 distributional RL 等改进组合起来（[Hessel et al., AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11796)）。但当前实现只有 Double、Dueling、Noisy 与 PER，没有 C51/QR 分布式价值头，且 `n_step=1`，所以学术写作中应称为“Rainbow-style DQN”，除非补齐并消融全部关键组件。

动作掩码在候选动作大量不可行时尤为重要。已有研究从理论和实验上说明了 invalid action masking 的合理性，并指出不可行动作比例越高，掩码的收益通常越明显（[Huang & Ontañón, 2020](https://arxiv.org/abs/2006.14171)）。本项目已经实现动作掩码，但仍应通过 masked/unmasked 对照量化它对非法动作率、样本效率和最终性能的贡献。

### 2.5 三维 UAV 航路、能耗与城市障碍

城市 UAV 航路不仅要最短，还要满足建筑净空、飞行高度、禁飞区、转角可行性和能耗限制。三维体素建图能统一表示建筑和动态空间约束；近期工作也在探索体素环境与 DRL 的结合（[ISPRS 2024](https://doi.org/10.5194/isprs-archives-XLVIII-4-2024-311-2024)）。路径规划综述指出，复杂动态环境下单一确定性算法很难同时兼顾最优性、实时性与多约束，因此混合启发式、分层规划和在线重规划仍是重要趋势（[Artificial Intelligence Review, 2024](https://doi.org/10.1007/s10462-024-10913-0)）。

本项目采用 3D voxel A*、26 邻域、禁止穿角、道路净空和有界搜索，具有比二维欧氏距离更真实的航路代价。但现阶段环境仍以静态建筑为主，尚未系统考虑风场、动态禁飞区、GPS/通信误差、感知延迟和电池退化。

### 2.6 从仿真走向可靠部署

前沿空地协同研究正从纯路径算法转向完整自治栈，包括 UAV/UGV 联合路由、移动充电/回收、ROS 2、PX4/MAVSDK 和户外场景验证（[Mondal et al., 2026](https://arxiv.org/abs/2607.07350)）。这表明高水平研究评价不再只看平均距离或奖励，还要考虑安全约束、失败恢复、感知通信、实时计算和 sim-to-real 鲁棒性。

## 3. 当前项目的研究定位

与现有工作相比，本项目已经具备以下组合优势：

- 使用真实新街口 333 节点路网，而非纯欧氏随机点；
- 同时考虑 CAV 有向道路、3 架 UAV、电量与移动回收；
- 任务具有动态释放、优先级、截止时间和不同服务时长；
- 使用事件驱动而非固定间隔决策；
- RL 不直接生成任意路线，而是在有界联合候选计划中选择；
- 对非法动作进行硬掩码，并在执行前重新验证三维航路；
- 包含 reward ledger、replay 事件组成和规划耗时诊断。

因此，本项目适合定位为：

> 面向城市动态道路巡检的事件驱动 CAV–UAV 协同决策框架，在真实路网和三维航路约束下，通过可行候选计划与深度强化学习实现在线任务分配、地面主动路由和空中移动回收。

## 4. 已完成 100k 基线暴露出的关键问题

正式运行完成 100,000 个 transition、99,001 次梯度更新。20 个评估种子上的最终平均完成率为 96.88%，超时率为 2.59%，平均奖励为 1833.43，且没有非有限 loss、非有限梯度或 commit failure。

但学习曲线说明“最终 checkpoint 未必是最佳 checkpoint”：

| Step | 平均奖励 | 完成率 | 超时率 | 平均响应时间/s | 能耗 |
|---:|---:|---:|---:|---:|---:|
| 10k | 1515.58 | 92.75% | 7.25% | 336.15 | 16349 |
| 30k | 1857.16 | 97.68% | 2.32% | 382.20 | 13595 |
| 60k | **1885.27** | **97.99%** | **1.40%** | 401.78 | 13663 |
| 80k | 1768.43 | 95.63% | 3.55% | 496.01 | 12891 |
| 100k | 1833.43 | 96.88% | 2.59% | 442.11 | 12721 |

这表明：

1. 30k–70k 后奖励与完成率进入平台期，后期存在波动或策略漂移；
2. 能耗持续改善，但平均响应时间在后期恶化，说明标量奖励内部存在权衡；
3. 最终策略约 58.2% 的地面联合动作是 `wait`，约 58.4% 的选择为 type-II，动作分布可能存在结构性偏置；
4. 训练只使用固定 seed 42，而评估使用 20 个种子，尚不能证明对训练随机性、任务分布或新城市具有稳健泛化；
5. 最大原始梯度范数较高，虽有裁剪且没有 NaN，但值得研究 reward scale、PER 极端样本与 target instability；
6. 目前没有多个独立训练 seed 的均值、置信区间和显著性检验，也缺少同计算预算下的强基线。

## 5. 建议实验：按优先级排序

### A 级：必须先做的可信性实验

#### A1. 多训练种子重复实验

- 训练种子：`[0, 1, 2, 3, 4]`，条件允许扩展到 10 个；
- 所有策略使用相同的 50 个评估种子；
- 报告 mean、standard deviation、95% bootstrap CI；
- 对奖励、完成率、超时率、响应时间和能耗做配对检验；
- 同时报告训练时间和在线决策时间。

这是将“一个成功运行”升级为“可信实验结论”的最低要求。

#### A2. Checkpoint 选择实验

将 10k–100k 的 10 个 checkpoint 全部放在相同的 50–100 个 common-random-number 场景上评估。比较：

- 最大奖励 checkpoint；
- 最大完成率 checkpoint；
- 最小高优先级响应时间 checkpoint；
- 最低能耗 checkpoint；
- 最终 checkpoint。

当前结果提示 60k checkpoint 很可能比 100k 更好。该实验成本低、信息量高，应最先执行。

#### A3. 算法组件消融

统一训练预算比较：

1. DQN；
2. Double DQN；
3. Dueling Double DQN；
4. + PER；
5. + Noisy Network；
6. + action masking；
7. 当前 Rainbow-style；
8. 完整 Distributional Rainbow。

动作掩码消融应额外报告非法动作、候选浪费率、commit failure 和训练样本效率。

#### A4. 与非学习基线公平比较

项目已有 nearest-task、priority-greedy、EDF、score-based、rolling-horizon、static-reopt 和 random-masked，可进一步增加：

- ALNS/VNS 滚动时域；
- 小规模 MILP/CP-SAT 最优或近优解；
- myopic one-step lookahead；
- Monte Carlo rollout / Multiple Scenario Approach。

所有方法必须使用相同任务流、三维航路代价、硬约束和计算时间预算。

### B 级：最有论文价值的算法实验

#### B1. 时间一致的 Distributional SMDP Rainbow（首选）

当前 replay 已存储 `delta_t`，可将固定一步折扣改为：

\[
\Gamma_t = \gamma^{\Delta t / \tau_{ref}}
\]

并实现 SMDP n-step return：

\[
G_t^{(n)} = r_t + \Gamma_t r_{t+1} + \Gamma_t\Gamma_{t+1}r_{t+2}+\cdots
\]

实验因素：

- fixed-decision discount vs time-aware discount；
- `n = 1, 3, 5`；
- scalar Q vs C51 vs QR-DQN；
- 均匀 replay vs PER；
- 不同事件持续时间分箱下的 TD error 校准。

创新点是让价值学习与事件驱动连续时间严格一致，而不是简单更换网络。

#### B2. 图注意力/GNN 状态编码

当前网络使用固定长度 1025 维向量和候选特征。可将道路、任务、CAV、UAV 和候选会合点表示为异构图：

- road node / task / CAV / UAV / rendezvous 五类节点；
- 道路连通、任务所在边、UAV 可达、CAV ETA 和回收匹配作为边；
- 用 GAT/Graph Transformer 编码，再对候选 joint plan 打分。

实验应验证：

- 在 333 节点训练，迁移到不同任务数或不同城区；
- 对节点编号置换是否不敏感；
- 小图训练到大图测试的尺度泛化；
- 相同参数量下与当前 MLP 的比较。

#### B3. 约束与风险敏感强化学习

将以下指标从软惩罚改为显式约束：

- 超时率不高于阈值；
- UAV 最低剩余电量；
- 高优先级响应时间；
- 回收不可行概率；
- A* 超时或规划预算超限率。

可比较 reward scalarization、Lagrangian constrained RL、CVaR/quantile risk 和 lexicographic policy。该方向能回答“平均性能好但极端场景是否安全”的问题。

#### B4. RL + 运筹优化混合框架

让 RL 负责事件级高层决策，ALNS/MILP/VNS 负责候选内部优化或局部修复：

- RL 选择 CAV 目标区、任务簇和 UAV 数量；
- 优化器计算具体 sortie、顺序和会合点；
- 学习价值函数作为搜索剪枝或候选排序器；
- 使用 imitation learning 从小规模最优解预训练。

这通常比完全端到端 RL 更容易满足约束，并能提供可解释的质量界。

#### B5. 分层或多智能体决策

可将当前单一 joint-plan 动作拆成：

1. 高层 CAV 路由策略；
2. 中层任务分配策略；
3. 低层 UAV sortie 与回收策略。

比较 centralized single-agent、CTDE MARL、parameter sharing 和通信受限 MARL。多智能体巡检研究已开始学习通信协议并考察故障容错（[Tong et al., 2024](https://arxiv.org/abs/2402.10222)），但空地异构移动回收仍有较大空间。

### C 级：环境真实性与泛化实验

#### C1. 不确定性与分布漂移

- CAV 边旅行时间加入时变拥堵和随机扰动；
- UAV 速度/能耗受风速风向影响；
- 服务时间服从随机分布；
- 任务到达率产生 rush-hour、突发事件和重尾分布；
- 训练与测试的任务率、优先级和空间热点不同。

建议报告 in-distribution、mild shift、severe shift 三档结果。

#### C2. 动态障碍和规划失败

- 动态禁飞区；
- 临时高楼施工、救援直升机走廊；
- voxel map 误差；
- A* 时间预算随机缩短；
- 被选 sortie 在执行前失效。

比较重新选择下一候选、局部重规划、保守风险掩码和 learned feasibility predictor。

#### C3. 通信与观测不完整

- CAV–UAV 通信延迟、丢包和带宽限制；
- 任务优先级或截止时间有观测误差；
- UAV 电量估计偏差；
- 用 RNN/Transformer belief state 将 MDP 扩展为 POMDP。

#### C4. 规模扩展

实验维度：

- CAV 数量：1、2、4；
- UAV 数量：1、3、6、10；
- 任务量：50、100、200、500；
- 路网规模：60、333、600、1000+ 节点；
- 城市：训练城区、邻近城区、完全未见城区。

除解质量外必须报告候选数、规划耗时、内存、每次决策延迟和实时率。

#### C5. 数字孪生与 sim-to-real

远期可将高层策略接入 ROS 2/Nav2 和 PX4/MAVSDK，先做软件在环，再做硬件在环。加入定位噪声、风场、控制延迟和通信中断，验证策略是否仍满足回收和电量安全约束。

## 6. 推荐的统一实验矩阵

| 维度 | 建议水平 |
|---|---|
| 训练种子 | 0–4，正式论文建议 10 个 |
| 评估种子 | 固定 50–100 个 common random numbers |
| 算法 | heuristics、DQN family、PPO/Maskable PPO、完整 SMDP Rainbow、GNN policy |
| 时间建模 | fixed interval、event-driven fixed discount、event-driven time-aware discount |
| 任务强度 | 0.5×、1×、1.5×、2× |
| 动态程度 | 静态、周期任务、Poisson、burst/heavy-tail |
| 交通 | 固定、时变、随机、突发阻断 |
| 航路 | 2D 欧氏、visibility graph、3D voxel A*、动态禁飞区 |
| 安全约束 | 电量、回收、截止时间、规划预算 |
| 地图 | 训练地图、同城新区域、跨城市 |

核心指标分为五组：

1. **任务质量**：完成率、超时率、优先级加权完成率、AoI；
2. **时效性**：平均/95 分位响应时间、高优先级响应时间；
3. **资源**：CAV 距离、UAV 能耗、悬停、等待和每 sortie 任务数；
4. **可靠性**：不可行回收、最低电量、commit failure、A* timeout；
5. **计算性**：训练时间、单决策延迟、候选生成时间、A* 时间和显存/内存。

统计上建议使用 paired common random numbers、bootstrap confidence intervals，并同时报告 effect size；不要只报告单个 seed 的最佳值。

## 7. 三条可形成论文的路线

### 路线一：稳妥且最贴合现有代码

**题目方向：** Time-aware Distributional SMDP Rainbow for Event-driven Dynamic CAV–UAV Patrol

贡献组合：

- 严格时间折扣；
- SMDP n-step；
- C51/QR distributional value；
- action masking + feasible joint-plan candidates；
- 多种子、强基线、真实路网验证。

该路线改动适中，理论叙事完整，最推荐优先做。

### 路线二：算法创新更强

**题目方向：** Heterogeneous Graph Policy for Scalable CAV–UAV Rendezvous and Dynamic Inspection

贡献组合：

- 异构图表示道路、任务、车辆与会合点；
- 图注意力候选打分；
- 尺度和跨地图泛化；
- 与 MLP、attention 和运筹基线比较。

### 路线三：应用与可靠性更强

**题目方向：** Risk-constrained Cooperative Road Inspection under Traffic, Wind and Communication Uncertainty

贡献组合：

- 交通、风场、通信和任务分布漂移；
- CVaR 或 constrained RL；
- 动态航路失效与恢复；
- safety/reliability 指标和数字孪生验证。

## 8. 推荐执行顺序

1. 评估现有 10 个 checkpoint，确认最佳停止点；
2. 完成 5 个训练种子的现有 Rainbow-style 基线；
3. 跑 heuristics、DQN family、masking 和 PER/Noisy 消融；
4. 实现 time-aware discount，先保持网络不变；
5. 增加 n-step 和 C51/QR-DQN；
6. 做任务强度、随机交通和分布漂移实验；
7. 再决定进入 GNN、constrained RL 或 sim-to-real 路线。

最不建议现在直接做的是：只替换一个“更热门”的 RL 算法，然后继续只跑 seed 42。当前最紧缺的不是算法名称，而是可信的多种子基线、时间一致性、强对照与泛化证据。
