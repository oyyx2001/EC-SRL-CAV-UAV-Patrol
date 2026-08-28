# IEEE T-ITS 期刊论文蓝图：动态城市交通巡检中的 CAV-UAV 协同调度

> 版本：2026-08-06  
> 任务边界：本报告只设计题目、论证主线、论文结构和实验计划；未修改项目代码，未启动强化学习训练。  
> 证据标签：**[代码实现]** 当前仓库可直接核查；**[会议论文]** 仅由会议稿报告；**[实验验证]** 当前结果文件或轻量测试可核查；**[建议补充]** 尚无充分实现或实验支撑。

## 0. 执行摘要

当前代码支持的最准确主线不是“纯 candidate-sortie 强化学习”，而是 **event-driven, feasibility-constrained joint CAV-UAV scheduling for dynamic urban traffic inspection（事件驱动、可行性约束的 CAV-UAV 动态城市交通巡检联合调度）**。Candidate sortie 是联合动作构造的底层组件；最终动作是同时包含 CAV 地面动作、UAV sortie 与回收插入评估的 `joint_plan`。**[代码实现]**

第一推荐题目为：

**Event-Driven Joint CAV-UAV Scheduling for Dynamic Urban Traffic Inspection via Feasibility-Masked Reinforcement Learning**  
**基于可行性掩码强化学习的动态城市交通巡检事件驱动 CAV-UAV 联合调度**

该题目比会议稿题目更忠实于当前实现：它不把所有任务窄化为 incident，不把 sortie 误写成完整动作，也不暗示默认启用了动态回收队列重排。期刊论文应把真实路网、三维 UAV 飞行约束、事件驱动 SMDP、联合地空候选计划、物理道路目标去重和时间感知折扣作为实质扩展；算法优越性、消融、敏感性和可扩展性仍需补充实验。**[建议补充]**

---

## 1. 完整论文提纲与逐节写作任务

### Title

- **核心问题：** 标题是否同时准确表达 ITS 应用、联合决策与技术抓手。
- **内容与依据：** 暂用第一推荐题目；避免 `full Rainbow`、`dynamic queue optimization` 和过窄的 `traffic incidents`。**[代码实现]**
- **写作产物：** 主标题不使用副标题也可；若保留副标题，前半句写问题，后半句写方法。
- **来源/新增/衔接/篇幅：** 会议题目只保留 CAV-UAV 与动态巡检语义，其他全部按期刊实现重定；约 12--18 个实词。

### Abstract

- **核心问题：** 在 180--250 词内说明为何难、解决什么、如何解决、用什么证据验证。
- **内容与依据：** 依次写动态异构任务与移动回收耦合、三层模型、candidate-sortie-based joint plans、feasibility mask、event-driven time-aware SMDP、真实路网实验。**[代码实现]**
- **结果约束：** 现在只能预留结果句，不能填“提升 X%”；最终从冻结实验表自动取多种子均值与 95% CI。**[建议补充]**
- **来源/新增/衔接/篇幅：** 问题动机可重写会议稿；joint plan、SMDP、真实路网和实证句为期刊新增；最后一句落到实时交通巡检价值。

### Index Terms

- **核心问题：** 关键词是否覆盖应用、系统、决策模型和学习方法。
- **建议内容：** `connected and automated vehicle (CAV)`, `unmanned aerial vehicle (UAV)`, `traffic inspection`, `cooperative scheduling`, `semi-Markov decision process`, `reinforcement learning`。
- **格式依据：** 采用 Word 参考稿的 `Index Terms—` 格式，并按 IEEE 术语表检查大小写。
- **衔接/篇幅：** 5--7 个词组；`candidate sortie` 可作为方法专名，但不替代 `traffic inspection`。

### Nomenclature / Notation Table

- **核心问题：** 是否需要独立符号表。
- **内容与依据：** 建议不设独立 Nomenclature，在 III-A 放表 I，列集合、索引、时间、能量、队列和 SMDP 符号。**[建议补充]**
- **图表：** `TABLE I: Main Notation`，只收录正文反复使用的 20--30 个符号。
- **衔接/篇幅：** 半栏到一栏；若符号最终超过 30 个，移至 Appendix A。

## I. INTRODUCTION

### A. Traffic Management Motivation and Operational Setting

- **核心问题：** 为什么动态道路巡检值得在 T-ITS 中研究。
- **具体内容：** 区分周期道路状态复查、突发事件确认、拥堵/异常扫描、道路障碍和关键时段巡检；说明时效、覆盖与资源成本并存。
- **依据与素材：** 会议稿问题动机可保留但必须重写；任务类型和时限使用当前配置，不能泛化为真实部署统计。**[会议论文][代码实现]**
- **图表与衔接：** 引出 Fig. 1 系统场景；末段从运营需求过渡到单 CAV—多 UAV 协同的优势与耦合。重要度高，约 0.6 页。

### B. Technical Challenges and Research Gap

- **核心问题：** 为什么现有静态路线、原子动作或固定步长 RL 不足。
- **具体内容：** 组合动作爆炸、三层空间映射、动态截止期、移动回收点/FIFO 拥塞、不规则决策时间、3D 可行性校验。
- **依据与素材：** 用当前 joint-plan 生成流程和事件模型解释难点；外部文献缺口必须在后续检索后用真实引用支撑。**[代码实现][建议补充]**
- **图表与衔接：** 可用一个小型组合规模例子，不填未经证明的大 O 结论；过渡到本文如何分层压缩动作。约 0.7 页。

### C. Proposed Framework and Contributions

- **核心问题：** 本文到底提出什么，与会议版的实质差异是什么。
- **具体内容：** 用一段话概括三层系统、candidate sortie、joint plan、mask、FIFO queue awareness 和 event-driven SMDP；随后列 3--4 条贡献。
- **依据与素材：** 使用第 9 节贡献文本，最终根据完成实验删减；不得写“first”或“optimal”。
- **图表与衔接：** 指向 Fig. 2 总体框架；结尾说明真实路网和多种子验证计划。重要度最高，约 0.7 页。

### D. Paper Organization

- **核心问题：** 读者如何沿章节理解从问题到证据的链条。
- **具体内容：** 一句对应一章：相关工作、模型、形式化、方法、实验、结果、结论。
- **依据与素材：** 与最终目录自动保持一致，定稿最后再写。
- **来源/新增/衔接/篇幅：** 纯期刊组织文字；约 0.15 页，不加入技术主张。

## II. RELATED WORK

### A. Dynamic Traffic Inspection and Patrol Scheduling

- **核心问题：** 动态事件确认、周期巡检和异构截止期任务在 ITS 调度中如何被建模。
- **具体内容：** 分类综述静态/动态巡检、周期重访、事件响应、AoI/SLA 指标，指出多数方法不同时处理移动起降平台和异构任务。
- **依据与素材：** 会议稿参考文献仅作线索，需重新检索 2022--2026 年 T-ITS/TRC/TS/RA-L 等文献。**[建议补充]**
- **图表与衔接：** Table II 相关工作分类；过渡到 CAV-UAV 系统。约 0.8 页。

### B. Cooperative Ground Vehicle-UAV Routing and Scheduling

- **核心问题：** 传统 truck-drone routing 与本文动态交通巡检有何联系和差异。
- **具体内容：** 比较固定客户配送、静态路径、移动会合、多 UAV 回收冲突、建筑飞行约束；强调本文是在线巡检而非一般配送 VRP。
- **依据与素材：** 当前代码提供差异定义，但“现有工作未覆盖”必须有文献矩阵支撑。**[代码实现][建议补充]**
- **图表与衔接：** Table II 增加是否动态、是否多 UAV、是否队列、是否真实路网/3D；引出动作设计难题。约 0.8 页。

### C. Reinforcement Learning with Structured and Masked Actions

- **核心问题：** RL 如何处理组合、动态和受约束动作。
- **具体内容：** 综述 action masking、candidate actions、参数化/层次动作、DQN/PPO 在路由调度中的使用；区分合法性约束与奖励惩罚。
- **依据与素材：** 方法定义来自代码；算法谱系和优缺点需要真实引用。避免把 masking 本身声称为新理论。**[建议补充]**
- **图表与衔接：** 用 Table II 定位本文为 candidate construction + joint plan + mask + SMDP；转入研究空白总结。约 0.8 页。

### D. Positioning of This Study

- **核心问题：** 本文相对三类文献的交集创新是什么。
- **具体内容：** 明确不是 CAV 交叉口控制、车辆博弈、普通 VRP 或以 UAV 局部路径为核心；强调动态交通任务和地空联合调度。
- **依据与素材：** 用一段对比句对应 Contributions 1--3，避免重复 Introduction。
- **图表与衔接：** 结束 Table II 的结论，直接引出三层系统模型。约 0.3 页。

## III. PROBLEM DESCRIPTION AND SYSTEM MODEL

### A. Three-Layer Urban Inspection System

- **核心问题：** CAV 路网、UAV 飞行空间和物理任务目标如何一致关联。
- **具体内容：** 定义有向图 $G=(V,E)$、3D free-space/voxel map、物理道路段集合与稳定 `edge_id` 映射；解释正反向存储边不重复产生目标。
- **依据与素材：** `env/road_network.py`、`env/task_generator.py`、edge registry、3D planner。**[代码实现][实验验证]**
- **图表与衔接：** Fig. 1 三层场景、Table I 符号；从空间结构过渡到任务释放。会议稿三层思想可重写，映射实现为期刊新增。约 1 页。

### B. Heterogeneous Dynamic Inspection Tasks

- **核心问题：** 不同任务如何释放、计时和完成。
- **具体内容：** 定义任务 $i$ 的道路目标、释放时刻、截止时刻、优先级、服务时长、重访周期和完成状态；区分 Poisson、固定时刻和周期批次。
- **依据与素材：** 默认配置中的 5 类任务；明确这些参数是仿真实验设置，不是城市实测统计。**[代码实现]**
- **图表与衔接：** Table III 任务类型与参数，Fig. 3 时间轴；接到地空运行流程。约 0.8 页。

### C. CAV and UAV Operational Processes

- **核心问题：** 一次决策后地面和空中代理如何演化。
- **具体内容：** CAV 路线推进与四类动作；UAV idle/flight/service/return/wait/recovered 状态；起飞、任务序列、会合与排空阶段。
- **依据与素材：** `env/cav_controller.py`、`env/drone.py`、`env/simulator.py`。**[代码实现]**
- **图表与衔接：** Fig. 3 动态事件流程/状态机；下一节把过程转成可行性约束。约 0.8 页。

### D. Assumptions and Scope

- **核心问题：** 模型在哪些边界内有效。
- **具体内容：** 单 CAV、多同构 UAV；已知路网/建筑；仿真生成任务；默认 FIFO 回收；通信可靠性和天气影响未显式建模；飞行服务能耗为配置模型。
- **依据与素材：** 只写代码能验证的假设，不把零起降/换电时间当作一般现实规律。**[代码实现]**
- **图表与衔接：** 可用短列表，不新增图；为 IV 的数学边界和 VII-E 局限埋点。约 0.35 页。

## IV. MATHEMATICAL FORMULATION

### A. Time, Travel, Service, and Energy Quantities

- **核心问题：** CAV 与 UAV 的时间/距离/能量如何可比并进入约束。
- **具体内容：** 定义 CAV 有向最短路时间，UAV 3D 分段时间，服务时长，爬升/巡航/下降与悬停能量，截止余量和电池余量。
- **依据与素材：** 当前路径规划和能耗配置；只形式化已实现量。**[代码实现]**
- **公式与衔接：** Eqs. (1)--(6)；由单段量进入 sortie 定义。会议稿简化公式可继承符号思路，3D 分段需新增。约 0.8 页。

### B. Multi-Task Sortie Feasibility

- **核心问题：** 哪个任务序列和回收点构成可执行 sortie。
- **具体内容：** 定义 $s=(u,v^L,\pi,v^R)$；约束任务唯一性/待处理状态、连接可达、截止期、能量、最大 3 任务以及返回 guard。
- **依据与素材：** `planner/sortie_generator.py` 和默认 candidate-generation 配置。**[代码实现]**
- **公式与衔接：** Eqs. (7)--(12)，配 Fig. 4；为联合计划组合提供可行执行单元。Candidate sortie 基本定义来自会议稿，筛选/3D/guard 为扩展。约 1 页。

### C. FIFO Recovery Synchronization and Congestion

- **核心问题：** 多 UAV 同时回收时，移动 CAV 与 FIFO 队列如何决定实际完成时间。
- **具体内容：** 定义预计到达、队列顺序、前序释放时刻、UAV 等待、CAV 等待、队列占用和既有任务延误；明确动态插入默认关闭。
- **依据与素材：** `env/recovery_scheduler.py`、joint-plan features、reward ledger。**[代码实现]**
- **公式与衔接：** Eqs. (13)--(18)，Fig. 5；会议稿 FIFO 递推可重构，joint-plan queue features 为期刊新增。约 0.9 页。

### D. Joint Scheduling Objective and Constraints

- **核心问题：** 系统希望优化什么，哪些条件不可违反。
- **具体内容：** 采用长期期望加权回报/多指标目标；服务完成与优先级收益对抗超时、AoI、能耗、地面距离和回收等待；列资源互斥、单任务服务、能量、时间和路网约束。
- **依据与素材：** `stage5_coordinated` reward 是可实现代理目标，不把它包装成精确多目标最优模型。**[代码实现]**
- **公式与衔接：** Eqs. (19)--(24)；说明组合动态问题随后用 SMDP 求解。约 0.8 页。

### E. Event-Driven SMDP Formulation

- **核心问题：** 不规则决策时刻下状态、动作、转移和折扣如何定义。
- **具体内容：** 定义决策 epoch $t_k$、状态 $x_k$、动态候选集合 $A(x_k)$、持续时间 $\Delta t_k$、区间奖励和终止/排空规则。
- **依据与素材：** `rl/smdp.py`、event coalescing 与 time-aware discount 测试。**[代码实现][实验验证]**
- **公式与衔接：** Eqs. (25)--(28)，包括 $\gamma_k=\gamma_{ref}^{\Delta t_k/\tau_{ref}}$；从优化模型进入具体求解框架。此节为关键期刊新增，约 0.9 页。

## V. PROPOSED METHOD

### A. Framework Overview and Decision Epochs

- **核心问题：** 每次事件触发后，系统以什么顺序构造和执行动作。
- **具体内容：** 合并同一时刻事件，更新任务/代理/队列，生成 sorties，构造 joint plans，计算 mask，网络选槽，精确物化与提交，再推进到下一事件。
- **依据与素材：** simulator、env wrapper、joint planner。**[代码实现][实验验证]**
- **图表与衔接：** Fig. 2 总框架与 Algorithm 1；从 SMDP 定义进入候选构造。期刊新增，约 0.7 页。

### B. Candidate Sortie Generation and Pruning

- **核心问题：** 如何在保留紧急/高优先级组合的同时控制枚举规模。
- **具体内容：** 任务预筛、最多 3 任务序列、增量截止/能量剪枝、回收点预筛、距离/CAV ETA 缓存、去重和 bounded top-K。
- **依据与素材：** `planner/sortie_generator.py`；默认每 UAV final top-12、完整评估上限 100。**[代码实现]**
- **图表与衔接：** Fig. 4、Algorithm 2；输出 sortie pool 给 V-C。会议稿核心可继承，优化剪枝和 3D 再验证为新增。约 1 页。

### C. Ground-Air Joint Plan Construction

- **核心问题：** 如何把 CAV 路线选择和 UAV sortie 合成固定大小动作集。
- **具体内容：** 构造 keep/wait/move/reroute 地面候选，与 sortie/回收配对；计算 type I--III、会合余量和代价；按配额、排序和去重填入 24 个非安全槽。
- **依据与素材：** `planner/joint_plan.py`、默认 joint-plan 配置、38 维 schema。**[代码实现]**
- **图表与衔接：** Fig. 6 joint-plan 结构、Table IV 计划类型；这是会议 sortie-only 动作的关键期刊扩展，接到 mask。约 1.1 页。

### D. Feasibility Mask and Safe Action

- **核心问题：** 网络如何只在当前可执行的动态动作槽中选择。
- **具体内容：** cache 中存在的有效计划为 1，未填充/不可行槽为 0；safe action 保证非空；mask 同时作用于行为选择和 Double-DQN target argmax。
- **依据与素材：** `rl/env_wrapper.py`、`rl/train_masked_dqn.py`、DQN 网络调用。**[代码实现]**
- **图表与衔接：** Fig. 7 状态—候选—mask 结构；为 V-F 的网络更新提供合法动作集。会议 mask 思想保留，对象从 sortie 扩为 joint plan。约 0.7 页。

### E. State and Candidate Feature Encoding

- **核心问题：** 1025 维状态如何向网络表达交通任务、地空资源和每个候选的差异。
- **具体内容：** 按全局、CAV、UAV、任务/队列和 25×38 candidate block 列出维度、缩放、clip 和缺失值；特别解释 event type、urgency、battery slack、recovery wait、plan type flags。
- **依据与素材：** env wrapper 和 `joint_plan_feature_schema.json`。**[代码实现]**
- **图表与衔接：** Table V 观测维度，Fig. 7 编码示意；避免逐个特征在正文堆公式，完整 schema 放 Appendix。约 0.8 页。

### F. Masked Rainbow-Style DQN with Time-Aware Discounting

- **核心问题：** 如何学习不规则时间下的动态候选动作价值。
- **具体内容：** 说明 dueling Q、Double-DQN selection/evaluation、NoisyNet exploration、PER、1-step replay 和 mask；明确没有 distributional head。
- **依据与素材：** `rl/dqn_networks.py`、`rl/replay_buffer.py`、`rl/train_masked_dqn.py`、`rl/smdp.py`。**[代码实现]**
- **公式与衔接：** Eqs. (29)--(33) masked target 与 PER loss；Algorithm 1 更新部分。与会议 “full Rainbow” 相比必须重写。约 1 页。

### G. Coordinated Reward and Accounting

- **核心问题：** 奖励如何反映任务服务、地空代价和队列拥塞且避免重复记账。
- **具体内容：** 列完成/优先级、超时/AoI、能耗/距离、回收等待/占用、路线改变和失败项；解释 interval-increment ledger、operation 与 drain reward。
- **依据与素材：** 默认 YAML、`rl/reward_ledger.py`、route-change reward 测试。**[代码实现][实验验证]**
- **公式与衔接：** Eq. (34)、Table VI 权重；结果章必须用 reward component 分解验证，而不是只报总 reward。约 0.8 页。

### H. Computational Procedure and Complexity

- **核心问题：** 方法能否在在线决策预算内生成动作并推理。
- **具体内容：** 给出任务预筛、序列枚举、回收点筛选、joint-plan 配对、网络前向和精确路径物化的流程及经验耗时；理论复杂度只按受限参数 $K,L,R,G$ 表示。
- **依据与素材：** 候选上限、规划预算和 `mean_plan_generation_time_ms`；不把平均值当最坏界。**[代码实现][实验验证]**
- **图表与衔接：** Algorithm 1/2 末尾，Table VII 运行时分解；连接实验中的实时性与规模测试。约 0.6 页。

## VI. EXPERIMENTAL DESIGN

### A. Study Area, Scenarios, and Task Configuration

- **核心问题：** 实验环境是否足以复现且符合城市交通巡检语义。
- **具体内容：** 新街口 333/601 路网、建筑/voxel 数据、1 CAV/3 UAV、速度/电池、10,800 s 释放 + 1,800 s 排空、5 类任务参数。
- **依据与素材：** 默认配置、地图数据和任务 registry。**[代码实现]**
- **图表与衔接：** Fig. 8 场景地图、Table VIII 参数；不声称任务率来自实测，过渡到比较方法。约 0.8 页。

### B. Compared Methods

- **核心问题：** 哪些基线能公平检验联合动作和学习价值。
- **具体内容：** Masked Rainbow-style DQN；修复后的 Maskable PPO/Masked DQN；priority/EDF/score/random；真正 joint-plan rolling horizon 与 static re-optimization；会议版 sortie-only 方法。
- **依据与素材：** 当前只有部分 DQN/启发式代码，PPO 和优化基线必须适配相同观测、mask、3D 可行性和计算预算。**[代码实现][建议补充]**
- **图表与衔接：** Table IX 基线定义和公平性；禁止在未实现前用“optimizer”称谓。约 0.7 页。

### C. Research Questions and Experiment Matrix

- **核心问题：** 每组实验对应哪条贡献，而非无目的堆图。
- **具体内容：** RQ1 总体性能，RQ2 表示/mask，RQ3 queue/SMDP，RQ4 鲁棒性与扩展性，RQ5 在线计算，RQ6 调度机制案例。
- **依据与素材：** 使用第 12 节实验矩阵，将已有和必补分开。**[建议补充]**
- **图表与衔接：** Table X RQ—实验—指标映射；为结果章保持同序。约 0.5 页。

### D. Metrics and Statistical Protocol

- **核心问题：** 如何避免单种子和选择性报告。
- **具体内容：** 完成/超时、高优先级响应、平均响应、weighted AoI、censoring；UAV 能耗/返航/等待、CAV 距离/等待、队列占用、计划时间和失败率。
- **依据与素材：** 当前 metric/reward ledger；正式方案至少 5 个独立训练 seed，每个 checkpoint 20 个固定 eval seed，报告均值、95% CI、配对检验和效应量。**[代码实现][建议补充]**
- **图表与衔接：** 预先确定 primary metrics，避免只挑有利指标；定义排空阶段和删失处理。约 0.6 页。

### E. Training, Evaluation, and Reproducibility

- **核心问题：** 如何确保配置、模型和结果能够追溯。
- **具体内容：** 固定 YAML、代码版本、reward signature、map/task/planner version；公开 seed 列表、训练步数、checkpoint 选择规则、硬件与 wall-clock。
- **依据与素材：** `params.json`、`replay_config.json`、progress、checkpoint 和汇总脚本已提供大部分字段。**[代码实现]**
- **图表与衔接：** Table XI 超参数；明确历史 `formal_seed42` 不是最终默认实验，最终重跑后冻结清单。约 0.6 页。

## VII. RESULTS AND DISCUSSION

### A. Overall Performance against Baselines

- **核心问题：** 提出方法是否在任务服务质量和地空成本间取得稳定优势。
- **具体内容：** 先报 primary metrics 与 CI，再报资源/队列/运行时；讨论是否存在 CAV 距离换取 UAV/响应改善。
- **依据与素材：** 当前只有 DQN 历史结果和单 seed SMDP 对照，正式基线表尚缺。**[实验验证][建议补充]**
- **图表与衔接：** Table XII 总体结果、Fig. 9 Pareto/误差条图；从总体结果转入机制消融。约 1.2 页。

### B. Ablation of Candidate Representation, Masking, and Queue Awareness

- **核心问题：** 三个关键机制各自贡献什么，是否只是同一创新重复包装。
- **具体内容：** 独立移除 multi-task sortie、joint CAV coupling、mask、queue features/penalties、time-aware discount；保持训练预算一致。
- **依据与素材：** 现有 reward freezing 不等于这些消融；需要新实验，必要时为无 mask 设计非法动作惩罚与安全统计。**[建议补充]**
- **图表与衔接：** Fig. 10 消融森林图、Table XIII；将机制影响连接到队列和可行性诊断。约 1 页。

### C. Robustness, Sensitivity, and Scalability

- **核心问题：** 策略对 UAV 数量、任务负载、异质性、电池和网络规模是否稳健。
- **具体内容：** 逐组单因素/小型析因设计，报告性能和候选生成/推理时间；区分重训泛化与 zero-shot 测试。
- **依据与素材：** 配置支持大多数参数切换，但网络规模和可变 UAV 数可能改变观测维度，需先确认实现。**[代码实现][建议补充]**
- **图表与衔接：** Fig. 11--12 敏感性与规模曲线；引到机制案例。约 1.2 页。

### D. Scheduling Mechanism and Representative Case Study

- **核心问题：** 策略为何做出某次 CAV 改道、multi-task sortie 或等待决定。
- **具体内容：** 选择固定 eval seed，画任务释放、CAV 轨迹、UAV sorties、回收队列甘特图；同步标注高优先级任务响应与计划类型。
- **依据与素材：** 已有 rollout/path 可视化和详细 ledger，具备制作初稿条件。**[实验验证]**
- **图表与衔接：** Fig. 13 时空调度图、Fig. 14 队列时间线；解释量化结果，而非只展示漂亮轨迹。约 1 页。

### E. Trade-offs, Limitations, and Deployment Implications

- **核心问题：** 方法的成本、失败边界和现实部署前提是什么。
- **具体内容：** 讨论 time-aware 的 CAV 距离代价、仿真任务分布、单 CAV、同构 UAV、确定性地图/通信、FIFO 默认、3D voxel 精度和训练成本。
- **依据与素材：** 单 seed 对照已提示地空成本转移；其余是模型边界，不虚构实地部署。**[实验验证][代码实现]**
- **图表与衔接：** 可放一张权衡表，无需新主图；为结论中的未来工作提供限定。约 0.7 页。

## VIII. CONCLUSION

### A. Main Findings and ITS Implications

- **核心问题：** 读者应记住的模型、方法和证据是什么。
- **具体内容：** 一段概括动态巡检问题，一段总结三层建模 + joint plans + mask + SMDP，一段报告最终已验证结果。
- **依据与素材：** 只使用冻结多种子实验结论，避免重复摘要中的所有数字。**[建议补充]**
- **衔接/篇幅：** 约 0.4 页；强调交通管理意义而非只强调网络结构。

### B. Limitations and Future Extensions

- **核心问题：** 哪些方向自然延伸但不应伪装成本论文已完成贡献。
- **具体内容：** 多 CAV、异构 UAV、真实流数据、通信/天气不确定性、动态回收插入、跨区域迁移和安全约束。
- **依据与素材：** MAPPO 和 insertion 仅能称 future work，不写“已验证”。**[建议补充]**
- **衔接/篇幅：** 约 0.2 页；具体但克制，不列过多泛化愿望。

## APPENDIX

### A. Detailed Feasibility and Recovery Recursions

- **核心问题：** 哪些推导太细但对复现有价值。
- **具体内容：** 3D segment 合成、deadline/battery slack、FIFO 递推、排空与 censoring 定义。
- **依据与素材：** 只搬运实际实现公式；可由会议公式重构，需与当前变量逐项比对。**[代码实现][会议论文]**
- **图表与衔接：** Eq. (A1)--(A8)；正文保留关键结果，附录给细节。约 1 页。

### B. Observation Features and Hyperparameter Details

- **核心问题：** 如何完整公开 38 维候选特征和训练配置而不挤占正文。
- **具体内容：** candidate feature schema、缩放/clip、全部奖励权重、路径预算、种子和版本签名。
- **依据与素材：** JSON/YAML/params 可直接生成表。**[代码实现]**
- **图表与衔接：** Table A-I/A-II；确保与发布配置自动校验。约 1--2 页。

### Acknowledgment, References, and Author Biographies

- **核心问题：** 是否符合 T-ITS 收尾格式和学术诚信要求。
- **具体内容：** Acknowledgment 只列真实资助/协助；References 采用 IEEE 数字格式；Biography 按期刊要求准备。
- **依据与素材：** 版式可参考 Word 稿，内容不得继承其占位或无关引用。
- **篇幅与顺序：** Appendix（若有）→ Acknowledgment → References → Biographies；投稿模板和最新 author guidelines 定稿时再核对。

### 9.1 建议总篇幅

| 部分 | 建议正文占比 | 重要度 |
|---|---:|---|
| Introduction + Related Work | 18%--22% | 高 |
| System Model + Formulation | 22%--25% | 高 |
| Proposed Method | 25%--30% | 最高 |
| Experimental Design | 12%--15% | 高 |
| Results and Discussion | 25%--30% | 最高 |
| Conclusion | 3%--5% | 中 |

实际页数应以 T-ITS 当期投稿要求和最终图表密度为准；不要为了模仿 5 页参考稿而压缩期刊证据。

---

## 2. 材料读取情况

### 1.1 已读取材料

| 材料 | 实际位置 | 读取与核查结果 | 用途 |
|---|---|---|---|
| 项目代码 | 当前工作区 `D:\OYYX\1. Experimental Code\Dynamic CAV-UAV` | 已审计环境、规划器、RL、配置、测试、结果和说明文件 | 论文事实的主要依据 |
| 会议论文 | `E:\weixing\xwechat_files\wxid_ice688g4r40x22_c0e1\temp\RWTemp\2026-07\9717c6ed631d931aa150fe8a5df54d9a\Dynamic CAV-UAV Cooperative Patrol for Urban Traffic Incidents A Candidate-Sortie Reinforcement Learning Framework.pdf` | 20 页，已提取全文并逐页检查渲染 | 研究基础、会议版本贡献与实验声明 |
| Word 格式参考稿 | `D:\OYYX\0. SCI Paper\2. SCI-多无人机的城市交通事件动态巡检\TITS_Dynamic_CAV_UAV_Paper_Framework_CLEAN.docx` | 5 页，已导出 PDF 并逐页检查 | 仅学习 T-ITS 版式、层级和写作形式 |
| 实验结果 | `results/`、`outputs/` | 约 319 个文件、181 MB；含 CSV、JSON、checkpoint、PNG 和汇总工作簿 | 确认当前可验证实验边界 |
| 项目说明 | `DEFAULT_TRAINING.md`、`ENVIRONMENT_SETUP.md`、`RESEARCH_BACKGROUND_AND_EXPERIMENT_ROADMAP.md`、各 README | 已主动搜索并结合代码核查 | 解释默认入口、环境和历史规划 |

### 1.2 验证动作

- 使用项目 `.venv` 对 `test_periodic_patrol_generation.py`、`test_incident_confirmation_generation.py`、`test_event_decision_coalescing.py`、`test_smdp_time_aware_discount.py` 和 `test_route_change_reward.py` 执行轻量测试，共 **36 passed**。**[实验验证]**
- 默认 Anaconda Python 缺少 `shapely` 和 `torch`，首次测试在收集阶段中止；切换到项目自带 `.venv` 后全部通过。这不是功能断言失败。**[实验验证]**
- 未运行新的训练、评估或超参数搜索；本文档中的性能值全部来自已有结果文件。**[实验验证]**

### 1.3 不纳入事实的材料

- Word 参考稿中的多 CAV、MARL、MAPPO、30--80 节点等内容属于旧写作规划，不能作为当前项目事实。
- 会议稿的 60 节点基线结果、PPO/DQN 比较、消融和敏感性结果未在当前结果目录发现一一对应的原始文件；只能标为 **[会议论文]**，不能标为当前代码已复现。
- 当前任务未进行外部数据库文献检索，因此本蓝图不生成具体参考文献条目，也不声称某项工作是“首次”。

---

## 3. 代码与论文事实摘要

### 2.1 代码结构与运行入口

| 审计项 | 当前事实 | 主要依据 | 证据等级 |
|---|---|---|---|
| 环境入口 | `rl/env_wrapper.py` 封装 `env/simulator.py`，按事件或固定间隔推进仿真 | `DynamicPatrolEnv`、`Simulator` | [代码实现] |
| DQN 训练入口 | `python -m rl.train_masked_dqn --config configs/formal_training.yaml`；正式配置继承默认配置 | `rl/train_masked_dqn.py`、`configs/default_training.yaml`、`configs/formal_training.yaml` | [代码实现] |
| PPO 入口 | `rl/train_ppo.py` 存在，但 CLI 默认仍引用已不存在的旧 `scenario_60node.yaml` | `rl/train_ppo.py` | [代码实现]；非当前主线 |
| MAPPO 入口 | `rl/train_mappo.py` 仅为扩展脚手架，未形成可用算法 | `rl/train_mappo.py` | [代码实现]；不可声称已实现 |
| 网络场景 | 新街口真实有向路网，333 节点、601 条存储边；1 CAV、3 UAV | `configs/default_training.yaml`、GraphML 与网络测试 | [代码实现][实验验证] |
| CAV 层 | CAV 在原始有向道路网络上，以路网最短路/路线状态移动 | `env/road_network.py`、`env/cav_controller.py`、`env/vehicle.py` | [代码实现] |
| UAV 层 | 三维 voxel A*，考虑建筑占据、爬升/巡航/下降、时间、距离和能耗 | `truck_uav_modules/uav_path_planner/`、默认配置 | [代码实现] |
| 任务目标层 | 用稳定 `edge_id` 注册物理道路巡检段，压缩正反向表示，避免同一路段重复生成目标 | `env/task_generator.py`、`data/xinjiekou/edge_task_registry.csv`、生成测试 | [代码实现][实验验证] |
| 仿真时域 | 任务释放 10,800 s；排空阶段启用，最多 1,800 s；默认事件驱动 | `configs/default_training.yaml` | [代码实现] |
| 终止条件 | 释放窗口结束且任务/在途过程排空则终止；排空上限触发截断，并记录 censored/backlog | `env/simulator.py`、`rl/env_wrapper.py` | [代码实现] |

### 2.2 状态、动作与候选计划

- **观测空间：** 总维度 1025，包含全局/车辆/UAV/任务统计以及 25 个动作槽对应的 38 维候选特征；特征名由 `joint_plan_feature_schema.json` 固定。**[代码实现]**
- **最终动作空间：** 离散 25 槽；槽 0 是 safe action，最多 24 个非安全候选。动作含 CAV 的 `keep_route`、`wait`、`move_to`、`reroute`，以及 UAV sortie 与回收计划，不是纯 sortie index。**[代码实现]**
- **计划类型：** safe、type I、type II、type III；默认配额分别约束非安全候选组成，防止单一类型占满动作槽。**[代码实现]**
- **Candidate sortie 生成：** 对每架 UAV 从待处理任务预筛选并枚举最多 3 任务的序列，联结起飞点、任务点和回收点，执行截止期、可达性、能量和回收同步检查；每架最多保留 12 个。**[代码实现]**
- **Joint plan 构造：** 将地面动作候选和 UAV sortie 配对，并计算会合余量、CAV/UAV 等待、现有队列延误、回收距离与插入代价，再去重、排序并填入固定槽。**[代码实现]**
- **Action mask：** 直接来自当次 joint-plan cache；不可行或未填充槽置为 false，safe 槽始终提供兜底。网络选动作和目标动作计算均使用 mask。**[代码实现]**

### 2.3 状态转移、回收与任务释放

- CAV 与 UAV 状态由离散事件推进；决策事件可合并，避免同一仿真时刻重复决策。**[代码实现][实验验证]**
- UAV sortie 包含起飞、顺序服务任务和返回动态会合点；选中计划在提交前可进行精确三维路径物化和再验证。**[代码实现]**
- 默认回收规则为 **FIFO**。`env/recovery_scheduler.py` 已有动态插入评估，但 `recovery_insertion.enabled: false`；因此当前默认系统不能宣称“动态队列重排”。**[代码实现]**
- 周期巡检：10 条物理道路，2 条/批，1800 s 重访，6 个周期，优先级 2。**[代码实现]**
- 动态任务：incident confirmation 为 2/h Poisson、优先级 10、截止 420 s；congestion 为 2/h 加固定释放时刻、截止 720 s；road obstruction 为 1/h、截止 300 s；critical-period patrol 在 900 s 固定释放。**[代码实现]**
- 同一物理道路目标的生成逻辑通过测试；但当前未提供真实交通管理系统的在线数据流，任务到达是仿真过程。**[实验验证][建议补充]**

### 2.4 奖励、算法与基线

- **奖励：** `stage5_coordinated` 按时间区间增量记账，含完成、优先级、超时、加权 AoI、飞行/悬停能耗、返航距离、CAV 距离、回收等待、队列占用、路线改变/反转、协调失败和非法动作。**[代码实现]**
- **主算法准确名称：** `Rainbow-style DQN` 或 `masked dueling double DQN with NoisyNet and prioritized replay`。已实现 Double DQN、Dueling、NoisyNet、PER 和 masking；`n_step=1`，无 C51/QR 分布式价值头，因此不应称“完整 Rainbow”。**[代码实现]**
- **时间处理：** 默认配置使用 event-driven SMDP 和 time-aware discount，形式为相对 60 s 参考间隔的时间折扣；实现与单元测试已存在。**[代码实现][实验验证]**
- **已有启发式：** nearest、priority、EDF、score、random-masked、rolling-horizon 和 static-reopt 名称存在。后两者当前本质是候选排序规则，且主要对应旧 sortie-only 接口，不是严格滚动优化器/静态重优化器。**[代码实现]**
- **公平性结论：** 在完成 joint-plan 适配、相同三维可行性检查和同等计算预算之前，不能把 rolling/static 当作强优化基线。**[建议补充]**

### 2.5 已保存实验指标与结果

| 结果组 | 已有内容 | 可以支持的结论 | 不能支持的结论 |
|---|---|---|---|
| `results/default_training/formal_seed42` | 100k、训练 seed 42、20 个评估 seed；完成率约 96.88%，超时率约 2.59%，平均计划生成约 5.65 ms | 系统可在默认联合动作场景训练/评估并输出完整 ledger | 不能代表 time-aware 默认配置；保存的 replay 为 `fixed_decision`，奖励签名也与当前 YAML 不完全一致 |
| `smdp_time_aware_discount/formal_100k` | fixed 与 time-aware 各 100k、同一训练 seed；均为 20 个评估 seed 汇总 | time-aware 在该单训练种子下响应、AoI、UAV 能耗、返航和回收等待更低 | 不能据此声称统计显著或跨训练种子稳健 |
| `reward_weight_freezing` | original/v2/v3，各 3 个训练 seed、20 个评估 seed，含训练诊断和奖励分量 | 奖励权重对行为和性能影响大；支持选择冻结配置的工程审计 | 不是候选动作、mask 或队列机制消融 |
| event decision coalescing | 对比文件、测试及指标 | 事件合并逻辑已实现且可核查 | 尚非论文主要算法基线 |
| 路径与案例可视化 | 地图、rollout、路径图、训练曲线、TD error、回放组成 | 可用于制作场景图和诊断图初稿 | 不能替代正式多种子对比 |

100k 单训练种子对照的可核查均值如下，仅用于描述已有资产，不应直接写成最终论文结论：

| 折扣 | 完成率 | 超时率 | 响应时间 (s) | 加权 AoI | UAV 总能耗 | UAV 返航距离 (m) | CAV 距离 (m) | 平均回收等待 (s) | 计划生成 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed decision | 0.9822 | 0.0178 | 250.47 | 2078.59 | 18932.04 | 37913.37 | 37654.86 | 84.39 | 2.85 |
| time aware | 0.9838 | 0.0162 | 223.06 | 2071.96 | 17118.24 | 26183.30 | 59286.94 | 37.05 | 2.32 |

解释边界：time-aware 显示出服务与 UAV 指标改善，但 CAV 行驶距离显著增加，表明存在地空成本转移；必须通过多训练种子和置信区间验证。**[实验验证][建议补充]**

---

## 4. 会议论文审计

### 3.1 会议论文结构与主张

- **题目：** *Dynamic CAV-UAV Cooperative Patrol for Urban Traffic Incidents: A Candidate-Sortie Reinforcement Learning Framework*. **[会议论文]**
- **问题：** 1 辆 CAV 搭载多 UAV，在动态释放的城市交通巡检任务中联合决定 UAV 多任务 sortie，并与移动 CAV 会合。**[会议论文]**
- **场景：** 60 节点、3600 s、固定 60 s 决策、3 类任务、sortie-only 动作、FIFO 尾插回收。**[会议论文]**
- **方法：** 候选多任务 sortie、可行性 action mask、队列感知回收估计，以及会议稿所称 Rainbow DQN。**[会议论文]**
- **数学内容：** 路网和任务定义、UAV sortie 时间/能量、回收同步递推、MDP 状态/动作/奖励和掩码。**[会议论文]**
- **实验：** 会议稿报告与 Masked PPO、Masked DQN、启发式比较，以及候选 sortie、mask、队列因素的消融和若干敏感性试验。**[会议论文]**
- **主要结论：** 会议稿声称该表示提高完成率和响应表现，queue-aware 与 mask 改善稳定性/可行性。当前仓库未发现可一一对应的 60 节点原始结果。**[会议论文]**

### 3.2 可继承、需重写与风险

| 内容 | 处理方式 | 原因 |
|---|---|---|
| 城市交通巡检动机与三层系统区分 | 保留思想、重写文字 | 与当前项目一致，但需改为异构巡检任务和真实路网 |
| Candidate sortie 基本定义 | 保留数学核心并扩展 | 当前仍是 joint plan 的底层构件 |
| 可行性掩码思想 | 保留并扩展至 joint-plan cache | 当前 mask 的对象已变化 |
| FIFO 回收同步递推 | 可保留为默认模型 | 当前默认仍是 FIFO；需删除暗示动态重排的表述 |
| 60 节点系统模型与状态/动作 | 全面重写 | 当前为 333 节点、事件驱动、25 槽 joint plan、1025 维观测 |
| “完整 Rainbow” | 改名并重写 | 当前无 distributional head，且 `n_step=1` |
| 实验设置、表格和数值 | 重新运行后重写 | 当前结果资产不构成会议数值的复现链 |
| 会议稿图表 | 只能重绘，不能原样复用 | 需反映新街口、三维路径、联合动作和新版指标 |

内容重复/创新不足风险：若期刊稿仍把贡献写成“candidate sortie + mask + queue-aware”三条，并沿用同一题目和同类实验，只增加篇幅，会被视为增量不足。期刊新增必须表现为 **决策问题、动作表示、时间模型、空间约束与验证体系的实质变化**。**[建议补充]**

---

## 5. Word 格式参考审计

- 页面为 IEEE Letter；标题、作者、Abstract 和 Index Terms 为单栏，正文切换为双栏。
- 摘要以 `Abstract—` 开始，关键词以 `Index Terms—` 开始。
- 一级标题居中、粗体、罗马数字、大写；二级标题使用 `A.`、粗斜体、左对齐。
- 公式居中、编号靠右；图题在图下，表题在表上；引用采用方括号数字。
- 参考稿没有独立 Nomenclature；包含 Appendix、Acknowledgment、References 和 Author Biographies。
- 公式、图表和交叉引用编号是手工文本，并非 OMML/SEQ/REF 自动域。后续正式稿建议使用 IEEE LaTeX 模板或 Word 自动域，避免手工编号错乱。**[建议补充]**
- 英文风格倾向先提出交通问题和研究缺口，再给系统模型、方法、实验；措辞正式、第三人称、少用口号式形容词。可学习结构，不继承其 MARL 等无关内容。

**Nomenclature 建议：** 当前不建议单列完整 Nomenclature。正文 III-A 放一张核心符号表即可；若最终数学符号超过约 30 个，再把符号表独立为 Nomenclature 或 Appendix A。

---

## 6. 代码—会议论文—期刊论文一致性对照

| 内容 | 当前代码 | 会议论文 | 期刊论文建议 | 是否需要补充 |
|---|---|---|---|---|
| 任务定位 | 周期巡检 + incident + congestion + obstruction + critical-period | 3 类 traffic incident/patrol | 用 dynamic urban traffic inspection / heterogeneous patrol tasks | 是，统一术语 |
| 路网 | 新街口 333 节点、601 存储边、有向 | 60 节点 | 采用当前真实路网并解释压缩物理边 | 是，场景图和统计 |
| CAV/UAV 数量 | 1 CAV、3 UAV 默认 | 1 CAV、多 UAV | 主问题仍为单 CAV—多 UAV；做 UAV 数量敏感性 | 是，敏感性 |
| UAV 飞行 | 3D voxel A* + 建筑避障 | 简化 2D/距离模型为主 | 将 3D 路径作为可行性层，不包装成核心调度创新 | 是，验证路径开销 |
| 目标道路 | `edge_id` 表示物理巡检段，去除方向重复 | 区分路网/任务层但不充分 | 明确三层映射和去重规则 | 已实现；需写清 |
| 决策时间 | event-driven，支持 time-aware SMDP | 固定 60 s MDP | 采用 event-driven SMDP | 是，多种子验证 |
| 动作 | 25 槽 joint plan；CAV 动作 + sortie + 回收 | sortie-only candidate action | Candidate sortie 是构件，joint plan 是正式动作 | 已实现；需重写 |
| Candidate sortie | 每 UAV top 12、最多 3 任务 | 核心动作表示 | 定位为分层候选构造的第一阶段 | 是，规模/消融 |
| Action mask | joint-plan cache 生成 | sortie 可行性 mask | 称 feasibility mask for joint plans | 是，无 mask 对照 |
| 回收机制 | 默认 FIFO；插入模块关闭 | FIFO 尾插、queue-aware | 只称 queue-aware evaluation；动态插入另作可选扩展 | 是，队列消融 |
| 观测 | 1025 维；25 × 38 候选特征等 | 旧状态结构 | 给出模块化维度表和归一化规则 | 已实现；需文档化 |
| 奖励 | `stage5_coordinated` 多分量 | 旧奖励 | 按当前 ledger 重写，并做奖励权重稳健性 | 部分已有 |
| RL 算法 | Rainbow-style DQN；Double/Dueling/Noisy/PER/mask，1-step | 声称 full Rainbow | 使用准确名称，避免 full Rainbow | 已实现；需改名 |
| PPO | 旧代码入口失配 | 作为主要比较 | 修复并适配 joint plan 后才能比较 | 是，需改代码 |
| Rolling/static | 排序启发式，旧接口 | 强基线名称 | 实现真正 RH/static re-optimization 或降低命名强度 | 是，需改代码 |
| 随机种子 | 默认训练 seed 42；正式 eval 1000--1019；另有 3-seed 权重实验 | 报告 100 次测试 | 最少 5 个训练 seed × 20 eval seed，并报告 CI | 是 |
| 指标 | 完成/超时/响应/AoI/能耗/距离/等待/队列/规划时间等 | 较少指标 | 使用当前 ledger，补充高优先级 SLA 和推理时间 | 大部分支持 |

### 5.1 名称对应关系

| 历史/会议名称 | 当前实现对应 | 论文中推荐名称 |
|---|---|---|
| Full Rainbow DQN | Double + Dueling + NoisyNet + PER + mask，`n_step=1`，无 distributional head | Masked Rainbow-style DQN |
| Candidate-sortie action | Candidate sortie 被装入联合 CAV-UAV plan | Candidate-sortie-based joint plan representation |
| Queue-aware recovery | FIFO 队列状态/等待/占用进入候选评估和奖励；动态插入关闭 | FIFO queue-aware recovery evaluation |
| Rolling-horizon optimization | 现为排序启发式 | Rolling-horizon heuristic；真正优化版待实现 |
| Static re-optimization | 现为排序启发式 | Static-priority heuristic；真正重优化版待实现 |
| Fixed-step MDP | 当前默认 event-driven + time-aware discount | Event-driven SMDP |

---

## 7. 推荐的论文核心叙事

### 6.1 故事线

1. **实际问题：** 交通管理部门需要对周期性道路状态和突发事件进行及时、异质的现场巡检；单 CAV 携带多 UAV 可结合地面续航与空中快速抵达。**[会议论文][代码实现]**
2. **现有困难：** 动态任务释放、道路方向与物理巡检目标不一致、建筑约束飞行、移动回收点和 FIFO 回收资源耦合，使原子任务动作或静态路线难以同时表达可行性与实时性。**[代码实现]**
3. **具体决策问题：** 在不规则事件时刻，从有限的 joint-plan 槽中选择一个同时规定 CAV 行为、UAV 多任务 sortie 和回收会合关系的可行计划。**[代码实现]**
4. **Candidate sortie 的作用：** 把组合爆炸的任务排列压缩为少量经截止期、能量和连通性筛选的 UAV 执行单元，但它本身不是完整动作。**[代码实现]**
5. **Action mask 的作用：** 将当期不可行、未填充或提交前失效的联合计划排除在探索和 Bellman 目标之外，降低无效动作学习。**[代码实现]**
6. **Recovery queue 的作用：** 显式计算现有 FIFO 队列造成的 UAV 等待、CAV 等待和队列延误，让出动选择考虑移动会合资源冲突；不声称默认动态重排。**[代码实现]**
7. **RL 的适用性：** 任务随机释放且决策间隔不规则，联合动作对未来 CAV 位置、UAV 可用性和回收拥塞有延迟影响，适合以事件驱动 SMDP 学习长期权衡。**[代码实现][建议补充验证]**
8. **ITS 意义：** 该框架把道路语义任务、路网移动和低空飞行约束连接到同一在线决策循环，为城市交通事件确认与重复巡检提供可部署的调度抽象。**[建议补充应用讨论]**
9. **一句话核心贡献：** *A feasibility-masked, event-driven joint-plan reinforcement learning framework that coordinates CAV motion, multi-task UAV sorties, and FIFO recovery congestion for heterogeneous urban traffic inspection.*
10. **避免过度声称：** 不写“首个”、不写“完整 Rainbow”、不写“最优”、不写“动态队列优化”、不把 3D A* 当作调度核心创新、不引用未复现的会议数值作为当前结果。

### 6.2 定位选择

| 候选定位 | 适配性 | 判断 |
|---|---|---|
| 动态 CAV-UAV 协同巡检调度 | 高 | 覆盖最完整，但需要加入 event-driven 和 joint scheduling 以突出期刊扩展 |
| Queue-aware candidate-sortie RL | 中 | 能体现机制，但容易把 FIFO 感知误解为动态队列优化，也会弱化 CAV 联合动作 |
| 面向异构动态任务的车机协同决策 | 高 | 交通含义强，但若不出现 joint scheduling，技术抓手不够明确 |
| **事件驱动、可行性约束的 CAV-UAV 动态交通巡检联合调度** | **最高** | 与 joint plan、SMDP、mask、真实路网和队列耦合同时一致 |

**推荐定位：** *Event-driven, feasibility-constrained joint CAV-UAV scheduling for dynamic urban traffic inspection.* 该定位最能体现当前代码相对会议稿的实质变化，又不会把尚未默认启用的动态回收插入包装成贡献。

---

## 8. 候选论文题目

| 排名 | 英文题目与中文翻译 | 强调点 | 优点 | 可能问题 | T-ITS 匹配 |
|---:|---|---|---|---|---|
| 1 | **Event-Driven Joint CAV-UAV Scheduling for Dynamic Urban Traffic Inspection via Feasibility-Masked Reinforcement Learning**  / **基于可行性掩码强化学习的动态城市交通巡检事件驱动 CAV-UAV 联合调度** | 事件驱动、联合动作、mask | 最忠实于当前实现，ITS 问题明确 | 未在题目出现 candidate sortie | 很高 |
| 2 | **Candidate-Sortie-Based Joint CAV-UAV Scheduling for Heterogeneous Urban Traffic Patrol Tasks** / **面向异构城市交通巡检任务的基于候选架次的 CAV-UAV 联合调度** | Candidate sortie + 异构任务 | 延续会议研究识别度，同时纠正纯 sortie 动作表述 | 未显式出现 RL 和 event-driven | 很高 |
| 3 | **Feasibility-Masked Reinforcement Learning for Online CAV-UAV Cooperative Traffic Inspection** / **面向在线 CAV-UAV 协同交通巡检的可行性掩码强化学习** | 在线决策、mask | 简洁，适合 T-ITS | “online”需用推理时间和动态释放实验支撑 | 很高 |
| 4 | **Queue-Aware Joint CAV-UAV Planning for Dynamic Urban Traffic Inspection** / **面向动态城市交通巡检的队列感知 CAV-UAV 联合规划** | 回收队列 | 突出移动回收资源耦合 | 易被理解为动态重排；默认仅 FIFO 感知 | 中高 |
| 5 | **Event-Driven Reinforcement Learning with Candidate Sorties for CAV-UAV Traffic Patrol** / **采用候选架次的 CAV-UAV 交通巡检事件驱动强化学习** | SMDP + sortie | 技术线清晰 | “patrol”可能弱化 incident confirmation | 高 |
| 6 | **Online Scheduling of a CAV and Multiple UAVs for Heterogeneous Traffic Inspection Tasks** / **面向异构交通巡检任务的单 CAV—多 UAV 在线调度** | 系统构型和任务 | 直接、应用导向、不过度承诺算法 | 技术创新点在题目中较弱 | 高 |
| 7 | **Joint Ground-Air Candidate Planning for Dynamic Urban Traffic Inspection** / **面向动态城市交通巡检的地空联合候选计划方法** | 地空联合候选 | 能准确表达 `joint_plan` | 未明确 CAV/UAV 与 RL | 中高 |
| 8 | **A Time-Aware SMDP Framework for Cooperative CAV-UAV Traffic Inspection** / **面向 CAV-UAV 协同交通巡检的时间感知半马尔可夫决策框架** | 不规则决策时间 | 体现会议到期刊的重要扩展 | 目前 time-aware 正式对照只有 1 个训练 seed | 高，补实验后 |
| 9 | **Coordinating CAV Motion, UAV Sorties, and Recovery Queues for Dynamic Traffic Inspection** / **动态交通巡检中的 CAV 运动、UAV 架次与回收队列协同** | 三重耦合 | 直观解释问题难点 | 未点明学习算法；queue 仍需谨慎 | 高 |
| 10 | **Dynamic CAV-UAV Cooperative Patrol for Urban Traffic Incidents: A Candidate-Sortie Reinforcement Learning Framework** / **面向城市交通事件的动态 CAV-UAV 协同巡检：候选架次强化学习框架** | 保留会议题目 | 连续性强、容易识别 | 范围偏窄；与会议题目重复；不符合当前 joint-plan 动作 | 中，不推荐原样采用 |

### 7.1 前三名与采用条件

1. **第一推荐：Event-Driven Joint CAV-UAV Scheduling for Dynamic Urban Traffic Inspection via Feasibility-Masked Reinforcement Learning.** 目前即可作为工作题目；最终需补 mask 消融和多种子结果。
2. **第二推荐：Candidate-Sortie-Based Joint CAV-UAV Scheduling for Heterogeneous Urban Traffic Patrol Tasks.** 若作者希望保留会议工作的术语连续性，这是最稳妥替代。
3. **第三推荐：Feasibility-Masked Reinforcement Learning for Online CAV-UAV Cooperative Traffic Inspection.** 只有在补足推理时间、动态负载与规模实验后，`Online` 才有充分证据。

---

## 9. 建议贡献设计

### Contribution 1 — 三层语义一致的动态交通巡检系统模型

- **缺口：** 有向路网边、物理巡检道路段和 UAV 空间目标容易被混为一体，导致正反向道路重复任务或不一致的地空距离。**[代码实现]**
- **方案：** 建立 CAV 有向路网层、UAV 三维飞行层和基于稳定 `edge_id` 的物理任务层，并定义异构任务释放、截止期和重访要求。
- **真正新增：** 相比会议 60 节点抽象场景，期刊版加入真实新街口网络、物理道路去重注册和建筑约束 3D 路径可行性。**[代码实现]**
- **验证：** 报告网络/任务映射统计、重复率检查、2D 距离估计与 3D 可行路径差异、路径规划失败/时间。
- **证据充分性：** 实现与生成测试充分；大规模统计图和 3D 代价对比仍需整理。**[实验验证][建议补充]**

### Contribution 2 — Candidate-sortie-based joint plan 与可行性掩码

- **缺口：** 直接枚举 CAV 动作 × UAV × 任务序列 × 回收点会产生组合爆炸，原子任务动作又无法表达多任务 sortie 和移动会合耦合。
- **方案：** 先生成经可行性剪枝的多任务 candidate sorties，再与 CAV 的 keep/wait/move/reroute 组合成固定 25 槽 joint plans，并由 cache 生成 action mask。**[代码实现]**
- **真正新增：** 相比会议稿的 sortie-only 动作，期刊版的选择单元是含 38 维特征的联合地空计划，且支持安全兜底、类型配额、提交前再验证和去重。
- **验证：** 候选数/生成时间/可行率随任务数变化；sortie-only、无联合动作、无 mask、无剪枝、不同 top-K 的消融。
- **证据充分性：** 代码充分，现有计划生成时间指标可用；核心消融尚缺。**[建议补充]**

### Contribution 3 — FIFO 回收拥塞感知的事件驱动 SMDP 决策

- **缺口：** 不规则任务和资源事件导致相邻决策间隔变化；忽略 FIFO 回收占用会低估 UAV/CAV 等待并产生短视 dispatch。
- **方案：** 在事件驱动 SMDP 中，把回收队列长度、锁定前缀、等待和既有延误纳入联合候选特征与奖励，并采用基于实际 `delta_t` 的时间感知折扣。**[代码实现]**
- **真正新增：** 从会议固定 60 s MDP 扩展为时间感知 SMDP，并将队列影响嵌入 joint-plan 评估；不声称默认动态改变 FIFO 顺序。
- **验证：** fixed-step/fixed-discount/time-aware 对照；移除队列特征与惩罚；不同 UAV 数和回收服务/等待参数下的拥塞曲线。
- **证据充分性：** 时间折扣有测试和单训练种子 100k 对照；队列与多种子因果验证不足。**[实验验证][建议补充]**

### Contribution 4 — 面向在线 ITS 的可复现实证体系

- **缺口：** 会议稿的小型抽象网络和有限指标难以说明真实路网可行性、实时性和策略代价转移。
- **方案：** 在 333 节点新街口场景，以多训练/评估种子比较 RL、启发式和重优化基线，报告任务服务、地空资源、回收拥塞、计算时间和典型调度轨迹。
- **真正新增：** 加入完整 reward ledger、排空/censoring 处理、计划生成时间、3D 路径失败和 CAV/UAV 成本分解。
- **验证：** 总体比较、消融、负载/UAV/电池/网络规模敏感性、统计置信区间和案例图。
- **证据充分性：** 指标管线和部分多种子结果已具备，但强基线、全消融和规模矩阵尚未完成。**[建议补充]**

**独立性检查：** Contribution 1 解决建模与语义一致性；Contribution 2 解决动作组合与可行探索；Contribution 3 解决不规则时间和回收拥塞；Contribution 4 解决现实性与证据质量。四条不是对同一“candidate sortie”概念的重复包装。若最终实验资源有限，可将 Contribution 4 合并到前 3 条的验证描述中，正文只列 3 条贡献。

---

## 10. 图、表、公式和算法规划

### 10.1 核心图清单

| 编号与英文名称 | 章节 | 想证明/解释什么 | 数据来源 | 当前制作条件 |
|---|---|---|---|---|
| Fig. 1. **Three-Layer CAV-UAV Urban Traffic Inspection System** | III-A | 清楚区分 CAV 有向路网、UAV 3D 飞行和物理道路任务 | GraphML、buildings、edge registry、现有地图图层 | **已具备**；需重新设计为论文图 |
| Fig. 2. **Event-Driven Joint Scheduling Framework** | V-A | 从事件到候选、mask、DQN、提交和状态推进的闭环 | simulator、planner、RL 模块 | **已具备逻辑**；需绘图 |
| Fig. 3. **Dynamic Task and Agent Event Timeline** | III-B/III-C | 展示任务释放、CAV 移动、UAV sortie、回收和排空 | event ledger、固定案例 seed | **部分具备**；需生成案例数据 |
| Fig. 4. **Candidate Multi-Task Sortie Construction and Pruning** | IV-B/V-B | 解释任务序列、3D 段、截止/能量剪枝和回收点 | sortie generator、3D path | **已具备逻辑** |
| Fig. 5. **FIFO Queue-Aware Mobile Recovery Synchronization** | IV-C | 说明 ETA、FIFO 顺序、UAV/CAV 等待和队列占用 | recovery scheduler | **已具备逻辑**；不可画成动态重排 |
| Fig. 6. **Candidate-Sortie-Based Joint Plan Representation** | V-C | 表明最终动作是 CAV 动作 + sortie + recovery，而非纯 sortie | joint planner、plan types | **已具备** |
| Fig. 7. **Observation, Candidate Slots, and Feasibility Mask** | V-D/V-E | 展示 1025 维观测、25 槽、38 维候选块和 mask | feature schema、env wrapper | **已具备** |
| Fig. 8. **Xinjiekou Road Network, Buildings, and Inspection Targets** | VI-A | 证明实验场景规模和三层空间落点 | 新街口地图资产 | **已具备**；需统一配色/图例 |
| Fig. 9. **Overall Performance with Confidence Intervals** | VII-A | 比较服务、地空成本和运行时 | 正式多种子实验 | **不具备**；需重跑 |
| Fig. 10. **Ablation Effects of Joint Plans, Masking, Queue Awareness, and SMDP Discounting** | VII-B | 分离各机制的边际影响 | 消融实验 | **不具备** |
| Fig. 11. **Sensitivity to UAV Fleet Size and Task Arrival Intensity** | VII-C | 检验拥塞与容量阈值 | 参数化评估/重训 | **不具备** |
| Fig. 12. **Scalability and Online Decision Time** | VII-C | 候选数、网络/任务规模与规划时间关系 | runtime ledger | **部分具备**；需系统采样 |
| Fig. 13. **Representative Spatiotemporal Joint Schedule** | VII-D | 解释策略如何同时安排 CAV 与 UAV | rollout 路径、任务日志 | **部分具备** |
| Fig. 14. **Recovery Queue and Task-Service Gantt Chart** | VII-D | 解释队列感知如何影响响应和等待 | recovery/task ledger | **部分具备** |

图形优先级：Fig. 1、2、4、6、8、9、10、13 为主文必需；Fig. 3、5、7、11、12、14 可按版面合并或移入补充材料。现有 PNG 可作数据和布局参考，但应统一字体、线宽、颜色、单位和双栏可读性。

### 10.2 核心表清单

| 编号与英文名称 | 章节 | 作用 | 数据来源 | 当前条件 |
|---|---|---|---|---|
| Table I. **Main Notation** | III-A | 统一三层模型和 SMDP 符号 | 本文公式 | 可制作 |
| Table II. **Comparison with Related Studies** | II | 支撑研究缺口，不依赖主观声称 | 正式文献检索 | 尚缺 |
| Table III. **Heterogeneous Task Definitions** | III-B | 区分释放、优先级、截止与服务时长 | 默认 YAML | 已具备 |
| Table IV. **Joint Plan Types and Feasibility Conditions** | V-C | 解释 safe/type I--III | joint planner | 已具备 |
| Table V. **Observation and Candidate Feature Dimensions** | V-E | 解释 1025 与 38 维 | schema/env wrapper | 已具备 |
| Table VI. **Coordinated Reward Components** | V-G | 明确符号、单位、权重和方向 | YAML/reward ledger | 已具备；最终配置需冻结 |
| Table VII. **Computational Procedure and Runtime Breakdown** | V-H | 支撑在线性 | profiling | 部分具备 |
| Table VIII. **Simulation and Agent Parameters** | VI-A | 完整复现 | YAML/地图 metadata | 已具备 |
| Table IX. **Compared Methods and Fairness Controls** | VI-B | 避免弱基线/错名 | 基线实现 | 尚需适配 |
| Table X. **Research Questions and Experiment Matrix** | VI-C | 让每个实验对应贡献 | 本蓝图第 12 节 | 可制作 |
| Table XI. **Training Hyperparameters and Seeds** | VI-E | 复现训练 | params/replay config | 已具备框架；最终重跑后更新 |
| Table XII. **Overall Multi-Seed Performance** | VII-A | 主结果 | 正式实验 | 尚缺 |
| Table XIII. **Ablation Results** | VII-B | 机制因果证据 | 消融 | 尚缺 |
| Table XIV. **Conference-to-Journal Extension Summary** | Introduction footnote/cover letter 或补充材料 | 清楚说明实质扩展 | 本蓝图第 13 节 | 已具备文字 |

### 10.3 建议数学内容

| 数学对象 | 保留/新增建议 | 证据与边界 |
|---|---|---|
| 集合与符号表 | 新增统一版本 | 只列代码中实际对象 |
| 三层映射 | 新增 | `edge_id`、有向路网、3D 位置；是期刊扩展重点 |
| 任务释放、截止与重访 | 保留并扩展 | 会议稿基础 + 当前 5 类任务 |
| CAV 行驶时间/距离 | 重写 | 基于有向路网，不用欧氏距离替代 |
| UAV sortie 时间、3D 距离与能耗 | 重写 | 对应 voxel A* 和分段能耗；局部路径不是核心创新 |
| Sortie 可行性 | 保留并扩展 | 截止、能量、可达、回收同步、最多 3 任务 |
| FIFO 回收递推 | 保留并修正 | 只描述 queue-aware FIFO，不写动态重排 |
| Joint-plan 动作定义 | 新增 | 正式动作需含 CAV 动作与回收信息 |
| 目标函数与约束 | 重写 | 用长期期望回报和实现约束，不伪造精确 MILP |
| SMDP 状态/动作/转移 | 新增 | event-driven 与 `delta_t` |
| 时间感知折扣 | 新增 | 已实现且有测试 |
| Action mask | 保留并扩展 | 对象从 sortie 改为 joint-plan slots |
| 奖励函数 | 全面重写 | 使用 `stage5_coordinated` 与 ledger |
| Bellman target/PER loss | 重写 | 准确描述 Rainbow-style DQN，不含 distributional/n-step>1 |

不建议为了“数学复杂度”引入代码无法求解或验证的完整 MILP、机会约束、博弈模型、通信可靠性模型或多 CAV 分配模型。

### 10.4 算法伪代码

1. **Algorithm 1: Event-Driven Masked Rainbow-Style DQN for Joint CAV-UAV Scheduling**  
   包含事件推进、候选生成、mask、选槽、再验证、transition discount、PER 更新和终止。
2. **Algorithm 2: Feasible Candidate Sortie and Joint Plan Generation**  
   包含任务预筛、序列扩展、增量剪枝、回收同步、地面配对、类型配额、去重和 top-K。
3. **Algorithm A1: FIFO Recovery Time Evaluation**（仅在正文公式过长时放附录）  
   只模拟 FIFO 队列；动态 insertion 可在 future-work 说明，不进入默认算法。

---

## 11. 已有实验清单与证据分级

| 已有实验 | 训练/评估规模 | 现有资产 | 论文可用方式 | 风险 |
|---|---|---|---|---|
| 默认 `formal_seed42` | 100k；1 train seed；20 eval seeds | model、summary、eval/training curve、ledger、config | 作为管线验证或历史预实验 | replay 仍为 fixed decision；奖励与当前 YAML 不一致 |
| 正式 SMDP discount 对照 | fixed/time-aware 各 100k；同一 train seed；20 eval seeds | summary、checkpoint、曲线、诊断、工作簿 | 作为预实验动机和地空代价讨论 | 只有 1 train seed，不能作最终显著性证据 |
| Reward weight freezing | 3 variants × 3 train seeds × 20 eval seeds | 全部 CSV/JSON/PNG | 说明奖励配置审计与敏感性 | 训练仅 20k；不是核心机制消融 |
| Event decision coalescing | 配置/结果/测试 | 测试和实验文件 | 支撑实现正确性，必要时放附录 | 不宜作为主结果 |
| Task-generation smoke tests | 周期与 incident | logs/tests | 支撑任务释放和物理边去重 | 不是策略性能实验 |
| Path/rollout visualizations | 若干固定案例 | PNG 与脚本 | Fig. 8/13 初稿 | 需固定最终模型与 seed 后重制 |

结论：已有实验可以支撑“系统已运行、实现逻辑可验证、time-aware 值得进一步检验”，但不能支撑“提出方法显著优于基线”或“各贡献均被消融验证”。

---

## 12. 必须补充的实验矩阵

优先级：P0 为投稿前必需，P1 为强烈建议，P2 为资源允许时补充。

| ID | 研究问题 | 自变量 | 对照组 | 主要指标 | 预期图表 | 当前支持/需改代码 | 优先级 |
|---|---|---|---|---|---|---|---|
| E1 | 完整方法是否优于可解释启发式 | 算法 | priority、EDF、score、nearest、random-masked | 完成、超时、响应、AoI、能耗、CAV 距离、等待、运行时 | Table XII、Fig. 9 | 启发式存在；需适配 joint-plan 公平接口 | **P0** |
| E2 | 是否优于学习基线 | 算法/网络 | Masked vanilla DQN、Maskable PPO | 同 E1 + 训练稳定性 | Table XII、训练曲线 | DQN 可裁剪；PPO 入口和配置需修复 | **P0** |
| E3 | 相比会议方法有何实质改善 | 动作/时间/场景版本 | 会议版 sortie-only + fixed step | 服务、资源、可行率、候选时间 | Table XII/XIV | 未发现当前可运行的完整会议复现入口；需重建适配 | **P0** |
| E4 | 真正滚动决策优化的竞争力如何 | 求解器与时域 | joint-plan rolling horizon、static re-optimization | primary metrics + wall-clock/timeout | Table IX/XII | 当前仅排序规则；需要实现或外接求解器 | **P0**，或在论文中降级主张 |
| E5 | Multi-task candidate sortie 是否必要 | max tasks per sortie = 1/2/3；allow multi-task | single-task action | 完成、响应、sortie 数、每 sortie 任务、候选时间 | Fig. 10 | 配置大体支持；确认观测一致 | **P0** |
| E6 | Joint CAV coupling 是否优于 sortie-only | 动作表示 | sortie-only、固定 CAV route、joint plan | 服务、CAV 距离、UAV 返回/等待、失败 | Fig. 10 | 需建立公平 sortie-only wrapper | **P0** |
| E7 | Action mask 的贡献是什么 | masked/unmasked | 无 mask + 同等非法动作处理 | 非法率、commit failure、回报、收敛、完成 | Fig. 10 | 训练器支持开关，但需保证 safe handling 公平 | **P0** |
| E8 | Queue awareness 是否改善回收拥塞 | queue features/reward on/off | queue-blind；理想无队列；FIFO-aware | recovery wait、queue occupancy、响应、能耗 | Fig. 10/14 | 需添加干净消融配置；不一定改核心代码 | **P0** |
| E9 | Time-aware SMDP 是否稳定优于 fixed discount | discount mode × time mode | fixed decision discount；必要时 fixed-step | primary + $\Delta t$、effective gamma、地空成本 | Fig. 10、Table XIII | 已支持且有预实验；需 ≥5 train seeds | **P0** |
| E10 | UAV 数量如何影响服务与回收拥塞 | UAV = 1,2,3,4,5 | 同算法同负载 | 完成、响应、队列、能耗、候选时间 | Fig. 11 | 配置可改；需确认固定观测维度/重训 | **P1** |
| E11 | 任务到达强度下的容量边界 | 动态 rate × 0.5,1,1.5,2 | 各算法 | SLA、backlog、censoring、队列、运行时 | Fig. 11 | YAML 支持；批量配置/重训或分布训练需补 | **P0** |
| E12 | 任务异质性影响如何 | 类型组合、priority/deadline mix | homogeneous；移除某类 | 分类型完成/响应、公平性、AoI | 分组柱/热图 | 指标需增加 per-task-type 汇总（若现有不足则改） | **P1** |
| E13 | 网络规模是否可扩展 | 60/中/333/更大节点，或子图规模 | 各方法 | 计划时间、候选数、内存、服务 | Fig. 12 | 现有 60/333 资产不完全统一；需场景适配 | **P1** |
| E14 | 电池容量的影响 | battery = 0.6/0.8/1.0/1.2× | 同算法 | 可行候选、完成、返航、最低余量、能耗 | sensitivity lines | YAML 支持；需重训/测试设计 | **P1** |
| E15 | 回收等待/服务参数的影响 | launch/recovery/swap time；queue penalty | 默认零时长与现实非零值 | queue/wait、响应、资源利用 | sensitivity lines | 配置已有字段；非零时长逻辑需测试 | **P1** |
| E16 | Candidate top-K 与剪枝是否兼顾质量/速度 | K、pool、max eval、optimization flag | exhaustive-small、无剪枝 | retained quality proxy、完成、生成时间 | Fig. 12 | 开关和上限已存在 | **P0** |
| E17 | 训练是否稳定 | train seed ≥5 | 算法间比较 | learning curve、final CI、TD/Q/gradient | curves + boxplots | 管线支持 | **P0** |
| E18 | 在线推理是否满足事件响应 | 任务数/候选数/网络规模 | 各算法/硬件 | p50/p95/p99 plan time、3D materialization time | Fig. 12/Table VII | 当前仅 mean；需加入/汇总分位数 | **P0** |
| E19 | 典型策略如何工作 | 固定 eval seeds 和事件案例 | 完整 vs queue-blind/heuristic | 时空轨迹、任务 SLA、queue Gantt | Fig. 13/14 | 日志和绘图基础已有 | **P1** |
| E20 | 动态回收插入是否值得成为扩展贡献 | insertion on/off | FIFO 默认 | 等待、existing delay、可行性、计算时间 | appendix table | 模块已实现但默认关闭；需专门测试正确性 | **P2**；不作为当前主贡献 |

### 12.1 正式统计协议

- 每个学习方法至少 **5 个独立训练 seed**；资源允许时采用 10 个。每个最终 checkpoint 在相同的 20 个 eval seed 上配对评估。
- 预先指定 primary endpoints：completion rate、timeout rate、high-priority response time、weighted AoI；其他为资源/机制指标。
- 报均值、标准差和 95% CI；同 eval seed 下做配对比较并给效应量。多重比较时说明校正方法。
- 同时报告 operation horizon、drain、pending/censored，不能只用排空后的完成率隐藏窗口末 backlog。
- 不用训练 episode reward 代替业务指标；不以单个最佳 seed 作为主结果。
- 固定最终代码、YAML、地图、任务 registry、reward signature 和 checkpoint 选择规则后再生成论文表格。

### 12.2 基线最低公平性要求

- 所有方法接收相同的任务释放、路网、3D 可行性、CAV/UAV 初态和 eval seeds。
- 所有方法必须输出同一 joint-plan 或经过同等提交前验证的动作，不能让 RL 承担 3D 约束而启发式使用直线距离特权。
- Rolling/static 方法必须说明规划时域、求解时间上限、终止 gap 和不可行处理；若只是排序，名称中明确 `heuristic`。
- PPO/DQN 比较需尽量匹配环境交互步数，并同时报告 wall-clock 与样本数。
- 会议方法复现需冻结为单独 legacy config，不能把新版 joint-plan 结果倒填到旧方法名下。

---

## 13. 会议论文到期刊论文扩展矩阵

| 会议论文内容 | 期刊论文处理方式 | 实质新增内容 | 必要新增实验 | 预计创新提升 |
|---|---|---|---|---|
| 60 节点抽象交通场景 | 替换并保留小场景作可比对照 | 333 节点新街口、建筑/voxel、物理边 registry | 场景统计、3D 可行性、规模对照 | 从概念验证到真实空间约束 |
| 3 类动态任务 | 扩展并统一术语 | 周期 + incident + congestion + obstruction + critical-period | 分任务类型、负载/异质性 | 更贴合 ITS 运营任务 |
| 固定 60 s MDP | 重写 | 事件触发、event coalescing、排空/censoring、time-aware SMDP | fixed/time-aware 多种子 | 处理不规则动态决策 |
| Sortie-only action | 保留 candidate sortie 作为下层构件 | CAV keep/wait/move/reroute + sortie + recovery 的 joint plan | sortie-only/fixed-route/joint 消融 | 决策问题发生实质扩展 |
| Candidate sortie | 扩展 | 3D 可达、增量剪枝、缓存、top-K、提交前精确物化 | K/剪枝/多任务消融、运行时 | 从表示概念到在线构造机制 |
| Sortie mask | 重写 | joint-plan cache mask + safe slot + masked target | mask on/off、非法/失败统计 | 与当前正式动作一致 |
| FIFO queue-aware recovery | 精确定义并限制主张 | 队列特征、等待/占用奖励、联合计划同步；insertion 仍为可选 | queue-blind、理想无队列、不同拥塞 | 强化耦合建模，但不夸大重排 |
| “Full Rainbow” | 纠正命名和算法描述 | time-aware masked Rainbow-style DQN | vanilla DQN、PPO、多种子稳定性 | 提高技术准确性，非新增算法组件 |
| 简化状态/奖励 | 全面重写 | 1025 维观测、38 维计划特征、stage5 ledger | reward/feature 诊断 | 支撑联合动作和可解释分析 |
| 会议基线结果 | 不直接搬用；完成复现后对照 | 同场景、同约束、同预算比较协议 | RL、heuristic、true RH/static | 建立可信证据链 |
| 会议消融/敏感性 | 只作实验设计线索 | 联合动作、SMDP、真实路网、运行时新消融 | E5--E18 | 避免简单增加篇幅 |
| 会议图表/公式 | 核心思想可重绘/重写 | 三层图、joint-plan 图、SMDP/3D/FIFO 新公式 | 与最终配置自动核对 | 消除旧版本不一致 |

### 13.1 必须明确回答的扩展问题

1. **期刊新增了什么？** 真实 333 节点三层场景、物理道路去重、建筑约束 3D 可行性、事件驱动 SMDP、时间感知折扣、candidate-sortie-based joint CAV-UAV plans、完整 ledger 和期刊级验证体系。
2. **哪些章节必须重写？** Abstract、Introduction contributions、System Model、Formulation、Method 中动作/状态/算法、全部 Experiments、Results、Conclusion；只有动机和 candidate/FIFO 基本思想可继承后重写。
3. **哪些图表可更新后使用？** 会议场景图、candidate sortie 图、queue 图、mask 图和框架图可沿用概念布局，但必须重新绘制并反映 joint plan、333 节点和 event-driven 流程；结果图表不能直接复用。
4. **哪些实验不足以支撑 T-ITS？** 单训练 seed、缺少强基线、会议数值无当前原始文件、没有 joint/mask/queue 核心消融、缺少负载/规模/运行时和置信区间。
5. **如何避免简单增页？** 让期刊版改变正式决策单元和时间模型，并用真实空间约束与多维验证回答新的 RQ，而不是只增加 related work 或参数表。
6. **最能体现实质扩展的贡献？** Contribution 2 的 joint-plan 动作扩展和 Contribution 3 的 event-driven queue-aware SMDP；Contribution 1 提供可信的真实三层交通语义。

### 13.2 重复发表风险控制

- 在投稿 cover letter 和论文脚注中声明会议版本并引用它，按 T-ITS 当前政策说明实质新增。
- 不逐段复制会议摘要、贡献、方法描述和结论；即使公式思想相同，也按新符号、动作和场景重新组织。
- 建议制作文本相似度自查，并记录哪些图/公式由会议版演化而来。
- 不把会议实验样本数或性能数字与期刊新场景混合；复现实验单列 `conference-version reproduction`。
- 最终新增内容比例不能只按页数判断，应以新问题定义、实现、实验和结论能否独立成立判断。

---

## 14. 当前材料缺口与最少待确认问题

### 14.1 已确认缺口

1. **期刊级外部文献集尚缺。** 当前未做系统文献检索，Related Work 的引用和“研究缺口”强度需下一阶段完成。
2. **会议实验复现链尚缺。** 没有找到与会议稿 60 节点主要表格逐项对应的原始 CSV/JSON/config。
3. **强基线尚缺。** PPO 配置入口失配；rolling/static 不是严格优化器，且旧接口与 joint plan 不公平。
4. **核心消融尚缺。** Candidate multi-task、joint CAV coupling、mask、queue awareness 的独立消融未见正式结果。
5. **统计证据不足。** 100k SMDP 正式对照只有 1 个训练 seed；已有 3-seed reward 实验只验证权重影响。
6. **规模与现实性证据不足。** 缺少任务负载、UAV 数、网络规模、电池、非零回收服务时间和运行时分位数实验。
7. **当前默认与历史结果不完全一致。** `formal_seed42` 使用 fixed-decision replay 且 reward signature 与当前 YAML 不同，不能当最终主结果。
8. **真实数据来源说明尚缺。** 路网/建筑可核查，但任务到达率和 deadline 是仿真配置，尚无交通管理部门数据标定说明。

### 14.2 最少待作者确认问题

1. 会议论文是否已正式发表；若已发表，请提供 DOI、会议名称、年份和版权版本，以便正确引用和制定扩展声明。
2. 60 节点会议实验的原始配置与结果是否存放在工作区之外；若有，只需提供唯一目录，不必手工整理。
3. 新街口路网、建筑和任务参数的来源与使用许可是什么；任务率/deadline 是否有实际业务依据，还是仅为仿真假设。
4. 期刊实验是否允许实现真正的 rolling-horizon/static re-optimization 与修复 PPO；若资源有限，需优先保证 E1、E2、E5--E9、E11、E16--E18。

除上述 4 个问题外，题目、主线、提纲和实验设计不需要额外澄清即可继续写作。

---

## 15. 下一阶段论文写作顺序

1. **冻结事实基线。** 将当前默认 YAML、地图/任务 registry、feature schema、reward signature 和算法名称形成一份 reproducibility manifest；历史结果单独归档，不覆盖。
2. **补文献与扩展声明。** 完成 T-ITS/TRC/TS/RA-L 等相关工作检索，建立 Table II；确认会议出版信息并写 conference-to-journal 声明。
3. **先写 III--V。** 先完成三层系统、任务、FIFO/SMDP 数学模型和 joint-plan 方法，因为这些最受代码约束，也决定后续实验变量。
4. **实现/校准 P0 基线和消融。** 优先 E1、E2、E5--E9、E16；若 E4 无法实现，降低 optimization baseline 声称并明确 heuristic 性质。
5. **运行正式多种子实验。** 统一 5+ train seeds × 20 eval seeds，冻结 checkpoint 选择，生成 Table XII/XIII 和 Fig. 9/10。
6. **补鲁棒性与在线性。** 至少完成任务强度 E11、计划生成效率 E16/E18；有资源再做 UAV 数/网络/电池。
7. **写 VI--VII。** 先按 RQ 填事实表，再写讨论；同时报告不利结果和 CAV/UAV 成本转移。
8. **制作 Fig. 1--8 和案例图。** 所有图从冻结配置/日志生成，统一 IEEE 双栏样式，避免手工修改数据。
9. **最后写 I、II、Abstract、Conclusion 和 Title。** 只有结果冻结后才填写摘要数字和最终贡献措辞。
10. **投稿前审计。** 核对代码—公式—表格—摘要一致性、会议文本相似度、统计报告、单位/符号、交叉引用和 T-ITS 最新模板要求。

### 建议的最小可投稿实验包

- 完整方法 + 4 个可解释启发式 + Masked DQN + 修复后的 PPO；
- 5 个训练 seed × 20 个共同评估 seed；
- multi-task sortie、joint plan、mask、queue awareness、time-aware discount 五组消融；
- 任务负载、top-K/剪枝、在线运行时三组压力测试；
- 一个固定 seed 的地空轨迹 + 回收队列案例；
- 全部 primary metrics、地空成本、queue、censoring 和 95% CI。

---

## 16. 最终建议

当前项目已经具备一篇 T-ITS 论文的**方法骨架和真实场景基础**，但现阶段最可靠的产出是“论文蓝图”，还不是可直接宣称优越性的结果稿。建议把期刊创新从会议稿的三个机制口号，升级为以下完整因果链：

> **physical-road-aware heterogeneous tasks → feasible multi-task sorties → fixed-size joint CAV-UAV plans → joint-plan action mask → FIFO recovery congestion features → event-driven time-aware SMDP learning → multi-seed service/resource/runtime evidence**

在该链条中，candidate sortie、mask 和 queue awareness 各有作用，但真正把会议工作提升为期刊工作的关键，是 **joint ground-air decision unit + event-driven time model + realistic three-layer validation**。在 P0 实验完成前，摘要和贡献中不要出现具体性能提升比例、显著性、实时部署或可扩展性结论。
