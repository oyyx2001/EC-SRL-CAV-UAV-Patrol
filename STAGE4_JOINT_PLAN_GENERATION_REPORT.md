# Stage 4 — Joint CAV–UAV Candidate Plan Generation

## 1. 阶段目标

阶段四把 UAV sortie、CAV 地面动作、恢复插入位置和恢复时序统一成一个可屏蔽的 `JointPlan` 动作，同时保持阶段一至三的独立运行路径。

## 2. 前三阶段实现状态

阶段一 Dynamic Recovery Insertion、阶段二 Event-driven SMDP、阶段三 Active CAV Routing 均保留。接入阶段四后，既有 54 项回归测试结果为 54/54 通过。

## 3. 阶段三主要收益与 UAV 返航代价

阶段三的主动地面控制可降低部分恢复等待，但地面会合点变化可能增加 UAV 返航距离。阶段四用同 sortie 的 keep-route 参考计划和硬 Return Guard 显式约束该代价。

## 4. 修改前动作空间

旧接口为 1 个 wait/no-op 加 3 UAV × 8 sortie 候选，共 25 个动作；实际 observation 维度为 219。

## 5. JointPlan 定义

`planner/joint_plan.py` 使用 dataclass 固定保存地面候选、sortie、插入位置、路线、恢复时序、任务/UAV、预测成本、特征、可行性、状态指纹和版本信息。

## 6. 四类计划定义

slot 0 是永远可用的 Safe Plan。Type I 是 keep-route 服务计划；Type II 是 wait/move rendezvous 计划；Type III 是小配额 reroute 计划。

## 7. GroundCandidate 生成方式

直接复用阶段三 `ActiveCAVController.generate_ground_candidates()`。联合模式只启用候选生成，不启用独立的 `decide()` 自动提交路径。

## 8. Sortie 生成方式

直接复用 `SortieGenerator.generate_all()`。UAV 始终从当前 CAV 节点立即发射；同时间多派遣只允许沿已提交路线继续生成 keep-route 计划。

## 9. Recovery insertion 集成

阶段一调度器新增 `evaluate_insertions(..., max_results, ground_plan_override, commit=False)`，旧 `evaluate()` 的排序、返回值和测试保持不变。

## 10. UAV Return Guard

参考计划为 keep-route + same sortie + best feasible insertion。主阈值为返航距离相对增幅 0.75，分母下限 100 m，并同时检查电量余量。消融配置可完全关闭护栏。

## 11. 候选规模控制

主配置生成至多 4 个地面候选，每地面候选至多 6 个 sortie，每对至多保留 1 个插入结果；非安全计划 K=24。

## 12. 类型配额

主配置 Type I/II/III 配额为 12/8/4。类型内先按确定性预评分排序，空余槽位由全局剩余候选确定性填充。

## 13. JointPlan feature schema

固定 38 维，覆盖任务收益 7、UAV 成本 7、CAV 成本 6、协调成本 7、系统状态 7、计划类别 4 个特征。连续值统一归一化、非有限值归零并裁剪到 [-5, 5]。顺序见 `joint_plan_feature_schema.json`。

## 14. Observation 结构

联合模式 global state 为 75 维，25×38 计划矩阵为 950 维，总 `obs_dim=1025`；主配置 `action_dim=25`。旧模式仍为 `obs_dim=219, action_dim=25`。

## 15. Action mask

slot 0 始终为 true；保留计划按 `feasible && mask` 设置；padding 特征全零且 mask=false。action index 与 plan slot index 在提交时再次校验。

## 16. JointPlanCache

缓存包含 decision time/epoch、state fingerprint、25 个槽位、feature matrix、mask、schedule versions 和 event sequence。observation、mask、step 读取同一冻结实例；状态过期直接拒绝。

## 17. 同时多 UAV 派遣处理

第一次联合动作提交地面计划后，同 timestamp 后续计划只能为 keep committed route，不能再次 wait/move/reroute 或修改 flexible rendezvous。Safe Plan 强制推进未来事件。

## 18. 联合计划无副作用评估

生成阶段只执行已有纯评估函数；不推进仿真时间、不派遣 UAV、不修改车辆/任务/恢复队列/RNG。候选与拒绝计数属于阶段四诊断日志。

## 19. 原子提交和回滚

提交前校验 fingerprint、版本、slot/mask，并快照 CAV、UAV、任务、恢复队列、版本、RNG 和相关日志。任一地面、插入、任务或发射步骤失败时恢复完整快照。事件队列在事务期间不调度新事件，故堆与 sequence 保持不变。

## 20. 修改文件清单

核心修改涉及 `planner/joint_plan.py`、`env/recovery_scheduler.py`、`env/cav_controller.py`、`env/simulator.py`、`rl/env_wrapper.py`、`rl/train_masked_dqn.py`、模型评估器和 metrics；新增 14 个阶段四配置、40 项专项测试、10 个微场景、实验/测试脚本与 feature schema。

## 21. 单元测试结果

最终执行 104/104 通过：阶段一 10、阶段二 19、阶段三 25、阶段四专项 40、微场景 10。结果记录于 `unit_test_results.csv` 与 `unit_test_summary.csv`。

## 22. 微场景结果

10 个微场景覆盖无任务、单任务、活动恢复队列、极端返航、低电量、locked prefix、padding、同时间路线锁、Safe 推进和注入失败回滚，10/10 通过；结果见 `micro_scenario_results.csv`。

## 23. 3-seed Smoke Test

J0/J1/J2/J3 的 3 个配对种子全部完成，无 step-limit、非法动作、提交失败或回滚。平均完成率分别为 0.908/0.950/0.916/0.910；输出见 `smoke_3seed_metrics.csv`。

## 24. 20-seed 结构实验

Legacy-C3 与 J0–J3 共 100 个完整 episode 全部完成。平均完成率为 0.932/0.902/0.911/0.917/0.902。J2 相对 J3 的 UAV 返航距离由 20,198.64 m 降至 19,201.57 m，平均减少 997.07 m（4.94%，13/20 个种子下降）；完成率增加 1.44 个百分点，Safe-only 仅增加 1.95 个百分点。J2 平均恢复等待 99.00 s，J3 为 96.82 s，体现了 guard 的返航—等待权衡。该确定性启发式结果只验证结构和稳定性，不作为最终 RL 论文结果。

## 25. K=8/16/24/32 敏感性

四个 K 配置分别使用 action_dim 9/17/25/33，每个 K 使用 3 个配对种子。四档完成率和选择结果一致；K=8/16 的平均截断次数为 4.33/1.00，K=24/32 均为 0。平均生成时间为 6.71/6.56/6.66/7.23 ms，K=16 的速度优势不足 2%，不足以抵消截断风险，因此保留 K=24。

## 26. Return Guard 消融

J2 与 J3 唯一区别是 Return Guard。20-seed 证据显示返航距离下降 4.94%，Safe-only 仅上升 1.95 个百分点。0.25/0.50/0.75/1.00 的 3-seed 敏感性结果已保存；小样本排序存在波动，主配置继续采用预注册的 0.75，而不依据 3-seed 结果追调阈值。

## 27. 短训练结果

Rainbow DQN 完成 1000 transitions、901 次梯度更新，replay 含 1000 条记录；训练曲线所有数值均有限。checkpoint 保存为 obs/action/candidate=1025/25/38、decision_mode=joint_plan、feature schema 38、replay schema v2。seed 1000 训练末评估完成率 0.725，独立加载后的 seed 1001 测试完成率 0.811，均无非法动作或提交失败。旧 219 维 checkpoint 会明确报出 dimension mismatch。

## 28. 候选生成计算时间

最终 20-seed J2 的 mean/P95 plan generation time 为 5.98/11.95 ms，mean/P95 单计划 evaluation time 为 0.253/0.416 ms，低于建议的 20/50 ms 门槛。feature clipping rate 为 0.355%，且所有输出特征均有限。

## 29. 是否达到阶段四验收标准

通过。证据包括：104/104 测试；21,393 个 cache epoch 每个严格 25 个唯一 slot；38,233 个 retained plan 可行率和 mask 有效率均为 100%；Safe Plan 存在率/可选率 100%；padding、slot、mask、stale cache、非法动作、提交失败和回滚均为 0；无负电量、无限零时间循环或非有限特征；SMDP 与旧三阶段回归全绿；正式实验与短训练均完成。

## 30. 推荐给阶段五的最终配置

推荐阶段五使用 K=24、Type I/II/III=12/8/4、Return Guard=0.75、distance floor=100 m、max ground=4、max sorties/ground=6、max insertion/pair=1。K=16 未获得明显速度收益且仍有截断，不建议替换。

## 31. 尚未解决的问题

阶段四结构验收已完成。仍需在阶段五解决的是：当前启发式 Safe-only rate 较高（J2 为 68.16%）、Legacy-C3 的完成率仍高于 J2，以及 Return Guard 带来的恢复等待权衡；这些应通过奖励重构与完整 RL 训练研究，不能用阶段四 3-seed 阈值调参替代。
