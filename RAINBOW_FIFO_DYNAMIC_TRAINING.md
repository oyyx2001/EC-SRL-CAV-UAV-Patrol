# Rainbow FIFO / Dynamic Insertion 配对训练

两份训练配置继承同一个公共配置，模型结构、候选数量、训练步数、随机种子模式、学习率、replay buffer、任务场景和奖励完全一致。两组训练行为的唯一差异是 recovery rule。

## 配置

- 公共配置：`configs/scenario_60node_stage2_rainbow_common.yaml`
- 修改前机制：`configs/scenario_60node_stage2_rainbow_fifo.yaml`
- 修改后机制：`configs/scenario_60node_stage2_rainbow_dynamic.yaml`

公共配置固定为 3 架 UAV、每架最多 8 个候选，因此 action dimension 为 `1 + 3 * 8 = 25`，与现有 Rainbow checkpoint 一致。

## 环境

在仓库根目录运行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$python='D:\ProgramData\Anaconda3\envs\gym\python.exe'
```

## 分别训练

FIFO（修改前）：

```powershell
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn `
  --config configs/scenario_60node_stage2_rainbow_fifo.yaml `
  --seed 42
```

Dynamic Insertion（修改后）：

```powershell
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn `
  --config configs/scenario_60node_stage2_rainbow_dynamic.yaml `
  --seed 42
```

默认训练步数为 1,000,000。模型分别保存到：

- `results/rl/dqn/stage2_rainbow_fifo_seed42/RainbowDQN_stage2_fifo_seed42.pt`
- `results/rl/dqn/stage2_rainbow_dynamic_seed42/RainbowDQN_stage2_dynamic_seed42.pt`

训练曲线、评估曲线、参数和 TensorBoard 日志也保存在对应 run 目录中。

## 公平比较

不要并行运行同一 GPU 上的两组完整训练。依次训练，并保持：

- 同一个 Git commit；
- 相同 `--seed`；
- 相同硬件和 device；
- 不额外覆盖任何超参数；
- 不修改公共配置、任务生成和奖励。

一个 seed 只能做功能和趋势检查。论文结果建议至少使用 3 个训练 seed。训练 seed 43/44 时，同时覆盖输出名称，避免覆盖 seed 42：

```powershell
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn --config configs/scenario_60node_stage2_rainbow_fifo.yaml --seed 43 --run-name stage2_rainbow_fifo_seed43 --model-name RainbowDQN_stage2_fifo_seed43.pt
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn --config configs/scenario_60node_stage2_rainbow_dynamic.yaml --seed 43 --run-name stage2_rainbow_dynamic_seed43 --model-name RainbowDQN_stage2_dynamic_seed43.pt
```

seed 44 同理。

## 短训练检查

提交前已使用以下方式对两份配置执行短步数训练检查：

```powershell
& $python -B -m dynamic_patrol_marl.rl.train_masked_dqn --config <config> --seed 42 --timesteps 10 --eval-seeds 1000 --device cpu --disable-tensorboard --out-dir results/stage2_training_smoke
```

短检查只验证初始化、环境交互、action mask、checkpoint 和结果写出，不代表模型性能。
