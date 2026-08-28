# Default Rainbow DQN training

The training entry loads `configs/default_training.yaml` automatically, so the
default scene is Xinjiekou.

```powershell
python -B -m rl.train_masked_dqn `
  --seed 42 `
  --max-episodes 30 `
  --disable-tensorboard
```

To train on Baijiahu with the same model, reward, task-generation, and
time-aware discount settings:

```powershell
python -B -m rl.train_masked_dqn `
  --config configs\baijiahu_training.yaml `
  --seed 42 `
  --max-episodes 30 `
  --disable-tensorboard
```

Each scene configuration supplies its output directory, run name, model name,
map, and task registry. Command-line options can still override run-specific
values.
