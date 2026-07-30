# Default Rainbow DQN training

The training entry now loads `configs/default_training.yaml` automatically.

```powershell
D:\ProgramData\Anaconda3\envs\gym\python.exe -B -m rl.train_masked_dqn `
  --seed 42 `
  --max-episodes 30 `
  --disable-tensorboard
```

The config supplies the default output directory, run name, model name, map,
task registry, reward weights, and Rainbow hyperparameters. To override one of
them, add the corresponding command-line option. To use another configuration,
add `--config configs\another_config.yaml`.
