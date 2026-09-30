# Default Rainbow DQN training

The training entry loads `configs/default_training.yaml` automatically, so the
default scene is Xinjiekou. Its committed sorties use hidden truncated-normal
handling times: launch 50--70 s (center 60 s) and recovery 25--35 s (center
30 s). Planning uses the conservative upper bounds 70 s and 35 s.

CAV planning uses deterministic stage-mean speeds: 30 km/h for 0--3600 s,
20 km/h for 3600--7200 s, and 30 km/h for 7200--12600 s. Physical execution
samples a hidden truncated-normal multiplier (`mean=1`, `std=0.1`, bounds
`[0.8, 1.2]`) when entering each road edge and resamples the remaining distance
when a stage boundary is crossed.

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
