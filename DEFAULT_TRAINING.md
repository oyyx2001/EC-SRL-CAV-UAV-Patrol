# Default Rainbow DQN training

`configs/default_training.yaml` is the canonical Xinjiekou configuration and now records the 300,000 decision-transition training budget used in the manuscript protocol. The default model is candidate-aware Rainbow DQN with 25 action slots, 38 features per candidate, elapsed-time discounting (`gamma=0.99`, reference interval 60 s), NoisyNet exploration after a 1,000-step warm-up, and prioritized replay.

The scene uses a 10,800 s task-release horizon followed by a drain period of up to 1,800 s. Planning uses launch and recovery upper bounds of 70 s and 35 s; execution samples hidden truncated-normal durations in the ranges 50--70 s and 25--35 s. CAV planning uses staged mean speeds of 30 km/h, 20 km/h, and 30 km/h for the three traffic intervals, with a hidden bounded multiplier in [0.8, 1.2].

Run the paper configuration with:

```powershell
python -B -m rl.train_masked_dqn `
  --config configs\default_training.yaml `
  --seed 42 `
  --timesteps 300000 `
  --disable-tensorboard
```

For a quick smoke test, override `--timesteps 2000` or use `--max-episodes 1`.
