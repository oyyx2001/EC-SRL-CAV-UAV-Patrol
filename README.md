# Dynamic CAV-UAV Patrol — Rainbow DQN release

This repository contains the reproducible runtime for event-driven urban
patrol by connected autonomous vehicles (CAVs) and UAVs. The public release
focuses on the environment, the joint-plan action space, and the Rainbow DQN
training/evaluation entry point.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m rl.train_masked_dqn --seed 42 --max-episodes 30 --disable-tensorboard
```

The default configuration is Xinjiekou and is loaded from
`configs/default_training.yaml`. To run the same Rainbow DQN workflow on the
Baijiahu scene:

```powershell
python -m rl.train_masked_dqn `
  --config configs\baijiahu_training.yaml `
  --seed 42 `
  --max-episodes 30 `
  --disable-tensorboard
```

For the CUDA 12.8 PyTorch environment used during development, install
`requirements-torch-cu128.txt` before `requirements.txt`. See
[`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) for the complete setup and
[`DEFAULT_TRAINING.md`](DEFAULT_TRAINING.md) for the training options.

## Repository layout

- `configs/`: the two scene configurations used by the Rainbow DQN workflow.
- `env/`: event-driven simulation, traffic, task, CAV, and UAV state logic.
- `planner/`: candidate sortie generation and joint CAV-UAV plan construction.
- `rl/`: the masked Rainbow DQN, SMDP discounting, replay buffer, and metrics.
- `data/`: the two scenario task registries.
- `truck_uav_modules/`: packaged road-network and 3-D voxel UAV path-planning
  runtime data.
- `tests/`: regression tests for the environment and training primitives.

Training outputs, checkpoints, caches, manuscript files, ablation studies,
baseline implementations, and plotting scripts are intentionally excluded
from this reviewer-facing release.

## Verification

```powershell
python -m rl.train_masked_dqn --help
python -m pytest -q tests
```
