# Dynamic CAV UAV Patrol

This repository contains the reviewer-facing implementation for **Event-Driven Candidate-Sortie Reinforcement Learning for Dynamic CAV-UAV Cooperative Traffic Patrol**. It packages the event-driven simulator, bounded joint-plan generator, queue-aware recovery insertion, candidate-aware Rainbow DQN, and the validated 3-D voxel A* revalidation step used by the paper.

The release contains only source code, reproducible scene configurations, required road/building/voxel assets, task registries, and regression tests. Manuscripts, checkpoints, training outputs, cached results, ablation scripts, and plotting artifacts are excluded.

## Quick start

Create an isolated environment and install the runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For the CUDA 12.8 PyTorch build used during development, install `requirements-torch-cu128.txt` before `requirements.txt`. CPU-only machines can use the official CPU PyTorch wheel instead.

Run a short smoke test:

```powershell
python -m rl.train_masked_dqn `
  --config configs\default_training.yaml `
  --seed 42 `
  --timesteps 2000 `
  --disable-tensorboard
```

Run the paper training budget on Xinjiekou:

```powershell
python -m rl.train_masked_dqn `
  --config configs\default_training.yaml `
  --seed 42 `
  --timesteps 300000 `
  --disable-tensorboard
```

Run the same workflow on Baijiahu:

```powershell
python -m rl.train_masked_dqn `
  --config configs\baijiahu_training.yaml `
  --seed 42 `
  --timesteps 300000 `
  --disable-tensorboard
```

The evaluation seeds in both scene configurations are the matched unseen seeds 1000--1019 used by the manuscript protocol. Training outputs are written below `results/`, which is ignored by Git.

## Model and configuration contract

- **Decision process:** event-driven joint-plan decisions with elapsed-time SMDP discounting (`gamma=0.99`, reference interval 60 s) and valid-action masking.
- **Action space:** 25 fixed slots: one safe plan plus up to 24 non-safe joint plans. Joint plans combine CAV actions (`keep_route`, `wait`, `move_to`, `reroute`), an optional multi-task UAV sortie, and a recovery-queue insertion.
- **Observation:** 75 global-state features plus 25 candidate slots × 38 candidate-specific features. The feature names and scales are recorded in [`joint_plan_feature_schema.json`](joint_plan_feature_schema.json).
- **Value network:** shared two-layer candidate encoder (128 units), two-layer global encoder (256 units), dueling value/advantage heads, factorized NoisyLinear exploration, Double DQN targets, prioritized replay, and Huber loss.
- **Candidate generation:** at most three tasks per sortie, top 12 feasible sorties per UAV, incremental lower-bound pruning, recovery-node screening, queue insertion, and final 3-D path revalidation for the selected sortie.
- **Fleet and operations:** one CAV, three UAVs, UAV battery capacity 1800, climb/cruise/descent speeds 3/10/2 m/s, stochastic 50--70 s launch and 25--35 s recovery durations, and staged CAV speeds of 30/20/30 km/h with a bounded 0.8--1.2 multiplier.
- **Scenes:** Xinjiekou uses 333 road nodes and 601 directed links; Baijiahu uses 129 nodes and 247 directed links. Both scenes include their compressed road graph, building footprints, launch/recovery nodes, task registry, and 3-D occupancy map.

## Repository layout

- `configs/` — canonical Xinjiekou and Baijiahu training configurations.
- `env/` — event processing, task streams, CAV/UAV state, traffic, and recovery scheduling.
- `planner/` — bounded sortie generation and joint candidate-plan construction.
- `rl/` — candidate-aware Rainbow DQN, replay, reward ledger, and SMDP discounting.
- `truck_uav_modules/` — ground-network access, 3-D voxel A* runtime, and packaged scene assets.
- `data/` — direct compressed-edge task registries for both scenes.
- `tests/` — regression tests for task generation, event coalescing, rewards, scene assets, and time-aware discounting.

## Verification

```powershell
python -m rl.train_masked_dqn --help
python -m pytest -q tests
```

The repository follows the current implementation as the executable source of truth. Before using published numerical results, reconcile any manuscript parameter that differs from the checked-in configuration and rerun the affected experiments.
