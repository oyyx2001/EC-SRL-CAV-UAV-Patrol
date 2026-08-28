# Dynamic CAV-UAV Patrol

An event-driven simulation and reinforcement-learning project for coordinated
urban patrol by connected autonomous vehicles (CAVs) and UAVs. The current
workflow supports the Xinjiekou and Baijiahu road networks, dynamic inspection
tasks, feasibility-masked joint plans, and time-aware SMDP training.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m rl.train_masked_dqn --seed 42 --max-episodes 30 --disable-tensorboard
```

Use `configs/baijiahu_training.yaml` to train on Baijiahu:

```powershell
python -m rl.train_masked_dqn `
  --config configs\baijiahu_training.yaml `
  --seed 42 `
  --max-episodes 30 `
  --disable-tensorboard
```

## Repository layout

- `configs/`: canonical training and ablation configurations.
- `env/`: road network, task generation, vehicle, UAV, and simulator logic.
- `planner/`: candidate sortie and joint-plan generation.
- `rl/`: SMDP environment wrappers, replay, networks, and training entry points.
- `tests/`: regression tests for the current training workflow.
- `truck_uav_modules/`: packaged ground-network and UAV path-planning modules.
- `data/`: task registries used by the current scenarios.
- `manuscript/`: current paper source, figures, and final rendered document.
- `scripts/`: experiment preparation, analysis, visualization, and smoke tests.

## Verification

Run the main regression suite from the repository root:

```powershell
python -m pytest -q tests
```

The standalone map and path-planning module has its own setup and tests in
[`truck_uav_modules/README.md`](truck_uav_modules/README.md).

## Documentation

- [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md): environment installation.
- [`DEFAULT_TRAINING.md`](DEFAULT_TRAINING.md): default and Baijiahu training.
- [`manuscript/Event_Driven_Candidate_Sortie_RL_TITS_Draft.md`](manuscript/Event_Driven_Candidate_Sortie_RL_TITS_Draft.md): current paper draft.

Only runtime inputs and current deliverables are versioned. Generated training
outputs, benchmark results, review renders, checkpoints, and temporary files
are excluded through `.gitignore`.
