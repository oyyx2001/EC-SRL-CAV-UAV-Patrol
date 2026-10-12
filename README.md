# EC-SRL: Dynamic CAV–UAV Cooperative Traffic Patrol

Research code for **Event-Driven Candidate-Sortie Reinforcement Learning for Dynamic CAV-UAV Cooperative Traffic Patrol**, by Yuxiang Ouyang, Shijia Zhang, Yikang Rui, and Bin Ran.

EC-SRL combines an event-driven patrol simulator, bounded candidate-sortie joint plans, proactive CAV motion, queue-aware recovery insertion, and 3-D UAV path revalidation. Its main policy, **CS-RDQN**, uses candidate-aware masked Rainbow DQN with Double DQN, dueling heads, NoisyNet, prioritized replay, and elapsed-time SMDP discounting.

This release provides the **main method and original pretrained policies**, not complete comparison baselines or ablation variants. **Author-owned code and both pretrained model weights use the MIT License.** Geographic inputs and their derivatives are not relicensed under MIT and are withheld pending source-rights verification.

**Reproducibility limitation:** a clean checkout can load the policies, but cannot run the original cases without authorized matching road/building/voxel/task-registry inputs. The author's existing local data and the previously verified Table VIII measurements are preserved. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md), [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Quick start: pretrained evaluation

Run commands from the repository root. The validated runtime is Windows, Python **3.9.21**, PyTorch **2.7.0+cpu**, and NumPy **1.26.4**. Dependencies are pinned to the tested versions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

Wheel options are in the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v270). No torchvision/torchaudio is needed. See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for the local OpenMP caveat and optional CUDA setup.

### Smoke test, not paper reproduction

Before either smoke or full evaluation, obtain authorized matching inputs, restore the unchanged scenario paths and check their identity:

```powershell
python -m tools.check_data_assets
```

This checks the 12 core case-input hashes recorded in `reproduction/evaluation_manifest.json`. Missing inputs are expected in a clean release and cause a nonzero exit. It does not download maps, substitute synthetic scenes, verify legal authorization or change simulation logic. File layouts are documented in [data/README.md](data/README.md) and [truck_uav_modules/data/README.md](truck_uav_modules/data/README.md).

The weights can be checked **without geographic data**:

```powershell
python -c "from dynamic_patrol_marl.rl.checkpoint_io import CASES, load_policy; [load_policy(c) for c in CASES]; print('Both original models loaded')"
```

After authorized data pass the identity check, run one **complete seed-1000 episode per case**:

```powershell
python -m rl.reproduce_paper --smoke --workers 2
```

Results go to `results/paper_smoke/`. This verifies execution, but cannot certify 20-seed statistics.

### Full Table VIII reproduction

```powershell
python -m rl.reproduce_paper --workers 4
```

This evaluates both original policies on **20 matched unseen seeds, 1000–1019**. Parameters are frozen, NoisyNet exploration is disabled in `eval()` mode, and selection is feasible-action masked argmax. No optimizer/replay buffer is created. Use `--workers 1` on resource-limited machines.

Individual cases:

```powershell
python -m rl.evaluate_checkpoint --case case1_xinjiekou --workers 2
python -m rl.evaluate_checkpoint --case case2_baijiahu --workers 2
```

The combined run saves under `results/paper_reproduction_20seeds/`:

- `per_seed_metrics.csv`: six metrics, seeds, episode counts, integrity diagnostics.
- `summary_mean_std.csv`: unweighted episode means, **sample SD (ddof=1)**, valid/missing counts.
- `reference_comparison.csv`: differences from manuscript means and SDs, without numerical correction.
- `paper_summary.csv`: wide mean table.
- `evaluation_manifest.json` and `episodes/*.json`: source, checkpoint, input-data and task-stream hashes, versions and resumable records.

Identical evaluations resume safely. Changes to source, checkpoints, data or the recorded environment require a new `--output-dir`. Generated results and new training checkpoints are ignored by Git.

## Verified performance

On 2026-10-11, fresh local evaluation using the author's original-case inputs covered all **40 case–seed combinations** and reproduced **all six means and sample SDs at Table VIII's two-decimal precision**. No measurements were adjusted. Those inputs are now withheld from the public Git index; this historical verification is not a claim that a data-free checkout can execute the cases.

The full evaluation was repeated on 2026-10-12 after offline voxel reconstruction
was restored: all 240 per-seed metric values and task-stream hashes are exactly
unchanged. `reproduction/` contains that latest-source validation and manifest.

| Metric | Case 1: Xinjiekou | Case 2: Baijiahu |
| --- | ---: | ---: |
| CR (%) ↑ | 90.27 ± 3.65 | 86.53 ± 3.95 |
| MRT (s) ↓ | 365.81 ± 28.08 | 436.60 ± 45.83 |
| HPRT (s) ↓ | 269.76 ± 30.08 | 274.06 ± 48.21 |
| ECT (units/task) ↓ | 230.23 ± 17.40 | 252.10 ± 18.12 |
| CTD (km) ↓ | 15.87 ± 5.61 | 12.88 ± 3.76 |
| CURW (s) ↓ | 1130.62 ± 435.81 | 887.68 ± 527.24 |

Validation CSVs are in [reproduction/](reproduction/). SD describes evaluation-realization variability for **one seed-42 trained policy**, not variability across training runs. Provenance and remaining qualifications are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md).

## Cases and checkpoints

| Public case | Config | Nodes | Physical links | Directed simulation arcs |
| --- | --- | ---: | ---: | ---: |
| `case1_xinjiekou` | `configs/case1_xinjiekou.yaml` | 333 | 601 | 1202 |
| `case2_baijiahu` | `configs/case2_baijiahu.yaml` | 129 | 247 | 494 |

The source graphs are undirected `MultiGraph`s; the simulator expands physical links into directed arcs. Both cases retain one CAV, three UAVs, battery capacity 1800, 25 action slots, original task/traffic randomization, a 10,800-s task-release period and up to 1,800 s of drain time.

The two approximately 0.50-MB policies are included directly:

- `checkpoints/cs_rdqn_case1_xinjiekou.pt`
- `checkpoints/cs_rdqn_case2_baijiahu.pt`

No external model download or Git LFS is needed. They contain original final **300k, seed-42 weights and buffers**, embedded evaluation configs and provenance. Optimizer state/private paths are omitted. Both weight artifacts are MIT-licensed with explicit author approval. The audit found no imported third-party pretrained backbone, but acquisition/use conditions for the training maps remain unresolved and the author license cannot override upstream data terms. See [checkpoints/README.md](checkpoints/README.md), [checkpoints/MODEL_CARD.md](checkpoints/MODEL_CARD.md) and `checkpoints/manifest.json`.

The observation remains **75 + 25 × 38 = 1025**. The original network actually uses **113 global-encoder inputs (75 + 38 safe-slot features)** and 24 shared non-safe candidate encodings. This split is preserved, not redesigned to accept only 75 global inputs.

## Repository structure

### Offline voxel reconstruction

With authorized local building, road-edge and launch-node GeoJSONs:

```powershell
python -m tools.rebuild_voxel_maps --scene all --output-dir results/voxel_rebuild --astar-trials 10
```

This independently derives the 10 x 10 x 20 m maps, without old 2-D grid metadata
or mapping CSVs. New outputs are compared against existing canonical arrays;
canonical assets are never overwritten. Both original scenes were locally
rebuilt with **zero differing voxels** (10,348 / 6,872 buildings), with 10 validated
A* paths each. Six Baijiahu supplied default heights are retained and disclosed.
See [VOXEL_REBUILD.md](VOXEL_REBUILD.md) for exact aggregation, clearance,
metadata-hash caveats and verification. The acquisition channel is reported as
a WeChat public account; permission evidence is still missing, so geographic
inputs and derivatives are not bundled or relicensed.

```text
configs/                 public cases and legacy filename aliases
checkpoints/             original-weight policies and provenance sidecars
env/                     event-driven traffic, tasks and CAV/UAV simulation
planner/                 sortie candidates, joint plans and recovery insertion
rl/                      main CS-RDQN training, loading and frozen evaluation
utils/                   configuration, timing and metric helpers
dynamic_patrol_marl/      compatibility package for existing imports
truck_uav_modules/       ground/3-D runtime; map-input README, not bundled maps
data/                    withheld task-registry layout and access notes
tools/                   data identity preflight and offline voxel reconstruction
tests/                   simulation, SMDP and release regression tests
reproduction/            fresh 20-seed validation CSVs and manifest
```

`truck_uav_modules/build_and_benchmark_3d_voxel_astar.py` is a **required A* runtime**, despite its filename. Shared compatibility hooks remain where deleting them could affect the main simulator; standalone baseline/ablation runners and state-only ablation networks are excluded.

## Optional training from scratch

Not required for pretrained reproduction. These commands also require authorized original-case inputs; run the data preflight first:

```powershell
python -m rl.train_masked_dqn --config configs/case1_xinjiekou.yaml --algo rainbow_dqn --seed 42 --timesteps 300000 --disable-tensorboard
python -m rl.train_masked_dqn --config configs/case2_baijiahu.yaml --algo rainbow_dqn --seed 42 --timesteps 300000 --disable-tensorboard
```

These preserve the existing training loop and do not overwrite release checkpoints. Historical training-entry YAMLs are unavailable; public configs preserve the audited runtime used for verified evaluation. Bitwise-identical retraining is **not** certified; see [DEFAULT_TRAINING.md](DEFAULT_TRAINING.md).

## Tests, citation and data

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m rl.train_masked_dqn --help
```

Environment/planning tests require the withheld inputs and were validated locally
(82 tests passed). A snapshot of the actual public file set, with no scene data,
also passed 21 independent tests (one Git-index check skipped in the export),
and both pretrained models loaded. This snapshot reused the author's installed
environment; it is not certification of a fresh installation or original-case
rollout. Run the data-independent tests separately:

```powershell
python -m pytest -q tests/test_license_release_scope.py tests/test_voxel_rebuild.py
```

The data preflight correctly reports missing inputs in that snapshot. Never
substitute maps or change measurements to hide the reproducibility limitation.

[CITATION.cff](CITATION.cff) records manuscript authors/title and repository metadata. The source code and named original models are **MIT-licensed** under [LICENSE](LICENSE); third-party libraries retain their own licenses. No MIT grant is applied to geographic data. Road-centreline metadata and building-height fields point to **Baidu-derived inputs**, but acquisition records and redistribution permission were not found. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for the evidence and official terms checked.

Data are excluded from the **current release tree**, not erased from existing
Git history: prior commits can still contain them. Publication of the code/model
release is author-approved; no history rewrite or force-push is included.
Source configs, numerical behavior, weights and measured results remain unchanged.
