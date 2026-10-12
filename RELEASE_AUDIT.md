# Local research-code release audit

## 2026-10-12 voxel reconstruction and approved publication

- Author authorized code publication and clarified acquisition through a WeChat
  public account. No article/account or redistribution permission was supplied;
  all 31 geographic/derived files remain excluded, physically unchanged locally.
- Restored the missing `rasterize_building_occupancy.py` and added
  `tools/rebuild_voxel_maps.py`. Extents and launch mappings derive from local
  GeoJSONs, without historical 2-D grids, off-repository files or prebuilt-map
  input to construction. New ignored outputs never overwrite canonical assets.
- Both scenes rebuilt exactly: 10,348 / 6,872 buildings; 77,427 / 51,719 occupied
  voxels; zero differing voxels and matching NPY hashes. Numerical metadata also
  matches (excluding nondeterministic build timing). A* validated 10 paths each.
- The core A* module changes only three offline loading functions; AST comparison
  confirmed all search/move/cost/coordinate functions unchanged. Historical 10 m
  benchmark defaults are retained; the new command builds the paper's 20 m z grid.
- Added 14 data-independent voxel tests; total local suite **82 passed**. The
  initial test run had a fixture dtype warning, subsequently corrected and retested.
- Exported the actual staged public file set into a new ignored directory with
  no map data: **21 independent tests passed, one Git-index check skipped**
  (no `.git` in the export), both policies loaded, and all 12 core inputs were
  explicitly reported missing with the expected nonzero preflight exit. The
  author's existing environment was reused; no fresh-install claim is made.
- Full frozen evaluation repeated: **40 episodes, 20 seeds 1000--1019 per case**.
  All 240 per-seed metric values and task streams are exactly unchanged; all six
  Table VIII means/sample SDs still match. No training/model/experiment edits.
- Latest source fingerprint:
  `7bb57e283b26f1743d58c295140d31a36185a72f847f1200d4d5d4ab5b435dff`.
  `reproduction/` records the latest run. Original run records remain in ignored
  `results/release_reproduction_final_20seeds/`.
- `VOXEL_REBUILD.md` records construction, verification and the six supplied
  default heights. Code/model MIT grants still do not establish map permissions.
- Normal fast-forward publication is authorized; **history rewriting/force-push
  is not**. Old Git commits may retain geographic files. Current-tree exclusion
  does not retract those historical copies or resolve upstream rights.

The sections below are historical, superseded where they describe then-current
test counts, source fingerprints, publication authorization or release gates.

## 2026-10-12 licensing/data-scope update (supersedes prior release-gate wording)

- Added the standard root MIT `LICENSE`, covering author-owned code and both original model artifacts by explicit author approval; `LICENSE_SCOPE.md` excludes geographic data and third-party library rights.
- Added `DATA_PROVENANCE.md`, `checkpoints/MODEL_CARD.md`, data-layout READMEs and `tools/check_data_assets.py`.
- Found Baidu road-centreline and `local_baidu_height` provenance indicators; no source redistribution authorization/asset license was located.
- Removed **31** geographic/derived inputs from the current Git index with `git rm --cached`; all local files and SHA-256 values were verified unchanged. This is not physical deletion.
- The remote HEAD inspected still contains those data in historical commits; no history rewriting, force-push or remote removal was performed.
- Runtime source, scenario configs, original model bytes and experimental results were not changed. Previously verified local reproduction remains valid, but a data-free checkout cannot execute the original cases.
- Code/model MIT grants do not override unresolved upstream input-data contracts. See `DATA_AVAILABILITY.md` for current scope and `DATA_PROVENANCE.md` for evidence.
- Post-update validation: **68 tests passed**. Both models load independently; all 12 local core-input hashes still match. The original runtime source fingerprint and both official checkpoint SHA-256 values are unchanged. No 20-seed evaluation or training was repeated solely for license/index/documentation changes.

The sections below retain the **2026-10-11 technical refactor record and its then-current file counts**; license/data-gate text is historical and superseded by this update.

## Outcome

Prepared locally on 2026-10-11 for **EC-SRL-CAV-UAV-Patrol**. No training/fine-tuning, manuscript/table editing, Git commit or GitHub push was performed. Changes are uncommitted in the existing checkout, based on local commit `92271ae49615eba2f9d3ee9ff16b7e8128e81460`; the remote HEAD inspected during preflight differed. No pull/reset/remote overwrite was used to replace the user's local project.

**Technical verification passed. Public redistribution still requires author license/data-permission decisions.** Both original main checkpoints and runtime data are present. Historical full training-entry YAMLs remain unavailable and are disclosed rather than recreated as supposedly verbatim originals.

The final complete frozen-policy run used 20 seeds 1000–1019 per case (40 episodes), with seed-42 / final-300k original policies. All six means and sample SDs match Table VIII at two decimals. The import-compatibility fix was followed by another complete evaluation: all 240 per-seed metric values and all task-stream hashes were unchanged.

## Validation results

| Check | Result |
| --- | --- |
| Regression tests | **60 passed**; no training started |
| Runtime package imports | **31 submodules imported**; package-aware and legacy planning imports also tested in fresh subprocesses |
| Python compilation | Passed for environment, planner, RL, utilities, compatibility package, planning modules and tests |
| Dependency check | `pip check`: no broken requirements; pinned direct/dev dependency dry-run satisfied in the tested environment |
| Config/assets | Both cases initialized; 333/129 nodes, 601/247 physical links, 1202/494 directed simulation arcs |
| Original/released models | All parameters/buffers and tested forward outputs bitwise identical; both original models loaded strictly |
| Policy contract | Frozen eval-mode masked argmax; NoisyNet disabled; no optimizer/replay initialization |
| Smoke test | Complete seed-1000 episodes for both cases; final full evaluation also covers those same seeds |
| Final paper evaluation | 40/40 completed, 20 valid observations for every metric/case; zero illegal actions, zero commit failures, zero truncated episodes |
| Planning diagnostics | 6 planning failures in Case 1 and 1 in Case 2 across the 20 episodes; retained as real simulator outcomes, not suppressed |
| Table VIII comparison | All 12 mean and all 12 sample-SD comparisons match at displayed precision |
| Data/source preservation | Environment, joint/sortie generation, recovery scheduling, reward, traffic, replay, SMDP, source maps and A* algorithm core unchanged |
| Privacy/ignore rules | Manuscripts, historical runs, archives and nonofficial model files excluded; published metadata contains no absolute machine paths/optimizer state; only the two official .pt files whitelisted |
| Citation | Verified manuscript title/authors, repository URL and YAML/basic CFF fields; no invented DOI/acceptance claim; external full CFF schema validation not run |
| Patch formatting | `git diff --check` passed |

Validated on Windows / Python 3.9.21 / torch 2.7.0+cpu / NumPy 1.26.4. The existing environment required the explicitly recorded process-local OpenMP workaround. Fresh installation, other platforms and GPU numerical reproduction have not been independently certified. See `ENVIRONMENT_SETUP.md` and `REPRODUCIBILITY.md`.

## Modified files (14)

- `.gitignore`: official model whitelist and historical-output exclusions.
- `README.md`: evaluation-first quick start, cases, original checkpoints, real validation results and release scope.
- `DEFAULT_TRAINING.md`: optional public-case training, missing historical YAML and noncertified retraining caveats.
- `ENVIRONMENT_SETUP.md`: actual tested runtime, dependency commands, CPU/CUDA distinction and OpenMP disclosure.
- `PAPER_ALIGNMENT.md`: current-PDF audit; obsolete PER/congestion warnings removed, unresolved description differences retained.
- `configs/default_training.yaml`, `configs/baijiahu_training.yaml`: compatibility filename aliases to the new public configurations.
- `requirements.txt`: direct dependencies pinned to the tested versions, including Torch; pytest moved to dev requirements.
- `requirements-torch-cu128.txt`: optional Torch wheel version aligned with the tested 2.7.0 API.
- `truck_uav_modules/requirements.txt`: uses root dependency definitions rather than conflicting duplicate version ranges.
- `rl/dqn_networks.py`: standalone state-only ablation networks removed; factory exposes only the main candidate-aware Rainbow network. Its encoder, dueling and NoisyNet implementation is unchanged.
- `rl/train_masked_dqn.py`: public default case and supported algorithm restricted to CS-RDQN; main training loop/calculations untouched.
- `truck_uav_modules/uav_path_planner/planner.py`, `voxel_astar.py`: package-aware imports added; original short-import runtime route and planning computations unchanged.

## Added files (24)

- `configs/case1_xinjiekou.yaml`, `configs/case2_baijiahu.yaml`.
- `rl/checkpoint_io.py`, `rl/evaluate_checkpoint.py`, `rl/reproduce_paper.py`, `rl/prepare_pretrained_release.py`.
- `checkpoints/cs_rdqn_case1_xinjiekou.pt`, `cs_rdqn_case2_baijiahu.pt`, their two `.json` sidecars, `checkpoints/manifest.json`, `checkpoints/README.md`.
- `requirements-dev.txt` and `tests/test_release_reproduction.py`.
- `CITATION.cff`, `DATA_AVAILABILITY.md`, `REPRODUCIBILITY.md`, this `RELEASE_AUDIT.md`.
- `reproduction/README.md`, `per_seed_metrics.csv`, `summary_mean_std.csv`, `reference_comparison.csv`, `paper_summary.csv`, `evaluation_manifest.json`.

The official checkpoints are 502799 bytes each; no LFS/Release download is required. Repackaging removes private metadata and optimizer state, not model tensors. Original/released-file/tensor hashes are retained in `checkpoints/manifest.json`.

## Removed from the live release, recoverable locally

No tracked file was permanently deleted. Existing local, previously untracked helpers were moved into `.codex_work/release_archive/`, which is Git-ignored:

- Scripts: `evaluate_ablations_50seeds.py`, `evaluate_joint_sensitivity.py`, `plot_ablation_degradation.py`, `plot_joint_sensitivity_boxplots.py`, `plot_joint_sensitivity_heatmaps.py`.
- Tests: `test_ablation_degradation_plot.py`, `test_ablation_evaluation.py`, `test_joint_sensitivity_boxplots.py`, `test_joint_sensitivity_evaluation.py`, `test_joint_sensitivity_heatmaps.py`.

Original experiment CSVs, plots, manuscript/PDF and source training models remain untouched locally. State-only network classes and non-main factory choices were removed from the active network source; the original network source is also archived for integrity checking. Baseline files already absent from this checkout were not falsely reported as deleted.

Required inheritance/compatibility modules and the historically named `build_and_benchmark_3d_voxel_astar.py` remain. Dormant shared simulator/trainer switches are deliberately retained where deleting them would risk changing main-method behavior; this is not a release of complete ablation pipelines.

## Outstanding author decisions

1. Select and authorize a software/checkpoint license. No LICENSE grant was guessed.
2. Verify original road/building sources, attribution and redistribution rights, including derived grids/registries.
3. Resolve manuscript wording on physical vs directed links, actual global-input split, geometry-dependent service times and UAV speed variability. Current PDF PER beta 0.4→1.0 and congestion priority 6 already agree.
4. Decide the final approved version/archive identifier before any later public release. No DOI/deposit or GitHub publication was created here.

The data-availability audit distinguishes local reproducibility from legal/public availability and supplies dataset-to-file mapping and units. These remaining decisions do not invalidate the verified frozen-policy Table VIII reproduction, but prevent an unconditional claim of publication-ready redistribution.
