# Reproducibility record

## Scope and outcome

On 2026-10-11 the two **original final seed-42 / 300k CS-RDQN policies** were evaluated afresh on 20 paired unseen seeds each, **1000–1019**, with the current audited simulator. All six means **and sample standard deviations** match Table VIII at two decimals. Forty unique case–seed episodes completed; all six metrics have 20 valid observations per case. No original measurement, reward, task seed or output was adjusted.

Validation uses a single trained policy per case. Between-evaluation-seed SD is not between-training-run SD and does not establish robustness to training initialization.

The fresh raw/summary/reference tables and manifest are in `reproduction/`.
On 2026-10-12 the complete 40-episode evaluation was repeated after restoring
offline voxel-generation helpers and three offline loader fallbacks. **All 240
per-seed metric values and all task-stream hashes are exactly unchanged** from
the 2026-10-11 validation, and all Table VIII means/SDs still match. Latest
runtime records are in the ignored
`results/release_reproduction_voxel_fix_20seeds_20261012/`; the earlier run remains
intact in `results/release_reproduction_final_20seeds/`. No training occurred.

## Policy and metric contract

- Frozen `eval()` network, all parameters `requires_grad=False`; NoisyNet disabled and epsilon=0.
- Feasible-action masked argmax, no optimizer/replay buffer, no training/fine-tuning.
- Original task generation and time-varying traffic; no task-load, horizon, recovery, reward or UAV-count overrides.
- CR = 100 × final completed / released; MRT and HPRT average response times over completed tasks; ECT = total UAV energy / completed tasks; CTD = CAV distance in meters / 1000; CURW = simulator cumulative recovery waiting.
- Statistics are unweighted episode means and sample SD (`ddof=1`). Undefined metrics remain missing, not zero; valid/missing counts are reported.
- The comparison checks the displayed two-decimal manuscript precision, not a permissive relative-error threshold. Smoke/partial evaluations cannot certify reproduction.

The evaluator hashes source, original-model release artifacts, runtime assets and per-seed task streams, and records package versions and numerical runtime settings. It refuses to mix cached records from a changed evaluation identity.

## Checkpoint provenance

Original artifacts were recovered from the corresponding `results/algorithm_comparison/*_random_traffic_300k_v1/rainbow_dqn/` runs. They are the final paper policies, not fleet-size validation-best models. Training seed 42, 300k budget, network dimensions, PER settings and reward signature were checked against original checkpoint metadata and matching run summaries.

| Case | Original artifact SHA-256 | Released artifact SHA-256 |
| --- | --- | --- |
| Case 1 | `42c587522c55fa055fc9c77a3120d4a072899bbdc1e751b1f4d8119f568f208f` | `f2d07f011a9ed8926b8e51280ef7b4c7e2643be588ba949fa55a48d9e0f38dd1` |
| Case 2 | `619b90b1a5d7a471d8ed14817a01ad3f07b14edd84532ccb0e3270776cd5dd31` | `8953d8632646b9ce06a8f1c2269c3b3d881a8bdc29947eaa3d0722313013ee98` |

Every original model parameter and buffer was compared bitwise to its released counterpart; old and release network forward outputs also matched bitwise on real observations. Optimizer state/private run paths were stripped. Embedded metadata and JSON sidecars retain provenance, feature schema, original replay/hyperparameters, config/reward signatures and stable tensor-content hashes.

**The historical training-entry YAMLs are missing.** Legacy retest manifests refer to `configs/ablations/formal/xinjiekou_full.yaml` and `baijiahu_full.yaml`. Current public YAMLs preserve the audited runtime and reproduce frozen-policy results, but this is not proof of bitwise-identical retraining or verbatim recovery of all historical training configuration text.

## Architecture and configuration qualifications

The raw observation is 75 + 25 × 38 = 1025 and there are 25 actions. The actual original network treats 24 non-safe slots as shared candidates and absorbs the safe slot's 38 features into its global input: **113 rather than 75 encoder inputs**. This is retained, including the original weights and tensor shapes; no architectural repair is applied to pretrained evaluation.

Public config standardization changes scenario labels only. Tests against the pre-refactor legacy entries verified identical seed-1000 task-stream hashes, observations and feasible-action masks in both cases. Environment, planner, recovery scheduler, reward ledger, SMDP, replay, traffic, data and the 3-D A* core were not rewritten.

Package-aware import compatibility was added to the existing 2-D planning interface and 3-D wrapper, preserving the original short-import route. Both namespaced and legacy imports now pass in fresh interpreters. Full evaluation was repeated after this change: every one of the 240 per-seed metric values and all task-stream hashes stayed identical. That 2026-10-11 source fingerprint was `52464eb30244191b64721c07ba977da4dade151ff1242ca26235487c9b861655`.

The 2026-10-12 source fingerprint is
`7bb57e283b26f1743d58c295140d31a36185a72f847f1200d4d5d4ab5b435dff`.
An AST comparison with the prior committed A* module identifies changes only
to `load_grid_metadata`, `load_launch_recovery_mapping` and
`load_and_clean_buildings`. Search/move/cost/coordinate-conversion definitions
are unchanged. New offline rasterization is not invoked by frozen evaluation.
Both scene arrays were independently reconstructed from GeoJSONs with zero
differing voxels and exact NPY hashes; see `VOXEL_REBUILD.md`.

PER beta is 0.4 → 1.0 over 200k; congestion priority is 6. Both match the **current supplied PDF**. Earlier alignment notes claiming beta 0.1 or priority 4 were outdated and have been replaced. See `PAPER_ALIGNMENT.md` for remaining service-time, speed-uncertainty and road-count wording differences.

## Runtime and limits

Validated on Windows / Python 3.9.21 / torch 2.7.0+cpu / NumPy 1.26.4, four evaluation workers with one Torch/BLAS thread per worker. Direct dependencies are pinned. The existing environment passed `pip check` but required `KMP_DUPLICATE_LIB_OK=TRUE`; the workaround is explicit in the manifest, not silently enabled by the evaluator.

The main planner retains physical-time planning budgets. On other/slower machines, wall-clock planning timeouts may alter outcomes even with identical random seeds; do not disable those budgets to force agreement. Inspect planning/commit failures, input/source hashes and dependency versions first if reproduction differs.

A clean installation, Linux/macOS and GPU numerical reproduction are not independently certified. The released policies omit optimizer state and cannot resume original training. No retraining or manuscript/table edits were performed. Code publication was subsequently authorized on 2026-10-12; the earlier no-push restriction no longer applies to the approved code/model release, but not to unlicensed geographic data or historical Git rewriting.

## Release gates

On 2026-10-12 the author selected **MIT for both code and the two original models**. Geographic assets remain intact locally, but **31 geographic/derived input files are removed from the public Git index** pending source-rights verification. Historical entry YAMLs remain unavailable. A clean checkout cannot run the original cases without authorized matching inputs; the previous local numerical verification does not promise otherwise.

Data acquisition/redistribution and any input-contract conditions on model outputs are still unresolved. `DATA_PROVENANCE.md` records Baidu source indicators, the author's reported WeChat-public-account acquisition channel, official terms, model-origin checks and the historical Git warning. Existing historical commits are not rewritten; an ordinary code-release commit withholds the 31 files from its current tree but cannot retract old copies. Licensing/data-index changes did not change the runtime fingerprint; the later offline-loader/helper additions changed the fingerprint as explicitly recorded above, not policy outputs, weights or measurements.
