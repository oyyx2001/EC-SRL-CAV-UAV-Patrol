# Original pretrained CS-RDQN policies

| Case | Artifact | Bytes | Training seed | Decision transitions |
| --- | --- | ---: | ---: | ---: |
| `case1_xinjiekou` | `cs_rdqn_case1_xinjiekou.pt` | 502799 | 42 | 300000 |
| `case2_baijiahu` | `cs_rdqn_case2_baijiahu.pt` | 502799 | 42 | 300000 |

These are repackaged **original final paper-model weights**, not new training, 500k checkpoints or fleet-size sensitivity models. All parameters and buffers, including NoisyNet buffers, are bitwise identical to the recovered originals. Removing optimizer state/private paths changes artifact bytes, **not network tensors**. `manifest.json` records original, release-file and tensor-content hashes.

Each .pt embeds a portable evaluation config, dimensions, feature schema, original hyperparameters, replay settings, training stages, reward metadata and integrity hashes. A .json sidecar accompanies it. Original final files omitted a top-level transition count; 300k is established by training stages **and the matching original run summary**, not merely a filename.

## Evaluate from the repository root

The weights load independently of scene data. Original-case evaluation requires authorized matching input assets, which are withheld from the public release. Run `python -m tools.check_data_assets` before these rollout commands; the local author's existing data were not deleted.

```powershell
python -m rl.evaluate_checkpoint --case case1_xinjiekou --smoke
python -m rl.evaluate_checkpoint --case case2_baijiahu --smoke
python -m rl.reproduce_paper --workers 4
```

The loader uses `torch.load(..., weights_only=True)`, verifies case/architecture, config and reward signatures, tensor hashes, and strict state-dict loading. It never initializes an optimizer/replay buffer. No historical result directory is required.

The raw observation is 1025 features (75 + 25 × 38), with 25 action slots. The original global encoder receives **113 inputs**, absorbing the safe slot's 38 features; candidate encoding is shared across 24 non-safe slots. Changing this split would invalidate the original model, so it was retained.

Historical training-entry YAMLs were not found. Embedded evaluation configs preserve the audited current runtime with standardized public case labels. They successfully reproduced all Table VIII means and sample SDs on 20 unseen seeds per case; they are **not claimed to be recovered verbatim training YAMLs**.

`rl.prepare_pretrained_release` is a maintainer-only packaging utility requiring hash-pinned originals and run summaries. It refuses missing/different source artifacts and never trains replacements. Evaluation users should not run it to create substitute checkpoints. Existing release files are not overwritten if their config/tensor hashes differ.

The files are below 1 MB and remain included directly. Both artifacts use **MIT** with explicit author approval on 2026-10-12; standard terms are in the root `LICENSE`. No checkpoint bytes or tensors were changed to add license notices. See `MODEL_CARD.md` for model-origin evidence and the unresolved upstream input-data conditions. No third-party pretrained weights were identified, but that does not establish legal clearance of every training-data contract.
