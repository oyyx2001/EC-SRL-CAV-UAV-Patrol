# License scope

The authors selected the **MIT License** for their source code and explicitly authorized MIT licensing of both original pretrained CS-RDQN artifacts on 2026-10-12. The standard terms are in [LICENSE](LICENSE).

## Covered author contributions

- Author-owned source code and associated software documentation in this release.
- `checkpoints/cs_rdqn_case1_xinjiekou.pt`.
- `checkpoints/cs_rdqn_case2_baijiahu.pt`.

No model tensors, embedded configurations or checkpoint bytes were changed to add this external license notice.

## Not relicensed under MIT

- Raw or processed geographic datasets, road networks, buildings, launch/recovery coordinates, derived occupancy grids and geometry-derived task registries.
- Third-party libraries installed separately from `requirements.txt`; their own licenses remain applicable.
- Content or rights the authors do not own or have authority to license.

The geographic inputs are excluded from the current Git index and covered by ignore rules pending source-rights verification. They remain in the author's local workspace; MIT is not a grant to acquire, use or redistribute them.

The model audit found no imported third-party pretrained backbone or model weights in the main training pipeline. This is not proof that every historical input-data acquisition/use condition permits model distribution. Author MIT authorization covers the authors' rights in the checkpoint artifacts and does not override upstream data contracts. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and [checkpoints/MODEL_CARD.md](checkpoints/MODEL_CARD.md).
