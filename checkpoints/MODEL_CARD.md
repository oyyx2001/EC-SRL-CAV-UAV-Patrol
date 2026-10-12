# Original CS-RDQN policy artifacts

## Model and license

| Artifact | Training | License |
| --- | --- | --- |
| `cs_rdqn_case1_xinjiekou.pt` | Seed 42, final 300k; resumes the same project's 160k checkpoint | MIT (author-approved 2026-10-12) |
| `cs_rdqn_case2_baijiahu.pt` | Seed 42, final 300k; no resume source in original metadata | MIT (author-approved 2026-10-12) |

Standard terms are in the repository [LICENSE](../LICENSE); scope is explained in [LICENSE_SCOPE.md](../LICENSE_SCOPE.md). License notices are external to the .pt blobs, so original released-file and tensor hashes remain unchanged.

## Intended use and evidence

The policies select masked candidate joint plans for the original one-CAV/three-UAV urban traffic-patrol simulation. They are not general-purpose pretrained models or safety-certified operational controllers. Architecture and normalization must match the embedded configuration; the actual global input split remains 113.

Original metadata, intermediate-model provenance and local training source were inspected. No external pretrained backbone or imported third-party model weights were identified. Required libraries are installed separately under their own licenses; no library code is bundled into the model state dictionaries.

On 2026-10-11, original-case frozen evaluation on seeds 1000–1019 per case matched all Table VIII means and sample SDs at two decimals. These are evaluation realizations of one trained policy per case, not independent training repetitions.

## Data and distribution qualifications

The training/evaluation scenes contain geographic inputs whose rights are unresolved. Baidu source indicators exist for road centrelines and heights; acquisition/contract records were not located. The author's MIT grant does not override third-party data rights or prove that all applicable input-data agreements permit training/model distribution. See [DATA_PROVENANCE.md](../DATA_PROVENANCE.md).

Original maps, voxel grids and task registries are withheld from the public Git index. Consequently the weights can load independently, but **a clean release cannot execute the original scenarios without authorized matching input data**. There is no automatic download or synthetic fallback. Use `python -m tools.check_data_assets` before evaluation.
