# Data and model redistribution audit

Audit date: **2026-10-12**. This is an evidence-based release-scope check, not a legal clearance or a claim that general online terms govern a specifically identified acquisition contract.

## Local evidence

**Author clarification (2026-10-12):** the files were obtained through a WeChat
public account (公众号). The account name/article URL, acquisition date/version,
upstream footprint provider and any license or permission document were not
supplied. This establishes a reported acquisition channel, **not** an open
redistribution grant or proof of rights-holder consent. The Baidu indicators
below are file metadata, not independent verification of the public account's
upstream source. Code publication is authorized; these geographic files and
derivatives remain withheld. Do not assign MIT to them.

| Input | Verified source indicator | Permission evidence | Release treatment |
| --- | --- | --- | --- |
| Xinjiekou road graphs/segments | `road_topology_summary.json` says repaired local **Baidu road centrelines** | No license, attribution file, acquisition record or written redistribution authorization found alongside assets | Withhold geographic files |
| Baijiahu road graphs/segments | Same Baidu-centreline note in the corresponding summary | Same unresolved status | Withhold geographic files |
| Xinjiekou building heights | All 10348 features have `height_source=local_baidu_height` | No redistribution permission found; footprint provider/acquisition provenance not independently established | Withhold entire building GeoJSON |
| Baijiahu building heights | 6866 features have `local_baidu_height`; 6 have `configured_default` | Defaulting six heights does not establish rights in the remaining heights or building geometry | Withhold entire building GeoJSON |
| Launch/recovery coordinates and compressed graphs | Derived from the above road network | No separate derivative-data permission found | Withhold derivatives |
| Voxel occupancy grids | Existing metadata describes building/road-derived voxelization and clearance | No independent derivative-data permission found | Withhold NPY/NPZ and associated spatial metadata |
| Edge-task registries | Geometry-derived road segment identifiers, lengths and task parameters | No independent derivative-data permission found | Withhold the two CSV inputs |

The `source` property in road-edge features names a **graph endpoint**, not a provider. Filenames mentioning OSM elsewhere in the project do not establish OSM provenance for these two final cases. No ODbL/OSM license is assigned to these inputs.

## Official terms checked

- [Baidu Maps Open Platform service terms](https://lbsyun.baidu.com/docs/pcsa?title=law/open/law): sections 3.3.2–3.3.4 place restrictions on obtaining/storing data, independent use, derivatives and public display.
- [Baidu Maps user service terms](https://map.baidu.com/zt/client/service/index.html): intellectual-property section reserves rights in map content and addresses prior permission.

These sources do **not** establish an open redistribution grant for the local files. The exact acquisition channel, version/date, provider contract and any separate written authorization remain unknown. The conservative decision is to withhold, not to declare infringement or to infer a free license.

## Model provenance and remaining limitation

The two final policy hashes match the original seed-42 / 300k artifacts. The main implementation constructs its own candidate-aware Rainbow network and optimizer; no torchvision/Transformers/Hugging Face backbone or third-party pretrained-weight import was found.

Case 1 resumes from `checkpoint_step_160000.pt`. That locally inspected intermediate checkpoint is the same project's `rainbow_dqn`, seed 42, 1025-observation/25-action model, and contains no further resume source. Case 2's original final checkpoint has no resume source. No imported external pretrained model was identified from these records.

Both trained artifacts are now MIT-licensed with the author's explicit approval; original bytes and tensor hashes are unchanged. **Training-data rights are a separate unresolved dependency.** The available records do not establish acquisition terms or whether they restrict training/model distribution. Do not present this audit as unconditional legal clearance of the weights. Obtain provider/rights-holder confirmation if the applicable agreement covers such outputs.

## Effective exclusion and history warning

All **31** previously tracked geographic/derived input files are removed from the current Git index using `git rm --cached`, not from disk. Ignore rules prevent automatic re-addition. Source interfaces and scenario configs retain the original paths and numerical settings.

The remote HEAD inspected **before the approved code publication** on this date
was `2026833b79dfede6a938bb36d4e44d3bf3fdbe87`; that historical commit contains all
31 input files. The new code release excludes them from its current tree.
Excluding current-version files **does not remove earlier commits or retract
historical remote copies**. No history rewriting or force-push is authorized or
performed. If historical public copies must be withdrawn, the author must
explicitly authorize a separate coordinated history/removal workflow.

## Required author evidence

1. Original road/building provider and acquisition channel/date/version, including footprint origin.
2. Applicable license/contract or written permission covering acquisition, research use, derivatives and public redistribution.
3. Whether that agreement imposes conditions on trained model outputs.
4. Required source attribution and any geographic-data publication conditions.

Until resolved, a clean release cannot independently run the original cases. Users must obtain authorized matching inputs and restore the exact configured paths. Checksums verify identity, **not permission**. Existing local Table VIII verification remains valid as a numerical record, not a claim of open-data access.
