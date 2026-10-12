# Code, model and data availability

## Code and pretrained models

The author selected **MIT** for author-owned code and explicitly approved MIT licensing of both original CS-RDQN weight artifacts on **2026-10-12**. The standard grant is in [LICENSE](LICENSE), with scope in [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Model/license notices do not modify any checkpoint bytes or tensors.

The main method, data-reading interfaces, public scenario configurations, original policies, frozen evaluation scripts, per-seed metrics and summary tables remain in the release. No training, simulation rewrite, numerical correction, Git commit or GitHub push was performed in this licensing update.

## Geographic inputs: withheld, not MIT

The author reports acquisition through a WeChat public account, but has not
provided its name/article, an upstream rights-holder record or a redistribution
license. Code publication has been authorized; this does not authorize a new
license grant for third-party maps. Local copies are preserved for validation.

A clean public version **does not contain the original geographic inputs or their derivatives**. The author’s existing local files remain unchanged at their configured paths, but the 31 previously tracked input files have been removed from the current Git index and ignored.

| Evidence/input | Expected path | Source/permission status | Current treatment |
| --- | --- | --- | --- |
| Roads, buildings and launch/recovery geometry | `truck_uav_modules/data/{xinjiekou,baijiahu}/` | Baidu road/height indicators found; acquisition channel, footprint source and redistribution authorization unverified | Withheld |
| Derived 3-D grids and spatial metadata | Scene `voxel_map_3d/` directories | Derived from the same road/building inputs; no separate permission found | Withheld |
| Geometry-derived task registries | `data/{xinjiekou,baijiahu}/edge_task_registry.csv` | Dependent on source geography; derivative redistribution terms unverified | Withheld |
| Original CS-RDQN policies | `checkpoints/*.pt` with sidecars/manifest | Author-approved MIT; no imported third-party pretrained model identified; upstream training-data conditions remain unresolved | Retained, with model-card qualification |
| Generated evaluation observations/statistics | `reproduction/*.csv`, `evaluation_manifest.json` | Author-generated numerical records, hashes, units and protocol retained | Retained; not original geographic data |

Road summaries identify repaired local **Baidu road centrelines**. Building heights identify `local_baidu_height` for all 10348 Xinjiekou features and 6866 Baijiahu features; six Baijiahu heights use configured defaults. This evidence does not establish a license for footprints/heights or permit redistribution. [DATA_PROVENANCE.md](DATA_PROVENANCE.md) records local evidence and the official Baidu terms checked, without assigning an unverified OSM/ODbL license.

Third-party libraries retain their own licenses. Neither these data nor libraries automatically inherit the code/model MIT grant.

## Reproducibility limitation and access workflow

The original policies were verified locally on **20 seeds 1000–1019 per case** on 2026-10-11; all six means and sample SDs match Table VIII at two decimals. These measurements remain unchanged. However, **a data-free checkout cannot run those original scenarios**.

1. Obtain authorized matching source/derived inputs under the actual provider's terms.
2. Restore the original file layout described in `data/README.md` and `truck_uav_modules/data/README.md`.
3. Run `python -m tools.check_data_assets` to compare the 12 core asset SHA-256 values with `reproduction/evaluation_manifest.json`.
4. Only then run smoke/full evaluation or optional training with the unchanged configs.

The preflight does not download/scrape data, generate substitute maps, change the environment or grant permissions. A checksum match confirms identity only. Model loading and licensing/preflight unit tests can run without scene data. Environment/rollout tests require authorized inputs; their success on the author's local data is not a claim of public-data availability.

The self-contained offline reconstruction code is now included; see
[VOXEL_REBUILD.md](VOXEL_REBUILD.md). It derives grids/mappings from local
GeoJSON inputs without the missing historical 2-D artifacts, and reproduces
both original occupancy arrays exactly. It saves only new ignored outputs,
does not grant data rights and is not an automatic download or replacement map.

Metrics use CR_pct (%), MRT_s/HPRT_s/CURW_s (s), ECT (model energy units/completed task) and CTD_km (km). Missing values are not replaced with zero. These are evaluation observations from one original seed-42 policy per case, not independent training repetitions.

## Historical copies and remaining evidence

Current-tree exclusion **does not erase Git history or retract historical remote
copies**. The pre-publication remote commit inspected on 2026-10-12
(`2026833b79dfede6a938bb36d4e44d3bf3fdbe87`) contains the 31 inputs. The approved
new code release withholds them from its tree. No history rewriting/force-push
was authorized or performed. Withdrawing historical copies requires separate
author direction; the ordinary removal commit is not a historical purge.

Still needed: actual data-provider/acquisition records, applicable license or written permission for research use and derivatives/redistribution, any conditions on training/model outputs, and required source attribution. The current evidence does not establish that all upstream data agreements permit model distribution; the author's MIT grant does not override them.

Historical training-entry YAMLs are unavailable, and no permanent dataset DOI/deposit or accepted-paper DOI was created. Technical reproduction and author license selection are distinct from third-party legal clearance.

## 中文核对

代码和两个模型权重已按你的明确授权采用 MIT，并已获准发布；第三方地图、建筑物及路网数据不属于 MIT。你提供的获取渠道是公众号，但尚无文章出处和再分发许可。31 个地图/衍生输入从新发布版本中排除，本地文件和校验值保持不变。建图代码已补全，两场景各 20 个种子的冻结复现结果仍一致；公开版缺少授权输入时不能独立运行原场景。旧 Git 历史可能仍有数据，未改写；撤回历史副本需要另行授权。还需补充提供方许可，并核实其是否约束训练模型的分发。
