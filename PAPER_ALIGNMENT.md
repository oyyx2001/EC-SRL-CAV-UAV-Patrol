# Alignment with the supplied manuscript

This audit uses the current supplied TITS manuscript, not the earlier draft summarized in the previous version of this file. Original checkpoint/training records and executable runtime are the primary implementation evidence; manuscript wording was not silently enforced by changing numerical settings.

| Item | Verified executable/original model | Current manuscript | Status |
| --- | --- | --- | --- |
| Training seed/budget | 42 / 300000, supported by original stages and run summaries | Seed 42 / 300000 | Aligned; older Case 1 default budgets do not override original run evidence |
| Raw observation/actions | 75 + 25 × 38 = 1025, 25 slots | Same raw representation | Aligned |
| Actual network input split | Global encoder 113 inputs = 75 + safe-slot 38; shared candidate encoder sees 24 non-safe slots | Raw global-state count reported as 75 | Clarify implementation split; no pretrained-network redesign |
| PER | alpha 0.6, beta 0.4 → 1.0 over 200k, epsilon 1e-6 | Table VII now reports the same | Aligned; obsolete beta 0.1 warning removed |
| Congestion priority | 6 | Table VI now reports 6 | Aligned; obsolete priority 4 warning removed |
| Incident service time | Geometry-based clip(length_m / 3 + 45, 60, 150) s | Table VI reports 60 s | Remaining wording mismatch; executable settings retained |
| Congestion service time | Geometry-based clip(length_m / 5 + 30, 45, 120) s | Table VI reports 75 s | Remaining wording mismatch; executable settings retained |
| UAV speed uncertainty | Nominal climb/cruise/descent 3/10/2 m/s; no UAV execution speed multiplier enabled | Table V lists UAV ±10% variability | Remaining wording mismatch; no new perturbation introduced |
| Road topology | Undirected source MultiGraph: 601/247 physical links; directed simulator: 1202/494 arcs | Table IV calls 601/247 directed links | Distinguish physical source links from directed simulation arcs |
| Historical full training YAMLs | Not found; non-private original model records and current audited evaluation configs retained | Not a runtime-config archive | Disclose missing historical entries; do not claim exact retraining provenance |
| Table VIII frozen evaluation | Both final original policies, 20 seeds 1000–1019 per case | Six means and SDs | All reproduced at two decimals in fresh evaluation |

No original experimental measurements or Table VIII/IX values were modified. No reward, action space, observation layout, task priorities/service-time formula, simulator or planning budgets were changed to force agreement. Technical evaluation reproduction does not resolve all manuscript description differences; these require an author wording decision.

See REPRODUCIBILITY.md, checkpoints/manifest.json and reproduction/reference_comparison.csv for evidence. Code/data licensing is a separate outstanding author decision.
