# Manuscript alignment notes

This file records the implementation details that must be kept synchronized with the supplied manuscript. It is part of the release so that a reviewer can distinguish the executable model from manuscript wording that still needs an author decision.

| Item | Checked-in implementation | Supplied manuscript draft |
| --- | --- | --- |
| Default training budget | 300,000 transitions in `configs/default_training.yaml` | 300,000 transitions |
| Candidate representation | 75 global features, 25 slots, 38 features per slot | 75 + 25 × 38 |
| PER | α=0.6, β from 0.4 to 1.0, ε=10⁻⁶ | The extracted Table VII formula image shows α=0.6, β from 0.4 to 0.1, ε=10⁻⁶ |
| Congestion priority | 6 in `edge_tasks.dynamic_tasks.congestion_anomaly_scan` | Table VI reports 4 |
| Incident service time | Geometry-based: `clip(length_m / 3 + 45, 60, 150)` seconds | Table VI reports 60 seconds |
| Congestion service time | Geometry-based: `clip(length_m / 5 + 30, 45, 120)` seconds | Table VI reports 75 seconds |
| UAV speed uncertainty | Configured climb/cruise/descent speeds are deterministic; no execution speed multiplier is enabled | Table V reports UAV speed variability of ±10% |

The repository follows the current implementation as its executable source of truth. If the manuscript is intended to define the final experiment instead, update the affected code/configuration and rerun all dependent training and evaluation before using the existing numerical tables. If the current implementation is intended to define the final experiment, update Table V--VII and the task-configuration description in the manuscript.
