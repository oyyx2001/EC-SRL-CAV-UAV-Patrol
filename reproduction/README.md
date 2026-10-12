# Fresh frozen-policy validation

Latest validation generated on 2026-10-12 by:

```powershell
python -m rl.reproduce_paper --workers 4 --output-dir results/release_reproduction_voxel_fix_20seeds_20261012
```

This folder contains unchanged copies of the generated CSVs and manifest, not a retyped reference table. There are 40 unique case–seed episodes (20 per case, seeds 1000–1019). `summary_mean_std.csv` and `reference_comparison.csv` retain full-precision values, sample SD (`ddof=1`), valid/missing counts, differences from Table VIII and displayed-precision match flags. `paper_summary.csv` is a compact mean-only pivot; consult the long summary for SDs.

All six means and sample SDs match the supplied Table VIII at two decimals in both cases. Runtime timing is diagnostic, not an experimental service-time metric. No metric values were adjusted. These are evaluation repetitions of one original seed-42 / final-300k policy per case, not 20 training repetitions.

The complete evaluation was repeated after the package-import compatibility fix
on 2026-10-11, then again after offline voxel-loader/helper completion on
2026-10-12. All 240 per-seed metric values and all task-stream hashes are exactly
unchanged between these runs. The files here contain the latest-source run;
the original measurements and prior local runtime records remain intact.

The manifest records the exact source, checkpoint and input-data hashes and numerical dependency versions. The author's existing CPU environment required the explicitly recorded process-local `KMP_DUPLICATE_LIB_OK=TRUE` workaround. Original per-episode JSON files remain in the ignored runtime output directory; re-evaluation recreates them without needing that historical directory.

**2026-10-12 access update:** the author approved MIT for source code and model weights, not third-party maps. The original geographic/derived input files are withheld from the current Git index. These numerical tables remain unchanged and describe the existing local validation; a data-free checkout cannot recreate the rollouts without authorized matching inputs. Run `python -m tools.check_data_assets` after lawful restoration. Its hash check verifies identity, not permission.
