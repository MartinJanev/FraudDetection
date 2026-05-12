# Concurrent CPU Fraud Detection Pipeline (Dask-based)

Updated: January 30, 2026

This directory contains the concurrent CPU version of the CMS Open Payments fraud detection pipeline. It mirrors the sequential logic but parallelizes work with Dask to reduce wall-clock time. Use the same configs and outputs as the baseline, with an additional workers dimension in the output paths.

## Requirements
- Python 3.11+
- Install dependencies from the repo root:
```bash
pip install -r ../requirements.txt
```

## Data Inputs
- Download CMS Open Payments CSVs (https://openpaymentsdata.cms.gov/).
- Place files in `../CMS_Open_Payements2024/`:
  - `2024GeneralPayements.csv`
  - `2024ResearchPayements.csv`
  - `2024OwnershipPayements.csv`

## Quick Start (general payments)
```bash
cd concurrentCPU

# Run end-to-end with the unified pipeline config
python run_pipeline.py --config configs/pipeline_general_2024.yaml --approach concurent_cpu --runs 1 --workers 16
```
Fractional sampling is controlled by the config (`phase1_clean.sampling.fraction`). Adjust `--workers` to set the Dask thread count.

## Outputs
- Run artifacts are written under `../output/concurent_cpu/<dataset>/fraction_XXpct/workers_<N>/`.
- Each run directory contains phase outputs and `reports/timings.json`.
- Aggregated metrics are in `aggregate/timing_summary.json` per fraction/workers.

## Configs
Unified pipeline configs live in `configs/` (`pipeline_general_2024.yaml`, `pipeline_research_2024.yaml`, `pipeline_ownership_2024.yaml`). They mirror the sequential configs with the same phase keys.

## Notes
- Matches sequential semantics; differences should be limited to performance.
- Chunk sizes and sampling fractions affect memory and timing; tune for your hardware.
- Workers > core count may hurt performance; start with 8–16.

For more detail on the baseline logic, see `../sequentialCPU/README.md`.
