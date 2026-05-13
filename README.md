# Parallel Fraud Detection in Healthcare Using Large-Scale Financial Relationship Graphs

A reproducible four-phase pipeline for CMS Open Payments data with a sequential CPU baseline and parallel/GPU tracks.

## Overview
- Clean CMS Open Payments CSVs, build payer→payee graphs, run NetworkX algorithms, and generate fraud risk scores.
- Phases: 1) Data Cleaning, 2) Graph Building, 3) Graph Algorithms, 4) Fraud Scoring.
- Design goal: identical scoring semantics across CPU and GPU implementations for fair comparisons.

## Status
- Phases 1–4 complete on sequential CPU (`sequentialCPU/`).

## Requirements
- Python 3.11+
- Dependencies: numpy, pandas, pyarrow, pyyaml, tqdm, networkx, scipy
- Install at the repo root:

```bash
pip install -r requirements.txt
```

## Data
- Source: CMS Open Payments (https://openpaymentsdata.cms.gov/)
- Place CSVs in `CMS_Open_Payements2024/`:
  - `2024GeneralPayements.csv`
  - `2024ResearchPayements.csv`
  - `2024OwnershipPayements.csv`

## Quickstart (sequential CPU example)
```bash
# Run the full sequential pipeline (module execution)
python -m sequentialCPU.run_pipeline --config sequentialCPU/configs/pipeline_general_2024.yaml --runs 1

# Run the full concurrent Numba pipeline (module execution)
python -m concurrentNumba.run_pipeline --config concurrentNumba/configs/pipeline_general_2024.yaml --max-workers 4 --runs 1
```

For research and ownership datasets, swap the config files:
- sequential: `sequentialCPU/configs/pipeline_research_2024.yaml`, `sequentialCPU/configs/pipeline_ownership_2024.yaml`
- concurrent: `concurrentNumba/configs/pipeline_research_2024.yaml`, `concurrentNumba/configs/pipeline_ownership_2024.yaml`

To run a single phase directly (module execution), e.g. Phase 1:
```bash
python -m sequentialCPU.01_clean_data --config sequentialCPU/configs/pipeline_general_2024.yaml
python -m concurrentNumba.01_clean_data --config concurrentNumba/configs/pipeline_general_2024.yaml
```

If you need to run the file path directly instead of `-m`, set `PYTHONPATH=.`:
```bash
PYTHONPATH=. python sequentialCPU/run_pipeline.py --config sequentialCPU/configs/pipeline_general_2024.yaml --runs 1
PYTHONPATH=. python concurrentNumba/run_pipeline.py --config concurrentNumba/configs/pipeline_general_2024.yaml --max-workers 4 --runs 1
```

## Outputs (CPU baseline)
- Cleaned data: `output/<dataset>/payments_clean/` plus `payments_rejected/` and `cleaning_report.json`.
- Graph artifacts: `output_cpu/graphs/<dataset>_bipartite/` (nodes, edges, manifests).
- Algorithms: degree, PageRank, and components parquet files under each graph folder.
- Scoring: `risk_scores.parquet` and `topk_risk_scores.parquet` under `.../scoring/`.

## Repository Layout
```
CMS_Open_Payements2024/   # Raw CMS CSVs
sequentialCPU/            # Sequential CPU pipeline (phases 1–4)
concurrentNumba/          # Concurrent Numba CPU pipeline (phases 1–4)
output/                   # Pipeline outputs
requirements.txt          # Python dependencies
README.md                 # This file
tools/                    # Utility scripts (e.g., synthetic data expander)
```

## Roadmap
- Parallel CPU/GPU implementations with matching semantics
- Performance benchmarking and scalability studies
- Optional visualization and feature engineering additions

## References
- CMS Open Payments: https://openpaymentsdata.cms.gov/
- CMS Data Dictionary: https://www.cms.gov/OpenPayments/Downloads/OpenPaymentsDataDictionary.pdf
- NetworkX: https://networkx.org/

---
