# Parallel Fraud Detection in Healthcare Using Large-Scale Financial Relationship Graphs

A reproducible four-phase pipeline for CMS Open Payments data with a sequential CPU baseline and parallel/GPU tracks.

## Overview
- Clean CMS Open Payments CSVs, build payer→payee graphs, run NetworkX algorithms, and generate fraud risk scores.
- Phases: 1) Data Cleaning, 2) Graph Building, 3) Graph Algorithms, 4) Fraud Scoring.
- Design goal: identical scoring semantics across CPU and GPU implementations for fair comparisons.

## Status
- Phases 1–4 complete on sequential CPU (`sequentialCPU/`).
- Parallel CPU and GPU implementations are under active development.

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
cd sequentialCPU

# Phase 1: Clean data
python 01_clean_data.py --config configs/config_general.yaml

# Phase 2: Build graph
python 02_build_graph.py --config configs/graph_general_2024.yaml

# Phase 3: Run graph algorithms
python 03_graph_algorithms.py --config configs/algos_general_2024.yaml

# Phase 4: Score fraud risk
python 04_fraud_scoring.py --config configs/score_general_2024.yaml

# Optional: view top scores
python -c "import pandas as pd; df = pd.read_parquet('../output_cpu/graphs/general_2024_bipartite/scoring/topk_risk_scores.parquet'); print(df[['node_id','node_type','risk_score','rank','in_weight','in_degree']].head(10))"
```
For research and ownership datasets, swap the config files (`config_research.yaml`, `config_ownership.yaml`, etc.).

## Outputs (CPU baseline)
- Cleaned data: `output/<dataset>/payments_clean/` plus `payments_rejected/` and `cleaning_report.json`.
- Graph artifacts: `output_cpu/graphs/<dataset>_bipartite/` (nodes, edges, manifests).
- Algorithms: degree, PageRank, and components parquet files under each graph folder.
- Scoring: `risk_scores.parquet` and `topk_risk_scores.parquet` under `.../scoring/`.

## Repository Layout
```
CMS_Open_Payements2024/   # Raw CMS CSVs
sequentialCPU/            # Sequential CPU pipeline (phases 1–4)
concurrentCPU/            # Parallel CPU pipeline (Dask-based)
parallelGPU/              # GPU pipeline (in progress)
output/                   # Cleaned data outputs
requirements.txt          # Python dependencies
README.md                 # This file
```

## Roadmap
- Parallel CPU/GPU implementations with matching semantics
- Performance benchmarking and scalability studies
- Optional visualization and feature engineering additions

## References
- CMS Open Payments: https://openpaymentsdata.cms.gov/
- CMS Data Dictionary: https://www.cms.gov/OpenPayments/Downloads/OpenPaymentsDataDictionary.pdf
- NetworkX: https://networkx.org/

For per-phase details, see `sequentialCPU/README.md` and the scripts in each pipeline directory.

---
**Last Updated:** January 30, 2026
