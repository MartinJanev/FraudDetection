# Parallel Fraud Detection in Healthcare Using Large-Scale Financial Relationship Graphs

A reproducible 4-phase pipeline for CMS Open Payments data with a sequential CPU baseline and planned GPU acceleration.

## Overview
- Cleans CMS Open Payments CSVs, builds payer→payee graphs, runs NetworkX algorithms, and generates fraud risk scores.
- Phases: (1) Data Cleaning, (2) Graph Building, (3) Graph Algorithms, (4) Fraud Scoring.
- Design goal: identical scoring semantics across CPU/GPU to compare performance.

## Status
- ✅ Phases 1-4 complete on sequential CPU (baseline in `sequentialCPU/`).
- 🎯 Next: GPU/parallel implementation for speed comparisons.

## Requirements
- Python 3.11+ (tested)
- Dependencies: numpy, pandas, pyarrow, pyyaml, tqdm, networkx, scipy
- Install once at repo root:

```bash
pip install -r requirements.txt
```

## Data
- Source: CMS Open Payments (https://openpaymentsdata.cms.gov/)
- Place CSVs in `CMS_Open_Payements2024/` (matches current folder name):
  - `2024GeneralPayements.csv`
  - `2024ResearchPayements.csv`
  - `2024OwnershipPayements.csv`

## Quickstart (sequential CPU baseline)
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
CMS_Open_Payements2024/   # Raw CMS CSVs (as named in this repo)
sequentialCPU/            # Full sequential CPU pipeline (Phases 1-4)
  configs/                # YAML configs for each phase
  01_clean_data.py        # Phase 1
  02_build_graph.py       # Phase 2
  03_graph_algorithms.py  # Phase 3
  04_fraud_scoring.py     # Phase 4
output/                   # Cleaned data (created by Phase 1)
output_cpu/               # Graphs, alg outputs, scores (created by Phases 2-4)
requirements.txt          # Python deps
README.md                 # This file
```

## Roadmap
- Parallel CPU/GPU implementations (Dask/CUDA) with identical scoring semantics
- Performance benchmarking and scalability reports
- Visualization dashboard and ML feature integration

## References
- CMS Open Payments: https://openpaymentsdata.cms.gov/
- CMS Data Dictionary: https://www.cms.gov/OpenPayments/Downloads/OpenPaymentsDataDictionary.pdf
- NetworkX: https://networkx.org/

For detailed phase-by-phase docs, see `sequentialCPU/README.md`. 

---
**Last Updated:** January 30, 2026
