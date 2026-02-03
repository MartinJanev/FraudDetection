# Sequential CPU Fraud Detection Pipeline

Updated: January 30, 2026

This directory contains the sequential CPU baseline for the CMS Open Payments fraud detection pipeline. It cleans raw CSVs, builds payer→payee graphs, runs NetworkX algorithms, and produces risk scores. Use this as the correctness baseline for parallel and GPU implementations.

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
cd sequentialCPU

# Phase 1: Clean data
python 01_clean_data.py --config configs/config_general.yaml

# Phase 2: Build graph
python 02_build_graph.py --config configs/graph_general_2024.yaml

# Phase 3: Graph algorithms
python 03_graph_algorithms.py --config configs/algos_general_2024.yaml

# Phase 4: Fraud scoring
python 04_fraud_scoring.py --config configs/score_general_2024.yaml

# Optional: view top scores
python -c "import pandas as pd; df = pd.read_parquet('../output_cpu/graphs/general_2024_bipartite/scoring/topk_risk_scores.parquet'); print(df[['node_id','node_type','risk_score','rank','in_weight','in_degree']].head(10))"
```
Swap configs for research/ownership: `config_research.yaml`, `config_ownership.yaml`, and matching graph/algos/score YAMLs.

## Outputs
- Cleaned data: `../output/<dataset>/payments_clean/` plus `payments_rejected/`, `cleaning_report.json`, `dataset_fingerprint.json`, manifests.
- Graphs: `../output_cpu/graphs/<dataset>_bipartite/` (`nodes.parquet`, `edges.parquet`, manifests).
- Algorithms: degree, PageRank, components under each graph folder.
- Scoring: `risk_scores.parquet`, `topk_risk_scores.parquet` under `.../scoring/`.

## Configs
Key YAMLs in `configs/`:
- Phase 1: `config_general.yaml`, `config_research.yaml`, `config_ownership.yaml`
- Phase 2: `graph_general_2024.yaml`, `graph_research_2024.yaml`, `graph_ownership_2024.yaml`
- Phase 3: `algos_general_2024.yaml`, `algos_research_2024.yaml`, `algos_ownership_2024.yaml`
- Phase 4: `score_general_2024.yaml`, `score_research_2024.yaml`, `score_ownership_2024.yaml`

Adjust paths, `chunk_size`, and output directories in these files as needed.

## Notes
- Deterministic normalization, hashing, and payee key generation for reproducibility.
- Dataset fingerprinting and config snapshots for auditability.
- Chunked processing for large CSVs; partitioned Parquet writes for speed.
- NetworkX algorithms feed robust z-score risk scoring.

## Troubleshooting
- Import/file errors: reinstall requirements and verify config paths and CSV placement.
- Memory issues: lower `chunk_size` in the Phase 1 config.
- Missing columns: use `column_mapping` in the Phase 1 config to match CSV headers.
- Too many small files: increase `chunk_size`.

For more context, see the root `README.md` and inline comments in the scripts.
