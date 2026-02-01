# Sequential CPU Fraud Detection Pipeline

**Status:** Production Ready (Phases 1-4) | **Updated:** January 30, 2026

This directory contains the full sequential CPU baseline for the CMS Open Payments fraud detection pipeline. It cleans raw CSVs, builds payer→payee graphs, runs NetworkX algorithms, and produces risk scores. Use this as the correctness baseline for parallel/GPU implementations.

## Requirements
- Python 3.11+
- Install dependencies once from repo root:
```bash
pip install -r ../requirements.txt
```

## Data Inputs
- Download CMS Open Payments CSVs (https://openpaymentsdata.cms.gov/).
- Place files in the repo-level `CMS_Open_Payements2024/` directory (name matches current folder):
  - `2024GeneralPayements.csv`
  - `2024ResearchPayements.csv`
  - `2024OwnershipPayements.csv`

## Quick Start (general payments example)
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

## Key Features (CPU baseline)
- Deterministic normalization, hashing, and payee key generation for reproducibility.
- Dataset fingerprinting and config snapshots for auditability.
- Chunked processing for large CSVs; partitioned Parquet writes for speed.
- NetworkX algorithms (degree, PageRank, connected components) feeding explainable robust z-score risk scoring.

## Performance Notes (reference)
- General (~12M rows): ~20–40 min on 8-core CPU, ~2 GB RAM with default `chunk_size=500000`.
- Research (~0.5M rows): ~1–3 min.
- Ownership (~5K rows): <1s.

Tune `chunk_size` if memory-constrained (e.g., 100k for 8 GB RAM).

## Troubleshooting
- **Import/File errors:** Re-run `pip install -r ../requirements.txt`; verify paths in configs and CSV placement.
- **Memory issues:** Lower `chunk_size` in the Phase 1 config.
- **Missing columns:** Use `column_mapping` overrides in the Phase 1 config to match your CSV headers.
- **Too many small files:** Increase `chunk_size` to reduce partitions.

For full pipeline details and background, see the root `README.md` and inline comments in each script.
