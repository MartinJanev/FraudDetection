# concurrentNumba – Fraud Detection Pipeline (Numba + ThreadPoolExecutor)

Parallel CPU pipeline built around **Numba JIT-compiled kernels**
and Python's `concurrent.futures.ThreadPoolExecutor` for chunk/file-level parallelism.

## Architecture

| Phase | Parallelism mechanism | Summary |
|-------|----------------------|---------|
| **Phase 1** – Clean | `ThreadPoolExecutor` over CSV chunks; Numba `@njit` for sampling hash + amount validation | Chunk-level cleaning with deterministic filtering |
| **Phase 2** – Build Graph | `ThreadPoolExecutor` over parquet files; Numba JIT for edge weight/count accumulation | File-level graph aggregation |
| **Phase 3** – Graph Algorithms | Numba `@njit(parallel=True)` kernels for degree computation; NetworKit fast path for PageRank + components with a NetworkX fallback | Fast degree features and optional native graph algorithms |
| **Phase 4** – Fraud Scoring | Numba `@njit(parallel=True)` for robust Z-score + weighted sum kernels; pure pandas I/O | Deterministic scoring and ranking |

## Dependencies

```
pip install numba numpy pandas pyarrow pyyaml networkx tqdm
```

> **Note:** Numba requires a compatible LLVM toolchain. On Windows, install via `conda install numba` or ensure LLVM/VS build tools are present.
> If Numba is unavailable the code falls back gracefully to pure-NumPy equivalents.
> NetworKit is now optional; install it separately if you want the OpenMP-accelerated PageRank/components backend.

## Usage

```bash
# Single run (research dataset, 10% sample, 8 workers)
python run_pipeline.py \
    --config configs/pipeline_research_2024.yaml \
    --approach concurrent_numba \
    --runs 3 \
    --max-workers 8

# General payments
python run_pipeline.py \
    --config configs/pipeline_general_2024.yaml \
    --approach concurrent_numba \
    --runs 5

# Ownership payments
python run_pipeline.py \
    --config configs/pipeline_ownership_2024.yaml \
    --approach concurrent_numba \
    --runs 1
```

## Output layout

```
output/concurrent_numba/<dataset>/<fraction_XXpct>/workers_N/
    <run_id>/
        phase1_clean/   cleaning_report.json, payments_clean/*.parquet
        phase2_graph/   graph_build_report.json, edges.parquet, nodes.parquet
        phase3_algos/   algos_report.json, degree.parquet, pagerank.parquet, components.parquet
        phase4_score/   fraud_scoring_report.json, risk_scores.parquet, topk_risk_scores.parquet
        reports/        timings.json, config_snapshot.yaml
    aggregate/
        timing_runs.csv
        timing_summary.json
        run_metrics.csv
        timing_runs_warmup_filtered.csv   (if runs >= 2)
```

## Numba JIT kernels

| File | Kernel | Purpose |
|------|--------|---------|
| `01_clean_data.py` | `_fast_hash_mask` | Parallel boolean sampling mask over uint64 hashes |
| `01_clean_data.py` | `_parse_amounts_kernel` | Parallel NaN / negative-amount validity check |
| `02_build_graph.py` | `_sum_weights_parallel` | Edge weight accumulation into label buckets |
| `02_build_graph.py` | `_count_labels` | Edge count accumulation |
| `03_graph_algorithms.py` | `_accumulate_out_weight` / `_accumulate_in_weight` | Parallel degree weight sums |
| `03_graph_algorithms.py` | `_count_codes` | Parallel degree counting |
| `04_fraud_scoring.py` | `_robust_z_numba` | Parallel robust Z-score |
| `04_fraud_scoring.py` | `_weighted_sum_scores` | Parallel weighted risk score |

All kernels fall back to equivalent pure-NumPy implementations if Numba is not installed.

## Sampling methods

- `fast_hash` (default): uses `pd.util.hash_pandas_object` + Numba parallel boolean mask → fastest
- `sha1`: identical to sequential baseline (slower but bit-exact reproducible)

