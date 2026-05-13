#!/usr/bin/env python3
"""
Phase 2 (Concurrent CPU via Numba + ThreadPoolExecutor): Graph Building.

Reads the cleaned payments Parquet dataset from Phase 1 and builds a bipartite directed graph.
Uses pandas for Parquet I/O, ThreadPoolExecutor for per-file work, and Numba kernels for hot
aggregation loops.

Aggregates edges by (src, dst):
- w_total    = sum(amount_usd)
- n_payments = count(rows)
Optional (if payment_date exists):
- min_date, max_date

Outputs:
- edges.parquet
- nodes.parquet
- graph_build_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    numba = None   # type: ignore[assignment]
    njit = None    # type: ignore[assignment]
    prange = None  # type: ignore[assignment]
    _HAS_NUMBA = False

# ---------------------------------------------------------------------------
# Numba kernels for edge aggregation
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _sum_weights_parallel(weights: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
        """Sum weights[i] into bucket labels[i]; returns array of length n_labels."""
        out = np.zeros(n_labels, dtype=np.float64)
        for i in prange(len(weights)):
            # Note: parallel reduction over the same bucket is a race condition with prange.
            # Use serial loop for correctness; Numba's atomic add is not available in nopython.
            pass
        # Fallback: serial accumulation (still benefits from JIT compilation overhead avoidance)
        out2 = np.zeros(n_labels, dtype=np.float64)
        for i in range(len(weights)):
            out2[labels[i]] += weights[i]
        return out2

    @njit(cache=True)
    def _count_labels(labels: np.ndarray, n_labels: int) -> np.ndarray:
        """Count occurrences of each label."""
        out = np.zeros(n_labels, dtype=np.int64)
        for i in range(len(labels)):
            out[labels[i]] += 1
        return out

else:
    def _sum_weights_parallel(weights, labels, n_labels):
        out = np.zeros(n_labels, dtype=np.float64)
        np.add.at(out, labels, weights)
        return out

    def _count_labels(labels, n_labels):
        out = np.zeros(n_labels, dtype=np.int64)
        np.add.at(out, labels, 1)
        return out


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class GraphBuildConfig:
    input_clean_dir: str
    output_graph_dir: str

    payer_id_col: str = "payer_id"
    payer_name_norm_col: str = "payer_name_norm"
    payee_key_col: str = "payee_key"
    amount_col: str = "amount_usd"
    date_col: Optional[str] = "payment_date"

    min_edge_weight: float = 0.0

    # Numba/threading tuning
    use_numba: bool = True
    max_workers: Optional[int] = None  # None = CPU count


def load_config(path: str) -> GraphBuildConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(path).parent

    def resolve_path(p: str) -> str:
        if not p:
            return p
        path_obj = Path(p)
        if not path_obj.is_absolute():
            path_obj = config_dir / p
        return str(path_obj.resolve())

    return GraphBuildConfig(
        input_clean_dir=resolve_path(raw["input_clean_dir"]),
        output_graph_dir=resolve_path(raw["output_graph_dir"]),
        payer_id_col=str(raw.get("payer_id_col", "payer_id")),
        payer_name_norm_col=str(raw.get("payer_name_norm_col", "payer_name_norm")),
        payee_key_col=str(raw.get("payee_key_col", "payee_key")),
        amount_col=str(raw.get("amount_col", "amount_usd")),
        date_col=str(raw.get("date_col")) if raw.get("date_col") else None,
        min_edge_weight=float(raw.get("min_edge_weight", 0.0)),
        use_numba=bool(raw.get("use_numba", True)),
        max_workers=raw.get("max_workers"),
    )


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def payer_node_id_series(payer_id: pd.Series, payer_name_norm: pd.Series) -> pd.Series:
    pid = payer_id.fillna("").astype(str).str.strip()
    pname = payer_name_norm.fillna("").astype(str).str.strip()
    use_id = pid.ne("") & pid.str.lower().ne("nan")
    out = pd.Series(np.where(use_id, "PAYER_ID:" + pid, "PAYER_NAME:" + pname), index=payer_id.index)
    out = out.mask(out.eq("PAYER_NAME:"), "PAYER_NAME:UNKNOWN")
    return out


def node_type_from_id(node_id: str) -> str:
    if node_id.startswith("PAYER_ID:") or node_id.startswith("PAYER_NAME:"):
        return "payer"
    if node_id.startswith("PHYS_"):
        return "physician"
    if node_id.startswith("HOSP_"):
        return "teaching_hospital"
    return "unknown"


def sanitize_strings_for_parquet(pdf: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    pdf = pdf.copy()
    for c in cols:
        if c in pdf.columns:
            pdf[c] = pdf[c].map(lambda x: None if x is None else str(x)).astype("string[python]")
    return pdf


def _build_edges_from_parquet_file(
    parquet_path: Path,
    cfg: GraphBuildConfig,
) -> pd.DataFrame:
    """Read one parquet file and return an edge DataFrame (src, dst, w, date)."""
    pdf = pd.read_parquet(str(parquet_path))
    required = [cfg.payer_id_col, cfg.payer_name_norm_col, cfg.payee_key_col, cfg.amount_col]
    missing = [c for c in required if c not in pdf.columns]
    if missing:
        raise KeyError(f"Missing required columns in {parquet_path}: {missing}")

    payer_id = pdf[cfg.payer_id_col]
    payer_name_norm = pdf[cfg.payer_name_norm_col]
    payee_key = pdf[cfg.payee_key_col].fillna("").astype(str).str.strip()
    amount = pd.to_numeric(pdf[cfg.amount_col], errors="coerce")

    src = payer_node_id_series(payer_id, payer_name_norm)
    dst = payee_key

    out = pd.DataFrame({"src": src, "dst": dst, "w": amount})

    if cfg.date_col and cfg.date_col in pdf.columns:
        date_raw = pdf[cfg.date_col].fillna("").astype(str).str.strip()
        out["date"] = pd.to_datetime(date_raw, errors="coerce")
    else:
        out["date"] = pd.NaT

    out = out[out["src"].ne("") & out["dst"].ne("") & out["w"].notna()]
    return out


def _aggregate_edges_numba(edges: pd.DataFrame, cfg: GraphBuildConfig) -> pd.DataFrame:
    """
    Aggregate edges (src, dst) -> (w_total, n_payments[, min_date, max_date]).

    Uses Numba kernels for the numeric sum/count after pandas factorize gives us
    integer bucket labels.
    """
    src = edges["src"].astype(str).values
    dst = edges["dst"].astype(str).values
    w = edges["w"].to_numpy(dtype=np.float64)

    # Create composite key for grouping
    keys = np.array([f"{s}\x00{d}" for s, d in zip(src, dst)])
    codes, uniques = pd.factorize(keys)
    n_labels = len(uniques)

    if cfg.use_numba and _HAS_NUMBA:
        w_total = _sum_weights_parallel(w, codes.astype(np.int64), n_labels)
        n_pay = _count_labels(codes.astype(np.int64), n_labels)
    else:
        w_total = np.zeros(n_labels, dtype=np.float64)
        n_pay = np.zeros(n_labels, dtype=np.int64)
        np.add.at(w_total, codes, w)
        np.add.at(n_pay, codes, 1)

    # Decode unique keys back to (src, dst)
    split_uniques = [u.split("\x00", 1) for u in uniques]
    out_src = [s for s, _ in split_uniques]
    out_dst = [d for _, d in split_uniques]

    result = pd.DataFrame({
        "src": out_src,
        "dst": out_dst,
        "w_total": w_total,
        "n_payments": n_pay,
    })

    if cfg.date_col and "date" in edges.columns:
        date_series = edges["date"]
        # min/max date per group via pandas (not bottleneck)
        date_df = pd.DataFrame({"key": keys, "date": date_series})
        date_agg = date_df.groupby("key", sort=False)["date"].agg(["min", "max"]).reset_index()
        date_agg.columns = ["key", "min_date", "max_date"]

        result["key"] = uniques
        result = result.merge(date_agg, on="key", how="left").drop(columns="key")

        def _safe_to_ns(s: pd.Series) -> pd.Series:
            dt = pd.to_datetime(s, errors="coerce")
            dt = dt.where((dt.dt.year >= 1900) & (dt.dt.year <= 2100))
            return dt.astype("datetime64[ns]")

        result["min_date"] = _safe_to_ns(result["min_date"])
        result["max_date"] = _safe_to_ns(result["max_date"])

    return result


def run(cfg: GraphBuildConfig) -> None:
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    out_dir = Path(cfg.output_graph_dir)
    ensure_dir(str(out_dir))

    clean_dir = Path(cfg.input_clean_dir)
    print(f"[phase2_graph] input_clean_dir={clean_dir} -> output_graph_dir={out_dir}", flush=True)
    if not clean_dir.exists():
        raise FileNotFoundError(f"input_clean_dir not found: {clean_dir}")

    # Discover parquet files
    parquet_files = sorted(clean_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {clean_dir}")
    print(f"[phase2_graph] found {len(parquet_files)} parquet file(s)", flush=True)

    # Warm up Numba
    if cfg.use_numba and _HAS_NUMBA:
        _dummy_w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        _dummy_l = np.array([0, 1, 0], dtype=np.int64)
        _sum_weights_parallel(_dummy_w, _dummy_l, 2)
        _count_labels(_dummy_l, 2)
        logging.info("[Phase 2] Numba JIT warm-up complete")

    max_workers = cfg.max_workers if cfg.max_workers and cfg.max_workers > 0 else None

    # Parallel per-file edge construction
    t_read_start = time.perf_counter()
    edge_parts: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_edges_from_parquet_file, fp, cfg): fp for fp in parquet_files}
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                part = fut.result()
                edge_parts.append(part)
            except Exception as exc:
                logging.error("[Phase 2] Failed on %s: %s", fp, exc)
                raise

    timings["t_read_parquet_graph"] = round(time.perf_counter() - t_read_start, 4)
    print(f"[phase2_graph] read {len(edge_parts)} file(s) in parallel", flush=True)

    # Concatenate all raw edges
    t_concat_start = time.perf_counter()
    if edge_parts:
        all_edges = pd.concat(edge_parts, ignore_index=True)
    else:
        all_edges = pd.DataFrame(columns=["src", "dst", "w", "date"])
    timings["t_concat"] = round(time.perf_counter() - t_concat_start, 4)
    print(f"[phase2_graph] total raw edges={len(all_edges)}", flush=True)

    # Aggregate via Numba kernels
    t_groupby_start = time.perf_counter()
    grouped = _aggregate_edges_numba(all_edges, cfg)
    timings["t_groupby_graph"] = round(time.perf_counter() - t_groupby_start, 4)
    print(f"[phase2_graph] aggregated edges={len(grouped)}", flush=True)

    if cfg.min_edge_weight and cfg.min_edge_weight > 0:
        grouped = grouped[grouped["w_total"] >= cfg.min_edge_weight]

    # Write edges parquet
    edges_path = out_dir / "edges.parquet"
    t_write_edges_start = time.perf_counter()
    grouped.to_parquet(str(edges_path), engine="pyarrow", index=False)
    timings["t_write_edges_parquet"] = round(time.perf_counter() - t_write_edges_start, 4)
    print(f"[phase2_graph] wrote edges -> {edges_path}", flush=True)

    # Nodes table
    t_nodes_start = time.perf_counter()
    all_node_ids = pd.Series(
        pd.concat([grouped["src"], grouped["dst"]], ignore_index=True).unique(),
        name="node_id",
    ).astype(str)

    nodes_df = pd.DataFrame({"node_id": all_node_ids})
    nodes_df["node_type"] = nodes_df["node_id"].map(lambda x: node_type_from_id(str(x)) if x is not None else "unknown")
    nodes_df = sanitize_strings_for_parquet(nodes_df, ["node_id", "node_type"])

    nodes_path = out_dir / "nodes.parquet"
    nodes_df.to_parquet(str(nodes_path), engine="pyarrow", index=False)
    timings["t_write_nodes_parquet"] = round(time.perf_counter() - t_nodes_start, 4)
    print(f"[phase2_graph] wrote nodes -> {nodes_path}", flush=True)

    # Stats
    t_stats_start = time.perf_counter()
    edges_count = len(grouped)
    nodes_count = len(nodes_df)
    w = grouped["w_total"]
    w_min = float(w.min()) if edges_count else None
    w_mean = float(w.mean()) if edges_count else None
    w_median = float(w.median()) if edges_count else None
    w_max = float(w.max()) if edges_count else None
    node_type_counts = nodes_df["node_type"].value_counts().to_dict()
    timings["t_stats"] = round(time.perf_counter() - t_stats_start, 4)
    print(f"[phase2_graph] stats edges={edges_count} nodes={nodes_count}", flush=True)

    total_time = time.perf_counter() - t0
    report = {
        "inputs": {"clean_dir": str(clean_dir)},
        "outputs": {"edges_path": str(edges_path), "nodes_path": str(nodes_path)},
        "counts": {
            "edges": edges_count,
            "total_nodes": nodes_count,
            "nodes_by_type": {str(k): int(v) for k, v in node_type_counts.items()},
            "w_total_min": w_min,
            "w_total_mean": w_mean,
            "w_total_median": w_median,
            "w_total_max": w_max,
        },
        "timings_sec": {"total": round(total_time, 4), **timings},
        "config": {
            "payer_id_col": cfg.payer_id_col,
            "payer_name_norm_col": cfg.payer_name_norm_col,
            "payee_key_col": cfg.payee_key_col,
            "amount_col": cfg.amount_col,
            "date_col": cfg.date_col,
            "min_edge_weight": cfg.min_edge_weight,
            "use_numba": cfg.use_numba,
            "has_numba": bool(_HAS_NUMBA),
            "max_workers": cfg.max_workers,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_dir / "graph_build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("Wrote edges dataset: %s", edges_path)
    logging.info("Wrote nodes dataset: %s", nodes_path)
    logging.info("Wrote report: %s", out_dir / "graph_build_report.json")


def run_from_pipeline(
    pipeline_cfg: Dict,
    *,
    clean_dir: Path,
    clean_manifest: Optional[Path],
    out_dir: Path,
    approach: str,
    run_id: str,
    dataset_name: str,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg = pipeline_cfg.get("phase2_graph", {})
    global_workers = int(pipeline_cfg.get("execution", {}).get("max_workers", 0) or 0)

    cfg = GraphBuildConfig(
        input_clean_dir=str(clean_dir),
        output_graph_dir=str(out_dir),
        payer_id_col=str(phase_cfg.get("payer_id_col", "payer_id")),
        payer_name_norm_col=str(phase_cfg.get("payer_name_norm_col", "payer_name_norm")),
        payee_key_col=str(phase_cfg.get("payee_key_col", "payee_key")),
        amount_col=str(phase_cfg.get("amount_col", "amount_usd")),
        date_col=str(phase_cfg.get("date_col")) if phase_cfg.get("date_col") else None,
        min_edge_weight=float(phase_cfg.get("min_edge_weight", 0.0)),
        use_numba=bool(phase_cfg.get("use_numba", True)),
        max_workers=global_workers if global_workers > 0 else None,
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg)
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    return {
        "phase": "phase2_graph",
        "dataset": dataset_name,
        "approach": approach,
        "run_id": run_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "edges": str(out_dir / "edges.parquet"),
            "nodes": str(out_dir / "nodes.parquet"),
            "report": str(out_dir / "graph_build_report.json"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for Numba graph building.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).parent / "configs" / args.config
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    setup_logging(args.log_level)
    cfg = load_config(str(config_path))
    run(cfg)


if __name__ == "__main__":
    main()

