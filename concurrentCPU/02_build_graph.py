#!/usr/bin/env python3
"""
Phase 2 (Concurrent CPU via Dask): Graph Building

Reads canonical cleaned payments Parquet dataset (Phase 1) and builds a bipartite directed graph:

    payer_node_id  ->  payee_key

Aggregates edges by (src, dst):
- w_total    = sum(amount_usd)
- n_payments = count(rows)
Optional (if payment_date exists):
- min_date, max_date

Outputs:
- edges.parquet (dataset)
- nodes.parquet
- graph_build_report.json

Paper-consistent intent:
- Partitioned storage (Parquet) + Dask groupby for parallel aggregation.
- Deterministic node IDs (same rules as your sequential version).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class GraphBuildConfig:
    input_clean_dir: str          # directory of Phase1 parquet dataset (payments_clean/)
    output_graph_dir: str

    payer_id_col: str = "payer_id"
    payer_name_norm_col: str = "payer_name_norm"
    payee_key_col: str = "payee_key"
    amount_col: str = "amount_usd"
    date_col: Optional[str] = "payment_date"

    min_edge_weight: float = 0.0

    # Dask tuning
    use_dask: bool = True
    dask_npartitions: int = 0
    scheduler: str = "threads"


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
        use_dask=bool(raw.get("use_dask", True)),
        dask_npartitions=int(raw.get("dask_npartitions", 0)),
        scheduler=str(raw.get("scheduler", "threads")),
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


def run(cfg: GraphBuildConfig) -> None:
    try:
        import dask
        import dask.dataframe as dd
    except Exception as e:
        raise ImportError("Dask is required for concurrent CPU Phase 2. Install: pip install dask[dataframe]") from e

    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    out_dir = Path(cfg.output_graph_dir)
    ensure_dir(str(out_dir))

    clean_dir = Path(cfg.input_clean_dir)
    if not clean_dir.exists():
        raise FileNotFoundError(f"input_clean_dir not found: {clean_dir}")

    # Read the parquet dataset produced by Phase 1
    t_read_start = time.perf_counter()
    ddf = dd.read_parquet(str(clean_dir), engine="pyarrow")
    if cfg.dask_npartitions and cfg.dask_npartitions > 0:
        ddf = ddf.repartition(npartitions=int(cfg.dask_npartitions))

    timings["t_read_parquet_graph"] = round(time.perf_counter() - t_read_start, 4)

    required = [cfg.payer_id_col, cfg.payer_name_norm_col, cfg.payee_key_col, cfg.amount_col]
    missing = [c for c in required if c not in ddf.columns]
    if missing:
        raise KeyError(f"Missing required columns in cleaned dataset: {missing}. Available: {list(ddf.columns)}")

    # Build base edge table per partition (pandas)
    def _mk_edges(pdf: pd.DataFrame) -> pd.DataFrame:
        payer_id = pdf[cfg.payer_id_col]
        payer_name_norm = pdf[cfg.payer_name_norm_col]
        payee_key = pdf[cfg.payee_key_col].fillna("").astype(str).str.strip()
        amount = pd.to_numeric(pdf[cfg.amount_col], errors="coerce")

        src = payer_node_id_series(payer_id, payer_name_norm)
        dst = payee_key

        out = pd.DataFrame({"src": src, "dst": dst, "w": amount})

        if cfg.date_col and cfg.date_col in pdf.columns:
            out["date"] = pdf[cfg.date_col].fillna("").astype(str).str.strip()
        else:
            out["date"] = ""

        out = out[out["src"].ne("") & out["dst"].ne("") & out["w"].notna()]
        return out

    t_edges_base_start = time.perf_counter()
    meta_edges = {"src": "object", "dst": "object", "w": "float64", "date": "object"}
    edges_base = ddf.map_partitions(_mk_edges, meta=meta_edges)
    timings["t_edges_base_graph"] = round(time.perf_counter() - t_edges_base_start, 4)

    t_groupby_start = time.perf_counter()

    # Parallel aggregation
    if cfg.date_col:
        grouped = edges_base.groupby(["src", "dst"]).agg(
            w_total=("w", "sum"),
            n_payments=("w", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        ).reset_index()
    else:
        grouped = edges_base.groupby(["src", "dst"]).agg(
            w_total=("w", "sum"),
            n_payments=("w", "size"),
        ).reset_index()

    timings["t_groupby_graph"] = round(time.perf_counter() - t_groupby_start, 4)

    if cfg.min_edge_weight and cfg.min_edge_weight > 0:
        grouped = grouped[grouped["w_total"] >= cfg.min_edge_weight]

    # Write edges as parquet dataset (partitioned)
    edges_path = out_dir / "edges.parquet"
    t_write_edges_start = time.perf_counter()
    grouped.to_parquet(str(edges_path), engine="pyarrow", write_index=False, schema="infer")
    timings["t_write_edges_parquet"] = round(time.perf_counter() - t_write_edges_start, 4)

    t_nodes_start = time.perf_counter()

    # Nodes table: unique(src ∪ dst)
    src_nodes = grouped["src"]
    dst_nodes = grouped["dst"]
    all_nodes = dd.concat([src_nodes, dst_nodes], axis=0).drop_duplicates().to_frame(name="node_id")

    def _node_types(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf["node_type"] = pdf["node_id"].astype(str).map(node_type_from_id)
        return pdf

    meta_nodes = {"node_id": "object", "node_type": "object"}
    nodes_ddf = all_nodes.map_partitions(_node_types, meta=meta_nodes)
    nodes_path = out_dir / "nodes.parquet"
    nodes_ddf.to_parquet(str(nodes_path), engine="pyarrow", write_index=False, schema="infer")
    timings["t_write_nodes_parquet"] = round(time.perf_counter() - t_nodes_start, 4)

    # Stats/report (compute a few key scalars)
    t_stats_start = time.perf_counter()
    edges_count = int(grouped.shape[0].compute())
    nodes_count = int(all_nodes.shape[0].compute())

    w = grouped["w_total"]
    w_min = float(w.min().compute()) if edges_count else None
    w_mean = float(w.mean().compute()) if edges_count else None
    w_median = float(w.quantile(0.5).compute()) if edges_count else None
    w_max = float(w.max().compute()) if edges_count else None

    # Node type counts
    node_type_counts = nodes_ddf["node_type"].value_counts().compute().to_dict()
    timings["t_stats"] = round(time.perf_counter() - t_stats_start, 4)

    total_time = time.perf_counter() - t0
    report = {
        "inputs": {
            "clean_dir": str(clean_dir),
        },
        "outputs": {
            "edges_path": str(edges_path),
            "nodes_path": str(nodes_path),
        },
        "counts": {
            "edges": edges_count,
            "total_nodes": nodes_count,
            "nodes_by_type": {str(k): int(v) for k, v in node_type_counts.items()},
            "w_total_min": w_min,
            "w_total_mean": w_mean,
            "w_total_median": w_median,
            "w_total_max": w_max,
        },
        "timings_sec": {
            "total": round(total_time, 4),
            **timings,
        },
        "config": {
            "payer_id_col": cfg.payer_id_col,
            "payer_name_norm_col": cfg.payer_name_norm_col,
            "payee_key_col": cfg.payee_key_col,
            "amount_col": cfg.amount_col,
            "date_col": cfg.date_col,
            "min_edge_weight": cfg.min_edge_weight,
            "use_dask": cfg.use_dask,
            "dask_npartitions": cfg.dask_npartitions,
            "scheduler": cfg.scheduler,
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
    clean_manifest: Optional[Path],  # kept for interface compatibility; unused in Dask version
    out_dir: Path,
    approach: str,
    run_id: str,
    dataset_name: str,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg = pipeline_cfg.get("phase2_graph", {})
    global_workers = int(pipeline_cfg.get("execution", {}).get("max_workers", 0) or 0)
    auto_parts = 0 if global_workers == 0 else max(4, global_workers * 4)

    cfg = GraphBuildConfig(
        input_clean_dir=str(clean_dir),
        output_graph_dir=str(out_dir),
        payer_id_col=str(phase_cfg.get("payer_id_col", "payer_id")),
        payer_name_norm_col=str(phase_cfg.get("payer_name_norm_col", "payer_name_norm")),
        payee_key_col=str(phase_cfg.get("payee_key_col", "payee_key")),
        amount_col=str(phase_cfg.get("amount_col", "amount_usd")),
        date_col=str(phase_cfg.get("date_col")) if phase_cfg.get("date_col") else None,
        min_edge_weight=float(phase_cfg.get("min_edge_weight", 0.0)),
        use_dask=bool(phase_cfg.get("use_dask", True)),
        dask_npartitions=int(phase_cfg.get("dask_npartitions", auto_parts)),
        scheduler=str(phase_cfg.get("scheduler", "threads")),
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
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
    ap.add_argument("--config", required=True, help="YAML config for concurrent CPU graph building.")
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
