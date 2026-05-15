#!/usr/bin/env python3
"""
Phase 2 (Sequential CPU): Graph Building
Reads canonical cleaned payments Parquet parts and builds a bipartite directed graph:

    payer_node_id  ->  payee_key

Aggregates edges by (payer_node_id, payee_key):
- w_total   = sum(amount_usd)
- n_payments = count(rows)
Optional (if payment_date exists, ISO strings):
- min_date, max_date

Outputs:
- edges.parquet
- nodes.parquet
- graph_build_report.json
- graph_build_manifest.json

Design goals:
- Deterministic IDs (CPU/GPU semantic alignment)
- Streaming-friendly (two-stage aggregation to avoid OOM)
- No fraud scoring, no graph algorithms here
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    input_clean_dir: str
    input_clean_manifest: Optional[str]
    output_graph_dir: str

    payer_id_col: str
    payer_name_norm_col: str
    payee_key_col: str
    amount_col: str
    date_col: Optional[str] = None

    min_edge_weight: float = 0.0

    tmp_dir_name: str = "_tmp_edge_parts"
    write_format: str = "parquet"
    output_dir_override: Optional[str] = None


def load_config(path: str) -> GraphBuildConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config_dir = Path(path).parent

    def resolve_path(p: str) -> str:
        if not p:
            return p
        path_obj = Path(p)
        if not path_obj.is_absolute():
            path_obj = config_dir / p
        return str(path_obj.resolve())

    input_clean_dir = resolve_path(raw["input_clean_dir"])
    output_graph_dir = resolve_path(raw["output_graph_dir"])
    input_clean_manifest = resolve_path(raw.get("input_clean_manifest")) if raw.get("input_clean_manifest") else None
    output_dir_override = resolve_path(raw.get("output_dir")) if raw.get("output_dir") else None

    return GraphBuildConfig(
        input_clean_dir=input_clean_dir,
        input_clean_manifest=input_clean_manifest,
        output_graph_dir=output_graph_dir,

        payer_id_col=str(raw.get("payer_id_col", "payer_id")),
        payer_name_norm_col=str(raw.get("payer_name_norm_col", "payer_name_norm")),
        payee_key_col=str(raw.get("payee_key_col", "payee_key")),
        amount_col=str(raw.get("amount_col", "amount_usd")),
        date_col=str(raw.get("date_col")) if raw.get("date_col") else None,

        min_edge_weight=float(raw.get("min_edge_weight", 0.0)),
        tmp_dir_name=str(raw.get("tmp_dir_name", "_tmp_edge_parts")),
        write_format=str(raw.get("write_format", "parquet")),
        output_dir_override=output_dir_override,
    )


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def list_parquet_parts(clean_dir: str) -> List[str]:
    p = Path(clean_dir)
    parts = sorted([str(x) for x in p.glob("part-*.parquet")])
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet found under {clean_dir}")
    return parts

def payer_node_id_series(payer_id: pd.Series, payer_name_norm: pd.Series) -> pd.Series:
    """
    Deterministic payer node ID:
      if payer_id present and non-empty -> PAYER_ID:<payer_id>
      else -> PAYER_NAME:<payer_name_norm>
    """
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

def write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)

def read_manifest_parts(manifest_path: str, clean_dir: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # Accept either {"parts": [...]} or a bare list
    parts = m.get("parts") if isinstance(m, dict) else m
    if not parts:
        raise ValueError(f"Manifest has no 'parts': {manifest_path}")
    out = []
    for part in parts:
        # Support entries as dicts {"part_file": "part-00000.parquet", ...}
        if isinstance(part, dict):
            part_file = part.get("part_file")
            if not part_file:
                continue
        else:
            part_file = part
        part_path = Path(part_file)
        if not part_path.is_absolute():
            part_path = Path(clean_dir) / part_file
        out.append(str(part_path))
    if not out:
        raise ValueError(f"Manifest parts could not be resolved: {manifest_path}")
    return out


def aggregate_part(
    part_path: str,
    cfg: GraphBuildConfig,
    part_idx: int,
    tmp_edges_dir: str
) -> Tuple[str, Dict[str, int]]:
    """
    Reads one cleaned part, aggregates edges locally, writes tmp aggregated edge part.
    Returns path to tmp part and simple counters.
    """
    df = pd.read_parquet(part_path)

    missing = [c for c in [cfg.payer_id_col, cfg.payer_name_norm_col, cfg.payee_key_col, cfg.amount_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {part_path}: {missing}")

    payer_id = df[cfg.payer_id_col]
    payer_name_norm = df[cfg.payer_name_norm_col]
    payee_key = df[cfg.payee_key_col]
    amount = pd.to_numeric(df[cfg.amount_col], errors="coerce")

    src = payer_node_id_series(payer_id, payer_name_norm)
    dst = payee_key.fillna("").astype(str).str.strip()

    base = pd.DataFrame({
        "src": src,
        "dst": dst,
        "w": amount,
    })

    if cfg.date_col and cfg.date_col in df.columns:
        d = df[cfg.date_col].fillna("").astype(str).str.strip()
        base["date"] = d
    else:
        base["date"] = ""

    base = base[base["dst"].ne("") & base["src"].ne("") & base["w"].notna()]
    rows_used = len(base)

    if cfg.date_col and cfg.date_col in df.columns:
        agg = base.groupby(["src", "dst"], as_index=False).agg(
            w_total=("w", "sum"),
            n_payments=("w", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
    else:
        agg = base.groupby(["src", "dst"], as_index=False).agg(
            w_total=("w", "sum"),
            n_payments=("w", "size"),
        )

    if cfg.min_edge_weight > 0:
        agg = agg[agg["w_total"] >= cfg.min_edge_weight]

    out_path = str(Path(tmp_edges_dir) / f"tmp-edges-part-{part_idx:05d}.parquet")
    write_parquet(agg, out_path)

    counters = {
        "rows_read": int(len(df)),
        "rows_used": int(rows_used),
        "tmp_edges": int(len(agg)),
    }
    return out_path, counters


def final_reduce(tmp_parts: List[str], cfg: GraphBuildConfig) -> pd.DataFrame:
    """
    Loads tmp aggregated edge parts and reduces them into final edges.
    This is the second aggregation stage: sum(w_total), sum(n_payments), min(min_date), max(max_date).
    """
    dfs = []
    for p in tmp_parts:
        dfs.append(pd.read_parquet(p))

    if not dfs:
        raise RuntimeError("No tmp edge parts to reduce.")

    all_edges = pd.concat(dfs, ignore_index=True)

    if "min_date" in all_edges.columns and "max_date" in all_edges.columns:
        edges = all_edges.groupby(["src", "dst"], as_index=False).agg(
            w_total=("w_total", "sum"),
            n_payments=("n_payments", "sum"),
            min_date=("min_date", "min"),
            max_date=("max_date", "max"),
        )
    else:
        edges = all_edges.groupby(["src", "dst"], as_index=False).agg(
            w_total=("w_total", "sum"),
            n_payments=("n_payments", "sum"),
        )

    if cfg.min_edge_weight > 0:
        edges = edges[edges["w_total"] >= cfg.min_edge_weight]

    return edges


def build_nodes(edges: pd.DataFrame, output_graph_dir: str) -> Tuple[str, Dict[str, int]]:
    """
    Build nodes table from edges (src + dst), with derived node_type.
    """
    src_nodes = edges["src"].dropna().astype(str)
    dst_nodes = edges["dst"].dropna().astype(str)
    all_nodes = pd.concat([src_nodes, dst_nodes], ignore_index=True).drop_duplicates()

    node_types = all_nodes.map(node_type_from_id)
    nodes_df = pd.DataFrame({
        "node_id": all_nodes,
        "node_type": node_types,
    })

    out_path = str(Path(output_graph_dir) / "nodes.parquet")
    write_parquet(nodes_df, out_path)

    counts = nodes_df["node_type"].value_counts(dropna=False).to_dict()
    return out_path, {
        "total_nodes": int(len(nodes_df)),
        "nodes_by_type": {str(k): int(v) for k, v in counts.items()},
    }


def run(cfg: GraphBuildConfig) -> None:
    t0 = time.perf_counter()

    output_dir = Path(cfg.output_dir_override) if cfg.output_dir_override else Path(cfg.output_graph_dir)
    ensure_dir(str(output_dir))

    tmp_edges_dir = output_dir / cfg.tmp_dir_name
    ensure_dir(str(tmp_edges_dir))

    if cfg.input_clean_manifest:
        parts = read_manifest_parts(cfg.input_clean_manifest, cfg.input_clean_dir)
    else:
        parts = list_parquet_parts(cfg.input_clean_dir)

    logging.info("Graph build input parts: %d", len(parts))
    logging.info("Output graph dir: %s", cfg.output_graph_dir)

    t_read_agg_start = time.perf_counter()
    tmp_paths: List[str] = []
    totals = {
        "rows_read": 0,
        "rows_used": 0,
        "tmp_edges": 0,
    }

    for idx, part in enumerate(parts):
        out_tmp, ctr = aggregate_part(part, cfg, idx, str(tmp_edges_dir))
        tmp_paths.append(out_tmp)
        for k in totals:
            totals[k] += ctr.get(k, 0)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(parts):
            logging.info(
                "Processed %d/%d parts | rows_used=%d | tmp_edges=%d",
                idx + 1, len(parts), totals["rows_used"], totals["tmp_edges"]
            )

    t_read_agg = time.perf_counter() - t_read_agg_start

    t_reduce_start = time.perf_counter()
    edges = final_reduce(tmp_paths, cfg)
    t_reduce = time.perf_counter() - t_reduce_start

    t_write_start = time.perf_counter()
    edges_path = str(output_dir / "edges.parquet")
    write_parquet(edges, edges_path)
    t_write_edges = time.perf_counter() - t_write_start

    t_nodes_start = time.perf_counter()
    nodes_path, node_counts = build_nodes(edges, str(output_dir))
    t_nodes = time.perf_counter() - t_nodes_start

    w = edges["w_total"].to_numpy()
    edge_stats = {
        "edges": int(len(edges)),
        "w_total_min": float(np.min(w)) if len(w) else None,
        "w_total_mean": float(np.mean(w)) if len(w) else None,
        "w_total_median": float(np.median(w)) if len(w) else None,
        "w_total_max": float(np.max(w)) if len(w) else None,
    }

    total_time = time.perf_counter() - t0
    report = {
        "counts": {
            "input_parts": len(parts),
            **totals,
            **edge_stats,
            **node_counts,
        },
        "timings_sec": {
            "read_and_local_aggregate": t_read_agg,
            "final_reduce": t_reduce,
            "write_edges": t_write_edges,
            "build_nodes": t_nodes,
            "total": total_time,
        },
        "config": {
            "payer_id_col": cfg.payer_id_col,
            "payer_name_norm_col": cfg.payer_name_norm_col,
            "payee_key_col": cfg.payee_key_col,
            "amount_col": cfg.amount_col,
            "date_col": cfg.date_col,
            "min_edge_weight": cfg.min_edge_weight,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = str(output_dir / "graph_build_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    manifest = {
        "tmp_edge_parts_dir": str(tmp_edges_dir),
        "tmp_edge_parts": [Path(p).name for p in tmp_paths],
        "final_edges": "edges.parquet",
        "final_nodes": "nodes.parquet",
    }
    manifest_path = str(output_dir / "graph_build_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logging.info("Wrote edges: %s", edges_path)
    logging.info("Wrote nodes: %s", nodes_path)
    logging.info("Wrote report: %s", report_path)


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
    cfg = GraphBuildConfig(
        input_clean_dir=str(clean_dir),
        input_clean_manifest=str(clean_manifest) if clean_manifest else None,
        output_graph_dir=str(out_dir),
        payer_id_col=str(phase_cfg.get("payer_id_col", "payer_id")),
        payer_name_norm_col=str(phase_cfg.get("payer_name_norm_col", "payer_name_norm")),
        payee_key_col=str(phase_cfg.get("payee_key_col", "payee_key")),
        amount_col=str(phase_cfg.get("amount_col", "amount_usd")),
        date_col=str(phase_cfg.get("date_col")) if phase_cfg.get("date_col") else None,
        min_edge_weight=float(phase_cfg.get("min_edge_weight", 0.0)),
        tmp_dir_name=str(phase_cfg.get("tmp_dir_name", "_tmp_edge_parts")),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        output_dir_override=str(out_dir),
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
            "manifest": str(out_dir / "graph_build_manifest.json"),
            "report": str(out_dir / "graph_build_report.json"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for CPU graph building.")
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
