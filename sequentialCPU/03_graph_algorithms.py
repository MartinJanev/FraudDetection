#!/usr/bin/env python3
"""
Phase 3 (Sequential CPU): Graph Algorithms
Reads edges.parquet from graph build and computes:
- Degree (in/out/total)
- PageRank (alpha configurable)
- Connected components (weak)
Outputs parquet files under graph directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class AlgoConfig:
    graph_dir: str
    pagerank_alpha: float = 0.85
    pagerank_tol: float = 1.0e-6
    pagerank_max_iter: int = 100
    compute_components: bool = True
    output_dir: Optional[str] = None


def load_config(path: str) -> AlgoConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config_dir = Path(path).parent

    def resolve_path(p: str) -> str:
        if not p:
            return p
        p_obj = Path(p)
        if not p_obj.is_absolute():
            p_obj = config_dir / p
        return str(p_obj.resolve())

    graph_dir = resolve_path(raw["graph_dir"])
    output_dir = resolve_path(raw.get("output_dir")) if raw.get("output_dir") else None

    return AlgoConfig(
        graph_dir=graph_dir,
        pagerank_alpha=float(raw.get("pagerank_alpha", 0.85)),
        pagerank_tol=float(raw.get("pagerank_tol", 1.0e-6)),
        pagerank_max_iter=int(raw.get("pagerank_max_iter", 100)),
        compute_components=bool(raw.get("compute_components", True)),
        output_dir=output_dir,
    )


def load_edges(graph_dir: str) -> pd.DataFrame:
    path = Path(graph_dir) / "edges.parquet"
    if not path.exists():
        raise FileNotFoundError(f"edges.parquet not found in {graph_dir}")
    df = pd.read_parquet(path)
    required = {"src", "dst", "w_total", "n_payments"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in edges.parquet: {missing}")
    return df


def node_type_from_id(node_id: str) -> str:
    if node_id.startswith("PAYER_ID:") or node_id.startswith("PAYER_NAME:"):
        return "payer"
    if node_id.startswith("PHYS_"):
        return "physician"
    if node_id.startswith("HOSP_"):
        return "teaching_hospital"
    return "unknown"


def compute_degrees(edges: pd.DataFrame) -> pd.DataFrame:
    out_deg = edges.groupby("src")["w_total"].agg(out_weight="sum", out_degree="size").reset_index().rename(columns={"src": "node_id"})
    in_deg = edges.groupby("dst")["w_total"].agg(in_weight="sum", in_degree="size").reset_index().rename(columns={"dst": "node_id"})

    degrees = pd.merge(out_deg, in_deg, on="node_id", how="outer").fillna(0)
    degrees["total_weight"] = degrees["in_weight"] + degrees["out_weight"]
    degrees["total_degree"] = degrees["in_degree"] + degrees["out_degree"]
    degrees["node_type"] = degrees["node_id"].map(node_type_from_id)

    return degrees


def compute_pagerank(edges: pd.DataFrame, alpha: float, tol: float, max_iter: int) -> pd.DataFrame:
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges[["src", "dst", "w_total"]].to_records(index=False))

    pr = nx.pagerank(G, alpha=alpha, tol=tol, max_iter=max_iter, weight="w_total")
    pagerank_df = pd.DataFrame({"node_id": list(pr.keys()), "pagerank": list(pr.values())})
    return pagerank_df


def compute_components(edges: pd.DataFrame) -> pd.DataFrame:
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges[["src", "dst", "w_total"]].to_records(index=False))

    undirected = G.to_undirected()
    comps = list(nx.connected_components(undirected))

    rows = []
    for cid, nodes in enumerate(comps):
        for n in nodes:
            rows.append((n, cid))
    components_df = pd.DataFrame(rows, columns=["node_id", "component_id"])
    return components_df


def run(cfg: AlgoConfig) -> None:
    t0 = time.perf_counter()

    edges = load_edges(cfg.graph_dir)

    logging.info("Loaded edges: %d", len(edges))

    t_deg_start = time.perf_counter()
    degrees = compute_degrees(edges)
    t_deg = time.perf_counter() - t_deg_start

    t_pr_start = time.perf_counter()
    pagerank_df = compute_pagerank(edges, cfg.pagerank_alpha, cfg.pagerank_tol, cfg.pagerank_max_iter)
    t_pr = time.perf_counter() - t_pr_start

    comp_df = None
    t_comp = None
    if cfg.compute_components:
        t_comp_start = time.perf_counter()
        comp_df = compute_components(edges)
        t_comp = time.perf_counter() - t_comp_start

    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.graph_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degrees_path = out_dir / "degree.parquet"
    pagerank_path = out_dir / "pagerank.parquet"
    components_path = out_dir / "components.parquet"

    degrees.to_parquet(degrees_path, index=False)
    pagerank_df.to_parquet(pagerank_path, index=False)
    if comp_df is not None:
        comp_df.to_parquet(components_path, index=False)

    total_time = time.perf_counter() - t0

    report = {
        "inputs": {
            "edges_path": str(Path(cfg.graph_dir) / "edges.parquet"),
            "edges": int(len(edges)),
        },
        "outputs": {
            "degree": str(degrees_path),
            "pagerank": str(pagerank_path),
            "components": str(components_path) if comp_df is not None else None,
        },
        "timings_sec": {
            "degree": t_deg,
            "pagerank": t_pr,
            "components": t_comp,
            "total": total_time,
        },
        "config": {
            "pagerank_alpha": cfg.pagerank_alpha,
            "pagerank_tol": cfg.pagerank_tol,
            "pagerank_max_iter": cfg.pagerank_max_iter,
            "compute_components": cfg.compute_components,
        },
    }

    report_path = out_dir / "algos_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("Wrote degree: %s", degrees_path)
    logging.info("Wrote pagerank: %s", pagerank_path)
    if comp_df is not None:
        logging.info("Wrote components: %s", components_path)
    logging.info("Wrote report: %s", report_path)


def run_from_pipeline(
    pipeline_cfg: Dict,
    *,
    graph_input_dir: Path,
    out_dir: Path,
    approach: str,
    run_id: str,
    dataset_name: str,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg = pipeline_cfg.get("phase3_algos", {})
    cfg = AlgoConfig(
        graph_dir=str(graph_input_dir),
        pagerank_alpha=float(phase_cfg.get("pagerank_alpha", 0.85)),
        pagerank_tol=float(phase_cfg.get("pagerank_tol", 1.0e-6)),
        pagerank_max_iter=int(phase_cfg.get("pagerank_max_iter", 100)),
        compute_components=bool(phase_cfg.get("compute_components", True)),
        output_dir=str(out_dir),
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg)
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    degree_path = out_dir / "degree.parquet"
    pagerank_path = out_dir / "pagerank.parquet"
    components_path = out_dir / "components.parquet"

    return {
        "phase": "phase3_algos",
        "dataset": dataset_name,
        "approach": approach,
        "run_id": run_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "degree_path": str(degree_path),
            "pagerank_path": str(pagerank_path),
            "components_path": str(components_path) if components_path.exists() else None,
            "report": str(out_dir / "algos_report.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Compute graph algorithms (CPU)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config for graph algorithms"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (INFO, DEBUG, ...)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).parent / "configs" / args.config
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(str(config_path))
    setup_logging(args.log_level)
    run(cfg)


if __name__ == "__main__":
    main()

