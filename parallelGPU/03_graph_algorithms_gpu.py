#!/usr/bin/env python3
"""
Phase 3 (GPU): Graph Algorithms (RAPIDS cuDF/cuGraph)

Inputs (from Phase 2, GPU-safe):
- edges.parquet with columns: src, dst, w_total, n_payments, min_date, max_date
- nodes.parquet with columns: node_id, node_type

Phase 3 contract:
- NO data cleaning
- Minimal graph load (construct cuGraph from edges)
- Run algorithms only
- Log metrics + timings
- Write outputs + manifest (repeatable, deterministic given same inputs)

Phase 2 symmetry:
- Uses output_dir_override pattern
- Writes {phase}_report.json + {phase}_manifest.json
- run_from_pipeline builds config from pipeline_cfg and forces output_dir_override=out_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml

# GPU deps
try:
    import cudf
    import cugraph
except Exception as e:  # pragma: no cover
    cudf = None
    cugraph = None
    _GPU_IMPORT_ERROR = e


# --- helpers (match Phase 2 style; fallback if not available) -----------------
try:
    # If you already have these in your project utils, keep this import.
    from utils.io import ensure_dir  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(p: str) -> None:
        Path(p).mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class AlgoConfig:
    # input
    input_graph_dir: str  # phase2_graph out_dir containing edges.parquet (+ nodes.parquet)
    # output (standalone default + pipeline override)
    output_algos_dir: str
    output_dir_override: Optional[str] = None

    # algo params
    pagerank_alpha: float = 0.85
    pagerank_tol: float = 1.0e-6
    pagerank_max_iter: int = 100
    compute_components: bool = True
    write_results_merged: bool = True


def load_config(path: str) -> AlgoConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(path).parent

    def resolve_path(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        p_obj = Path(p)
        if not p_obj.is_absolute():
            p_obj = config_dir / p
        return str(p_obj.resolve())

    # Default dirs relative to the CONFIG FILE (your requirement)
    default_graph_dir = (config_dir / "output" / "phase2_graph").resolve()
    default_out_dir = (config_dir / "output" / "phase3_algos_gpu").resolve()

    # Pipeline YAML mode: take params from phase3_algos section
    is_pipeline = "phase3_algos" in raw and ("graph_dir" not in raw and "input_graph_dir" not in raw)
    if is_pipeline:
        p3 = raw.get("phase3_algos", {}) or {}

        graph_dir = resolve_path(p3.get("graph_dir") or p3.get("input_graph_dir")) or str(default_graph_dir)
        out_dir = resolve_path(p3.get("output_dir") or p3.get("output_algos_dir")) or str(default_out_dir)

        return AlgoConfig(
            input_graph_dir=str(graph_dir),
            output_algos_dir=str(out_dir),
            output_dir_override=str(out_dir),  # IMPORTANT: pipeline-style override
            pagerank_alpha=float(p3.get("pagerank_alpha", 0.85)),
            pagerank_tol=float(p3.get("pagerank_tol", 1.0e-6)),
            pagerank_max_iter=int(p3.get("pagerank_max_iter", 100)),
            compute_components=bool(p3.get("compute_components", True)),
            write_results_merged=bool(p3.get("write_results_merged", True)),
        )

    # Phase YAML mode: allow top-level graph_dir/output_dir, but still default nicely
    graph_dir = resolve_path(raw.get("graph_dir") or raw.get("input_graph_dir")) or str(default_graph_dir)
    out_dir = resolve_path(raw.get("output_dir") or raw.get("output_algos_dir")) or str(default_out_dir)

    return AlgoConfig(
        input_graph_dir=str(graph_dir),
        output_algos_dir=str(out_dir),
        output_dir_override=resolve_path(raw.get("output_dir_override")) or str(out_dir),
        pagerank_alpha=float(raw.get("pagerank_alpha", 0.85)),
        pagerank_tol=float(raw.get("pagerank_tol", 1.0e-6)),
        pagerank_max_iter=int(raw.get("pagerank_max_iter", 100)),
        compute_components=bool(raw.get("compute_components", True)),
        write_results_merged=bool(raw.get("write_results_merged", True)),
    )


def _require_gpu() -> None:
    if cudf is None or cugraph is None:
        raise RuntimeError(
            "GPU libraries not available. Ensure RAPIDS is installed "
            "(cudf, cugraph) in this environment.\n"
            f"Import error: {_GPU_IMPORT_ERROR}"
        )


def load_edges(graph_dir: str) -> "cudf.DataFrame":
    _require_gpu()

    path = Path(graph_dir) / "edges.parquet"
    if not path.exists():
        raise FileNotFoundError(f"edges.parquet not found in {graph_dir}")

    df = cudf.read_parquet(str(path))

    required = {"src", "dst", "w_total", "n_payments"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in edges.parquet: {missing}")

    # Phase 3 rule: no cleaning/renumbering.
    if not cudf.api.types.is_integer_dtype(df["src"].dtype) or not cudf.api.types.is_integer_dtype(df["dst"].dtype):
        raise TypeError(
            "Phase 2 must output integer node IDs for GPU (src/dst). "
            f"Got dtypes: src={df['src'].dtype}, dst={df['dst'].dtype}"
        )

    # Ensure weight is float for PageRank stability
    if not cudf.api.types.is_float_dtype(df["w_total"].dtype):
        df["w_total"] = df["w_total"].astype("float32")

    return df


def compute_degrees(edges: "cudf.DataFrame") -> "cudf.DataFrame":
    # Weighted out-degree + out-degree count
    out_deg = (
        edges.groupby("src")
        .agg({"w_total": "sum", "dst": "count"})
        .reset_index()
        .rename(columns={"src": "node_id", "w_total": "out_weight", "dst": "out_degree"})
    )

    # Weighted in-degree + in-degree count
    in_deg = (
        edges.groupby("dst")
        .agg({"w_total": "sum", "src": "count"})
        .reset_index()
        .rename(columns={"dst": "node_id", "w_total": "in_weight", "src": "in_degree"})
    )

    deg = out_deg.merge(in_deg, on="node_id", how="outer")

    for col in ["out_weight", "out_degree", "in_weight", "in_degree"]:
        if col not in deg.columns:
            deg[col] = 0
        deg[col] = deg[col].fillna(0)

    deg["total_weight"] = deg["in_weight"] + deg["out_weight"]
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]

    deg["out_degree"] = deg["out_degree"].astype("int64")
    deg["in_degree"] = deg["in_degree"].astype("int64")
    deg["total_degree"] = deg["total_degree"].astype("int64")

    return deg


def build_cugraph_digraph(edges: "cudf.DataFrame") -> "cugraph.Graph":
    # cuGraph DiGraph is deprecated; use Graph(directed=True) with renumber disabled
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        edges,
        source="src",
        destination="dst",
        edge_attr="w_total",
        renumber=False,
    )
    return G


def compute_pagerank(G: "cugraph.DiGraph", alpha: float, tol: float, max_iter: int) -> "cudf.DataFrame":
    pr = cugraph.pagerank(
        G,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
        weight="w_total",
    )
    return pr.rename(columns={"vertex": "node_id"})


def compute_components(edges: "cudf.DataFrame") -> "cudf.DataFrame":
    UG = cugraph.Graph(directed=False)
    UG.from_cudf_edgelist(
        edges,
        source="src",
        destination="dst",
        edge_attr=None,
        renumber=False,
    )
    comps = cugraph.connected_components(UG)
    return comps.rename(columns={"vertex": "node_id", "labels": "component_id"})


def run(cfg: AlgoConfig) -> None:
    _require_gpu()
    t0_total = time.perf_counter()

    output_dir = Path(cfg.output_dir_override) if cfg.output_dir_override else Path(cfg.output_algos_dir)
    ensure_dir(str(output_dir))

    graph_dir = Path(cfg.input_graph_dir)
    edges_path_in = graph_dir / "edges.parquet"
    nodes_path_in = graph_dir / "nodes.parquet"  # optional for Phase 3, but should exist per contract

    logging.info("Graph input dir: %s", str(graph_dir))
    logging.info("Output algos dir: %s", str(output_dir))

    edges = load_edges(str(graph_dir))
    edge_count = int(len(edges))
    logging.info("Loaded edges: %d", edge_count)

    # Degree (cuDF)
    t0 = time.perf_counter()
    degrees = compute_degrees(edges)
    t_deg = time.perf_counter() - t0

    # Build graph once
    t0 = time.perf_counter()
    G = build_cugraph_digraph(edges)
    t_build = time.perf_counter() - t0

    # PageRank
    t0 = time.perf_counter()
    pagerank_df = compute_pagerank(G, cfg.pagerank_alpha, cfg.pagerank_tol, cfg.pagerank_max_iter)
    t_pr = time.perf_counter() - t0

    # Components (optional)
    comp_df = None
    t_comp = None
    if cfg.compute_components:
        t0 = time.perf_counter()
        comp_df = compute_components(edges)
        t_comp = time.perf_counter() - t0

    # Write artifacts
    degree_path = output_dir / "degree.parquet"
    pagerank_path = output_dir / "pagerank.parquet"
    components_path = output_dir / "components.parquet"
    results_path = output_dir / "results.parquet"

    t0 = time.perf_counter()
    degrees.to_parquet(str(degree_path), index=False)
    pagerank_df.to_parquet(str(pagerank_path), index=False)
    if comp_df is not None:
        comp_df.to_parquet(str(components_path), index=False)
    t_write = time.perf_counter() - t0

    # Optional merged results for Phase 4
    merged_rows = None
    if cfg.write_results_merged:
        t0 = time.perf_counter()
        merged = degrees.merge(pagerank_df, on="node_id", how="left")
        if comp_df is not None:
            merged = merged.merge(comp_df, on="node_id", how="left")
        merged_rows = int(len(merged))
        merged.to_parquet(str(results_path), index=False)
        t_merge_write = time.perf_counter() - t0
    else:
        results_path = None
        t_merge_write = None

    total_time = time.perf_counter() - t0_total

    # Counts (Phase-2-like)
    counts = {
        "edges": edge_count,
        "degree_rows": int(len(degrees)),
        "pagerank_rows": int(len(pagerank_df)),
        "components_rows": int(len(comp_df)) if comp_df is not None else 0,
        "merged_rows": int(merged_rows) if merged_rows is not None else 0,
    }

    report = {
        "inputs": {
            "graph_dir": str(graph_dir),
            "edges_path": str(edges_path_in),
            "nodes_path": str(nodes_path_in) if nodes_path_in.exists() else None,
        },
        "outputs": {
            "degree_path": str(degree_path),
            "pagerank_path": str(pagerank_path),
            "components_path": str(components_path) if comp_df is not None else None,
            "results_path": str(results_path) if results_path is not None else None,
        },
        "counts": counts,
        "timings_sec": {
            "degree": float(t_deg),
            "graph_build": float(t_build),
            "pagerank": float(t_pr),
            "components": float(t_comp) if t_comp is not None else None,
            "write_artifacts": float(t_write),
            "merge_and_write_results": float(t_merge_write) if t_merge_write is not None else None,
            "total": float(total_time),
        },
        "config": {
            "pagerank_alpha": cfg.pagerank_alpha,
            "pagerank_tol": cfg.pagerank_tol,
            "pagerank_max_iter": cfg.pagerank_max_iter,
            "compute_components": cfg.compute_components,
            "write_results_merged": cfg.write_results_merged,
            "renumber": False,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = output_dir / "algos_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Small stable manifest like Phase 2
    phase_manifest = {
        "final_degree": "degree.parquet",
        "final_pagerank": "pagerank.parquet",
        "final_components": "components.parquet" if cfg.compute_components else None,
        "final_results": "results.parquet" if cfg.write_results_merged else None,
        "report": "algos_report.json",
    }
    phase_manifest_path = output_dir / "graph_algos_manifest.json"
    with open(phase_manifest_path, "w", encoding="utf-8") as f:
        json.dump(phase_manifest, f, indent=2)

    logging.info("Wrote degree: %s", degree_path)
    logging.info("Wrote pagerank: %s", pagerank_path)
    if comp_df is not None:
        logging.info("Wrote components: %s", components_path)
    if results_path is not None:
        logging.info("Wrote results: %s", results_path)
    logging.info("Wrote report: %s", report_path)
    logging.info("Wrote phase manifest: %s", phase_manifest_path)


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
        input_graph_dir=str(graph_input_dir),
        output_algos_dir=str(out_dir),            # default doesn't matter; override is what matters
        output_dir_override=str(out_dir),         # Phase-2 symmetry: pipeline decides folder
        pagerank_alpha=float(phase_cfg.get("pagerank_alpha", 0.85)),
        pagerank_tol=float(phase_cfg.get("pagerank_tol", 1.0e-6)),
        pagerank_max_iter=int(phase_cfg.get("pagerank_max_iter", 100)),
        compute_components=bool(phase_cfg.get("compute_components", True)),
        write_results_merged=bool(phase_cfg.get("write_results_merged", True)),
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg)
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    manifest = {
        "phase": "phase3_algos",
        "dataset": dataset_name,
        "approach": approach,
        "run_id": run_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "degree": str(out_dir / "degree.parquet"),
            "pagerank": str(out_dir / "pagerank.parquet"),
            "components": str(out_dir / "components.parquet") if (out_dir / "components.parquet").exists() else None,
            "results": str(out_dir / "results.parquet") if (out_dir / "results.parquet").exists() else None,
            "phase_manifest": str(out_dir / "graph_algos_manifest.json"),
            "report": str(out_dir / "algos_report.json"),
        },
    }

    # Pipeline-level manifest.json (same idea you already use)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: Compute graph algorithms (GPU RAPIDS)")
    ap.add_argument("--config", required=True, help="YAML config for GPU graph algorithms.")
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
