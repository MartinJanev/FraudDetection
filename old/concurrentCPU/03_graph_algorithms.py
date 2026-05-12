#!/usr/bin/env python3
"""\
Phase 3 (Concurrent CPU via NetworKit + Dask): Graph Algorithms

This version aligns with the paper's *parallel CPU* idea:
- Use partitioned Parquet and Dask to parallelize degree aggregation.
- Use NetworKit (OpenMP) to parallelize PageRank + connected components.

Reads:   <graph_dir>/edges.parquet  (src, dst, w_total, n_payments)
Writes:  degree.parquet, pagerank.parquet, components.parquet, algos_report.json
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
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

try:
    import networkit as nk
except Exception as e:
    raise ImportError(
        "NetworKit is required for this script (exclusive backend). "
        "Install: pip install networkit"
    ) from e

try:
    import dask
    import dask.dataframe as dd

    _HAS_DASK = True
except Exception:
    dd = None
    dask = None
    _HAS_DASK = False


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

    # NetworKit/OpenMP threads; 0 = library default
    max_threads: int = 0

    # Degree computation backend
    degrees_backend: str = "dask"  # {"dask", "pandas"}

    # Dask tuning
    dask_scheduler: str = "threads"  # {"threads", "processes", "single-threaded"}
    dask_npartitions: int = 0  # 0 = let Dask decide


def load_config(path: str) -> AlgoConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config_dir = Path(path).parent

    def resolve(p: str) -> str:
        if not p:
            return p
        p_obj = Path(p)
        if not p_obj.is_absolute():
            p_obj = config_dir / p
        return str(p_obj.resolve())

    return AlgoConfig(
        graph_dir=resolve(raw["graph_dir"]),
        pagerank_alpha=float(raw.get("pagerank_alpha", 0.85)),
        pagerank_tol=float(raw.get("pagerank_tol", 1.0e-6)),
        pagerank_max_iter=int(raw.get("pagerank_max_iter", 100)),
        compute_components=bool(raw.get("compute_components", True)),
        output_dir=resolve(raw.get("output_dir")) if raw.get("output_dir") else None,
        max_threads=int(raw.get("max_threads", 0)),
        degrees_backend=str(raw.get("degrees_backend", "dask")).lower(),
        dask_scheduler=str(raw.get("dask_scheduler", "threads")).lower(),
        dask_npartitions=int(raw.get("dask_npartitions", 0)),
    )


def _edges_path(graph_dir: str) -> Path:
    return Path(graph_dir) / "edges.parquet"


def load_edges_pandas(graph_dir: str) -> pd.DataFrame:
    path = _edges_path(graph_dir)
    if not path.exists():
        raise FileNotFoundError(f"edges.parquet not found in {graph_dir}")

    df = pd.read_parquet(path, columns=["src", "dst", "w_total", "n_payments"])

    required = {"src", "dst", "w_total", "n_payments"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in edges.parquet: {missing}")

    return df


def load_edges_dask(graph_dir: str, *, npartitions: int = 0):
    if not _HAS_DASK:
        raise ImportError(
            "Dask is not installed, but degrees_backend='dask' was requested. "
            "Install: pip install dask[dataframe]"
        )

    path = _edges_path(graph_dir)
    if not path.exists():
        raise FileNotFoundError(f"edges.parquet not found in {graph_dir}")

    ddf = dd.read_parquet(str(path), columns=["src", "dst", "w_total", "n_payments"])
    if npartitions and npartitions > 0:
        ddf = ddf.repartition(npartitions=int(npartitions))
    return ddf


def compute_degrees_pandas(edges: pd.DataFrame) -> pd.DataFrame:
    out_deg = (
        edges.groupby("src")["w_total"]
        .agg(out_weight="sum", out_degree="size")
        .reset_index()
        .rename(columns={"src": "node_id"})
    )
    in_deg = (
        edges.groupby("dst")["w_total"]
        .agg(in_weight="sum", in_degree="size")
        .reset_index()
        .rename(columns={"dst": "node_id"})
    )

    deg = pd.merge(out_deg, in_deg, on="node_id", how="outer").fillna(0)
    deg["total_weight"] = deg["in_weight"] + deg["out_weight"]
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]
    return deg


def compute_degrees_dask(
    ddf,
    *,
    scheduler: str,
) -> pd.DataFrame:
    """Compute degree features using parallel Dask groupby, then materialize to pandas."""

    # out-degree + out-weight
    out = ddf.groupby("src").agg({"w_total": "sum", "dst": "count"}).rename(
        columns={"w_total": "out_weight", "dst": "out_degree"}
    )

    # in-degree + in-weight
    inn = ddf.groupby("dst").agg({"w_total": "sum", "src": "count"}).rename(
        columns={"w_total": "in_weight", "src": "in_degree"}
    )

    # Convert indices to a common key name.
    out = out.reset_index().rename(columns={"src": "node_id"})
    inn = inn.reset_index().rename(columns={"dst": "node_id"})

    with dask.config.set(scheduler=scheduler):
        out_pd, in_pd = dask.compute(out, inn)

    deg = pd.merge(out_pd, in_pd, on="node_id", how="outer").fillna(0)
    deg["total_weight"] = deg["in_weight"] + deg["out_weight"]
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]

    return deg


def _nk_build_graph(edges: pd.DataFrame) -> Tuple["nk.graph.Graph", pd.Index]:
    """Build a NetworKit directed weighted graph from (src,dst,w_total)."""

    # Map node ids to contiguous ints for NetworKit.
    nodes = pd.Index(
        pd.concat([edges["src"], edges["dst"]], ignore_index=True)
        .astype(str)
        .unique()
    )
    node_to_int = {n: i for i, n in enumerate(nodes)}

    g = nk.graph.Graph(n=len(nodes), weighted=True, directed=True)

    # Fast row iteration.
    for s, d, w in edges[["src", "dst", "w_total"]].itertuples(index=False, name=None):
        if pd.isna(w):
            continue
        u = node_to_int.get(str(s))
        v = node_to_int.get(str(d))
        if u is None or v is None:
            continue
        g.addEdge(u, v, float(w))

    return g, nodes


def compute_pagerank(
    edges: pd.DataFrame,
    alpha: float,
    tol: float,
    max_iter: int,
    *,
    max_threads: int,
) -> pd.DataFrame:
    if max_threads > 0:
        nk.setNumberOfThreads(int(max_threads))

    g, nodes = _nk_build_graph(edges)

    # NetworKit PageRank: (Graph, damping, tol)
    pr = nk.centrality.PageRank(g, float(alpha), float(tol))

    # Best-effort max-iteration control across versions.
    if hasattr(pr, "setMaxIterations"):
        pr.setMaxIterations(int(max_iter))
    elif hasattr(pr, "maxIterations"):
        try:
            pr.maxIterations = int(max_iter)
        except Exception:
            pass

    pr.run()

    return pd.DataFrame({"node_id": nodes.tolist(), "pagerank": pr.scores()})


def compute_components(edges: pd.DataFrame, *, max_threads: int) -> pd.DataFrame:
    if max_threads > 0:
        nk.setNumberOfThreads(int(max_threads))

    g, nodes = _nk_build_graph(edges)

    # Weak components => undirected view
    ug = nk.graphtools.toUndirected(type(g)) if isinstance(g, type) else nk.graphtools.toUndirected(g)
    cc = nk.components.ConnectedComponents(ug)
    cc.run()

    rows = []
    for cid, members in enumerate(cc.getComponents()):
        for u in members:
            rows.append((nodes[int(u)], cid))

    return pd.DataFrame(rows, columns=["node_id", "component_id"])


def _maybe_set_dask_threads(max_threads: int) -> None:
    """Best-effort: limit Dask threadpool parallelism to match experiment settings."""
    if max_threads <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))


def run(cfg: AlgoConfig) -> None:
    t0 = time.perf_counter()

    # Keep thread counts consistent across numerical kernels.
    if cfg.max_threads > 0:
        nk.setNumberOfThreads(int(cfg.max_threads))
        _maybe_set_dask_threads(int(cfg.max_threads))

    edges_path = _edges_path(cfg.graph_dir)
    print(f"[phase3_algos] graph_dir={cfg.graph_dir} | edges={edges_path}", flush=True)

    if cfg.max_threads and cfg.max_threads > 0:
        nk.setNumberOfThreads(cfg.max_threads)
        print(f"[phase3_algos] NetworKit threads={nk.getMaxNumberOfThreads()}", flush=True)

    # --- Degree features (parallel if dask backend) ---
    t_deg_start = time.perf_counter()
    degrees_backend_used = cfg.degrees_backend

    if cfg.degrees_backend == "dask":
        print(f"[phase3_algos] computing degrees via Dask (scheduler={cfg.dask_scheduler}, nparts={cfg.dask_npartitions or 'auto'})", flush=True)
        ddf_edges = load_edges_dask(cfg.graph_dir, npartitions=cfg.dask_npartitions)
        degrees = compute_degrees_dask(ddf_edges, scheduler=cfg.dask_scheduler)
        edges_count = int(ddf_edges.shape[0].compute())
    elif cfg.degrees_backend == "pandas":
        print("[phase3_algos] computing degrees via pandas", flush=True)
        edges_pd = load_edges_pandas(cfg.graph_dir)
        edges_count = int(len(edges_pd))
        degrees = compute_degrees_pandas(edges_pd)
    else:
        raise ValueError("degrees_backend must be one of {'dask','pandas'}")

    t_deg = time.perf_counter() - t_deg_start

    print(f"[phase3_algos] degree rows={len(degrees)}", flush=True)

    # --- Graph algorithms (NetworKit; require edges in-memory as pandas) ---
    t_edges_pd_start = time.perf_counter()
    edges_pd = load_edges_pandas(cfg.graph_dir)
    t_edges_pd = time.perf_counter() - t_edges_pd_start

    logging.info("Loaded edges (pandas for NetworKit): %d", len(edges_pd))

    # Node count used in NetworKit (unique over src ∪ dst)
    src_series = edges_pd["src"] if isinstance(edges_pd["src"], pd.Series) else pd.Series(edges_pd["src"])
    dst_series = edges_pd["dst"] if isinstance(edges_pd["dst"], pd.Series) else pd.Series(edges_pd["dst"])
    n_nodes = int(pd.unique(pd.concat([src_series, dst_series], axis=0)).size)

    t_pr_start = time.perf_counter()
    pagerank_df = compute_pagerank(
        edges_pd,
        cfg.pagerank_alpha,
        cfg.pagerank_tol,
        cfg.pagerank_max_iter,
        max_threads=cfg.max_threads,
    )
    t_pr = time.perf_counter() - t_pr_start

    comp_df = None
    t_comp = None
    if cfg.compute_components:
        t_comp_start = time.perf_counter()
        comp_df = compute_components(edges_pd, max_threads=cfg.max_threads)
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
            "edges_path": str(_edges_path(cfg.graph_dir)),
            "edges": int(edges_count),
            "nodes": int(n_nodes),
        },
        "outputs": {
            "degree": str(degrees_path),
            "pagerank": str(pagerank_path),
            "components": str(components_path) if comp_df is not None else None,
        },
        "timings_sec": {
            "degree": float(t_deg),
            "edges_to_pandas_for_graph": float(t_edges_pd),
            "pagerank": float(t_pr),
            "components": float(t_comp) if t_comp is not None else None,
            "total": float(total_time),
        },
        "config": {
            "pagerank_alpha": cfg.pagerank_alpha,
            "pagerank_tol": cfg.pagerank_tol,
            "pagerank_max_iter": cfg.pagerank_max_iter,
            "compute_components": cfg.compute_components,
            "max_threads": cfg.max_threads,
            "degrees_backend": cfg.degrees_backend,
            "dask_scheduler": cfg.dask_scheduler,
            "dask_npartitions": cfg.dask_npartitions,
        },
        "runtime": {
            "backend_used": "networkit",
            "degrees_backend_used": degrees_backend_used,
            "has_dask": bool(_HAS_DASK),
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

    # After writing outputs
    print(f"[phase3_algos] wrote degree -> {degrees_path}", flush=True)
    print(f"[phase3_algos] wrote pagerank -> {pagerank_path}", flush=True)
    if comp_df is not None:
        print(f"[phase3_algos] wrote components -> {components_path}", flush=True)

    print(f"[phase3_algos] stats degree_min={degrees['total_degree'].min()} degree_max={degrees['total_degree'].max()}", flush=True)
    print(f"[phase3_algos] timings total={total_time:.2f}s", flush=True)


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
    global_workers = int(pipeline_cfg.get("execution", {}).get("max_workers", 0) or 0)
    max_threads = int(global_workers)

    cfg = AlgoConfig(
        graph_dir=str(graph_input_dir),
        pagerank_alpha=float(phase_cfg.get("pagerank_alpha", 0.85)),
        pagerank_tol=float(phase_cfg.get("pagerank_tol", 1.0e-6)),
        pagerank_max_iter=int(phase_cfg.get("pagerank_max_iter", 100)),
        compute_components=bool(phase_cfg.get("compute_components", True)),
        output_dir=str(out_dir),
        max_threads=max_threads,
        degrees_backend=str(phase_cfg.get("degrees_backend", "dask")),
        dask_scheduler=str(phase_cfg.get("dask_scheduler", "threads")),
        dask_npartitions=int(phase_cfg.get("dask_npartitions", 0)),
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg)
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    return {
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
            "components": str(out_dir / "components.parquet"),
            "report": str(out_dir / "algos_report.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Graph algorithms (Concurrent CPU)")
    parser.add_argument("--config", required=True, help="Path to YAML config for graph algorithms")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...)")
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
