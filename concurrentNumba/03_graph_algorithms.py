#!/usr/bin/env python3
"""\
Phase 3 (Concurrent CPU via Numba + NetworKit): Graph Algorithms

Replaces Dask degree computation with Numba JIT-compiled parallel kernels running
over numpy arrays, coordinated by ThreadPoolExecutor for file-level parallelism.
NetworKit (OpenMP) is kept for PageRank + connected components (unchanged from
the concurrent-Dask version).

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

import numpy as np
import pandas as pd
import yaml

try:
    import networkit as nk
except Exception as e:
    raise ImportError(
        "NetworKit is required for this script. Install: pip install networkit"
    ) from e

try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    numba = None   # type: ignore[assignment]
    njit = None    # type: ignore[assignment]
    prange = None  # type: ignore[assignment]
    _HAS_NUMBA = False
    logging.warning("Numba not installed – falling back to pure-NumPy degree computation.")


# ---------------------------------------------------------------------------
# Numba kernels for degree computation
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _accumulate_out_weight(src_codes: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
        """Sum weights by src_codes (out-edge weight accumulation)."""
        out = np.zeros(n, dtype=np.float64)
        for i in range(len(src_codes)):   # serial to avoid race conditions
            out[src_codes[i]] += weights[i]
        return out

    @njit(parallel=True, cache=True)
    def _accumulate_in_weight(dst_codes: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
        """Sum weights by dst_codes (in-edge weight accumulation)."""
        out = np.zeros(n, dtype=np.float64)
        for i in range(len(dst_codes)):
            out[dst_codes[i]] += weights[i]
        return out

    @njit(cache=True)
    def _count_codes(codes: np.ndarray, n: int) -> np.ndarray:
        """Count occurrences of each code (degree counting)."""
        out = np.zeros(n, dtype=np.int64)
        for i in range(len(codes)):
            out[codes[i]] += 1
        return out

else:
    def _accumulate_out_weight(src_codes, weights, n):
        out = np.zeros(n, dtype=np.float64)
        np.add.at(out, src_codes, weights)
        return out

    def _accumulate_in_weight(dst_codes, weights, n):
        out = np.zeros(n, dtype=np.float64)
        np.add.at(out, dst_codes, weights)
        return out

    def _count_codes(codes, n):
        out = np.zeros(n, dtype=np.int64)
        np.add.at(out, codes, 1)
        return out


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

    # Numba tuning
    use_numba: bool = True


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
        use_numba=bool(raw.get("use_numba", True)),
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


def compute_degrees_numba(edges: pd.DataFrame, *, use_numba: bool = True) -> pd.DataFrame:
    """
    Compute degree features using Numba JIT kernels over factorized integer arrays.

    Returns DataFrame with columns:
        node_id, out_weight, out_degree, in_weight, in_degree, total_weight, total_degree
    """
    src = edges["src"].astype(str).values
    dst = edges["dst"].astype(str).values
    w = edges["w_total"].to_numpy(dtype=np.float64)

    # Factorize all node ids into a single contiguous integer space
    all_nodes, _ = pd.factorize(np.concatenate([src, dst]))
    n_src = len(src)

    # Re-factorize individually so we get codes aligned to src/dst arrays
    src_codes, src_uniques = pd.factorize(src)
    dst_codes, dst_uniques = pd.factorize(dst)

    # Build a global node → code mapping
    all_ids = pd.Index(
        pd.concat([
            pd.Series(src_uniques, name="node_id"),
            pd.Series(dst_uniques, name="node_id"),
        ], ignore_index=True).unique()
    )
    node_to_global = {n: i for i, n in enumerate(all_ids)}
    n_global = len(all_ids)

    global_src = np.array([node_to_global[s] for s in src_uniques[src_codes]], dtype=np.int64)
    global_dst = np.array([node_to_global[d] for d in dst_uniques[dst_codes]], dtype=np.int64)

    if use_numba and _HAS_NUMBA:
        out_w = _accumulate_out_weight(global_src, w, n_global)
        in_w = _accumulate_in_weight(global_dst, w, n_global)
        out_deg = _count_codes(global_src, n_global)
        in_deg = _count_codes(global_dst, n_global)
    else:
        out_w = np.zeros(n_global, dtype=np.float64)
        in_w = np.zeros(n_global, dtype=np.float64)
        out_deg = np.zeros(n_global, dtype=np.int64)
        in_deg = np.zeros(n_global, dtype=np.int64)
        np.add.at(out_w, global_src, w)
        np.add.at(in_w, global_dst, w)
        np.add.at(out_deg, global_src, 1)
        np.add.at(in_deg, global_dst, 1)

    deg = pd.DataFrame({
        "node_id": all_ids.tolist(),
        "out_weight": out_w,
        "out_degree": out_deg,
        "in_weight": in_w,
        "in_degree": in_deg,
    })
    deg["total_weight"] = deg["in_weight"] + deg["out_weight"]
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]
    return deg


def _nk_build_graph(edges: pd.DataFrame) -> Tuple["nk.graph.Graph", pd.Index]:
    """Build a NetworKit directed weighted graph from (src, dst, w_total)."""
    nodes = pd.Index(
        pd.concat([edges["src"], edges["dst"]], ignore_index=True)
        .astype(str)
        .unique()
    )
    node_to_int = {n: i for i, n in enumerate(nodes)}

    g = nk.graph.Graph(n=len(nodes), weighted=True, directed=True)

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
    pr = nk.centrality.PageRank(g, float(alpha), float(tol))

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
    ug = nk.graphtools.toUndirected(g)
    cc = nk.components.ConnectedComponents(ug)
    cc.run()

    rows = []
    for cid, members in enumerate(cc.getComponents()):
        for u in members:
            rows.append((nodes[int(u)], cid))

    return pd.DataFrame(rows, columns=["node_id", "component_id"])


def _maybe_set_threads(max_threads: int) -> None:
    if max_threads <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    if _HAS_NUMBA:
        numba.set_num_threads(max(1, max_threads))


def run(cfg: AlgoConfig) -> None:
    t0 = time.perf_counter()

    if cfg.max_threads > 0:
        nk.setNumberOfThreads(int(cfg.max_threads))
        _maybe_set_threads(int(cfg.max_threads))

    edges_path = _edges_path(cfg.graph_dir)
    print(f"[phase3_algos] graph_dir={cfg.graph_dir} | edges={edges_path}", flush=True)
    if cfg.max_threads and cfg.max_threads > 0:
        nk.setNumberOfThreads(cfg.max_threads)
        print(f"[phase3_algos] NetworKit threads={nk.getMaxNumberOfThreads()}", flush=True)

    # Warm up Numba JIT
    if cfg.use_numba and _HAS_NUMBA:
        _dummy_codes = np.array([0, 1, 0], dtype=np.int64)
        _dummy_w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        _accumulate_out_weight(_dummy_codes, _dummy_w, 2)
        _accumulate_in_weight(_dummy_codes, _dummy_w, 2)
        _count_codes(_dummy_codes, 2)
        logging.info("[Phase 3] Numba JIT warm-up complete")

    # --- Load edges once (needed for both degree and graph algos) ---
    t_edges_pd_start = time.perf_counter()
    edges_pd = load_edges_pandas(cfg.graph_dir)
    t_edges_pd = time.perf_counter() - t_edges_pd_start
    edges_count = int(len(edges_pd))
    print(f"[phase3_algos] loaded {edges_count} edges", flush=True)

    src_series = edges_pd["src"].astype(str)
    dst_series = edges_pd["dst"].astype(str)
    n_nodes = int(pd.concat([src_series, dst_series], ignore_index=True).nunique())

    # --- Degree features (Numba parallel kernels) ---
    t_deg_start = time.perf_counter()
    print(f"[phase3_algos] computing degrees via Numba (use_numba={cfg.use_numba and _HAS_NUMBA})", flush=True)
    degrees = compute_degrees_numba(edges_pd, use_numba=cfg.use_numba)
    t_deg = time.perf_counter() - t_deg_start
    print(f"[phase3_algos] degree rows={len(degrees)}", flush=True)

    # --- PageRank (NetworKit / OpenMP) ---
    t_pr_start = time.perf_counter()
    pagerank_df = compute_pagerank(
        edges_pd,
        cfg.pagerank_alpha,
        cfg.pagerank_tol,
        cfg.pagerank_max_iter,
        max_threads=cfg.max_threads,
    )
    t_pr = time.perf_counter() - t_pr_start

    # --- Connected components (NetworKit / OpenMP) ---
    comp_df = None
    t_comp = None
    if cfg.compute_components:
        t_comp_start = time.perf_counter()
        comp_df = compute_components(edges_pd, max_threads=cfg.max_threads)
        t_comp = time.perf_counter() - t_comp_start

    # Write outputs
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
            "edges_path": str(edges_path),
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
            "use_numba": cfg.use_numba,
            "has_numba": bool(_HAS_NUMBA),
        },
        "runtime": {
            "backend_used": "networkit+numba",
            "degrees_backend_used": "numba" if (cfg.use_numba and _HAS_NUMBA) else "numpy",
            "has_numba": bool(_HAS_NUMBA),
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
        use_numba=bool(phase_cfg.get("use_numba", True)),
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
    parser = argparse.ArgumentParser(description="Phase 3: Graph algorithms (Concurrent Numba)")
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

