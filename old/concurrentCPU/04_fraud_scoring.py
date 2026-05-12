#!/usr/bin/env python3
"""
Phase 4 (CPU): Fraud Scoring / Risk Ranking (Deterministic)

Consumes graph algorithm outputs (Phase 3), currently:
- degree.parquet (node_id, in_weight/in_degree/out_weight/out_degree OR in_w/in_deg/out_w/out_deg)
Optionally:
- pagerank.parquet (node_id, pagerank)

Also consumes (if needed to attach node_type):
- nodes.parquet (node_id, node_type)

Produces:
- risk_scores.parquet
- topk_risk_scores.parquet
- fraud_scoring_report.json

"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dask.dataframe.io.parquet.core import apply_filters


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class ScoreConfig:
    graph_dir: str

    # Inputs
    degree_path: str = ""
    nodes_path: str = ""          # optional override; used if node_type missing in degree
    pagerank_path: str = ""       # optional override

    # Output
    output_dir: Optional[str] = None
    top_k: int = 200

    # Scope
    score_node_types: List[str] = None

    # Method
    method: str = "robust_z_logsum"
    eps: float = 1e-9

    # Weights (degree-based)
    w_in_w: float = 0.55
    w_in_deg: float = 0.25
    w_out_w: float = 0.10
    w_out_deg: float = 0.10

    # Optional PageRank contribution (off by default)
    use_pagerank: bool = False
    w_pagerank: float = 0.0

    # Filters
    min_in_w: float = 0.0
    min_in_deg: int = 0


def load_config(path: str) -> ScoreConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(path).parent

    def resolve_path(p: str) -> str:
        if not p:
            return ""
        p_obj = Path(p)
        if not p_obj.is_absolute():
            p_obj = config_dir / p
        return str(p_obj.resolve())

    graph_dir = resolve_path(raw["graph_dir"])
    output_dir = resolve_path(raw.get("output_dir")) if raw.get("output_dir") else None

    degree_path = resolve_path(raw.get("degree_path", "")) if raw.get("degree_path") else ""
    nodes_path = resolve_path(raw.get("nodes_path", "")) if raw.get("nodes_path") else ""
    pagerank_path = resolve_path(raw.get("pagerank_path", "")) if raw.get("pagerank_path") else ""

    score_node_types = raw.get("score_node_types", ["physician", "teaching_hospital"])

    return ScoreConfig(
        graph_dir=graph_dir,
        degree_path=degree_path,
        nodes_path=nodes_path,
        pagerank_path=pagerank_path,
        output_dir=output_dir if output_dir else graph_dir,
        top_k=int(raw.get("top_k", 200)),
        score_node_types=list(score_node_types),
        method=str(raw.get("method", "robust_z_logsum")),
        eps=float(raw.get("eps", 1e-9)),
        w_in_w=float(raw.get("w_in_w", 0.55)),
        w_in_deg=float(raw.get("w_in_deg", 0.25)),
        w_out_w=float(raw.get("w_out_w", 0.10)),
        w_out_deg=float(raw.get("w_out_deg", 0.10)),
        use_pagerank=bool(raw.get("use_pagerank", False)),
        w_pagerank=float(raw.get("w_pagerank", 0.0)),
        min_in_w=float(raw.get("min_in_w", 0.0)),
        min_in_deg=int(raw.get("min_in_deg", 0)),
    )


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def median_abs_deviation(x: np.ndarray, eps: float = 1e-9) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad + eps)


def robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.median(x)
    mad = median_abs_deviation(x, eps=eps)
    return 0.6745 * (x - med) / mad


def _normalize_degree_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
      in_weight/in_degree/out_weight/out_degree
    or:
      in_w/in_deg/out_w/out_deg
    Normalize to in_w/in_deg/out_w/out_deg.
    """
    mapping = {
        "in_weight": "in_w",
        "in_degree": "in_deg",
        "out_weight": "out_w",
        "out_degree": "out_deg",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def _attach_node_type_if_missing(df: pd.DataFrame, cfg: ScoreConfig) -> pd.DataFrame:
    """
    If node_type is missing from degree output, attach it from nodes.parquet.
    """
    if "node_type" in df.columns:
        return df

    # Try explicit override first, otherwise default to graph_dir/nodes.parquet
    nodes_path = cfg.nodes_path or str(Path(cfg.graph_dir) / "nodes.parquet")
    if not Path(nodes_path).exists():
        raise KeyError(
            "degree.parquet does not contain node_type, and nodes.parquet was not found.\n"
            f"Tried: {nodes_path}\n"
            "Fix: ensure Phase 2 writes nodes.parquet, or add nodes_path in Phase 4 config."
        )

    nodes = pd.read_parquet(nodes_path)
    if "node_id" not in nodes.columns or "node_type" not in nodes.columns:
        raise KeyError(
            f"nodes.parquet missing required columns node_id/node_type. Available: {nodes.columns.tolist()}"
        )

    out = df.merge(nodes[["node_id", "node_type"]], on="node_id", how="left")
    if out["node_type"].isna().any():
        # Keep going; but reportability is useful, so log it.
        missing = int(out["node_type"].isna().sum())
        logging.warning("node_type missing for %d nodes after join with nodes.parquet", missing)

    return out


def _attach_pagerank_if_enabled(df: pd.DataFrame, cfg: ScoreConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Optionally attach pagerank. Returns (df, stats).
    """
    stats = {"pagerank_rows": 0, "pagerank_missing": 0}

    if not cfg.use_pagerank or cfg.w_pagerank == 0.0:
        return df, stats

    pr_path = cfg.pagerank_path or str(Path(cfg.graph_dir) / "pagerank.parquet")
    if not Path(pr_path).exists():
        raise FileNotFoundError(
            f"use_pagerank=true but pagerank.parquet not found: {pr_path}"
        )

    pr = pd.read_parquet(pr_path)
    if "node_id" not in pr.columns or "pagerank" not in pr.columns:
        raise KeyError(
            f"pagerank.parquet missing required columns node_id/pagerank. Available: {pr.columns.tolist()}"
        )

    stats["pagerank_rows"] = int(len(pr))

    out = df.merge(pr[["node_id", "pagerank"]], on="node_id", how="left")
    stats["pagerank_missing"] = int(out["pagerank"].isna().sum())
    out["pagerank"] = pd.to_numeric(out["pagerank"], errors="coerce").fillna(0.0)

    return out, stats


def score_table(df: pd.DataFrame, cfg: ScoreConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns (scored_df, stats_dict)
    """
    df = _normalize_degree_columns(df)
    df = _attach_node_type_if_missing(df, cfg)
    df, pr_stats = _attach_pagerank_if_enabled(df, cfg)

    required = ["node_id", "node_type", "in_w", "in_deg", "out_w", "out_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"degree.parquet missing columns: {missing}. Available: {df.columns.tolist()}")

    if cfg.score_node_types is None:
        score_types = ["physician", "teaching_hospital"]
    else:
        score_types = cfg.score_node_types

    df = df[df["node_type"].isin(score_types)].copy()

    in_w = pd.to_numeric(df["in_w"], errors="coerce").fillna(0.0).to_numpy()
    in_deg = pd.to_numeric(df["in_deg"], errors="coerce").fillna(0.0).to_numpy()
    out_w = pd.to_numeric(df["out_w"], errors="coerce").fillna(0.0).to_numpy()
    out_deg = pd.to_numeric(df["out_deg"], errors="coerce").fillna(0.0).to_numpy()

    if cfg.method != "robust_z_logsum":
        raise ValueError(f"Unsupported scoring method: {cfg.method}")

    f_in_w = np.log1p(in_w)
    f_in_deg = np.log1p(in_deg)
    f_out_w = np.log1p(out_w)
    f_out_deg = np.log1p(out_deg)

    z_in_w = robust_z(f_in_w, eps=cfg.eps)
    z_in_deg = robust_z(f_in_deg, eps=cfg.eps)
    z_out_w = robust_z(f_out_w, eps=cfg.eps)
    z_out_deg = robust_z(f_out_deg, eps=cfg.eps)

    risk = (
        cfg.w_in_w * z_in_w
        + cfg.w_in_deg * z_in_deg
        + cfg.w_out_w * z_out_w
        + cfg.w_out_deg * z_out_deg
    )

    z_pr = None
    if cfg.use_pagerank and cfg.w_pagerank != 0.0:
        pr = pd.to_numeric(df["pagerank"], errors="coerce").fillna(0.0).to_numpy()
        f_pr = np.log1p(pr)
        z_pr = robust_z(f_pr, eps=cfg.eps)
        risk = risk + cfg.w_pagerank * z_pr
        df["pagerank"] = pr
        df["z_pagerank"] = z_pr

    risk_shifted = risk - np.min(risk) if len(risk) else risk

    df["risk_score"] = risk_shifted
    df["z_in_w"] = z_in_w
    df["z_in_deg"] = z_in_deg
    df["z_out_w"] = z_out_w
    df["z_out_deg"] = z_out_deg

    df["rank"] = df["risk_score"].rank(ascending=False, method="dense").astype(int)
    df = df.sort_values(["risk_score", "in_w", "in_deg"], ascending=False)

    stats = {
        "rows_scored": float(len(df)),
        "risk_min": float(df["risk_score"].min()) if len(df) else None,
        "risk_mean": float(df["risk_score"].mean()) if len(df) else None,
        "risk_median": float(df["risk_score"].median()) if len(df) else None,
        "risk_max": float(df["risk_score"].max()) if len(df) else None,
        "pagerank_rows": float(pr_stats.get("pagerank_rows", 0)),
        "pagerank_missing": float(pr_stats.get("pagerank_missing", 0)),
        "use_pagerank": bool(cfg.use_pagerank and cfg.w_pagerank != 0.0),
    }

    return df, stats


def score_nodes(df, cfg):
    scored, stats = score_table(df, cfg)
    # persist stats for caller if needed
    return scored


def run(cfg: ScoreConfig) -> None:
    t0 = time.perf_counter()
    print(f"[phase4_score] graph_dir={cfg.graph_dir} | degree={cfg.degree_path or 'auto'} | pagerank={'on' if cfg.use_pagerank else 'off'}", flush=True)

    degree_path = Path(cfg.degree_path) if cfg.degree_path else Path(cfg.graph_dir) / "degree.parquet"
    if not degree_path.exists():
        raise FileNotFoundError(f"Degree file not found: {degree_path}")

    t_read0 = time.perf_counter()
    df = pd.read_parquet(degree_path)
    t_read = time.perf_counter() - t_read0
    print(f"[phase4_score] loaded degree rows={len(df)}", flush=True)

    df = _normalize_degree_columns(df)
    df = _attach_node_type_if_missing(df, cfg)
    df, pr_stats = _attach_pagerank_if_enabled(df, cfg)
    print(f"[phase4_score] pagerank rows={pr_stats.get('pagerank_rows', 0)} missing={pr_stats.get('pagerank_missing', 0)}", flush=True)

    # apply simple min filters inline (since we already have df in memory)
    df = df[df["in_w"].fillna(0.0) >= cfg.min_in_w]
    df = df[df["in_deg"].fillna(0).astype(int) >= cfg.min_in_deg]
    print(f"[phase4_score] after filters rows={len(df)}", flush=True)

    scores = score_nodes(df, cfg)
    print(f"[phase4_score] scored rows={len(scores)}", flush=True)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_path = out_dir / "risk_scores.parquet"
    topk_path = out_dir / "topk_risk_scores.parquet"
    scores.to_parquet(risk_path, index=False)
    scores.head(cfg.top_k).to_parquet(topk_path, index=False)
    print(f"[phase4_score] wrote risk -> {risk_path}", flush=True)
    print(f"[phase4_score] wrote topk -> {topk_path}", flush=True)

    total_time = time.perf_counter() - t0
    print(f"[phase4_score] timings total={total_time:.2f}s", flush=True)


def run_from_pipeline(
    pipeline_cfg: Dict,
    *,
    graph_algos_dir: Path,
    out_dir: Path,
    approach: str,
    run_id: str,
    dataset_name: str,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg = pipeline_cfg.get("phase4_score", {})

    # In your pipeline, graph_algos_dir is phase3_algos output dir.
    # nodes.parquet typically lives in phase2_graph, but we can infer it:
    # phase3_algos is a sibling of phase2_graph under the run directory.
    nodes_guess = graph_algos_dir.parent / "phase2_graph" / "nodes.parquet"

    cfg = ScoreConfig(
        graph_dir=str(graph_algos_dir),
        degree_path=str(Path(graph_algos_dir) / "degree.parquet"),
        nodes_path=str(nodes_guess) if nodes_guess.exists() else str(phase_cfg.get("nodes_path", "")),
        pagerank_path=str(Path(graph_algos_dir) / "pagerank.parquet"),
        output_dir=str(out_dir),
        top_k=int(phase_cfg.get("top_k", 200)),
        score_node_types=list(phase_cfg.get("score_node_types", ["physician", "teaching_hospital"])),
        method=str(phase_cfg.get("method", "robust_z_logsum")),
        eps=float(phase_cfg.get("eps", 1e-9)),
        w_in_w=float(phase_cfg.get("w_in_w", 0.55)),
        w_in_deg=float(phase_cfg.get("w_in_deg", 0.25)),
        w_out_w=float(phase_cfg.get("w_out_w", 0.10)),
        w_out_deg=float(phase_cfg.get("w_out_deg", 0.10)),
        use_pagerank=bool(phase_cfg.get("use_pagerank", False)),
        w_pagerank=float(phase_cfg.get("w_pagerank", 0.0)),
        min_in_w=float(phase_cfg.get("min_in_w", 0.0)),
        min_in_deg=int(phase_cfg.get("min_in_deg", 0)),
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg)
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    out_scores = out_dir / "risk_scores.parquet"
    out_topk = out_dir / "topk_risk_scores.parquet"
    out_report = out_dir / "fraud_scoring_report.json"

    return {
        "phase": "phase4_score",
        "dataset": dataset_name,
        "approach": approach,
        "run_id": run_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "risk_scores": str(out_scores),
            "topk": str(out_topk),
            "report": str(out_report),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: Fraud Scoring / Risk Ranking (CPU)")
    ap.add_argument("--config", required=True, help="YAML config for Phase 4 scoring.")
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
