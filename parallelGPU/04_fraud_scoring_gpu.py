#!/usr/bin/env python3
"""
Phase 4 (GPU-friendly): Fraud Scoring / Risk Ranking

Semantics mirror sequentialCPU/04_fraud_scoring.py:
- Score using degree features only (no cleaning, deterministic)
- Outputs: risk_scores.parquet, topk_risk_scores.parquet, fraud_scoring_report.json
- run_from_pipeline is orchestrated by run_pipeline.py

GPU note: computation is light; we use pandas/NumPy for portability while reading
Parquet via cudf if available to keep symmetry, but semantics identical.
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

try:
    import cudf  # type: ignore
except Exception:
    cudf = None


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class ScoreConfig:
    graph_dir: str
    degree_path: str
    output_dir: str

    top_k: int = 200
    score_node_types: List[str] = None
    method: str = "robust_z_logsum"
    eps: float = 1e-9
    w_in_w: float = 0.55
    w_in_deg: float = 0.25
    w_out_w: float = 0.10
    w_out_deg: float = 0.10
    min_in_w: float = 0.0
    min_in_deg: int = 0


def median_abs_deviation(x: np.ndarray, eps: float = 1e-9) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad + eps)


def robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.median(x)
    mad = median_abs_deviation(x, eps=eps)
    return 0.6745 * (x - med) / mad


def score_table(df: pd.DataFrame, cfg: ScoreConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    column_mapping = {
        "in_weight": "in_w",
        "in_degree": "in_deg",
        "out_weight": "out_w",
        "out_degree": "out_deg",
    }
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["node_id", "node_type", "in_w", "in_deg", "out_w", "out_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"degree.parquet missing columns: {missing}. Available: {df.columns.tolist()}")

    df = df[df["node_type"].isin(cfg.score_node_types)].copy()
    df = df[df["in_w"].fillna(0.0) >= cfg.min_in_w]
    df = df[df["in_deg"].fillna(0).astype(int) >= cfg.min_in_deg]

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
    }
    return df, stats


def load_degree(degree_path: Path) -> pd.DataFrame:
    if not degree_path.exists():
        raise FileNotFoundError(f"Degree file not found: {degree_path}")
    if cudf is not None:
        try:
            return cudf.read_parquet(str(degree_path)).to_pandas()
        except Exception:
            pass
    return pd.read_parquet(degree_path)


def run(cfg: ScoreConfig) -> None:
    t0 = time.perf_counter()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    degree_path = Path(cfg.degree_path)
    deg = load_degree(degree_path)

    t_score0 = time.perf_counter()
    scored, stats = score_table(deg, cfg)
    t_score = time.perf_counter() - t_score0

    out_scores = Path(cfg.output_dir) / "risk_scores.parquet"
    out_topk = Path(cfg.output_dir) / "topk_risk_scores.parquet"
    out_report = Path(cfg.output_dir) / "fraud_scoring_report.json"

    t_write0 = time.perf_counter()
    scored.to_parquet(out_scores, index=False)
    scored.head(cfg.top_k).to_parquet(out_topk, index=False)
    t_write = time.perf_counter() - t_write0

    total = time.perf_counter() - t0

    report = {
        "inputs": {
            "graph_dir": cfg.graph_dir,
            "degree_path": str(degree_path),
        },
        "outputs": {
            "risk_scores": str(out_scores),
            "topk": str(out_topk),
        },
        "scoring": {
            "method": cfg.method,
            "score_node_types": cfg.score_node_types,
            "weights": {
                "w_in_w": cfg.w_in_w,
                "w_in_deg": cfg.w_in_deg,
                "w_out_w": cfg.w_out_w,
                "w_out_deg": cfg.w_out_deg,
            },
            "filters": {
                "min_in_w": cfg.min_in_w,
                "min_in_deg": cfg.min_in_deg,
            },
        },
        "stats": stats,
        "timings_sec": {
            "score_compute": round(t_score, 4),
            "write_outputs": round(t_write, 4),
            "total": round(total, 4),
        },
        "topk_preview": scored[["node_id", "node_type", "risk_score", "rank", "in_w", "in_deg"]].head(20).to_dict(orient="records"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("Wrote risk scores: %s", out_scores)
    logging.info("Wrote top-%d: %s", cfg.top_k, out_topk)
    logging.info("Wrote report: %s", out_report)


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
    cfg = ScoreConfig(
        graph_dir=str(graph_algos_dir),
        degree_path=str(Path(graph_algos_dir) / "degree.parquet"),
        output_dir=str(out_dir),
        top_k=int(phase_cfg.get("top_k", 200)),
        score_node_types=list(phase_cfg.get("score_node_types", ["physician", "teaching_hospital"])),
        method=str(phase_cfg.get("method", "robust_z_logsum")),
        eps=float(phase_cfg.get("eps", 1e-9)),
        w_in_w=float(phase_cfg.get("w_in_w", 0.55)),
        w_in_deg=float(phase_cfg.get("w_in_deg", 0.25)),
        w_out_w=float(phase_cfg.get("w_out_w", 0.10)),
        w_out_deg=float(phase_cfg.get("w_out_deg", 0.10)),
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
    ap = argparse.ArgumentParser(description="Phase 4: Fraud Scoring / Risk Ranking (GPU-friendly)")
    ap.add_argument("--config", required=True, help="YAML config for Phase 4 scoring.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = ScoreConfig(
        graph_dir=str(Path(raw["graph_dir"]).resolve()),
        degree_path=str(Path(raw.get("degree_path", Path(raw["graph_dir"]) / "degree.parquet")).resolve()),
        output_dir=str(Path(raw.get("output_dir", Path(raw["graph_dir"]) / "phase4_score")).resolve()),
        top_k=int(raw.get("top_k", 200)),
        score_node_types=list(raw.get("score_node_types", ["physician", "teaching_hospital"])),
        method=str(raw.get("method", "robust_z_logsum")),
        eps=float(raw.get("eps", 1e-9)),
        w_in_w=float(raw.get("w_in_w", 0.55)),
        w_in_deg=float(raw.get("w_in_deg", 0.25)),
        w_out_w=float(raw.get("w_out_w", 0.10)),
        w_out_deg=float(raw.get("w_out_deg", 0.10)),
        min_in_w=float(raw.get("min_in_w", 0.0)),
        min_in_deg=int(raw.get("min_in_deg", 0)),
    )
    run(cfg)


if __name__ == "__main__":
    main()
