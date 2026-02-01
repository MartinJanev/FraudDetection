#!/usr/bin/env python3
"""
End-to-end concurrent CPU pipeline runner.
Chains phases 1–4 in-process; can run multiple runs concurrently via threads.

UPDATED: runs are executed sequentially; concurrency must happen *inside phases*.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import concurrent.futures
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_FILES = {
    "clean": "01_clean_data.py",
    "graph": "02_build_graph.py",
    "algos": "03_graph_algorithms.py",
    "score": "04_fraud_scoring.py",
}


def load_phase_module(filename: str, module_name: str):
    path = Path(__file__).parent / filename
    spec = importlib_util.spec_from_file_location(module_name, path)
    module = importlib_util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PHASE_MODULES = {
    "clean": load_phase_module(PHASE_FILES["clean"], "phase1_clean"),
    "graph": load_phase_module(PHASE_FILES["graph"], "phase2_graph"),
    "algos": load_phase_module(PHASE_FILES["algos"], "phase3_algos"),
    "score": load_phase_module(PHASE_FILES["score"], "phase4_score"),
}


def unique_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(2)
    return f"{ts}-{suffix}"


def resolve_path_maybe(base: Path, p: str) -> Path:
    path_obj = Path(p)
    if not path_obj.is_absolute():
        path_obj = base / p
    return path_obj.resolve()


def load_pipeline_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg or {}
    cfg["__config_dir"] = str(path.parent.resolve())
    return cfg


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def append_timing_row(csv_path: Path, row: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "run_id",
        "dataset",
        "approach",
        "phase1_clean",
        "phase2_graph",
        "phase3_algos",
        "phase4_score",
        "total",
        "started_utc",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def compute_summary(csv_path: Path, summary_path: Path) -> None:
    if not csv_path.exists():
        return
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    metrics = ["phase1_clean", "phase2_graph", "phase3_algos", "phase4_score", "total"]
    summary = {"runs": len(rows)}
    for m in metrics:
        vals = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
        if not vals:
            continue
        summary[m] = {
            "mean_sec": mean(vals),
            "std_sec": pstdev(vals) if len(vals) > 1 else 0.0,
            "min_sec": min(vals),
            "max_sec": max(vals),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)



def _safe_read_json(p: Path) -> Dict:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def append_run_metrics_row(csv_path: Path, row: Dict[str, object]) -> None:
    """Append a single run's metrics row to output/<approach>/<dataset>/aggregate/run_metrics.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    fieldnames = [
        "run_id",
        "dataset",
        "approach",
        "scale_mode",
        "scale_value",
        "rows_in",
        "rows_valid",
        "rows_rejected",
        "nodes",
        "edges",
        "phase1_total",
        "phase2_total",
        "phase3_total",
        "phase4_total",
        "alg_degree",
        "alg_pagerank",
        "alg_components",
        "total_pipeline",
        "started_utc",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def extract_run_metrics(*, run_dir: Path, pipeline_cfg: Dict, dataset_name: str, approach: str, run_id: str, total_pipeline: float, started_utc: str) -> Dict[str, object]:
    """Extract metrics from phase reports produced by phases 1–4."""
    dataset_cfg = pipeline_cfg.get("dataset", {}) or {}
    scale_mode = str(dataset_cfg.get("scale_mode", "none"))
    scale_value = dataset_cfg.get("scale_value", None)

    p1 = _safe_read_json(run_dir / "phase1_clean" / "cleaning_report.json")
    p2 = _safe_read_json(run_dir / "phase2_graph" / "graph_build_report.json")
    p3 = _safe_read_json(run_dir / "phase3_algos" / "algos_report.json")
    p4 = _safe_read_json(run_dir / "phase4_score" / "fraud_scoring_report.json")

    rows_in = p1.get("rows_in")
    rows_valid = p1.get("rows_valid")
    rows_rejected = p1.get("rows_rejected")

    counts = p2.get("counts", {}) or {}
    nodes = counts.get("total_nodes")
    edges = counts.get("edges")

    p1_total = (p1.get("timings_sec", {}) or {}).get("total")
    p2_total = (p2.get("timings_sec", {}) or {}).get("total")
    p3_total = (p3.get("timings_sec", {}) or {}).get("total")
    p4_total = (p4.get("timings_sec", {}) or {}).get("total")

    p3_tim = p3.get("timings_sec", {}) or {}
    alg_degree = p3_tim.get("degree")
    alg_pagerank = p3_tim.get("pagerank")
    alg_components = p3_tim.get("components")

    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "approach": approach,
        "scale_mode": scale_mode,
        "scale_value": scale_value,
        "rows_in": rows_in,
        "rows_valid": rows_valid,
        "rows_rejected": rows_rejected,
        "nodes": nodes,
        "edges": edges,
        "phase1_total": p1_total,
        "phase2_total": p2_total,
        "phase3_total": p3_total,
        "phase4_total": p4_total,
        "alg_degree": alg_degree,
        "alg_pagerank": alg_pagerank,
        "alg_components": alg_components,
        "total_pipeline": round(float(total_pipeline), 4),
        "started_utc": started_utc,
    }


def run_once(
    pipeline_cfg: Dict,
    *,
    approach: str,
    out_root: Path,
    dataset_name: str,
) -> Dict:
    run_id = unique_run_id()
    run_dir = out_root / run_id
    phase_dirs = {
        "phase1_clean": run_dir / "phase1_clean",
        "phase2_graph": run_dir / "phase2_graph",
        "phase3_algos": run_dir / "phase3_algos",
        "phase4_score": run_dir / "phase4_score",
        "reports": run_dir / "reports",
        "logs": run_dir / "logs",
        "figs": run_dir / "figs",
    }
    ensure_dirs(list(phase_dirs.values()))

    # Save config snapshot
    cfg_snapshot_path = phase_dirs["reports"] / "config_snapshot.yaml"
    with open(cfg_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline_cfg, f)

    timings = {}
    phase_reports = []

    t_total_start = time.perf_counter()

    # Phase 1: clean
    t0 = time.perf_counter()
    clean_res = PHASE_MODULES["clean"].run_from_pipeline(
        pipeline_cfg,
        out_dir=phase_dirs["phase1_clean"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase1_clean"] = time.perf_counter() - t0
    phase_reports.append(clean_res)

    clean_manifest = Path(clean_res["artifacts"].get("clean_manifest")) if clean_res.get("artifacts") else None
    if clean_manifest and not clean_manifest.exists():
        clean_manifest = None

    # Phase 2: graph build
    t0 = time.perf_counter()
    graph_res = PHASE_MODULES["graph"].run_from_pipeline(
        pipeline_cfg,
        clean_dir=Path(clean_res["artifacts"]["clean_dir"]),
        clean_manifest=clean_manifest,
        out_dir=phase_dirs["phase2_graph"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase2_graph"] = time.perf_counter() - t0
    phase_reports.append(graph_res)

    # Phase 3: algorithms
    t0 = time.perf_counter()
    alg_res = PHASE_MODULES["algos"].run_from_pipeline(
        pipeline_cfg,
        graph_input_dir=Path(graph_res["artifacts"]["edges"]).parent,
        out_dir=phase_dirs["phase3_algos"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase3_algos"] = time.perf_counter() - t0
    phase_reports.append(alg_res)

    # Phase 4: scoring
    t0 = time.perf_counter()
    score_res = PHASE_MODULES["score"].run_from_pipeline(
        pipeline_cfg,
        graph_algos_dir=phase_dirs["phase3_algos"],
        out_dir=phase_dirs["phase4_score"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase4_score"] = time.perf_counter() - t0
    phase_reports.append(score_res)

    timings["total"] = time.perf_counter() - t_total_start

    timings_path = phase_dirs["reports"] / "timings.json"
    run_report = {
        "run_id": run_id,
        "dataset": dataset_name,
        "approach": approach,
        "start_utc": phase_reports[0].get("start_utc"),
        "end_utc": phase_reports[-1].get("end_utc"),
        "phase_reports": phase_reports,
        "timings_sec": timings,
    }
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)

    return {
        "run_id": run_id,
        "timings": timings,
        "report_path": str(timings_path),
        "start_utc": run_report["start_utc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concurrent CPU pipeline end-to-end")
    parser.add_argument("--config", required=True, help="Path to unified pipeline YAML config")
    parser.add_argument("--approach", default="concurent_cpu", help="Approach name for output folder")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute")
    parser.add_argument("--max-workers", type=int, default=None, help="Global CPU concurrency. Overrides YAML execution.max_workers if set.")
    parser.add_argument("--log-level", default="INFO", help="Unused placeholder for future logging")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline_cfg = load_pipeline_config(config_path)

    # --- Global concurrency resolution (single source of truth) ---
    pipeline_cfg.setdefault("execution", {})
    yaml_workers = pipeline_cfg["execution"].get("max_workers")
    if args.max_workers is not None:
        pipeline_cfg["execution"]["max_workers"] = int(args.max_workers)
    elif yaml_workers is not None:
        pipeline_cfg["execution"]["max_workers"] = int(yaml_workers)
    else:
        pipeline_cfg["execution"]["max_workers"] = 0  # 0 = library defaults

    dataset_cfg = pipeline_cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name") or dataset_cfg.get("dataset_name") or "dataset")

    out_root_cfg = pipeline_cfg.get("output", {})
    root_dir = out_root_cfg.get("root_dir", "output")
    out_root = Path(root_dir)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root = out_root / args.approach / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    runs_cfg = pipeline_cfg.get("runs", {})
    runs = int(runs_cfg.get("n_runs", args.runs) or args.runs)

    # NOTE: max_workers for concurrent runs is no longer used; kept for compatibility
    max_workers = pipeline_cfg["execution"].get("max_workers", 0)
    if max_workers is None:
        max_workers = 0
    max_workers = max(0, int(max_workers))

    aggregate_dir = out_root / "aggregate"
    timing_csv = aggregate_dir / "timing_runs.csv"
    timing_summary = aggregate_dir / "timing_summary.json"
    run_metrics_csv = aggregate_dir / "run_metrics.csv"

    def build_row(res: Dict) -> Dict:
        return {
            "run_id": res["run_id"],
            "dataset": dataset_name,
            "approach": args.approach,
            "phase1_clean": round(res["timings"].get("phase1_clean", 0.0), 4),
            "phase2_graph": round(res["timings"].get("phase2_graph", 0.0), 4),
            "phase3_algos": round(res["timings"].get("phase3_algos", 0.0), 4),
            "phase4_score": round(res["timings"].get("phase4_score", 0.0), 4),
            "total": round(res["timings"].get("total", 0.0), 4),
            "started_utc": res.get("start_utc"),
        }

    # IMPORTANT: concurrency must happen *inside phases*, not by running multiple full pipelines in parallel.
    rows = []
    for _ in tqdm(range(runs), desc="Pipeline runs", unit="run"):
        res = run_once(
            pipeline_cfg,
            approach=args.approach,
            out_root=out_root,
            dataset_name=dataset_name,
        )
        rows.append(build_row(res))
        run_dir = out_root / res["run_id"]
        metrics_row = extract_run_metrics(
            run_dir=run_dir,
            pipeline_cfg=pipeline_cfg,
            dataset_name=dataset_name,
            approach=args.approach,
            run_id=res["run_id"],
            total_pipeline=res["timings"].get("total", 0.0),
            started_utc=res.get("start_utc"),
        )
        append_run_metrics_row(run_metrics_csv, metrics_row)

    # Write timing rows sequentially to avoid file contention
    for row in rows:
        append_timing_row(timing_csv, row)

    compute_summary(timing_csv, timing_summary)

    # Optional: warm-up filtering (skip first run) to reduce cache/JIT noise.
    # This is safe and does not affect the original timing_runs.csv.
    if len(rows) >= 2:
        warmup_filtered = rows[1:]
        warmup_csv = aggregate_dir / "timing_runs_warmup_filtered.csv"
        with open(warmup_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(warmup_filtered)

    print(f"Completed {len(rows)} run(s). Results under {out_root}")
    print(f"Aggregate timings: {timing_csv}")
    print(f"Summary: {timing_summary}")


if __name__ == "__main__":
    main()
