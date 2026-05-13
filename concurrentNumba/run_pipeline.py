#!/usr/bin/env python3
"""
End-to-end concurrent Numba CPU pipeline runner.
Chains phases 1–4 in-process.

Parallel execution lives *inside* each phase:
  - Phase 1: ThreadPoolExecutor over CSV chunks + Numba JIT kernels
  - Phase 2: ThreadPoolExecutor over parquet files + Numba aggregation kernels
  - Phase 3: Numba JIT degree kernels + optional NetworKit (OpenMP) for PageRank/components, with a NetworkX fallback
  - Phase 4: Numba JIT scoring kernels
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import secrets
import sys
import time
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_FILES = {
    "clean": "01_clean_data.py",
    "graph": "02_build_graph.py",
    "algos": "03_graph_algorithms.py",
    "score": "04_fraud_scoring.py",
}

try:
    from .numba_kernels import _HAS_NUMBA, warmup_all_kernels
except ImportError:
    from numba_kernels import _HAS_NUMBA, warmup_all_kernels  # type: ignore[import]


def load_phase_module(filename: str, module_name: str):
    path = Path(__file__).parent / filename
    spec = importlib_util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load phase module: {path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _load_phase_modules() -> Dict[str, object]:
    """Deferred phase module loading (avoids import-time side effects)."""
    package = __package__ or "concurrentNumba"
    return {
        "clean": load_phase_module(PHASE_FILES["clean"], f"{package}.phase1_clean"),
        "graph": load_phase_module(PHASE_FILES["graph"], f"{package}.phase2_graph"),
        "algos": load_phase_module(PHASE_FILES["algos"], f"{package}.phase3_algos"),
        "score": load_phase_module(PHASE_FILES["score"], f"{package}.phase4_score"),
    }

import shutil


def cleanup_intermediate_data(run_out_dir: Path):
    """Deletes large intermediate .parquet files but keeps timing and JSON reports."""
    print(f"🧹 Cleaning up large intermediate data in: {run_out_dir}")

    # Folders to completely delete (they only contain parquet data)
    heavy_dirs = [
        run_out_dir / "phase1_clean" / "payments_clean",
        run_out_dir / "phase2_graph" / "_tmp_edge_parts"
    ]

    for d in heavy_dirs:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)

    # Remove any isolated .parquet files in the phase directories
    for phase in ["phase1_clean", "phase2_graph", "phase3_algos", "phase4_score"]:
        phase_dir = run_out_dir / phase
        if phase_dir.exists():
            for parquet_file in phase_dir.glob("*.parquet"):
                parquet_file.unlink()

    print("✅ Cleanup complete. Kept timing metrics and JSON reports.")


def fraction_label_from_fraction(fraction: float) -> str:
    pct = int(round(max(0.0, fraction) * 100))
    return f"fraction_{pct:02d}pct"


def unique_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(2)
    return f"{ts}-{suffix}"


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
    summary: Dict[str, object] = {"runs": len(rows)}
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


def extract_run_metrics(
        *,
        run_dir: Path,
        pipeline_cfg: Dict,
        dataset_name: str,
        approach: str,
        run_id: str,
        total_pipeline: float,
        started_utc: str,
) -> Dict[str, object]:
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
        phase_modules: Dict[str, object],
) -> Dict:
    run_id = unique_run_id()
    print(f"=== Run {run_id} | dataset={dataset_name} | approach={approach} ===", flush=True)
    run_dir = out_root / run_id
    phase_dirs = {
        "phase1_clean": run_dir / "phase1_clean",
        "phase2_graph": run_dir / "phase2_graph",
        "phase3_algos": run_dir / "phase3_algos",
        "phase4_score": run_dir / "phase4_score",
        "reports": run_dir / "reports",
    }
    ensure_dirs(list(phase_dirs.values()))
    print(f"[{run_id}] Outputs -> {run_dir}", flush=True)

    cfg_snapshot_path = phase_dirs["reports"] / "config_snapshot.yaml"
    with open(cfg_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline_cfg, f)

    timings = {}
    phase_reports = []

    t_total_start = time.perf_counter()

    # Phase 1: clean
    print(f"[{run_id}] Starting phase1_clean", flush=True)
    t0 = time.perf_counter()
    clean_res = phase_modules["clean"].run_from_pipeline(
        pipeline_cfg,
        out_dir=phase_dirs["phase1_clean"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase1_clean"] = time.perf_counter() - t0
    print(f"[{run_id}] Finished phase1_clean in {timings['phase1_clean']:.2f}s", flush=True)
    print(f"[{run_id}] Clean output -> {clean_res['artifacts'].get('clean_dir')}", flush=True)

    clean_manifest_val = None
    if clean_res.get("artifacts"):
        cm = clean_res["artifacts"].get("clean_manifest")
        if cm:
            clean_manifest_val = Path(cm)
            if not clean_manifest_val.exists():
                clean_manifest_val = None

    # Phase 2: graph build
    print(f"[{run_id}] Starting phase2_graph", flush=True)
    t0 = time.perf_counter()
    graph_res = phase_modules["graph"].run_from_pipeline(
        pipeline_cfg,
        clean_dir=Path(clean_res["artifacts"]["clean_dir"]),
        clean_manifest=clean_manifest_val,
        out_dir=phase_dirs["phase2_graph"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase2_graph"] = time.perf_counter() - t0
    print(f"[{run_id}] Finished phase2_graph in {timings['phase2_graph']:.2f}s", flush=True)
    print(f"[{run_id}] Graph output -> {graph_res['artifacts'].get('edges')}", flush=True)

    # Phase 3: algorithms
    print(f"[{run_id}] Starting phase3_algos", flush=True)
    t0 = time.perf_counter()
    alg_res = phase_modules["algos"].run_from_pipeline(
        pipeline_cfg,
        graph_input_dir=Path(graph_res["artifacts"]["edges"]).parent,
        out_dir=phase_dirs["phase3_algos"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase3_algos"] = time.perf_counter() - t0
    print(f"[{run_id}] Finished phase3_algos in {timings['phase3_algos']:.2f}s", flush=True)
    phase_reports.append(alg_res)

    # Phase 4: scoring
    print(f"[{run_id}] Starting phase4_score", flush=True)
    t0 = time.perf_counter()
    score_res = phase_modules["score"].run_from_pipeline(
        pipeline_cfg,
        graph_algos_dir=phase_dirs["phase3_algos"],
        out_dir=phase_dirs["phase4_score"],
        approach=approach,
        run_id=run_id,
        dataset_name=dataset_name,
    )
    timings["phase4_score"] = time.perf_counter() - t0
    print(f"[{run_id}] Finished phase4_score in {timings['phase4_score']:.2f}s", flush=True)
    phase_reports.append(score_res)

    timings["total"] = time.perf_counter() - t_total_start
    print(f"[{run_id}] Pipeline done in {timings['total']:.2f}s", flush=True)

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
    parser = argparse.ArgumentParser(description="Run concurrent Numba CPU pipeline end-to-end")
    parser.add_argument("--config", required=True, help="Path to unified pipeline YAML config")
    parser.add_argument("--approach", default="concurrent_numba", help="Approach name for output folder")
    parser.add_argument("--runs", type=int, default=None, help="Number of runs to execute (overrides config)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Global CPU concurrency. Overrides YAML execution.max_workers if set.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline_cfg = load_pipeline_config(config_path)

    pipeline_cfg.setdefault("execution", {})
    yaml_workers = pipeline_cfg["execution"].get("max_workers")
    if args.max_workers is not None:
        pipeline_cfg["execution"]["max_workers"] = int(args.max_workers)
    elif yaml_workers is not None:
        pipeline_cfg["execution"]["max_workers"] = int(yaml_workers)
    else:
        pipeline_cfg["execution"]["max_workers"] = 0

    dataset_cfg = pipeline_cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name") or dataset_cfg.get("dataset_name") or "dataset")
    payment_type = dataset_cfg.get("payment_type") or dataset_cfg.get("type")

    out_root_cfg = pipeline_cfg.get("output", {})
    root_dir = out_root_cfg.get("root_dir", "output")
    out_root = Path(root_dir)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    sampling_fraction = float(pipeline_cfg.get("phase1_clean", {}).get("sampling", {}).get("fraction", 1.0))
    fraction_dir = fraction_label_from_fraction(sampling_fraction)

    worker_label = pipeline_cfg["execution"].get("max_workers", 0)
    worker_dir = f"workers_{worker_label if worker_label > 0 else 'auto'}"

    out_root = out_root / args.approach / dataset_name / fraction_dir / worker_dir
    out_root.mkdir(parents=True, exist_ok=True)

    runs_cfg = pipeline_cfg.get("runs", {})
    config_runs = int(runs_cfg.get("n_runs", 1))
    runs = args.runs if args.runs is not None else config_runs

    max_workers = pipeline_cfg["execution"].get("max_workers", 0)
    if max_workers is None:
        max_workers = 0
    max_workers = max(0, int(max_workers))

    print(
        "Configured run:",
        f"dataset={dataset_name}",
        f"payment_type={payment_type}",
        f"runs={runs}",
        f"approach={args.approach}",
        f"max_workers={max_workers}",
        f"output_root={out_root}",
        sep=" | ",
        flush=True,
    )

    aggregate_dir = out_root / "aggregate"
    timing_csv = aggregate_dir / "timing_runs.csv"
    timing_summary = aggregate_dir / "timing_summary.json"
    run_metrics_csv = aggregate_dir / "run_metrics.csv"

    def build_row(res: Dict) -> Dict[str, object]:
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

    phase_modules = _load_phase_modules()

    if _HAS_NUMBA:
        use_numba_cfg = bool(
            pipeline_cfg.get("phase1_clean", {}).get("use_numba", True)
            or pipeline_cfg.get("phase4_score", {}).get("use_numba", True)
        )
        if use_numba_cfg:
            logging.info("[Startup] Pre-warming Numba kernels ...")
            warmup_all_kernels()

    rows = []
    for _ in tqdm(range(runs), desc="Pipeline runs", unit="run"):
        res = run_once(
            pipeline_cfg,
            approach=args.approach,
            out_root=out_root,
            dataset_name=dataset_name,
            phase_modules=phase_modules,
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
            started_utc=str(res.get("start_utc") or ""),
        )
        append_run_metrics_row(run_metrics_csv, metrics_row)

    for row in rows:
        append_timing_row(timing_csv, row)

    compute_summary(timing_csv, timing_summary)

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

    for r in rows:
        cleanup_intermediate_data(out_root / r["run_id"])


if __name__ == "__main__":
    main()
