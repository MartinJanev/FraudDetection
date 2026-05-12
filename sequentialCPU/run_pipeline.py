#!/usr/bin/env python3
"""
End-to-end sequential CPU pipeline runner.
Chains phases 1–4 in-process, writes per-run artifacts under
output/<approach>/<dataset>/<run_id>/, and aggregates timings.
"""

from __future__ import annotations

import argparse
import csv
import json
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

import shutil
import glob
from pathlib import Path


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


def run_once(
    pipeline_cfg: Dict,
    *,
    approach: str,
    out_root: Path,
    dataset_name: str,
    phases_to_run: List[str],
    run_dir_override: Path | None = None,
) -> Dict:
    run_dir = Path(run_dir_override) if run_dir_override else out_root / unique_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    phase_dirs = {
        "phase1_clean": run_dir / "phase1_clean",
        "phase2_graph": run_dir / "phase2_graph",
        "phase3_algos": run_dir / "phase3_algos",
        "phase4_score": run_dir / "phase4_score",
        "reports": run_dir / "reports",
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
    clean_res = None
    if "phase1_clean" in phases_to_run:
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
    else:
        # Reuse existing outputs
        clean_res = {
            "artifacts": {
                "clean_dir": str(phase_dirs["phase1_clean"] / "payments_clean"),
                "rejected_dir": str(phase_dirs["phase1_clean"] / "payments_rejected"),
                "clean_manifest": str(phase_dirs["phase1_clean"] / "payments_clean_manifest.json"),
            },
            "start_utc": None,
            "end_utc": None,
        }

    clean_manifest = Path(clean_res["artifacts"].get("clean_manifest")) if clean_res.get("artifacts") else None
    if clean_manifest and not clean_manifest.exists():
        clean_manifest = None


    # Diplay to terminal that phase 1 is complete
    print(f"Phase 1 complete. Cleaned data at: {clean_res['artifacts']['clean_dir']}")

    # Phase 2: graph build
    graph_res = None
    if "phase2_graph" in phases_to_run:
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
    else:
        graph_res = {
            "artifacts": {
                "edges": str(phase_dirs["phase2_graph"] / "edges.parquet"),
                "nodes": str(phase_dirs["phase2_graph"] / "nodes.parquet"),
            },
            "start_utc": None,
            "end_utc": None,
        }

    # Diplay to terminal that phase 2 is complete
    print(f"Phase 2 complete. Graph edges at: {graph_res['artifacts']['edges']}")

    # Phase 3: algorithms
    alg_res = None
    if "phase3_algos" in phases_to_run:
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
    else:
        alg_res = {
            "artifacts": {
                "degree_path": str(phase_dirs["phase3_algos"] / "degree.parquet"),
                "pagerank_path": str(phase_dirs["phase3_algos"] / "pagerank.parquet"),
            },
            "start_utc": None,
            "end_utc": None,
        }

    # Diplay to terminal that phase 3 is complete
    print(f"Phase 3 complete. Algorithm outputs at: {phase_dirs['phase3_algos']}")


    # Phase 4: scoring
    score_res = None
    if "phase4_score" in phases_to_run:
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
        "start_utc": phase_reports[0].get("start_utc") if phase_reports else None,
        "end_utc": phase_reports[-1].get("end_utc") if phase_reports else None,
        "phase_reports": phase_reports,
        "timings_sec": timings,
        "phases_run": phases_to_run,
    }
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)


    # Display to terminal that the entire pipeline run is complete
    print(f"Pipeline run complete. All artifacts under: {run_dir}")

    return {
        "run_id": run_id,
        "timings": timings,
        "report_path": str(timings_path),
        "start_utc": run_report["start_utc"],
    }


def fraction_label_from_fraction(fraction: float) -> str:
    pct = int(round(max(0.0, fraction) * 100))
    return f"fraction_{pct:02d}pct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential CPU pipeline end-to-end")
    parser.add_argument("--config", required=True, help="Path to unified pipeline YAML config")
    parser.add_argument("--approach", default="seq_cpu", help="Approach name for output folder")
    parser.add_argument("--runs", type=int, default=None, help="Number of runs to execute (overrides config)")
    parser.add_argument("--phases", default="phase1_clean,phase2_graph,phase3_algos,phase4_score", help="Comma-separated phases to run")
    parser.add_argument("--reuse-run-dir", default=None, help="Existing run directory to reuse (skip generating new run_id)")
    parser.add_argument("--log-level", default="INFO", help="Unused placeholder for future logging")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline_cfg = load_pipeline_config(config_path)
    dataset_cfg = pipeline_cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name") or dataset_cfg.get("dataset_name") or "dataset")

    out_root_cfg = pipeline_cfg.get("output", {})
    root_dir = out_root_cfg.get("root_dir", "output")
    sampling_fraction = float(pipeline_cfg.get("phase1_clean", {}).get("sampling", {}).get("fraction", 1.0))
    fraction_dir = fraction_label_from_fraction(sampling_fraction)

    out_root = Path(root_dir)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root = out_root / args.approach / dataset_name / fraction_dir
    out_root.mkdir(parents=True, exist_ok=True)

    config_runs = int(pipeline_cfg.get("runs", {}).get("n_runs", 1))
    runs = args.runs if args.runs is not None else config_runs

    aggregate_dir = out_root / "aggregate"
    timing_csv = aggregate_dir / "timing_runs.csv"
    timing_summary = aggregate_dir / "timing_summary.json"

    phases_to_run = [p.strip() for p in args.phases.split(",") if p.strip()]
    valid_phases = {"phase1_clean", "phase2_graph", "phase3_algos", "phase4_score"}
    for p in phases_to_run:
        if p not in valid_phases:
            raise ValueError(f"Invalid phase: {p}. Valid: {sorted(valid_phases)}")

    reuse_dir = Path(args.reuse_run_dir) if args.reuse_run_dir else None

    run_records = []
    for _ in tqdm(range(runs), desc="Pipeline runs", unit="run"):
        res = run_once(
            pipeline_cfg,
            approach=args.approach,
            out_root=out_root,
            dataset_name=dataset_name,
            phases_to_run=phases_to_run,
            run_dir_override=reuse_dir,
        )
        t = res.get("timings", {})
        row = {
            "run_id": res["run_id"],
            "dataset": dataset_name,
            "approach": args.approach,
            "phase1_clean": round(t.get("phase1_clean", 0.0), 4),
            "phase2_graph": round(t.get("phase2_graph", 0.0), 4),
            "phase3_algos": round(t.get("phase3_algos", 0.0), 4),
            "phase4_score": round(t.get("phase4_score", 0.0), 4),
            "total": round(t.get("total", 0.0), 4),
            "started_utc": res.get("start_utc"),
        }
        append_timing_row(timing_csv, row)
        run_records.append(row)

    compute_summary(timing_csv, timing_summary)

    print(f"Completed {len(run_records)} run(s). Results under {out_root}")
    print(f"Aggregate timings: {timing_csv}")
    print(f"Summary: {timing_summary}")

    for r in run_records:
            cleanup_intermediate_data(reuse_dir if reuse_dir else out_root / r["run_id"])

if __name__ == "__main__":
    main()
