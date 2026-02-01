#!/usr/bin/env python3
"""
Phase 5: Visualization / Results Aggregation
Reads per-phase reports for multiple approaches and emits paper-ready tables and figures.
Inputs are only the JSON reports (no heavy data files).
Outputs are written to paper_results/<dataset>_*.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yaml


@dataclass(frozen=True)
class ApproachPaths:
    label: str
    phase1_report: str
    phase2_report: str
    phase3_report: str
    phase4_report: str


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg_dir = Path(path).parent
    def resolve(p: str) -> str:
        p_obj = Path(p)
        return str((cfg_dir / p_obj).resolve()) if not p_obj.is_absolute() else str(p_obj)

    approaches: Dict[str, ApproachPaths] = {}
    for name, entry in raw["approaches"].items():
        approaches[name] = ApproachPaths(
            label=str(entry.get("label", name)),
            phase1_report=resolve(entry["phase1_report"]),
            phase2_report=resolve(entry["phase2_report"]),
            phase3_report=resolve(entry["phase3_report"]),
            phase4_report=resolve(entry["phase4_report"]),
        )

    return {
        "dataset": str(raw["dataset"]),
        "baseline": str(raw["baseline"]),
        "output_dir": str(resolve(raw.get("output_dir", "../paper_results"))),
        "approaches": approaches,
    }


def read_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_timings(a: ApproachPaths) -> Dict[str, float]:
    p1 = read_json(a.phase1_report)
    p2 = read_json(a.phase2_report)
    p3 = read_json(a.phase3_report)
    p4 = read_json(a.phase4_report)

    def get_total(report: Dict, phase_name: str) -> float:
        timings = report.get("timings_sec") or {}
        if isinstance(timings, dict):
            if "total" in timings:
                return float(timings["total"])
            numeric = [v for v in timings.values() if isinstance(v, (int, float))]
            if numeric:
                return float(sum(numeric))

        legacy = report.get("timings") or {}
        if isinstance(legacy, dict):
            for k in ("total", "total_seconds", "total_time_seconds"):
                if k in legacy:
                    return float(legacy[k])
            numeric = [v for k, v in legacy.items() if isinstance(v, (int, float)) and ("sec" in k or "second" in k)]
            if numeric:
                return float(sum(numeric))

        for k in ("total_time_seconds", "total_time", "total_seconds", "runtime_sec", "runtime_seconds"):
            if k in report:
                return float(report[k])

        logging.warning("%s missing timings; defaulting to 0", phase_name)
        return 0.0

    return {
        "t_clean": get_total(p1, "phase1"),
        "t_graph_build": get_total(p2, "phase2"),
        "t_algos": get_total(p3, "phase3"),
        "t_scoring": get_total(p4, "phase4"),
    }


def extract_sizes(a: ApproachPaths) -> Dict[str, float]:
    p1 = read_json(a.phase1_report)
    p2 = read_json(a.phase2_report)
    p4 = read_json(a.phase4_report)

    counts = p2.get("counts", {})
    edges = counts.get("edges") or counts.get("total_edges")
    nodes = counts.get("total_nodes") or counts.get("nodes")

    return {
        "rows_valid": float(p1.get("rows_valid", 0)),
        "nodes": float(nodes) if nodes is not None else 0.0,
        "edges": float(edges) if edges is not None else 0.0,
        "rows_scored": float(p4.get("stats", {}).get("rows_scored", 0)),
    }


def write_runtime_csv(dataset: str, output_dir: Path, baseline: str, rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    base_total_series = df.loc[df["approach"] == baseline, "t_total"]
    if base_total_series.empty:
        raise KeyError(f"Baseline '{baseline}' not found in approaches")
    base_total = float(base_total_series.iloc[0])
    if base_total <= 0:
        logging.warning("Baseline '%s' total time missing/zero; speedup will be NaN", baseline)
        df["speedup_vs_seq"] = float("nan")
    else:
        df["speedup_vs_seq"] = base_total / df["t_total"].replace({0: float("nan")})
    out_path = output_dir / f"{dataset}_runtime_summary.csv"
    df.to_csv(out_path, index=False)
    logging.info("Wrote %s", out_path)
    return df


def write_sizes_csv(dataset: str, output_dir: Path, rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_path = output_dir / f"{dataset}_graph_sizes.csv"
    df.to_csv(out_path, index=False)
    logging.info("Wrote %s", out_path)
    return df


def plot_runtime_breakdown(dataset: str, output_dir: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    phases = ["t_clean", "t_graph_build", "t_algos", "t_scoring"]
    bottoms = [0] * len(df)
    for phase in phases:
        ax.bar(df["approach"], df[phase], bottom=bottoms, label=phase.replace("t_", ""))
        bottoms = [b + v for b, v in zip(bottoms, df[phase])]
    ax.set_ylabel("Seconds")
    ax.set_title(f"Runtime breakdown: {dataset}")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"fig_runtime_breakdown_{dataset}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def plot_speedup(dataset: str, output_dir: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["approach"], df["speedup_vs_seq"], color="#4c72b0")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_title(f"Speedup: {dataset}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    out_path = output_dir / f"fig_speedup_{dataset}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def write_run_summary(dataset: str, output_dir: Path, rows: List[Dict]) -> None:
    summary = {r["approach"]: {k: r[k] for k in r if k != "approach"} for r in rows}
    out_path = output_dir / f"run_summary_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "approaches": summary}, f, indent=2)
    logging.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5: visualize/aggregate results")
    ap.add_argument("--config", required=True, help="YAML config with approach report paths")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    baseline = cfg["baseline"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_runtime: List[Dict] = []
    rows_sizes: List[Dict] = []

    for name, paths in cfg["approaches"].items():
        try:
            timings = extract_timings(paths)
            sizes = extract_sizes(paths)
        except FileNotFoundError as e:
            logging.warning("Skipping approach %s due to missing report: %s", name, e)
            continue

        t_total = timings["t_clean"] + timings["t_graph_build"] + timings["t_algos"] + timings["t_scoring"]
        rows_runtime.append({
            "approach": name,
            **timings,
            "t_total": t_total,
        })
        rows_sizes.append({
            "approach": name,
            **sizes,
        })

    runtime_df = write_runtime_csv(dataset, output_dir, baseline, rows_runtime)
    sizes_df = write_sizes_csv(dataset, output_dir, rows_sizes)

    plot_runtime_breakdown(dataset, output_dir, runtime_df)
    plot_speedup(dataset, output_dir, runtime_df)
    write_run_summary(dataset, output_dir, rows_runtime)

    logging.info("Rows runtime:\n%s", runtime_df)
    logging.info("Rows sizes:\n%s", sizes_df)


if __name__ == "__main__":
    main()

