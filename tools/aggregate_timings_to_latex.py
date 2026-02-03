#!/usr/bin/env python3
"""
Aggregate timing_summary.json files into LaTeX-friendly rows and TikZ coordinates.

- Scans output/<approach>/<dataset>/fraction_XXpct[/workers_N]/aggregate/timing_summary.json
- Uses phase1_clean rows_in from the first run's cleaning_report.json to report number of payments.
- Emits table rows (per approach) and TikZ coordinate lists (total vs payments).

Supported approach variants by default:
  * seq_cpu (Sequential CPU)
  * concurent_cpu workers 8,16,32 (Concurrent CPU threads)
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "output"
FRACTION_ORDER = [5, 10, 20, 30, 40, 60, 100]


@dataclass
class Variant:
    approach: str
    label: str
    workers: Optional[int] = None  # None for seq_cpu


DEFAULT_VARIANTS: List[Variant] = [
    Variant("seq_cpu", "Sequential CPU"),
    Variant("concurent_cpu", "Concurrent CPU (8 threads)", workers=8),
    Variant("concurent_cpu", "Concurrent CPU (16 threads)", workers=16),
    Variant("concurent_cpu", "Concurrent CPU (32 threads)", workers=32),
]


def load_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def fraction_key(name: str) -> int:
    m = re.search(r"fraction_(\d+)pct", name)
    return int(m.group(1)) if m else 0


def pct_from_fraction_label(name: str) -> Optional[int]:
    m = re.search(r"fraction_(\d+)pct", name)
    return int(m.group(1)) if m else None


def first_run_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name != "aggregate":
            return p
    return None


def rows_in_from_clean(base: Path) -> Optional[int]:
    run_dir = first_run_dir(base)
    if not run_dir:
        return None
    report = load_json(run_dir / "phase1_clean" / "cleaning_report.json")
    rows_in = report.get("rows_in")
    if isinstance(rows_in, int):
        return rows_in
    try:
        return int(rows_in)
    except Exception:
        return None


def list_fractions(dataset_dir: Path) -> List[str]:
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("fraction_")], key=fraction_key)


def summary_path(variant: Variant, dataset: str, fraction: str) -> Path:
    if variant.workers is None:
        return OUTPUT_ROOT / variant.approach / dataset / fraction / "aggregate" / "timing_summary.json"
    workers_dir = f"workers_{variant.workers}"
    return OUTPUT_ROOT / variant.approach / dataset / fraction / workers_dir / "aggregate" / "timing_summary.json"


def base_run_path(variant: Variant, dataset: str, fraction: str) -> Path:
    if variant.workers is None:
        return OUTPUT_ROOT / variant.approach / dataset / fraction
    workers_dir = f"workers_{variant.workers}"
    return OUTPUT_ROOT / variant.approach / dataset / fraction / workers_dir


def collect_variant_rows(dataset: str, variant: Variant) -> List[Dict[str, object]]:
    dataset_dir = OUTPUT_ROOT / variant.approach / dataset
    fractions = list_fractions(dataset_dir)
    rows: List[Dict[str, object]] = []
    for frac in fractions:
        summary = load_json(summary_path(variant, dataset, frac))
        total = (summary.get("total") or {}).get("mean_sec")
        if total is None:
            continue
        rows_in = rows_in_from_clean(base_run_path(variant, dataset, frac))
        rows.append(
            {
                "approach_label": variant.label,
                "dataset": dataset,
                "fraction": frac,
                "fraction_pct": pct_from_fraction_label(frac),
                "rows_in": rows_in,
                "phase1": (summary.get("phase1_clean") or {}).get("mean_sec"),
                "phase2": (summary.get("phase2_graph") or {}).get("mean_sec"),
                "phase3": (summary.get("phase3_algos") or {}).get("mean_sec"),
                "phase4": (summary.get("phase4_score") or {}).get("mean_sec"),
                "total": total,
            }
        )
    return rows


def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "--"
    return f"{x:.{decimals}f}"


def emit_table(rows: List[Dict[str, object]]) -> str:
    lines = []
    lines.append("% Approach & Data Size & Payments & Clean & Graph & Algos & Score & Total")
    for r in rows:
        payments = f"{r['rows_in']:,}" if r.get("rows_in") else "--"
        frac_pct = r.get("fraction_pct")
        frac_disp = f"{frac_pct/100:.2f}" if frac_pct is not None else "--"
        line = " ".join(
            [
                f"{r['approach_label']} & {frac_disp} & {payments} & {fmt_num(r['phase1'])} & {fmt_num(r['phase2'])} & {fmt_num(r['phase3'])} & {fmt_num(r['phase4'])} & {fmt_num(r['total'])} \\",
            ]
        )
        lines.append(line)
    return "\n".join(lines)


def emit_tikz(rows: List[Dict[str, object]], dataset: str) -> str:
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_label.setdefault(r["approach_label"], []).append(r)
    for lst in by_label.values():
        lst.sort(key=lambda x: fraction_key(x["fraction"]))

    lines = [f"% TikZ coordinates for dataset {dataset}"]
    for label, lst in by_label.items():
        lines.append(f"% {label}")
        coords = []
        for r in lst:
            if r.get("rows_in") is None or r.get("total") is None:
                continue
            coords.append(f"({r['rows_in']},{fmt_num(r['total'], 3)})")
        coord_block = "\n".join(coords)
        lines.append("coordinates {")
        lines.append(coord_block)
        lines.append("};")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate timing summaries into LaTeX-friendly snippets.")
    parser.add_argument("--dataset", default="research_2024", help="Dataset name under output/<approach>/ (e.g., research_2024)")
    parser.add_argument("--mode", choices=["table", "tikz", "both"], default="both", help="What to emit")
    parser.add_argument("--approach", action="append", help="Restrict to approaches (seq_cpu or concurent_cpu)")
    parser.add_argument("--fractions", action="append", help="Restrict to fractions (e.g., 0.05 0.1 0.2 or 5 10 20)")
    args = parser.parse_args()

    def normalize_fraction_list(raw_list: Optional[List[str]]) -> Optional[List[int]]:
        if not raw_list:
            return None
        vals: List[int] = []
        for item in raw_list:
            try:
                v = float(item)
                if v <= 1:
                    vals.append(int(round(v * 100)))
                else:
                    vals.append(int(round(v)))
            except ValueError:
                continue
        return sorted(set(vals))

    allowed_fracs = normalize_fraction_list(args.fractions)

    variants = [v for v in DEFAULT_VARIANTS if (not args.approach) or (v.approach in args.approach)]
    all_rows: List[Dict[str, object]] = []
    for v in variants:
        all_rows.extend(collect_variant_rows(args.dataset, v))

    if allowed_fracs:
        all_rows = [r for r in all_rows if (r.get("fraction_pct") in allowed_fracs)]

    # stable ordering: approach label then fraction order
    all_rows.sort(key=lambda r: (r["approach_label"], FRACTION_ORDER.index(r["fraction_pct"]) if r.get("fraction_pct") in FRACTION_ORDER else 999))

    if not all_rows:
        print("No timing summaries found.")
        return

    if args.mode in ("table", "both"):
        print("% LaTeX table rows")
        print(emit_table(all_rows))
    if args.mode in ("tikz", "both"):
        if args.mode == "both":
            print("\n% ----\n")
        print(emit_tikz(all_rows, args.dataset))


if __name__ == "__main__":
    main()

