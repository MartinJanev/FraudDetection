#!/usr/bin/env python3
"""sequentialCPU/01_clean_data.py

Phase 1 (Sequential CPU): Extract + Clean CMS Open Payments into canonical schema.

Design: single thread, single process, zero Numba.  Everything that the
concurrentNumba pipeline parallelises across a ThreadPoolExecutor is done here
in a plain Python ``for`` loop over chunks.  The transformation logic
(normalisation, payee-key derivation, SHA-1 record IDs) is bit-for-bit
identical so outputs can be compared directly.

Module layout
-------------
normalizers.py   Stateless scalar helpers: normalize_name, normalize_zip5,
                 safe_float, parse_date, stable_hash_hex.
                 Also owns NORMALIZATION_VERSION / HASH_FUNCTION constants.

config.py        CleanConfig dataclass + YAML loaders (pipeline-style and
                 legacy flat).  max_workers is always forced to 1.

fingerprint.py   compute_file_fingerprint + create_dataset_fingerprint for
                 reproducibility tracking.

schema.py        CANONICAL_COLS, CMS_COLUMN_MAPPINGS, make_canonical_meta,
                 get_column_mapping, sanitize_for_parquet.

transforms.py    build_payee_fields_vectorized (fully vectorised, no row
                 loops), canonicalize_chunk.

01_clean_data.py (this file) Orchestrator: sequential chunk loop with tqdm
                 progress, parquet writing, manifest, cleaning report, CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .config import CleanConfig, load_config
from .fingerprint import create_dataset_fingerprint
from .normalizers import (
    NORMALIZATION_VERSION, NORMALIZATION_DESCRIPTION, HASH_FUNCTION,
)
from .schema import CANONICAL_COLS, make_canonical_meta
from .transforms import canonicalize_chunk


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_partitioned_parquet(df: pd.DataFrame, output_dir: Path, part_num: int) -> None:
    """Write *df* as a single named parquet partition file.

    Files are named ``part-{part_num:05d}.parquet`` so they sort correctly and
    can be read as a dataset by Phase 2.
    """
    part_file = output_dir / f"part-{part_num:05d}.parquet"
    df.to_parquet(str(part_file), index=False)


# ---------------------------------------------------------------------------
# Sampling (head-slice, deterministic — no random indexes)
# ---------------------------------------------------------------------------

def _apply_sampling(canon: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return the first ``fraction * len(canon)`` rows of *canon*.

    This head-slice approach is:
    * **Deterministic** — same rows every run for the same input order.
    * **Cache-friendly** — reads contiguous memory, no index scatter.
    * **No random state** — seed is intentionally unused here.

    When ``fraction >= 0.9999`` the full DataFrame is returned unchanged.
    """
    if fraction >= 0.9999 or len(canon) == 0:
        return canon
    n_keep = max(1, int(round(len(canon) * fraction)))
    return canon.iloc[:n_keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main phase runner
# ---------------------------------------------------------------------------

def run(cfg: CleanConfig, config_path: Optional[str] = None) -> None:
    """Execute Phase 1 sequential cleaning and write all artefacts.

    Progress is reported via:
    * ``tqdm`` chunk progress bar (one bar per input file).
    * ``logging.info`` stage banners at each major milestone.

    No threads, no processes, no Numba are used.

    Parameters
    ----------
    cfg:
        Fully resolved pipeline configuration.  ``cfg.max_workers`` is ignored
        — this pipeline is always single-threaded.
    config_path:
        When provided, the YAML is copied to ``<output_dir>/config_used.yaml``
        and a dataset fingerprint is written to
        ``<output_dir>/dataset_fingerprint.json``.
    """
    t_phase_start = time.perf_counter()
    timings: Dict[str, float] = {}

    out_dir = Path(cfg.output_dir)
    ensure_dir(str(out_dir))

    # ── Stage 1: Reproducibility artefacts ───────────────────────────────────
    logging.info("[Phase 1] ── Stage 1/5: Writing reproducibility artefacts ──")
    if config_path:
        fingerprint = create_dataset_fingerprint(cfg, config_path)
        fp_path = out_dir / "dataset_fingerprint.json"
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)
        logging.info("[Phase 1] Wrote fingerprint: %s", fp_path)

        config_copy = out_dir / "config_used.yaml"
        with open(config_path, "r", encoding="utf-8") as src, \
             open(config_copy, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        logging.info("[Phase 1] Saved config snapshot: %s", config_copy)

    # ── Output paths ─────────────────────────────────────────────────────────
    clean_dir  = out_dir / "payments_clean"
    rej_dir    = out_dir / "payments_rejected"
    clean_csv  = out_dir / "payments_clean.csv"
    rej_csv    = out_dir / "payments_rejected.csv"
    report_path = out_dir / "cleaning_report.json"

    if cfg.write_format == "parquet":
        ensure_dir(str(clean_dir))
        if not cfg.skip_rejected:
            ensure_dir(str(rej_dir))
        for p in clean_dir.glob("part-*.parquet"):
            p.unlink()
        if not cfg.skip_rejected:
            for p in rej_dir.glob("part-*.parquet"):
                p.unlink()
    else:
        if clean_csv.exists():
            clean_csv.unlink()
        if not cfg.skip_rejected and rej_csv.exists():
            rej_csv.unlink()

    if report_path.exists():
        report_path.unlink()

    # ── Accumulators ─────────────────────────────────────────────────────────
    rows_in           = 0
    rows_after_sample = 0
    rows_valid        = 0
    rows_rejected     = 0
    part_num          = 0
    amounts: List[np.ndarray] = []
    physician_count   = 0
    hospital_count    = 0
    missing_npi       = 0
    missing_profile   = 0
    missing_hosp_id   = 0
    manifest_clean: List[Dict]    = []
    manifest_rejected: List[Dict] = []

    do_sampling = cfg.sampling_fraction < 0.9999

    logging.info(
        "[Phase 1] ── Stage 2/5: Reading + cleaning chunks ──  "
        "(dataset=%s year=%d sampling=%.3f)",
        cfg.dataset_type, cfg.program_year, cfg.sampling_fraction,
    )
    t_read_start = time.perf_counter()

    for fpath in cfg.input_files:
        fpath = str(fpath)
        fname = Path(fpath).name

        # Count rows upfront so tqdm shows accurate totals
        logging.info("[Phase 1] Counting rows in %s ...", fname)
        with open(fpath, "r", encoding="utf-8") as fh:
            total_rows = sum(1 for _ in fh) - 1  # subtract header
        total_chunks = max(1, (total_rows + cfg.chunk_size - 1) // cfg.chunk_size)
        logging.info(
            "[Phase 1] %s: %d rows → %d chunks of %d",
            fname, total_rows, total_chunks, cfg.chunk_size,
        )

        reader = pd.read_csv(
            fpath,
            chunksize=cfg.chunk_size,
            low_memory=False,
            dtype=str,          # ingest as string; we parse deterministically
            encoding="utf-8",
            sep=",",
            engine="c",
        )

        chunk_iter = (
            _tqdm(reader, total=total_chunks, desc=f"  Cleaning {fname}", unit="chunk")
            if _HAS_TQDM else reader
        )

        for chunk in chunk_iter:
            rows_in += len(chunk)
            canon = canonicalize_chunk(chunk, cfg, source_file=fname)

            # Deterministic head-slice sampling
            if do_sampling:
                canon = _apply_sampling(canon, cfg.sampling_fraction)

            rows_after_sample += len(canon)
            if len(canon) == 0:
                continue

            valid_df = canon[canon["is_valid"]].copy()
            rej_df   = canon[~canon["is_valid"]].copy()

            rows_valid    += len(valid_df)
            rows_rejected += len(rej_df)

            if len(valid_df) > 0:
                amounts.append(valid_df["amount_usd"].to_numpy())
                physician_count += int((valid_df["payee_type"] == "physician").sum())
                hospital_count  += int((valid_df["payee_type"] == "teaching_hospital").sum())

                phys_rows = valid_df[valid_df["payee_type"] == "physician"]
                hosp_rows = valid_df[valid_df["payee_type"] == "teaching_hospital"]
                missing_npi     += int(phys_rows["physician_npi"].isna().sum())
                missing_profile += int(phys_rows["physician_profile_id"].isna().sum())
                missing_hosp_id += int(hosp_rows["teaching_hospital_id"].isna().sum())

            # ── Write outputs ────────────────────────────────────────────────
            if cfg.write_format == "parquet":
                if len(valid_df) > 0:
                    _write_partitioned_parquet(valid_df, clean_dir, part_num)
                    manifest_clean.append({
                        "part_file":   f"part-{part_num:05d}.parquet",
                        "row_count":   len(valid_df),
                        "source_file": fname,
                    })
                if len(rej_df) > 0 and not cfg.skip_rejected:
                    _write_partitioned_parquet(rej_df, rej_dir, part_num)
                    manifest_rejected.append({
                        "part_file":   f"part-{part_num:05d}.parquet",
                        "row_count":   len(rej_df),
                        "source_file": fname,
                    })
                part_num += 1
            else:
                # CSV append mode (header written only on first chunk)
                header_clean = not clean_csv.exists()
                if len(valid_df) > 0:
                    valid_df.to_csv(str(clean_csv), mode="a", index=False, header=header_clean)
                if len(rej_df) > 0 and not cfg.skip_rejected:
                    header_rej = not rej_csv.exists()
                    rej_df.to_csv(str(rej_csv), mode="a", index=False, header=header_rej)

    timings["t_read_and_clean"] = round(time.perf_counter() - t_read_start, 4)
    logging.info(
        "[Phase 1] Stage 2 done | rows_in=%d rows_sampled=%d rows_valid=%d | elapsed=%.2fs",
        rows_in, rows_after_sample, rows_valid, timings["t_read_and_clean"],
    )

    # ── Stage 3: Manifests ────────────────────────────────────────────────────
    logging.info("[Phase 1] ── Stage 3/5: Writing partition manifests ──")
    if cfg.write_format == "parquet":
        clean_manifest_path = out_dir / "payments_clean_manifest.json"
        with open(clean_manifest_path, "w", encoding="utf-8") as f:
            json.dump({"total_parts": len(manifest_clean), "total_rows": rows_valid, "parts": manifest_clean}, f, indent=2)
        logging.info("[Phase 1] Wrote clean manifest (%d parts)", len(manifest_clean))

        if not cfg.skip_rejected:
            rej_manifest_path = out_dir / "payments_rejected_manifest.json"
            with open(rej_manifest_path, "w", encoding="utf-8") as f:
                json.dump({"total_parts": len(manifest_rejected), "total_rows": rows_rejected, "parts": manifest_rejected}, f, indent=2)
            logging.info("[Phase 1] Wrote rejected manifest (%d parts)", len(manifest_rejected))

    # ── Stage 4: Amount statistics ────────────────────────────────────────────
    logging.info("[Phase 1] ── Stage 4/5: Computing statistics ──")
    t_stats_start = time.perf_counter()
    if amounts:
        all_amt = np.concatenate(amounts)
        amt_stats = {
            "min":    float(np.nanmin(all_amt)),
            "mean":   float(np.nanmean(all_amt)),
            "median": float(np.nanmedian(all_amt)),
            "max":    float(np.nanmax(all_amt)),
        }
    else:
        amt_stats = {"min": None, "mean": None, "median": None, "max": None}

    phys_npi_pct     = (missing_npi     / physician_count * 100) if physician_count else None
    phys_prof_pct    = (missing_profile / physician_count * 100) if physician_count else None
    hosp_id_pct      = (missing_hosp_id / hospital_count  * 100) if hospital_count  else None
    timings["t_stats"] = round(time.perf_counter() - t_stats_start, 4)

    # ── Stage 5: Cleaning report ──────────────────────────────────────────────
    logging.info("[Phase 1] ── Stage 5/5: Writing cleaning report ──")
    total_time = time.perf_counter() - t_phase_start

    report = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "rows_in":            int(rows_in),
        "rows_after_sampling": int(rows_after_sample),
        "rows_valid":         int(rows_valid),
        "rows_rejected":      int(rows_rejected) if not cfg.skip_rejected else 0,
        "valid_rate": float(rows_valid / rows_after_sample) if rows_after_sample else None,
        "sampling": {
            "fraction":          float(cfg.sampling_fraction),
            "seed":              cfg.sampling_seed,
            "enabled":           do_sampling,
            "strategy":          "head_slice_first_x_percent",
            "rows_before":       int(rows_in),
            "rows_after":        int(rows_after_sample),
            "retention_rate":    float(rows_after_sample / rows_in) if rows_in else None,
        },
        "amount_usd_stats": amt_stats,
        "payee_counts": {
            "physicians":         int(physician_count),
            "teaching_hospitals": int(hospital_count),
        },
        "identifier_missingness": {
            "physician_npi_missing_count":     int(missing_npi),
            "physician_npi_missing_pct":       float(phys_npi_pct)  if phys_npi_pct  is not None else None,
            "physician_profile_missing_count": int(missing_profile),
            "physician_profile_missing_pct":   float(phys_prof_pct) if phys_prof_pct is not None else None,
            "hospital_id_missing_count":       int(missing_hosp_id),
            "hospital_id_missing_pct":         float(hosp_id_pct)   if hosp_id_pct   is not None else None,
        },
        "timings_sec": {"total": round(total_time, 4), **timings},
        "reproducibility": {
            "normalization_version":     NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function":             HASH_FUNCTION,
            "fingerprint_file":          "dataset_fingerprint.json",
            "config_snapshot_file":      "config_used.yaml",
            "max_workers":               1,
            "uses_numba":                False,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("[Phase 1] ══ Cleaning completed in %.2fs ══", total_time)
    if cfg.write_format == "parquet":
        logging.info("  Clean partitions : %s (%d parts)", clean_dir, part_num)
        if not cfg.skip_rejected:
            logging.info("  Rejected partitions: %s (%d parts)", rej_dir, part_num)
    else:
        logging.info("  Clean CSV        : %s", clean_csv)
        if not cfg.skip_rejected:
            logging.info("  Rejected CSV     : %s", rej_csv)
    logging.info("  Report           : %s", report_path)


# ---------------------------------------------------------------------------
# Pipeline entry-point (called from run_pipeline.py)
# ---------------------------------------------------------------------------

def run_from_pipeline(
    pipeline_cfg: Dict,
    *,
    out_dir: Path,
    approach: str,
    run_id: str,
    dataset_name: str,
) -> Dict:
    """Construct a ``CleanConfig`` from a pipeline dict and run Phase 1.

    Returns a timing / artefact summary dict consumed by the pipeline runner.
    ``max_workers`` from the pipeline config is silently ignored and kept at 1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg   = pipeline_cfg.get("phase1_clean", {}) or {}
    dataset_cfg = pipeline_cfg.get("dataset", {}) or {}
    inputs_cfg  = pipeline_cfg.get("inputs", {}) or {}
    config_dir  = Path(pipeline_cfg.get("__config_dir", Path.cwd()))

    payment_type = str(dataset_cfg.get("payment_type", "general_payment"))
    type_map = {
        "general":          "general_payment",
        "general_payment":  "general_payment",
        "research":         "research_payment",
        "research_payment": "research_payment",
        "ownership":        "ownership",
    }
    dataset_type = type_map.get(payment_type, payment_type)
    program_year = int(phase_cfg.get(
        "program_year", dataset_cfg.get("year", dataset_cfg.get("program_year", 2024))
    ))

    raw_inputs = inputs_cfg.get("csv_files") or inputs_cfg.get("csv_glob", [])
    if isinstance(raw_inputs, str):
        raw_inputs = [raw_inputs]
    if not raw_inputs:
        raise ValueError("pipeline_cfg.inputs.csv_files is required for phase1_clean")

    input_files: List[str] = []
    for f in list(raw_inputs):
        p = Path(f)
        if not p.is_absolute():
            p = config_dir / f
        input_files.append(str(p.resolve()))

    sampling_cfg = phase_cfg.get("sampling", {}) or {}

    cfg = CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=input_files,
        output_dir=str(out_dir),
        chunk_size=int(phase_cfg.get("chunk_size", 500_000)),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        keep_source_file=bool(phase_cfg.get("keep_source_file", True)),
        column_mapping=phase_cfg.get("column_mapping"),
        max_workers=1,   # always 1 — sequential pipeline
        skip_rejected=bool(phase_cfg.get("skip_rejected", False)),
        sampling_fraction=float(sampling_cfg.get("fraction", 1.0)),
        sampling_seed=sampling_cfg.get("seed"),
    )

    cfg_snapshot = {
        "dataset_type":      cfg.dataset_type,
        "program_year":      cfg.program_year,
        "input_files":       cfg.input_files,
        "output_dir":        cfg.output_dir,
        "chunk_size":        cfg.chunk_size,
        "write_format":      cfg.write_format,
        "keep_source_file":  cfg.keep_source_file,
        "column_mapping":    cfg.column_mapping,
        "sampling_fraction": cfg.sampling_fraction,
        "sampling_seed":     cfg.sampling_seed,
        "max_workers":       1,
    }
    cfg_path = out_dir / "config_from_pipeline.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_snapshot, f)

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg, config_path=str(cfg_path))
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    return {
        "phase":    "phase1_clean",
        "dataset":  dataset_name,
        "approach": approach,
        "run_id":   run_id,
        "start_utc": start_ts,
        "end_utc":   end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "clean_dir":          str(out_dir / "payments_clean"),
            "rejected_dir":       None if cfg.skip_rejected else str(out_dir / "payments_rejected"),
            "clean_manifest":     str(out_dir / "payments_clean_manifest.json"),
            "rejected_manifest":  None if cfg.skip_rejected else str(out_dir / "payments_rejected_manifest.json"),
            "report":             str(out_dir / "cleaning_report.json"),
            "fingerprint":        str(out_dir / "dataset_fingerprint.json"),
        },
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 – CMS Open Payments cleaning (Sequential CPU)")
    parser.add_argument("--config",    required=True, help="Path to YAML config for Phase 1 cleaning.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).parent / "configs" / args.config
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(str(config_path))
    run(cfg, config_path=str(config_path))


if __name__ == "__main__":
    main()

