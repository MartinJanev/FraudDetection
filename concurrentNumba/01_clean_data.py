#!/usr/bin/env python3
"""concurrentNumba/01_clean_data.py

Phase 1 (Concurrent CPU via Numba + ThreadPoolExecutor): Extract + Clean CMS Open Payments
into canonical schema.

Module layout
-------------
numba_kernels.py    Numba JIT kernels (_fast_hash_mask, _parse_amounts_kernel, _robust_z_kernel)
                    and their pure-NumPy fallbacks.

config.py           CleanConfig dataclass + YAML loader (pipeline-style and legacy flat).

normalizers.py      Stateless scalar helpers: normalize_name, normalize_zip5, safe_float,
                    parse_date, stable_hash_hex.  Also holds NORMALIZATION_VERSION constants.

schema.py           Canonical column list, CMS column mappings, make_canonical_meta,
                    get_column_mapping, sanitize_for_parquet.

transforms.py       build_payee_fields_vectorized, canonicalize_partition,
                    sampling_mask_from_record_ids.

01_clean_data.py    (this file) Orchestrator: ThreadPoolExecutor chunk pipeline,
                    progress reporting, parquet writing, report JSON, CLI entry-point.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import yaml

import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
)

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _tqdm = None          # type: ignore[assignment]
    _HAS_TQDM = False

# Sub-module imports
from config import CleanConfig, load_config                          # type: ignore[import]
from normalizers import NORMALIZATION_VERSION, NORMALIZATION_DESCRIPTION, HASH_FUNCTION, FINGERPRINT_MB  # type: ignore[import]
from numba_kernels import _HAS_NUMBA, warmup_kernels                 # type: ignore[import]
from schema import CANONICAL_COLS, make_canonical_meta, sanitize_for_parquet  # type: ignore[import]
from transforms import canonicalize_partition                         # type: ignore[import]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _make_pbar(desc: str, total: Optional[int] = None, unit: str = "chunk"):
    """Return a tqdm progress bar when tqdm is available, else a no-op context manager."""
    if _HAS_TQDM:
        return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)
    import contextlib
    return contextlib.nullcontext()


def _write_parquet_partitioned(df: pd.DataFrame, out_dir: Path, chunk_size: int) -> None:
    """Write *df* as multiple Parquet part files to avoid a single huge file.

    Files are named ``part-0000.parquet``, ``part-0001.parquet``, …

    Parameters
    ----------
    df:
        DataFrame to write.
    out_dir:
        Destination directory (created if absent).
    chunk_size:
        Maximum rows per part file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n == 0:
        df.to_parquet(str(out_dir / "part-0000.parquet"), index=False, engine="pyarrow")
        return
    n_parts = max(1, (n + chunk_size - 1) // chunk_size)
    for i in range(n_parts):
        part = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        part.to_parquet(str(out_dir / f"part-{i:04d}.parquet"), index=False, engine="pyarrow")


# ---------------------------------------------------------------------------
# Per-chunk processor
# ---------------------------------------------------------------------------

def _process_chunk(
    chunk_df: pd.DataFrame,
    cfg: CleanConfig,
    source_file: str,
    chunk_idx: int,
    do_sampling: bool,
    sampling_stage: str,
) -> pd.DataFrame:
    """Process a single CSV chunk.

    Pipeline:
    1. (Optional) Sample the first ``fraction * len(chunk)`` rows when
       ``sampling_stage == "raw"``.  Uses a deterministic head-slice so the
       sample is always the first X % of the chunk rather than random indexes.
    2. Canonicalise via ``canonicalize_partition``.
    3. (Optional) Sample the first ``fraction * len(canon)`` rows when
       ``sampling_stage == "canonical"``.

    Parameters
    ----------
    chunk_df:
        Raw CSV chunk (all columns as ``str``).
    cfg:
        Pipeline configuration.
    source_file:
        Basename of the originating CSV (stored in output when
        ``cfg.keep_source_file`` is ``True``).
    chunk_idx:
        Zero-based chunk counter (unused here; reserved for logging).
    do_sampling:
        ``True`` when ``cfg.sampling_fraction < 0.9999``.
    sampling_stage:
        ``"raw"`` or ``"canonical"`` – controls when the slice is applied.
    """
    if do_sampling and sampling_stage == "raw":
        frac = cfg.sampling_fraction
        if frac < 0.9999:
            # Deterministic head-slice: first X % of each chunk
            n_keep = max(1, int(round(len(chunk_df) * frac)))
            chunk_df = chunk_df.iloc[:n_keep].reset_index(drop=True)

    canon = canonicalize_partition(chunk_df, cfg, source_file)

    if do_sampling and sampling_stage == "canonical":
        # Deterministic head-slice: first X % of canonicalised rows
        n_keep = max(1, int(round(len(canon) * cfg.sampling_fraction)))
        canon = canon.iloc[:n_keep]

    return canon


# ---------------------------------------------------------------------------
# Main phase runner
# ---------------------------------------------------------------------------

def run(cfg: CleanConfig, config_path: Optional[str] = None) -> None:
    """Execute Phase 1 cleaning and write all artefacts to ``cfg.output_dir``.

    Progress is reported via:
    * ``tqdm`` progress bars (chunk submission + completion) when tqdm is
      installed.
    * ``logging.info`` milestones for every major stage regardless.

    Parameters
    ----------
    cfg:
        Fully resolved pipeline configuration.
    config_path:
        When provided, the YAML is copied verbatim to
        ``<output_dir>/config_used.yaml`` for reproducibility.
    """
    t_phase_start = time.perf_counter()
    timings: Dict[str, float] = {}

    out_dir = Path(cfg.output_dir)
    ensure_dir(str(out_dir))

    if config_path:
        with open(config_path, "r", encoding="utf-8") as src, \
             open(out_dir / "config_used.yaml", "w", encoding="utf-8") as dst:
            dst.write(src.read())

    clean_dir = out_dir / "payments_clean"
    rej_dir   = out_dir / "payments_rejected"

    if cfg.write_format != "parquet":
        raise ValueError("Numba pipeline expects parquet output (write_format=parquet).")

    if clean_dir.exists():
        shutil.rmtree(clean_dir, ignore_errors=True)
    if rej_dir.exists() and not cfg.skip_rejected:
        shutil.rmtree(rej_dir, ignore_errors=True)

    ensure_dir(str(clean_dir))
    if not cfg.skip_rejected:
        ensure_dir(str(rej_dir))

    max_workers = cfg.max_workers if cfg.max_workers and cfg.max_workers > 0 else None
    logging.info(
        "[Phase 1] Numba concurrent cleaning started | max_workers=%s | chunk_size=%s",
        max_workers, cfg.chunk_size,
    )

    # Warm up Numba JIT (first call compiles kernels)
    if cfg.use_numba and _HAS_NUMBA:
        warmup_kernels()

    do_sampling   = cfg.sampling_fraction < 0.9999
    sampling_stage = (cfg.sampling_stage or "canonical").lower().strip()

    # ------------------------------------------------------------------
    # Stage 1 – Read CSV files, submit chunks to thread pool
    # ------------------------------------------------------------------
    t_read_start = time.perf_counter()
    rows_in_val: int = 0
    all_clean_parts: List[pd.DataFrame] = []
    futures_map: Dict = {}

    logging.info("[Phase 1] ── Stage 1/5: Submitting CSV chunks to thread pool ──")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        chunk_idx = 0
        for fpath in cfg.input_files:
            source_file = Path(fpath).name if cfg.keep_source_file else ""
            reader = pd.read_csv(
                fpath,
                dtype=str,
                chunksize=cfg.chunk_size,
                encoding="utf-8",
            )
            with _make_pbar(f"  Submitting chunks from {Path(fpath).name}", unit="chunk") as submit_bar:
                for chunk_df in reader:
                    rows_in_val += len(chunk_df)
                    fut = pool.submit(
                        _process_chunk,
                        chunk_df,
                        cfg,
                        source_file,
                        chunk_idx,
                        do_sampling,
                        sampling_stage,
                    )
                    futures_map[fut] = chunk_idx
                    chunk_idx += 1
                    if _HAS_TQDM and submit_bar is not None:
                        submit_bar.update(1)

        total_chunks = chunk_idx
        logging.info(
            "[Phase 1] ── Stage 2/5: Cleaning %d chunks (%d rows total) ──",
            total_chunks, rows_in_val,
        )

        with _make_pbar("  Cleaning chunks", total=total_chunks, unit="chunk") as done_bar:
            for fut in as_completed(futures_map):
                cidx = futures_map[fut]
                try:
                    result = fut.result()
                    all_clean_parts.append(result)
                except Exception as exc:
                    logging.error("[Phase 1] Chunk %d failed: %s", cidx, exc)
                    raise
                finally:
                    if _HAS_TQDM and done_bar is not None:
                        done_bar.update(1)

    timings["t_read_and_canon"] = round(time.perf_counter() - t_read_start, 4)
    logging.info(
        "[Phase 1] Stage 2 done | parts=%d | elapsed=%.2fs",
        len(all_clean_parts), timings["t_read_and_canon"],
    )

    # ------------------------------------------------------------------
    # Stage 3 – Concatenate all cleaned parts
    # ------------------------------------------------------------------
    logging.info("[Phase 1] ── Stage 3/5: Concatenating %d parts ──", len(all_clean_parts))
    t_concat_start = time.perf_counter()
    if all_clean_parts:
        canon_full = pd.concat(all_clean_parts, ignore_index=True)
    else:
        canon_full = make_canonical_meta(cfg)
    timings["t_concat"] = round(time.perf_counter() - t_concat_start, 4)
    logging.info(
        "[Phase 1] Stage 3 done | total_rows=%d | elapsed=%.2fs",
        len(canon_full), timings["t_concat"],
    )

    rows_after_sampling_val = len(canon_full)

    valid_df    = canon_full[canon_full["is_valid"] == True].copy()
    rejected_df = canon_full[canon_full["is_valid"] == False].copy()

    # ------------------------------------------------------------------
    # Stage 4 – Sanitise for Parquet and write
    # ------------------------------------------------------------------
    logging.info(
        "[Phase 1] ── Stage 4/5: Sanitising | valid=%d rejected=%d ──",
        len(valid_df), len(rejected_df),
    )
    t_sanitize_start = time.perf_counter()
    valid_df = sanitize_for_parquet(valid_df)
    if not cfg.skip_rejected:
        rejected_df = sanitize_for_parquet(rejected_df)
    timings["t_sanitize"] = round(time.perf_counter() - t_sanitize_start, 4)
    logging.info("[Phase 1] Sanitisation done | elapsed=%.2fs", timings["t_sanitize"])

    t_write_start = time.perf_counter()
    logging.info("[Phase 1] Writing clean dataset (%d rows)...", len(valid_df))
    _write_parquet_partitioned(valid_df, clean_dir, cfg.chunk_size)
    timings["t_write_parquet_valid"] = round(time.perf_counter() - t_write_start, 4)
    logging.info(
        "[Phase 1] Clean parquet written | elapsed=%.2fs", timings["t_write_parquet_valid"]
    )

    if not cfg.skip_rejected:
        t_write_rej = time.perf_counter()
        logging.info("[Phase 1] Writing rejected dataset (%d rows)...", len(rejected_df))
        _write_parquet_partitioned(rejected_df, rej_dir, cfg.chunk_size)
        timings["t_write_parquet_rejected"] = round(time.perf_counter() - t_write_rej, 4)
        logging.info(
            "[Phase 1] Rejected parquet written | elapsed=%.2fs",
            timings["t_write_parquet_rejected"],
        )

    # ------------------------------------------------------------------
    # Stage 5 – Compute summary statistics
    # ------------------------------------------------------------------
    rows_valid_val = rows_rej_val = 0
    amt_min_val = amt_mean_val = amt_median_val = amt_max_val = None
    phys_val = hosp_val = 0
    phys_missing_npi_val = phys_missing_prof_val = hosp_missing_id_val = 0

    logging.info("[Phase 1] ── Stage 5/5: Computing statistics ──")
    if cfg.compute_stats:
        t_stats_start = time.perf_counter()
        rows_valid_val = len(valid_df)
        rows_rej_val   = len(rejected_df) if not cfg.skip_rejected else 0

        if rows_valid_val:
            amt = pd.to_numeric(valid_df["amount_usd"], errors="coerce")
            amt_min_val  = float(amt.min())
            amt_mean_val = float(amt.mean())
            amt_max_val  = float(amt.max())
            if cfg.stats_compute_median:
                amt_median_val = float(amt.median())

            phys_val = int((valid_df["payee_type"] == "physician").sum())
            hosp_val = int((valid_df["payee_type"] == "teaching_hospital").sum())

            if phys_val:
                phys_rows = valid_df[valid_df["payee_type"] == "physician"]
                phys_missing_npi_val  = int(phys_rows["physician_npi"].isna().sum())
                phys_missing_prof_val = int(phys_rows["physician_profile_id"].isna().sum())
            if hosp_val:
                hosp_rows = valid_df[valid_df["payee_type"] == "teaching_hospital"]
                hosp_missing_id_val = int(hosp_rows["teaching_hospital_id"].isna().sum())

        timings["t_stats"] = round(time.perf_counter() - t_stats_start, 4)

    total_time = time.perf_counter() - t_phase_start

    # ------------------------------------------------------------------
    # Write cleaning report
    # ------------------------------------------------------------------
    report = {
        "dataset_type":  cfg.dataset_type,
        "program_year":  cfg.program_year,
        "rows_in":       rows_in_val,
        "rows_after_sampling": rows_after_sampling_val,
        "rows_valid":    rows_valid_val,
        "rows_rejected": rows_rej_val if not cfg.skip_rejected else 0,
        "valid_rate": (
            float(rows_valid_val / rows_in_val)
            if rows_in_val and rows_valid_val is not None else None
        ),
        "amount_usd_stats": {
            "min":    amt_min_val,
            "mean":   amt_mean_val,
            "median": amt_median_val,
            "max":    amt_max_val,
        },
        "payee_counts": {
            "physicians":         int(phys_val),
            "teaching_hospitals": int(hosp_val),
        },
        "identifier_missingness": {
            "physician_npi_missing_count":    int(phys_missing_npi_val),
            "physician_npi_missing_pct":      (phys_missing_npi_val / phys_val * 100.0) if phys_val else None,
            "physician_profile_missing_count": int(phys_missing_prof_val),
            "physician_profile_missing_pct":  (phys_missing_prof_val / phys_val * 100.0) if phys_val else None,
            "hospital_id_missing_count":      int(hosp_missing_id_val),
            "hospital_id_missing_pct":        (hosp_missing_id_val / hosp_val * 100.0) if hosp_val else None,
        },
        "timings_sec": {"total": round(total_time, 4), **timings},
        "sampling": {
            "fraction": float(cfg.sampling_fraction),
            "seed":     cfg.sampling_seed,
            "method":   cfg.sampling_method,
            "enabled":  cfg.sampling_fraction < 0.9999,
            "strategy": "head_slice_first_x_percent",
        },
        "stats_config": {
            "compute_stats":    bool(cfg.compute_stats),
            "compute_median":   bool(cfg.stats_compute_median),
            "median_method":    cfg.stats_median_method,
            "pandas_threshold": int(cfg.stats_pandas_threshold),
        },
        "reproducibility": {
            "normalization_version":     NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function": HASH_FUNCTION,
            "use_numba":    cfg.use_numba,
            "has_numba":    bool(_HAS_NUMBA),
            "chunk_size":   cfg.chunk_size,
            "max_workers":  cfg.max_workers,
        },
        "outputs": {
            "payments_clean_dir":    str(clean_dir),
            "payments_rejected_dir": str(rej_dir) if not cfg.skip_rejected else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("[Phase 1] ══ Cleaning completed in %.2fs ══", total_time)
    logging.info("  Clean dataset  : %s", clean_dir)
    if not cfg.skip_rejected:
        logging.info("  Rejected dataset: %s", rej_dir)
    logging.info("  Report         : %s", out_dir / "cleaning_report.json")


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
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg   = pipeline_cfg.get("phase1_clean", {}) or {}
    dataset_cfg = pipeline_cfg.get("dataset", {}) or {}
    inputs_cfg  = pipeline_cfg.get("inputs", {}) or {}
    config_dir  = Path(pipeline_cfg.get("__config_dir", Path.cwd()))
    scale_cfg   = dataset_cfg.get("scale", {}) or {}

    global_workers = int(pipeline_cfg.get("execution", {}).get("max_workers", 1) or 1)

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

    sampling_cfg = (phase_cfg.get("sampling", {}) or {})
    stats_cfg    = (phase_cfg.get("stats", {}) or {})

    cfg = CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=input_files,
        output_dir=str(out_dir),
        use_numba=bool(phase_cfg.get("use_numba", True)),
        chunk_size=int(phase_cfg.get("chunk_size", 250_000)),
        max_workers=int(global_workers),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        keep_source_file=bool(phase_cfg.get("keep_source_file", True)),
        skip_rejected=bool(phase_cfg.get("skip_rejected", True)),
        sampling_fraction=float(sampling_cfg.get("fraction", 1.0)),
        sampling_seed=sampling_cfg.get("seed"),
        sampling_method=str(sampling_cfg.get("method", "sha1")),
        sampling_stage=str(sampling_cfg.get("stage", "canonical")),
        scale_mode=str(scale_cfg.get("mode", scale_cfg.get("scale_mode", "none"))),
        scale_value=scale_cfg.get("value", scale_cfg.get("scale_value", None)),
        scale_key_cols=scale_cfg.get("key_cols", None),
        scale_enabled=bool(scale_cfg.get("enabled", False)),
        scale_fraction=float(scale_cfg.get("fraction", 1.0)),
        scale_seed=int(scale_cfg.get("seed", 123)),
        scale_method=str(scale_cfg.get("method", "hash_record_id")),
        column_mapping=phase_cfg.get("column_mapping"),
        stats_compute_median=bool(stats_cfg.get("compute_median", False)),
        stats_median_method=str(stats_cfg.get("median_method", "exact")),
        stats_pandas_threshold=int(stats_cfg.get("use_pandas_if_rows_lt", 0)),
        compute_stats=bool(phase_cfg.get("compute_stats", True)),
    )

    cfg_snapshot = {
        "dataset_type":     cfg.dataset_type,
        "program_year":     cfg.program_year,
        "input_files":      cfg.input_files,
        "output_dir":       cfg.output_dir,
        "write_format":     cfg.write_format,
        "keep_source_file": cfg.keep_source_file,
        "skip_rejected":    cfg.skip_rejected,
        "sampling_fraction": cfg.sampling_fraction,
        "sampling_seed":    cfg.sampling_seed,
        "sampling_method":  cfg.sampling_method,
        "sampling_stage":   cfg.sampling_stage,
        "use_numba":  cfg.use_numba,
        "chunk_size": cfg.chunk_size,
        "max_workers": cfg.max_workers,
        "stats": {
            "compute_stats":   cfg.compute_stats,
            "compute_median":  cfg.stats_compute_median,
            "median_method":   cfg.stats_median_method,
            "use_pandas_if_rows_lt": cfg.stats_pandas_threshold,
        },
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
            "clean_dir":    str(out_dir / "payments_clean"),
            "rejected_dir": str(out_dir / "payments_rejected"),
            "report":       str(out_dir / "cleaning_report.json"),
        },
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 – CMS Open Payments cleaning (Numba)")
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

