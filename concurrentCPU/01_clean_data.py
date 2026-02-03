#!/usr/bin/env python3
"""concurrentCPU/01_clean_data.py

Phase 1 (Concurrent CPU via Dask): Extract + Clean CMS Open Payments into canonical schema.

Fixes included (critical for responsiveness and speed):
- Single-pass stats: collapse multiple .compute() calls into one dask.compute() batch.
- Optional median: quantile(0.5) is expensive; default remains configurable via YAML.
- Persist once for reuse: materialize (persist) the sanitized valid/rejected frames once and reuse for both writes and stats.
  This prevents recomputation of the entire DAG multiple times.
- Optional pandas stats fast-path for small datasets (stats-only; does NOT affect parquet outputs).
- Optional deterministic sampling method:
    * sha1 (default): matches sequential semantics (slow but equivalent)
    * fast_hash: uses pandas hash_pandas_object (much faster, deterministic, but NOT sha1-identical)
- Dask worker control: if scheduler=threads and max_workers>0, uses a ThreadPoolExecutor sized to max_workers.

Important: Concurrency is inside Dask partitions; do NOT run multiple full pipelines in parallel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

# dask imports happen inside run() so the module can import even if dask is missing

import warnings

warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually")

# ---------------------------
# Reproducibility Constants
# ---------------------------

NORMALIZATION_VERSION = "v1"
NORMALIZATION_DESCRIPTION = "punctuation-strip + uppercase + whitespace collapse"
HASH_FUNCTION = "sha1"
FINGERPRINT_MB = 10


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class CleanConfig:
    dataset_type: str  # "general_payment" | "research_payment" | "ownership"
    program_year: int
    input_files: List[str]
    output_dir: str

    # Dask tuning
    use_dask: bool = True
    blocksize: str = "256MB"  # read_csv blocksize
    dask_npartitions: int = 0  # 0 = let Dask decide
    scheduler: str = "threads"  # threads/processes/synchronous
    persist: bool = True  # cache sanitized frames in memory before write/stats

    # Output
    write_format: str = "parquet"  # parquet recommended for concurrent pipeline
    keep_source_file: bool = True
    skip_rejected: bool = True

    # Sequential-equivalent sampling (applied after canonicalization, per record_id)
    sampling_fraction: float = 1.0
    sampling_seed: Optional[int] = None
    sampling_method: str = "sha1"  # "sha1" (sequential-equivalent) or "fast_hash" (faster, not sha1-identical)
    sampling_stage: str = "raw"  # "canonical" (existing behavior) or "raw" (sample before canonicalization)

    # Legacy scale knobs (kept for compatibility)
    scale_mode: str = "none"
    scale_value: Optional[float] = None
    scale_key_cols: Optional[List[str]] = None
    scale_enabled: bool = False
    scale_fraction: float = 1.0
    scale_seed: int = 123
    scale_method: str = "hash_record_id"

    # Optional overrides
    column_mapping: Optional[Dict[str, str]] = None
    max_workers: Optional[int] = None

    # Stats tuning
    stats_compute_median: bool = False
    stats_median_method: str = "exact"  # "exact" or "approx"
    stats_pandas_threshold: int = 0  # 0 disables pandas fast-path
    compute_stats: bool = True  # allow turning stats off for bulk experiments


# ---------------------------
# Config parsing
# ---------------------------


def _parse_pipeline_style_config(raw: Dict, config_dir: Path) -> CleanConfig:
    dataset = raw.get("dataset", {})
    payment_type = str(dataset.get("payment_type", "general")).lower()
    type_map = {
        "general": "general_payment",
        "general_payment": "general_payment",
        "research": "research_payment",
        "research_payment": "research_payment",
        "ownership": "ownership",
    }
    dataset_type = type_map.get(payment_type, payment_type)
    program_year = int(dataset.get("program_year", dataset.get("year", 2024)))

    inputs = raw.get("inputs", {})
    input_files = inputs.get("csv_files", []) or inputs.get("csv_glob", [])
    if isinstance(input_files, str):
        input_files = [input_files]

    resolved_inputs: List[str] = []
    for f in input_files:
        p = Path(f)
        if not p.is_absolute():
            p = config_dir / p
        resolved_inputs.append(str(p.resolve()))
    if not resolved_inputs:
        raise KeyError("inputs.csv_files is required in pipeline config")

    out_cfg = raw.get("output", {})
    root_dir = out_cfg.get("root_dir", "output")
    out_dir = Path(root_dir)
    if not out_dir.is_absolute():
        out_dir = config_dir.parent.parent / out_dir
    dataset_name = dataset.get("name", "dataset")

    # keep your folder name convention
    output_dir = out_dir / "concurent_cpu" / dataset_name / "phase1_clean"

    phase_cfg = raw.get("phase1_clean", {})
    scale_cfg = dataset.get("scale", {}) or {}
    sampling_cfg = (phase_cfg.get("sampling", {}) or {})
    stats_cfg = (phase_cfg.get("stats", {}) or {})
    exec_cfg = raw.get("execution", {}) or {}

    max_workers = exec_cfg.get("max_workers")
    if max_workers is not None:
        try:
            max_workers = int(max_workers)
        except Exception:
            max_workers = None

    return CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=resolved_inputs,
        output_dir=str(output_dir),
        use_dask=bool(phase_cfg.get("use_dask", True)),
        blocksize=str(phase_cfg.get("blocksize", "256MB")),
        dask_npartitions=int(phase_cfg.get("dask_npartitions", 0)),
        scheduler=str(phase_cfg.get("scheduler", "threads")),
        persist=bool(phase_cfg.get("persist", True)),
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
        max_workers=max_workers,
        stats_compute_median=bool(stats_cfg.get("compute_median", False)),
        stats_median_method=str(stats_cfg.get("median_method", "exact")),
        stats_pandas_threshold=int(stats_cfg.get("use_pandas_if_rows_lt", 0)),
        compute_stats=bool(phase_cfg.get("compute_stats", True)),
    )


def load_config(path: str) -> CleanConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(path).parent
    if "dataset_type" not in raw:
        return _parse_pipeline_style_config(raw, config_dir)

    def resolve_list(files: List[str]) -> List[str]:
        out: List[str] = []
        for fp in files:
            p = Path(fp)
            if not p.is_absolute():
                p = config_dir / p
            out.append(str(p.resolve()))
        return out

    sampling_cfg = (raw.get("sampling", {}) or {})
    stats_cfg = (raw.get("stats", {}) or {})

    return CleanConfig(
        dataset_type=str(raw["dataset_type"]),
        program_year=int(raw["program_year"]),
        input_files=resolve_list(list(raw["input_files"])),
        output_dir=str((config_dir / raw["output_dir"]).resolve())
        if not Path(raw["output_dir"]).is_absolute()
        else str(Path(raw["output_dir"]).resolve()),
        use_dask=bool(raw.get("use_dask", True)),
        blocksize=str(raw.get("blocksize", "256MB")),
        dask_npartitions=int(raw.get("dask_npartitions", 0)),
        scheduler=str(raw.get("scheduler", "threads")),
        persist=bool(raw.get("persist", True)),
        write_format=str(raw.get("write_format", "parquet")),
        keep_source_file=bool(raw.get("keep_source_file", True)),
        skip_rejected=bool(raw.get("skip_rejected", True)),
        sampling_fraction=float(sampling_cfg.get("fraction", 1.0)),
        sampling_seed=sampling_cfg.get("seed"),
        sampling_method=str(sampling_cfg.get("method", "sha1")),
        sampling_stage=str(sampling_cfg.get("stage", "canonical")),
        column_mapping=raw.get("column_mapping"),
        stats_compute_median=bool(stats_cfg.get("compute_median", False)),
        stats_median_method=str(stats_cfg.get("median_method", "exact")),
        stats_pandas_threshold=int(stats_cfg.get("use_pandas_if_rows_lt", 0)),
        compute_stats=bool(raw.get("compute_stats", True)),
    )


# ---------------------------
# Normalization helpers
# ---------------------------

_PUNCT_RE = re.compile(r"[.,;:()\[\]{}'\"`]+")


def normalize_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper()


def normalize_zip5(z: Optional[str]) -> Optional[str]:
    if z is None:
        return None
    z = re.sub(r"\D+", "", str(z))
    if len(z) < 5:
        return None
    return z[:5]


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        s = s.replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        if s in {"", "-", "."}:
            return None
        return float(s)
    except Exception:
        return None


def parse_date(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt).date()
            return dt.isoformat()
        except ValueError:
            continue
    return None


def stable_hash_hex(parts: Iterable[Optional[str]]) -> str:
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


# ---------------------------
# Canonical columns + CMS mappings
# ---------------------------

CANONICAL_COLS = [
    "record_id",
    "program_year",
    "dataset_type",
    "source_file",
    "payer_name_raw",
    "payer_name_norm",
    "payer_id",
    "payer_state",
    "payer_country",
    "payee_type",
    "payee_key",
    "physician_npi",
    "physician_profile_id",
    "physician_name_raw",
    "physician_name_norm",
    "physician_specialty",
    "physician_primary_type",
    "physician_state",
    "physician_zip5",
    "teaching_hospital_id",
    "teaching_hospital_name_raw",
    "teaching_hospital_name_norm",
    "teaching_hospital_state",
    "teaching_hospital_zip5",
    "amount_usd",
    "payment_date",
    "payment_quarter",
    "nature_of_payment",
    "form_of_payment",
    "payment_context",
    "product_name",
    "associated_covered_drug_or_device_flag",
    "is_product_related",
    "is_valid",
    "drop_reason",
]

CMS_COLUMN_MAPPINGS = {
    "general_payment": {
        "amount": "Total_Amount_of_Payment_USDollars",
        "date": "Date_of_Payment",
        "payer_name": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "phys_npi": "Covered_Recipient_NPI",
        "phys_profile_id": "Covered_Recipient_Profile_ID",
        "phys_last": "Covered_Recipient_Last_Name",
        "phys_first": "Covered_Recipient_First_Name",
        "phys_middle": "Covered_Recipient_Middle_Name",
        "phys_specialty": "Covered_Recipient_Specialty_1",
        "phys_primary_type": "Covered_Recipient_Primary_Type_1",
        "phys_state": "Recipient_State",
        "phys_zip": "Recipient_Zip_Code",
        "hospital_id": "Teaching_Hospital_ID",
        "hospital_name": "Teaching_Hospital_Name",
        "hospital_ccn": "Teaching_Hospital_CCN",
        "nature": "Nature_of_Payment_or_Transfer_of_Value",
        "form": "Form_of_Payment_or_Transfer_of_Value",
        "context": "Contextual_Information",
        "product": "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "drug_device_flag": "Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "recipient_type": "Covered_Recipient_Type",
    },
    "research_payment": {
        "amount": "Total_Amount_of_Payment_USDollars",
        "date": "Date_of_Payment",
        "payer_name": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "phys_npi": "Covered_Recipient_NPI",
        "phys_profile_id": "Covered_Recipient_Profile_ID",
        "phys_last": "Covered_Recipient_Last_Name",
        "phys_first": "Covered_Recipient_First_Name",
        "phys_middle": "Covered_Recipient_Middle_Name",
        "phys_specialty": "Covered_Recipient_Specialty_1",
        "phys_primary_type": "Covered_Recipient_Primary_Type_1",
        "phys_state": "Recipient_State",
        "phys_zip": "Recipient_Zip_Code",
        "hospital_id": "Teaching_Hospital_ID",
        "hospital_name": "Teaching_Hospital_Name",
        "hospital_ccn": "Teaching_Hospital_CCN",
        "nature": "Nature_of_Payment_or_Transfer_of_Value",
        "form": "Form_of_Payment_or_Transfer_of_Value",
        "context": "Contextual_Information",
        "product": "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "drug_device_flag": "Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "recipient_type": "Covered_Recipient_Type",
    },
    "ownership": {
        "amount": "Total_Amount_Invested_USDollars",
        "payer_name": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "phys_npi": "Physician_NPI",
        "phys_profile_id": "Physician_Profile_ID",
        "phys_last": "Physician_Last_Name",
        "phys_first": "Physician_First_Name",
        "phys_middle": "Physician_Middle_Name",
        "phys_specialty": "Physician_Specialty",
        "phys_primary_type": "Physician_Primary_Type",
        "phys_state": "Recipient_State",
        "phys_zip": "Recipient_Zip_Code",
    },
}


def make_canonical_meta(_: CleanConfig) -> pd.DataFrame:
    S = "string[python]"
    dtypes = {
        "record_id": S,
        "program_year": "Int16",
        "dataset_type": S,
        "source_file": S,
        "payer_name_raw": S,
        "payer_name_norm": S,
        "payer_id": S,
        "payer_state": S,
        "payer_country": S,
        "payee_type": S,
        "payee_key": S,
        "physician_npi": S,
        "physician_profile_id": S,
        "physician_name_raw": S,
        "physician_name_norm": S,
        "physician_specialty": S,
        "physician_primary_type": S,
        "physician_state": S,
        "physician_zip5": S,
        "teaching_hospital_id": S,
        "teaching_hospital_name_raw": S,
        "teaching_hospital_name_norm": S,
        "teaching_hospital_state": S,
        "teaching_hospital_zip5": S,
        "amount_usd": "float64",
        "payment_date": S,
        "payment_quarter": "Int8",
        "nature_of_payment": S,
        "form_of_payment": S,
        "payment_context": S,
        "product_name": S,
        "associated_covered_drug_or_device_flag": S,
        "is_product_related": "bool",
        "is_valid": "bool",
        "drop_reason": S,
    }
    data = {c: pd.Series([], dtype=dtypes.get(c, S)) for c in CANONICAL_COLS}
    return pd.DataFrame(data)


def get_column_mapping(df: pd.DataFrame, dataset_type: str, custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    base_mapping = custom_mapping if custom_mapping else CMS_COLUMN_MAPPINGS.get(dataset_type)
    if base_mapping is None:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    actual_cols = set(df.columns)
    result: Dict[str, Optional[str]] = {}
    missing_cols: List[str] = []
    for key, col_name in base_mapping.items():
        if col_name in actual_cols:
            result[key] = col_name
        else:
            result[key] = None
            if key in ["amount", "payer_name"]:
                missing_cols.append(col_name)
    if missing_cols:
        raise ValueError(f"Required columns missing from CSV: {missing_cols}")
    return result


def sanitize_for_parquet(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf is None or len(pdf) == 0:
        return pdf

    float_cols = {"amount_usd"}
    int8_cols = {"payment_quarter"}
    bool_cols = {"is_product_related", "is_valid"}
    intlike_cols = {"program_year"}

    for c in pdf.columns:
        s = pdf[c]
        if c in float_cols:
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("float64")
            continue
        if c in bool_cols:
            if s.dtype.name == "boolean":
                pdf[c] = s.fillna(False).astype(bool)
            else:
                try:
                    pdf[c] = s.fillna(False).map(bool).astype(bool)
                except Exception:
                    pdf[c] = False
            continue
        if c in int8_cols:
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("Int8")
            continue
        if c in intlike_cols:
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("Int16")
            continue

        def _clean_cell(x):
            if x is None:
                return None
            if isinstance(x, float) and np.isnan(x):
                return None
            if isinstance(x, str):
                return x
            return str(x)

        pdf[c] = s.map(_clean_cell).astype("string[python]")

    return pdf


def build_payee_fields_vectorized(df: pd.DataFrame, colmap: Dict[str, Optional[str]]) -> Dict[str, pd.Series]:
    n = len(df)

    hosp_id_col = colmap.get("hospital_id")
    hosp_name_col = colmap.get("hospital_name")
    recipient_type_col = colmap.get("recipient_type")

    is_hospital = pd.Series([False] * n, index=df.index)

    if hosp_id_col and hosp_id_col in df.columns:
        hosp_id_orig = df[hosp_id_col]
        hosp_id_mask = (
            hosp_id_orig.notna()
            & hosp_id_orig.astype(str).str.strip().ne("")
            & hosp_id_orig.astype(str).str.lower().ne("nan")
        )
        is_hospital |= hosp_id_mask

    if hosp_name_col and hosp_name_col in df.columns:
        hosp_name_orig = df[hosp_name_col]
        hosp_name_mask = (
            hosp_name_orig.notna()
            & hosp_name_orig.astype(str).str.strip().ne("")
            & hosp_name_orig.astype(str).str.lower().ne("nan")
        )
        is_hospital |= hosp_name_mask

    if recipient_type_col and recipient_type_col in df.columns:
        recip_type = df[recipient_type_col].astype(str).str.upper()
        is_hospital |= recip_type.str.contains("HOSPITAL", na=False)

    hosp_id_series = (
        df[hosp_id_col].astype(str).str.strip() if hosp_id_col and hosp_id_col in df.columns else pd.Series([None] * n, index=df.index)
    )
    hosp_id_series = hosp_id_series.replace("", None).replace("nan", None)

    hosp_name_raw = (
        df[hosp_name_col].astype(str).str.strip() if hosp_name_col and hosp_name_col in df.columns else pd.Series([None] * n, index=df.index)
    )
    hosp_name_raw = hosp_name_raw.replace("", None).replace("nan", None)
    hosp_name_norm = hosp_name_raw.map(lambda x: normalize_name(x) if x else None)

    state_col = colmap.get("phys_state")
    zip_col = colmap.get("phys_zip")
    hosp_state = df[state_col] if state_col and state_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_zip5 = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n, index=df.index)

    npi_col = colmap.get("phys_npi")
    profile_col = colmap.get("phys_profile_id")

    npi_series = df[npi_col] if npi_col and npi_col in df.columns else pd.Series([None] * n, index=df.index)
    npi_clean = npi_series.astype(str).str.replace(r"\D+", "", regex=True)
    npi_clean = npi_clean.replace("", None).replace("nan", None)

    profile_series = df[profile_col] if profile_col and profile_col in df.columns else pd.Series([None] * n, index=df.index)
    profile_clean = profile_series.astype(str).str.strip()
    profile_clean = profile_clean.replace("", None).replace("nan", None)

    first_col = colmap.get("phys_first")
    middle_col = colmap.get("phys_middle")
    last_col = colmap.get("phys_last")

    first = df[first_col].astype(str).str.strip() if first_col and first_col in df.columns else pd.Series([""] * n, index=df.index)
    middle = df[middle_col].astype(str).str.strip() if middle_col and middle_col in df.columns else pd.Series([""] * n, index=df.index)
    last = df[last_col].astype(str).str.strip() if last_col and last_col in df.columns else pd.Series([""] * n, index=df.index)

    first = first.replace("nan", "").replace("NaN", "")
    middle = middle.replace("nan", "").replace("NaN", "")
    last = last.replace("nan", "").replace("NaN", "")

    phys_name_raw = (first + " " + middle + " " + last).str.replace(r"\s+", " ", regex=True).str.strip()
    phys_name_raw = phys_name_raw.replace("", None).replace("nan", None).replace("NaN", None)
    phys_name_norm = phys_name_raw.map(lambda x: normalize_name(x) if x else None)

    specialty_col = colmap.get("phys_specialty")
    primary_type_col = colmap.get("phys_primary_type")
    phys_specialty = df[specialty_col] if specialty_col and specialty_col in df.columns else pd.Series([None] * n, index=df.index)
    phys_primary_type = df[primary_type_col] if primary_type_col and primary_type_col in df.columns else pd.Series([None] * n, index=df.index)
    phys_state = df[state_col] if state_col and state_col in df.columns else pd.Series([None] * n, index=df.index)
    phys_zip5 = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n, index=df.index)

    # Build payee keys with minimal per-row Python work to cut memory overhead on large partitions
    payee_key = pd.Series([None] * n, index=df.index, dtype=object)
    payee_type = pd.Series(["physician"] * n, index=df.index)
    payee_type = payee_type.where(~is_hospital, "teaching_hospital")

    hosp_idx = df.index[is_hospital]
    if len(hosp_idx):
        hosp_has_id = (
            hosp_id_series.loc[hosp_idx].notna()
            & hosp_id_series.loc[hosp_idx].astype(str).str.strip().ne("")
            & hosp_id_series.loc[hosp_idx].astype(str).str.lower().ne("nan")
        )
        if hosp_has_id.any():
            payee_key.loc[hosp_idx[hosp_has_id]] = (
                "HOSP_ID:" + hosp_id_series.loc[hosp_idx[hosp_has_id]].astype(str)
            )
        hosp_hash_idx = hosp_idx[~hosp_has_id]
        if len(hosp_hash_idx):
            hname = hosp_name_norm.loc[hosp_hash_idx]
            hzip = hosp_zip5.loc[hosp_hash_idx]
            payee_key.loc[hosp_hash_idx] = [
                f"HOSP_NAMEZIP:{stable_hash_hex([hn if pd.notna(hn) else None, hz if pd.notna(hz) else None])}"
                for hn, hz in zip(hname, hzip)
            ]

    phys_idx = df.index[~is_hospital]
    if len(phys_idx):
        npi_clean_phys = npi_clean.loc[phys_idx]
        prof_clean_phys = profile_clean.loc[phys_idx]
        npi_mask = (
            npi_clean_phys.notna()
            & npi_clean_phys.astype(str).str.strip().ne("")
            & npi_clean_phys.astype(str).str.lower().ne("nan")
        )
        prof_mask = (
            prof_clean_phys.notna()
            & prof_clean_phys.astype(str).str.strip().ne("")
            & prof_clean_phys.astype(str).str.lower().ne("nan")
        )

        if npi_mask.any():
            payee_key.loc[phys_idx[npi_mask]] = "PHYS_NPI:" + npi_clean_phys.loc[phys_idx[npi_mask]].astype(str)
        prof_only_idx = phys_idx[~npi_mask & prof_mask]
        if len(prof_only_idx):
            payee_key.loc[prof_only_idx] = "PHYS_PROF:" + prof_clean_phys.loc[prof_only_idx].astype(str)

        hash_idx = phys_idx[~npi_mask & ~prof_mask]
        if len(hash_idx):
            pname = phys_name_norm.loc[hash_idx]
            pzip = phys_zip5.loc[hash_idx]
            payee_key.loc[hash_idx] = [
                f"PHYS_NAMEZIP:{stable_hash_hex([pn if pd.notna(pn) else None, pz if pd.notna(pz) else None])}"
                for pn, pz in zip(pname, pzip)
            ]

    return {
        "payee_type": payee_type,
        "payee_key": payee_key,
        "physician_npi": npi_clean.where(~is_hospital, None),
        "physician_profile_id": profile_clean.where(~is_hospital, None),
        "physician_name_raw": phys_name_raw.where(~is_hospital, None),
        "physician_name_norm": phys_name_norm.where(~is_hospital, None),
        "physician_specialty": phys_specialty.where(~is_hospital, None),
        "physician_primary_type": phys_primary_type.where(~is_hospital, None),
        "physician_state": phys_state.where(~is_hospital, None),
        "physician_zip5": phys_zip5.where(~is_hospital, None),
        "teaching_hospital_id": hosp_id_series.where(is_hospital, None),
        "teaching_hospital_name_raw": hosp_name_raw.where(is_hospital, None),
        "teaching_hospital_name_norm": hosp_name_norm.where(is_hospital, None),
        "teaching_hospital_state": hosp_state.where(is_hospital, None),
        "teaching_hospital_zip5": hosp_zip5.where(is_hospital, None),
    }


def canonicalize_partition(df: pd.DataFrame, cfg: CleanConfig, source_file: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    colmap = get_column_mapping(df, cfg.dataset_type, cfg.column_mapping)

    payer_name_col = colmap.get("payer_name")
    amount_col = colmap.get("amount")
    if not payer_name_col or not amount_col:
        raise ValueError("Required payer_name/amount column mapping missing")

    payer_raw = df[payer_name_col].astype("object")
    payer_norm = payer_raw.map(lambda x: normalize_name(x))

    amount_raw = df[amount_col]
    amount = amount_raw.map(safe_float).astype("float64")

    date_col = colmap.get("date")
    date_raw = df[date_col] if date_col else pd.Series([None] * len(df), index=df.index)
    payment_date = date_raw.map(parse_date)

    nature_col = colmap.get("nature")
    form_col = colmap.get("form")
    context_col = colmap.get("context")
    product_col = colmap.get("product")
    flag_col = colmap.get("drug_device_flag")

    nature = df[nature_col] if nature_col else pd.Series([None] * len(df), index=df.index)
    form = df[form_col] if form_col else pd.Series([None] * len(df), index=df.index)
    context = df[context_col] if context_col else pd.Series([None] * len(df), index=df.index)
    product = df[product_col] if product_col else pd.Series([None] * len(df), index=df.index)
    flag = df[flag_col] if flag_col else pd.Series([None] * len(df), index=df.index)

    payer_id_col = colmap.get("payer_id")
    payer_state_col = colmap.get("payer_state")
    payer_country_col = colmap.get("payer_country")

    payer_id = df[payer_id_col] if payer_id_col else pd.Series([None] * len(df), index=df.index)
    payer_state = df[payer_state_col] if payer_state_col else pd.Series([None] * len(df), index=df.index)
    payer_country = df[payer_country_col] if payer_country_col else pd.Series([None] * len(df), index=df.index)

    payee_fields = build_payee_fields_vectorized(df, colmap)

    flag_clean = flag.astype(str).str.strip().str.upper()
    is_product_related = (
        flag_clean.notna() & (flag_clean != "") & (flag_clean != "NAN") & (flag_clean != "NONE")
    )

    def quarter_from_iso(d: Optional[str]) -> Optional[int]:
        if d is None:
            return None
        try:
            m = int(d.split("-")[1])
            return (m - 1) // 3 + 1
        except Exception:
            return None

    payment_quarter = payment_date.map(quarter_from_iso)

    is_valid = (
        amount.notna()
        & (amount >= 0)
        & payer_norm.notna()
        & (payer_norm.astype(str).str.len() > 0)
        & payee_fields["payee_key"].notna()
        & (payee_fields["payee_key"].astype(str).str.len() > 0)
    )

    drop_reason = pd.Series([None] * len(df), dtype="string", index=df.index)
    drop_reason = drop_reason.mask(amount.isna(), "missing_amount")
    drop_reason = drop_reason.mask(amount.notna() & (amount < 0), "negative_amount")
    drop_reason = drop_reason.mask(payer_norm.isna() | (payer_norm.astype(str).str.len() == 0), "missing_payer")
    drop_reason = drop_reason.mask(
        payee_fields["payee_key"].isna() | (payee_fields["payee_key"].astype(str).str.len() == 0),
        "missing_payee",
    )
    drop_reason = drop_reason.where(~is_valid, drop_reason)

    # record_id (kept identical semantics; per-partition loop)
    hash_inputs: List[List[Optional[str]]] = []
    for i in range(len(payer_norm)):
        hash_inputs.append(
            [
                str(cfg.program_year),
                cfg.dataset_type,
                payer_norm.iloc[i],
                payee_fields["payee_key"].iloc[i],
                str(amount.iloc[i]) if pd.notna(amount.iloc[i]) else None,
                payment_date.iloc[i],
                str(nature.iloc[i]) if pd.notna(nature.iloc[i]) else None,
            ]
        )
    record_id = pd.Series([stable_hash_hex(h) for h in hash_inputs], index=df.index, dtype="string")

    out = pd.DataFrame(
        {
            "record_id": record_id,
            "program_year": np.int16(cfg.program_year),
            "dataset_type": cfg.dataset_type,
            "source_file": source_file if cfg.keep_source_file else None,
            "payer_name_raw": payer_raw.astype("string[python]"),
            "payer_name_norm": payer_norm.astype("string[python]"),
            "payer_id": payer_id.astype("string", errors="ignore"),
            "payer_state": payer_state.astype("string", errors="ignore"),
            "payer_country": payer_country.astype("string", errors="ignore"),
            "payee_type": payee_fields["payee_type"].astype("string[python]"),
            "payee_key": payee_fields["payee_key"].astype("string[python]"),
            "physician_npi": payee_fields["physician_npi"].astype("string[python]"),
            "physician_profile_id": payee_fields["physician_profile_id"].astype("string", errors="ignore"),
            "physician_name_raw": payee_fields["physician_name_raw"].astype("string", errors="ignore"),
            "physician_name_norm": payee_fields["physician_name_norm"].astype("string", errors="ignore"),
            "physician_specialty": payee_fields["physician_specialty"].astype("string", errors="ignore"),
            "physician_primary_type": payee_fields["physician_primary_type"].astype("string", errors="ignore"),
            "physician_state": payee_fields["physician_state"].astype("string", errors="ignore"),
            "physician_zip5": payee_fields["physician_zip5"].astype("string", errors="ignore"),
            "teaching_hospital_id": payee_fields["teaching_hospital_id"].astype("string", errors="ignore"),
            "teaching_hospital_name_raw": payee_fields["teaching_hospital_name_raw"].astype("string", errors="ignore"),
            "teaching_hospital_name_norm": payee_fields["teaching_hospital_name_norm"].astype("string", errors="ignore"),
            "teaching_hospital_state": payee_fields["teaching_hospital_state"].astype("string", errors="ignore"),
            "teaching_hospital_zip5": payee_fields["teaching_hospital_zip5"].astype("string", errors="ignore"),
            "amount_usd": amount,
            "payment_date": pd.Series(payment_date, index=df.index, dtype="string[python]"),
            "payment_quarter": payment_quarter,
            "nature_of_payment": nature.astype("string", errors="ignore"),
            "form_of_payment": form.astype("string", errors="ignore"),
            "payment_context": context.astype("string", errors="ignore"),
            "product_name": product.astype("string", errors="ignore"),
            "associated_covered_drug_or_device_flag": flag.astype("string", errors="ignore"),
            "is_product_related": is_product_related.astype(bool),
            "is_valid": is_valid.astype(bool),
            "drop_reason": drop_reason.astype("string", errors="ignore"),
        }
    )

    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = None

    return out[CANONICAL_COLS]


# ---------------------------
# Sampling helpers
# ---------------------------


def sampling_mask_from_record_ids(record_ids: pd.Series, fraction: float, seed: Optional[int], method: str) -> pd.Series:
    """Return deterministic boolean mask selecting approximately `fraction` of rows.

    method:
      - sha1: matches original semantics (rid+salt -> sha1 -> uniform)
      - fast_hash: uses pandas hash_pandas_object (faster, deterministic, but not sha1-identical)

    NOTE: If you must be bit-identical to sequential, keep method='sha1'.
    """
    if fraction >= 0.9999:
        return pd.Series([True] * len(record_ids), index=record_ids.index)
    salt = "" if seed is None else str(seed)

    rid = record_ids.astype(str)

    if method.lower() == "fast_hash":
        # Deterministic within pandas; faster than per-row sha1.
        # Map to [0,1) via modulo.
        h = pd.util.hash_pandas_object(rid + salt, index=False).astype("uint64")
        vals = (h % np.uint64(10_000_000)).astype("uint64")
        return pd.Series(vals.values < int(fraction * 10_000_000), index=record_ids.index)

    # sha1 (sequential-equivalent): still Python-level but kept for correctness.
    hashes = rid.map(lambda x: hashlib.sha1((x + salt).encode("utf-8")).hexdigest())
    values = hashes.map(lambda hx: int(hx[:15], 16) / float(16**15))
    return values < fraction


# ---------------------------
# Phase runner
# ---------------------------


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def run(cfg: CleanConfig, config_path: Optional[str] = None) -> None:
    if not cfg.use_dask:
        raise ValueError("use_dask=false is not supported in this concurrent CPU script (intentionally).")

    try:
        import dask
        import dask.dataframe as dd
        from dask import compute
        from dask.diagnostics import ProgressBar
    except Exception as e:
        raise ImportError("Dask is required for concurrent CPU Phase 1. Install: pip install dask[dataframe]") from e

    t_phase_start = time.perf_counter()
    timings: Dict[str, float] = {}

    out_dir = Path(cfg.output_dir)
    ensure_dir(str(out_dir))

    # Save config snapshot
    if config_path:
        with open(config_path, "r", encoding="utf-8") as src, open(out_dir / "config_used.yaml", "w", encoding="utf-8") as dst:
            dst.write(src.read())

    clean_dir = out_dir / "payments_clean"
    rej_dir = out_dir / "payments_rejected"

    if cfg.write_format != "parquet":
        raise ValueError("Concurrent pipeline expects parquet output (write_format=parquet).")

    if clean_dir.exists():
        shutil.rmtree(clean_dir, ignore_errors=True)
    if rej_dir.exists() and not cfg.skip_rejected:
        shutil.rmtree(rej_dir, ignore_errors=True)

    ensure_dir(str(clean_dir))
    if not cfg.skip_rejected:
        ensure_dir(str(rej_dir))

    # Configure dask scheduler and worker pool (threads)
    scheduler = (cfg.scheduler or "threads").lower().strip()
    if scheduler == "threads" and cfg.max_workers and int(cfg.max_workers) > 0:
        pool = ThreadPoolExecutor(max_workers=int(cfg.max_workers))
        dask.config.set(pool=pool)
        dask.config.set(scheduler="threads")
        logging.info("[Phase 1] Dask scheduler=threads pool_size=%d", int(cfg.max_workers))
    else:
        dask.config.set(scheduler=scheduler)
        logging.info("[Phase 1] Dask scheduler=%s", scheduler)

    # Read CSV(s)
    t_read_start = time.perf_counter()
    logging.info("[Phase 1] Concurrent cleaning started | max_workers=%s | blocksize=%s", cfg.max_workers, cfg.blocksize)

    ddf_list = []
    for fpath in cfg.input_files:
        part = dd.read_csv(
            fpath,
            dtype=str,
            blocksize=cfg.blocksize,
            assume_missing=True,
            encoding="utf-8",
        )
        part["__source_file"] = Path(fpath).name if cfg.keep_source_file else ""
        ddf_list.append(part)

    ddf = dd.concat(ddf_list, axis=0, interleave_partitions=True)
    if cfg.dask_npartitions and cfg.dask_npartitions > 0:
        ddf = ddf.repartition(npartitions=int(cfg.dask_npartitions))

    # Optional early sampling on raw dataframe to shrink work up-front
    do_sampling = cfg.sampling_fraction < 0.9999
    sampling_stage = (cfg.sampling_stage or "canonical").lower().strip()
    if do_sampling and sampling_stage == "raw":
        logging.info(
            "[Phase 1] Applying RAW sampling fraction=%.4f seed=%s (pre-canonical)",
            cfg.sampling_fraction,
            cfg.sampling_seed,
        )
        ddf = ddf.sample(frac=float(cfg.sampling_fraction), random_state=cfg.sampling_seed, replace=False)

    logging.info("[Phase 1] Read complete | partitions=%d", ddf.npartitions)

    # Canonicalize
    t_canon_start = time.perf_counter()
    meta = make_canonical_meta(cfg)

    def _canon_part(pdf: pd.DataFrame) -> pd.DataFrame:
        # Avoid expensive mode() unless needed; most of the time __source_file is constant per partition.
        source_file = ""
        if cfg.keep_source_file and "__source_file" in pdf.columns and len(pdf):
            try:
                source_file = str(pdf["__source_file"].iloc[0])
            except Exception:
                source_file = ""
        return canonicalize_partition(
            pdf.drop(columns=["__source_file"], errors="ignore"),
            cfg,
            source_file=source_file,
        )

    canon = ddf.map_partitions(_canon_part, meta=meta)

    # Optional sampling
    if do_sampling and sampling_stage == "canonical":
        logging.info(
            "[Phase 1] Applying sampling fraction=%.4f seed=%s method=%s",
            cfg.sampling_fraction,
            cfg.sampling_seed,
            cfg.sampling_method,
        )

        def _sample_part(pdf: pd.DataFrame) -> pd.DataFrame:
            if len(pdf) == 0:
                return pdf
            mask = sampling_mask_from_record_ids(
                pdf["record_id"],
                float(cfg.sampling_fraction),
                cfg.sampling_seed,
                cfg.sampling_method,
            )
            return pdf.loc[mask]

        canon = canon.map_partitions(_sample_part, meta=meta)

    # Schema guard
    missing = [c for c in CANONICAL_COLS if c not in canon.columns]
    if missing:
        raise RuntimeError(
            "Canonical schema mismatch after map_partitions. "
            f"Missing columns: {missing}. canon.columns={list(canon.columns)}"
        )

    timings["t_canon_graph"] = round(time.perf_counter() - t_canon_start, 4)
    logging.info("[Phase 1] Canonicalization graph built | partitions=%d", canon.npartitions)

    valid = canon[canon["is_valid"] == True]
    rejected = canon[canon["is_valid"] == False]

    # Sanitize
    t_sanitize_start = time.perf_counter()
    valid = valid.map_partitions(sanitize_for_parquet, meta=make_canonical_meta(cfg))
    if not cfg.skip_rejected:
        rejected = rejected.map_partitions(sanitize_for_parquet, meta=make_canonical_meta(cfg))
    timings["t_sanitize_graph"] = round(time.perf_counter() - t_sanitize_start, 4)

    # Persist ONCE so writes + stats reuse the same materialized partitions (avoids recomputation)
    t_persist_start = time.perf_counter()
    if cfg.persist:
        logging.info("[Phase 1] Persisting sanitized datasets (enables reuse for writes+stats)...")
        with ProgressBar():
            valid_p = valid.persist()
            rej_p = rejected.persist() if not cfg.skip_rejected else None
        timings["t_persist"] = round(time.perf_counter() - t_persist_start, 4)
    else:
        logging.info("[Phase 1] Skipping persist (memory-saving mode); writes/stats will compute from graph")
        valid_p = valid
        rej_p = rejected if not cfg.skip_rejected else None
        timings["t_persist"] = 0.0

    # Write Parquet
    t_write_start = time.perf_counter()
    logging.info("[Phase 1] Writing clean dataset...")
    with ProgressBar():
        valid_p.to_parquet(str(clean_dir), engine="pyarrow", write_index=False)
    timings["t_write_parquet_valid"] = round(time.perf_counter() - t_write_start, 4)

    if not cfg.skip_rejected:
        t_write_rej = time.perf_counter()
        logging.info("[Phase 1] Writing rejected dataset...")
        with ProgressBar():
            assert rej_p is not None
            rej_p.to_parquet(str(rej_dir), engine="pyarrow", write_index=False)
        timings["t_write_parquet_rejected"] = round(time.perf_counter() - t_write_rej, 4)

    # Stats (single compute)
    rows_in_val = rows_after_sampling_val = rows_valid_val = rows_rej_val = None
    amt_min_val = amt_mean_val = amt_median_val = amt_max_val = None
    phys_val = hosp_val = 0
    phys_missing_npi_val = phys_missing_prof_val = hosp_missing_id_val = 0
    pandas_stats_used = False

    if cfg.compute_stats:
        t_stats_start = time.perf_counter()
        logging.info("[Phase 1] Computing stats (single-pass reductions)...")

        # Decide pandas fast path based on rows_valid (cheap on persisted)
        if cfg.stats_pandas_threshold and int(cfg.stats_pandas_threshold) > 0:
            with ProgressBar():
                rows_valid_val = int(valid_p.shape[0].compute())
            if rows_valid_val <= int(cfg.stats_pandas_threshold):
                pandas_stats_used = True
                vpdf_pd = valid_p.compute()
                rows_in_val = int(ddf.shape[0].compute())
                rows_after_sampling_val = int(canon.shape[0].compute())
                rows_rej_val = 0
                if not cfg.skip_rejected and rej_p is not None:
                    rows_rej_val = int(rej_p.shape[0].compute())

                amt_pd = pd.to_numeric(vpdf_pd["amount_usd"], errors="coerce")
                if len(amt_pd):
                    amt_min_val = float(amt_pd.min())
                    amt_mean_val = float(amt_pd.mean())
                    amt_max_val = float(amt_pd.max())
                    if cfg.stats_compute_median:
                        amt_median_val = float(amt_pd.median())

                phys_val = int((vpdf_pd["payee_type"] == "physician").sum())
                hosp_val = int((vpdf_pd["payee_type"] == "teaching_hospital").sum())
                if phys_val:
                    phys_missing_npi_val = int(vpdf_pd[vpdf_pd["payee_type"] == "physician"]["physician_npi"].isna().sum())
                    phys_missing_prof_val = int(vpdf_pd[vpdf_pd["payee_type"] == "physician"]["physician_profile_id"].isna().sum())
                if hosp_val:
                    hosp_missing_id_val = int(vpdf_pd[vpdf_pd["payee_type"] == "teaching_hospital"]["teaching_hospital_id"].isna().sum())

        if not pandas_stats_used:
            # Build lazy expressions off persisted frames (important)
            rows_in_expr = ddf.shape[0]
            rows_after_sampling_expr = canon.shape[0]
            rows_valid_expr = valid_p.shape[0]
            rows_rej_expr = rej_p.shape[0] if (not cfg.skip_rejected and rej_p is not None) else 0

            import dask.dataframe as dd  # local

            amt_series = dd.to_numeric(valid_p["amount_usd"], errors="coerce")
            amt_min_expr = amt_series.min()
            amt_mean_expr = amt_series.mean()
            amt_max_expr = amt_series.max()

            compute_median = bool(cfg.stats_compute_median)
            median_method = (cfg.stats_median_method or "exact").lower()
            if compute_median:
                # "approx" can still map to quantile in many dask versions; kept for future extension.
                amt_median_expr = amt_series.quantile(0.5)
            else:
                amt_median_expr = None

            phys_expr = (valid_p["payee_type"] == "physician").sum()
            hosp_expr = (valid_p["payee_type"] == "teaching_hospital").sum()

            vpdf = valid_p[["payee_type", "physician_npi", "physician_profile_id", "teaching_hospital_id"]]
            phys_missing_npi_expr = vpdf[vpdf["payee_type"] == "physician"]["physician_npi"].isna().sum()
            phys_missing_prof_expr = vpdf[vpdf["payee_type"] == "physician"]["physician_profile_id"].isna().sum()
            hosp_missing_id_expr = vpdf[vpdf["payee_type"] == "teaching_hospital"]["teaching_hospital_id"].isna().sum()

            to_compute = [
                rows_in_expr,
                rows_after_sampling_expr,
                rows_valid_expr,
                rows_rej_expr,
                amt_min_expr,
                amt_mean_expr,
                amt_max_expr,
                phys_expr,
                hosp_expr,
                phys_missing_npi_expr,
                phys_missing_prof_expr,
                hosp_missing_id_expr,
            ]
            if compute_median:
                to_compute.append(amt_median_expr)

            with ProgressBar():
                results = compute(*to_compute)

            i = 0
            rows_in_val = int(results[i]); i += 1
            rows_after_sampling_val = int(results[i]); i += 1
            rows_valid_val = int(results[i]); i += 1
            rows_rej_val = int(results[i]) if not cfg.skip_rejected else 0; i += 1

            if rows_valid_val:
                amt_min_val = float(results[i]); i += 1
                amt_mean_val = float(results[i]); i += 1
                amt_max_val = float(results[i]); i += 1
            else:
                i += 3

            phys_val = int(results[i]) if rows_valid_val else 0; i += 1
            hosp_val = int(results[i]) if rows_valid_val else 0; i += 1

            phys_missing_npi_val = int(results[i]) if phys_val else 0; i += 1
            phys_missing_prof_val = int(results[i]) if phys_val else 0; i += 1
            hosp_missing_id_val = int(results[i]) if hosp_val else 0; i += 1

            if compute_median:
                amt_median_val = float(results[i]) if rows_valid_val else None

        timings["t_stats"] = round(time.perf_counter() - t_stats_start, 4)

    total_time = time.perf_counter() - t_phase_start

    report = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "rows_in": rows_in_val,
        "rows_after_sampling": rows_after_sampling_val,
        "rows_valid": rows_valid_val,
        "rows_rejected": rows_rej_val if rows_rej_val is not None else (0 if cfg.skip_rejected else None),
        "valid_rate": (float(rows_valid_val / rows_in_val) if rows_in_val and rows_valid_val is not None else None),
        "amount_usd_stats": {
            "min": amt_min_val,
            "mean": amt_mean_val,
            "median": amt_median_val,
            "max": amt_max_val,
        },
        "payee_counts": {
            "physicians": int(phys_val),
            "teaching_hospitals": int(hosp_val),
        },
        "identifier_missingness": {
            "physician_npi_missing_count": int(phys_missing_npi_val),
            "physician_npi_missing_pct": (phys_missing_npi_val / phys_val * 100.0) if phys_val else None,
            "physician_profile_missing_count": int(phys_missing_prof_val),
            "physician_profile_missing_pct": (phys_missing_prof_val / phys_val * 100.0) if phys_val else None,
            "hospital_id_missing_count": int(hosp_missing_id_val),
            "hospital_id_missing_pct": (hosp_missing_id_val / hosp_val * 100.0) if hosp_val else None,
        },
        "timings_sec": {"total": round(total_time, 4), **timings},
        "sampling": {
            "fraction": float(cfg.sampling_fraction),
            "seed": cfg.sampling_seed,
            "method": cfg.sampling_method,
            "enabled": cfg.sampling_fraction < 0.9999,
        },
        "stats_config": {
            "compute_stats": bool(cfg.compute_stats),
            "compute_median": bool(cfg.stats_compute_median),
            "median_method": cfg.stats_median_method,
            "pandas_threshold": int(cfg.stats_pandas_threshold),
            "pandas_used": bool(pandas_stats_used),
        },
        "reproducibility": {
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function": HASH_FUNCTION,
            "use_dask": cfg.use_dask,
            "blocksize": cfg.blocksize,
            "dask_npartitions": cfg.dask_npartitions,
            "scheduler": cfg.scheduler,
            "max_workers": cfg.max_workers,
        },
        "outputs": {
            "payments_clean_dir": str(clean_dir),
            "payments_rejected_dir": str(rej_dir) if not cfg.skip_rejected else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("[Phase 1] Cleaning completed in %.2fs", total_time)
    logging.info("Wrote clean dataset: %s", clean_dir)
    if not cfg.skip_rejected:
        logging.info("Wrote rejected dataset: %s", rej_dir)
    logging.info("Wrote report: %s", out_dir / "cleaning_report.json")


def run_from_pipeline(
    pipeline_cfg: Dict,
    *,
    out_dir: Path,
    approach: str,
    run_id: str,
    dataset_name: str,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg = pipeline_cfg.get("phase1_clean", {}) or {}
    dataset_cfg = pipeline_cfg.get("dataset", {}) or {}
    inputs_cfg = pipeline_cfg.get("inputs", {}) or {}
    config_dir = Path(pipeline_cfg.get("__config_dir", Path.cwd()))
    scale_cfg = dataset_cfg.get("scale", {}) or {}

    global_workers = int(pipeline_cfg.get("execution", {}).get("max_workers", 1) or 1)
    max_workers = int(global_workers)
    auto_parts = 0 if global_workers == 0 else max(4, global_workers * 4)

    payment_type = str(dataset_cfg.get("payment_type", "general_payment"))
    type_map = {
        "general": "general_payment",
        "general_payment": "general_payment",
        "research": "research_payment",
        "research_payment": "research_payment",
        "ownership": "ownership",
    }
    dataset_type = type_map.get(payment_type, payment_type)
    program_year = int(phase_cfg.get("program_year", dataset_cfg.get("year", dataset_cfg.get("program_year", 2024))))

    raw_inputs = inputs_cfg.get("csv_files")
    if not raw_inputs:
        raw_inputs = inputs_cfg.get("csv_glob", [])
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
    stats_cfg = (phase_cfg.get("stats", {}) or {})

    cfg = CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=input_files,
        output_dir=str(out_dir),
        use_dask=bool(phase_cfg.get("use_dask", True)),
        blocksize=str(phase_cfg.get("blocksize", "256MB")),
        dask_npartitions=int(phase_cfg.get("dask_npartitions", auto_parts)),
        scheduler=str(phase_cfg.get("scheduler", "threads")),
        persist=bool(phase_cfg.get("persist", True)),
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
        max_workers=max_workers,
        stats_compute_median=bool(stats_cfg.get("compute_median", False)),
        stats_median_method=str(stats_cfg.get("median_method", "exact")),
        stats_pandas_threshold=int(stats_cfg.get("use_pandas_if_rows_lt", 0)),
        compute_stats=bool(phase_cfg.get("compute_stats", True)),
    )

    cfg_snapshot = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "input_files": cfg.input_files,
        "output_dir": cfg.output_dir,
        "write_format": cfg.write_format,
        "keep_source_file": cfg.keep_source_file,
        "skip_rejected": cfg.skip_rejected,
        "sampling_fraction": cfg.sampling_fraction,
        "sampling_seed": cfg.sampling_seed,
        "sampling_method": cfg.sampling_method,
        "sampling_stage": cfg.sampling_stage,
        "use_dask": cfg.use_dask,
        "blocksize": cfg.blocksize,
        "dask_npartitions": cfg.dask_npartitions,
        "scheduler": cfg.scheduler,
        "max_workers": cfg.max_workers,
        "persist": cfg.persist,
        "stats": {
            "compute_stats": cfg.compute_stats,
            "compute_median": cfg.stats_compute_median,
            "median_method": cfg.stats_median_method,
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
        "phase": "phase1_clean",
        "dataset": dataset_name,
        "approach": approach,
        "run_id": run_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "clean_dir": str(out_dir / "payments_clean"),
            "rejected_dir": str(out_dir / "payments_rejected"),
            "report": str(out_dir / "cleaning_report.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config for Phase 1 cleaning.")
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
