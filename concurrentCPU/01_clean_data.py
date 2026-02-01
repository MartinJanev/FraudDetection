#!/usr/bin/env python3
"""
Phase 1 (Concurrent CPU via Dask): Extract + Clean CMS Open Payments into canonical schema.

Paper-consistent intent:
- Partitioned storage (Parquet dataset).
- Parallel ingestion + preprocessing (Dask DataFrame).
- Deterministic canonicalization semantics (same canonical columns, same payee_key logic).

Outputs (under output_dir):
- payments_clean/         (Parquet dataset, partitioned by Dask)
- payments_rejected/      (optional Parquet dataset)
- cleaning_report.json
- dataset_fingerprint.json
- config_used.yaml (snapshot)

Important:
- This code uses your existing canonicalization functions (pandas-on-partition).
- Concurrency is inside Dask partitions; do NOT run many full pipelines in parallel.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

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

    # Output
    write_format: str = "parquet"  # parquet recommended for concurrent pipeline
    keep_source_file: bool = True
    skip_rejected: bool = True

    # Dataset scaling (Phase 1 only)
    # scale:
    #   enabled: bool
    #   fraction: float (0-1]
    #   seed: int
    #   method: "hash_record_id"
    # Legacy-scale fields (pre-canonical sampling) kept for compatibility
    scale_mode: str = "none"
    scale_value: Optional[float] = None
    scale_key_cols: Optional[List[str]] = None
    # New record_id-based sampling
    scale_enabled: bool = False
    scale_fraction: float = 1.0
    scale_seed: int = 123
    scale_method: str = "hash_record_id"

    # Optional overrides
    column_mapping: Optional[Dict[str, str]] = None
    max_workers: Optional[int] = None


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

    resolved_inputs = []
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
    return CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=resolved_inputs,
        output_dir=str(output_dir),
        use_dask=bool(phase_cfg.get("use_dask", True)),
        blocksize=str(phase_cfg.get("blocksize", "256MB")),
        dask_npartitions=int(phase_cfg.get("dask_npartitions", 0)),
        scheduler=str(phase_cfg.get("scheduler", "threads")),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        keep_source_file=bool(phase_cfg.get("keep_source_file", True)),
        skip_rejected=bool(phase_cfg.get("skip_rejected", True)),
        # legacy scale params (unused when method is hash_record_id)
        scale_mode=str(scale_cfg.get("mode", scale_cfg.get("scale_mode", "none"))),
        scale_value=scale_cfg.get("value", scale_cfg.get("scale_value", None)),
        scale_key_cols=scale_cfg.get("key_cols", None),
        # new sampling knob (disabled by default)
        scale_enabled=bool(scale_cfg.get("enabled", False)),
        scale_fraction=float(scale_cfg.get("fraction", 1.0)),
        scale_seed=int(scale_cfg.get("seed", 123)),
        scale_method=str(scale_cfg.get("method", "hash_record_id")),
        column_mapping=phase_cfg.get("column_mapping"),
    )


def load_config(path: str) -> CleanConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(path).parent
    if "dataset_type" not in raw:
        return _parse_pipeline_style_config(raw, config_dir)

    def resolve_list(files: List[str]) -> List[str]:
        out = []
        for fp in files:
            p = Path(fp)
            if not p.is_absolute():
                p = config_dir / p
            out.append(str(p.resolve()))
        return out

    return CleanConfig(
        dataset_type=str(raw["dataset_type"]),
        program_year=int(raw["program_year"]),
        input_files=resolve_list(list(raw["input_files"])),
        output_dir=str((config_dir / raw["output_dir"]).resolve()) if not Path(
            raw["output_dir"]).is_absolute() else str(Path(raw["output_dir"]).resolve()),
        use_dask=bool(raw.get("use_dask", True)),
        blocksize=str(raw.get("blocksize", "256MB")),
        dask_npartitions=int(raw.get("dask_npartitions", 0)),
        scheduler=str(raw.get("scheduler", "threads")),
        write_format=str(raw.get("write_format", "parquet")),
        keep_source_file=bool(raw.get("keep_source_file", True)),
        skip_rejected=bool(raw.get("skip_rejected", True)),
        column_mapping=raw.get("column_mapping"),
    )


# ---------------------------
# Dataset Fingerprinting
# ---------------------------

def compute_file_fingerprint(filepath: str, hash_mb: int = FINGERPRINT_MB) -> Dict:
    path = Path(filepath)
    if not path.exists():
        return {"file": str(path.name), "exists": False, "error": "File not found"}

    stat = path.stat()
    file_size = stat.st_size
    modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    hash_obj = hashlib.sha1()
    bytes_to_hash = hash_mb * 1024 * 1024
    bytes_hashed = 0
    try:
        with open(filepath, "rb") as f:
            while bytes_hashed < bytes_to_hash:
                chunk = f.read(min(8192, bytes_to_hash - bytes_hashed))
                if not chunk:
                    break
                hash_obj.update(chunk)
                bytes_hashed += len(chunk)
        partial_hash = hash_obj.hexdigest()
    except Exception as e:
        partial_hash = f"error: {e}"

    return {
        "file": str(path.name),
        "absolute_path": str(path.absolute()),
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "modified_utc": modified_time,
        "partial_hash": partial_hash,
        "hash_method": f"{HASH_FUNCTION}(first_{hash_mb}MB)",
    }


def create_dataset_fingerprint(config: CleanConfig, config_path: str) -> Dict:
    input_fingerprints = [compute_file_fingerprint(fpath) for fpath in config.input_files]
    with open(config_path, "r", encoding="utf-8") as f:
        config_snapshot = yaml.safe_load(f)

    return {
        "fingerprint_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function": HASH_FUNCTION,
        },
        "config_snapshot": config_snapshot,
        "input_files": input_fingerprints,
        "processing_params": {
            "dataset_type": config.dataset_type,
            "program_year": config.program_year,
            "write_format": config.write_format,
            "keep_source_file": config.keep_source_file,
            "use_dask": config.use_dask,
            "blocksize": config.blocksize,
            "dask_npartitions": config.dask_npartitions,
            "scheduler": config.scheduler,
        },
    }


# ---------------------------
# Normalization helpers
# ---------------------------

_PUNCT_RE = re.compile(r"[.,;:()\\[\\]{}'\"`]+")


def normalize_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\\s+", " ", s)
    return s.upper()


def normalize_zip5(z: Optional[str]) -> Optional[str]:
    if z is None:
        return None
    z = re.sub(r"\\D+", "", str(z))
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
        s = re.sub(r"[^0-9.\\-]", "", s)
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
# Canonical columns + CMS mappings (unchanged from your current script)
# ---------------------------

def make_canonical_meta(cfg: CleanConfig) -> pd.DataFrame:
    # Use pandas StringDtype so Dask meta_nonempty doesn't inject `object()` sentinels.
    S = "string[python]"   # safe & pyarrow-friendly
    B = "bool"
    I = "int16"
    F = "float64"
    I8 = "Int8"  # pandas nullable int8

    dtypes = {
        "record_id": S,
        "program_year": I,
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

        "amount_usd": F,
        "payment_date": S,
        "payment_quarter": I8,

        "nature_of_payment": S,
        "form_of_payment": S,
        "payment_context": S,
        "product_name": S,
        "associated_covered_drug_or_device_flag": S,

        "is_product_related": B,
        "is_valid": B,
        "drop_reason": S,
    }

    data = {}
    for c in CANONICAL_COLS:
        dt = dtypes.get(c, S)
        data[c] = pd.Series([], dtype=dt)
    return pd.DataFrame(data)


CANONICAL_COLS = [
    "record_id", "program_year", "dataset_type", "source_file",
    "payer_name_raw", "payer_name_norm", "payer_id", "payer_state", "payer_country",
    "payee_type", "payee_key",
    "physician_npi", "physician_profile_id", "physician_name_raw", "physician_name_norm",
    "physician_specialty", "physician_primary_type", "physician_state", "physician_zip5",
    "teaching_hospital_id", "teaching_hospital_name_raw", "teaching_hospital_name_norm",
    "teaching_hospital_state", "teaching_hospital_zip5",
    "amount_usd", "payment_date", "payment_quarter",
    "nature_of_payment", "form_of_payment", "payment_context",
    "product_name", "associated_covered_drug_or_device_flag", "is_product_related",
    "is_valid", "drop_reason",
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


def get_column_mapping(df: pd.DataFrame, dataset_type: str, custom_mapping: Optional[Dict[str, str]] = None) -> Dict[
    str, Optional[str]]:
    if custom_mapping:
        base_mapping = custom_mapping
    elif dataset_type in CMS_COLUMN_MAPPINGS:
        base_mapping = CMS_COLUMN_MAPPINGS[dataset_type]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    actual_cols = set(df.columns)
    result = {}
    missing_cols = []
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
    """
    Ensure all columns are PyArrow-serializable:
    - numeric columns: coerce to numeric
    - bool columns: coerce to bool (nullable -> bool with False for missing if needed)
    - string/object columns: convert any non-(str/None/NaN) to string, and normalize NaN->None
    """
    if pdf is None or len(pdf) == 0:
        # keep schema stable even for empty partitions
        return pdf

    # Columns we want strongly typed
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
            # Convert to real bool; missing -> False (or keep as pandas boolean if you prefer)
            # Using pandas BooleanDtype can also work, but plain bool is simplest for parquet.
            if s.dtype.name == "boolean":
                pdf[c] = s.fillna(False).astype(bool)
            else:
                pdf[c] = s.astype(bool, errors="ignore")
                # if that didn't convert cleanly, force via truthy mapping
                if pdf[c].dtype != bool:
                    pdf[c] = s.fillna(False).map(bool).astype(bool)
            continue

        if c in int8_cols:
            # Coerce '1' -> 1 and None/NaN -> <NA>, then to pandas nullable Int8
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("Int8")
            continue

        # Everything else: treat as string-ish object column.
        # Replace NaN with None and stringify any weird Python objects.
        def _clean_cell(x):
            if x is None:
                return None
            # pandas missing
            if isinstance(x, float) and np.isnan(x):
                return None
            # already clean
            if isinstance(x, str):
                return x
            # for anything else (including <object object at ...>), stringify
            return str(x)

        pdf[c] = s.map(_clean_cell).astype("string[python]")

    return pdf


def build_payee_fields_vectorized(df: pd.DataFrame, colmap: Dict[str, Optional[str]]) -> Dict[str, pd.Series]:
    # (This function is kept semantically identical to your current version.)
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

    hosp_id_series = df[hosp_id_col].astype(
        str).str.strip() if hosp_id_col and hosp_id_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_id_series = hosp_id_series.replace("", None).replace("nan", None)

    hosp_name_raw = df[hosp_name_col].astype(
        str).str.strip() if hosp_name_col and hosp_name_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_name_raw = hosp_name_raw.replace("", None).replace("nan", None)
    hosp_name_norm = hosp_name_raw.map(lambda x: normalize_name(x) if x else None)

    state_col = colmap.get("phys_state")
    zip_col = colmap.get("phys_zip")
    hosp_state = df[state_col] if state_col and state_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_zip5 = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n,
                                                                                                    index=df.index)

    npi_col = colmap.get("phys_npi")
    profile_col = colmap.get("phys_profile_id")

    npi_series = df[npi_col] if npi_col and npi_col in df.columns else pd.Series([None] * n, index=df.index)
    npi_clean = npi_series.astype(str).str.replace(r"\\D+", "", regex=True)
    npi_clean = npi_clean.replace("", None).replace("nan", None)

    profile_series = df[profile_col] if profile_col and profile_col in df.columns else pd.Series([None] * n,
                                                                                                 index=df.index)
    profile_clean = profile_series.astype(str).str.strip()
    profile_clean = profile_clean.replace("", None).replace("nan", None)

    first_col = colmap.get("phys_first")
    middle_col = colmap.get("phys_middle")
    last_col = colmap.get("phys_last")

    first = df[first_col].astype(str).str.strip() if first_col and first_col in df.columns else pd.Series([""] * n,
                                                                                                          index=df.index)
    middle = df[middle_col].astype(str).str.strip() if middle_col and middle_col in df.columns else pd.Series([""] * n,
                                                                                                              index=df.index)
    last = df[last_col].astype(str).str.strip() if last_col and last_col in df.columns else pd.Series([""] * n,
                                                                                                      index=df.index)

    first = first.replace("nan", "").replace("NaN", "")
    middle = middle.replace("nan", "").replace("NaN", "")
    last = last.replace("nan", "").replace("NaN", "")

    phys_name_raw = (first + " " + middle + " " + last).str.replace(r"\\s+", " ", regex=True).str.strip()
    phys_name_raw = phys_name_raw.replace("", None).replace("nan", None).replace("NaN", None)
    phys_name_norm = phys_name_raw.map(lambda x: normalize_name(x) if x else None)

    specialty_col = colmap.get("phys_specialty")
    primary_type_col = colmap.get("phys_primary_type")
    phys_specialty = df[specialty_col] if specialty_col and specialty_col in df.columns else pd.Series([None] * n,
                                                                                                       index=df.index)
    phys_primary_type = df[primary_type_col] if primary_type_col and primary_type_col in df.columns else pd.Series(
        [None] * n, index=df.index)
    phys_state = df[state_col] if state_col and state_col in df.columns else pd.Series([None] * n, index=df.index)
    phys_zip5 = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n,
                                                                                                    index=df.index)

    payee_key = pd.Series([None] * n, index=df.index, dtype=object)
    payee_type = pd.Series(["physician"] * n, index=df.index)
    payee_type = payee_type.where(~is_hospital, "teaching_hospital")

    for idx in df.index[is_hospital]:
        hid = hosp_id_series.loc[idx]
        if pd.notna(hid) and str(hid).strip() and str(hid).lower() != "nan":
            payee_key.loc[idx] = f"HOSP_ID:{hid}"
        else:
            hname = hosp_name_norm.loc[idx]
            hzip = hosp_zip5.loc[idx]
            payee_key.loc[
                idx] = f"HOSP_NAMEZIP:{stable_hash_hex([hname if pd.notna(hname) else None, hzip if pd.notna(hzip) else None])}"

    for idx in df.index[~is_hospital]:
        npi = npi_clean.loc[idx]
        prof = profile_clean.loc[idx]
        if pd.notna(npi) and str(npi).strip() and str(npi).lower() != "nan":
            payee_key.loc[idx] = f"PHYS_NPI:{npi}"
        elif pd.notna(prof) and str(prof).strip() and str(prof).lower() != "nan":
            payee_key.loc[idx] = f"PHYS_PROF:{prof}"
        else:
            pname = phys_name_norm.loc[idx]
            pzip = phys_zip5.loc[idx]
            payee_key.loc[
                idx] = f"PHYS_NAMEZIP:{stable_hash_hex([pname if pd.notna(pname) else None, pzip if pd.notna(pzip) else None])}"

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
            flag_clean.notna()
            & (flag_clean != "")
            & (flag_clean != "NAN")
            & (flag_clean != "NONE")
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
        payee_fields["payee_key"].isna() | (payee_fields["payee_key"].astype(str).str.len() == 0), "missing_payee")
    drop_reason = drop_reason.where(~is_valid, drop_reason)

    # record_id (kept identical to your existing semantics; loop is per-partition)
    hash_inputs = []
    for i in range(len(payer_norm)):
        hash_inputs.append([
            str(cfg.program_year),
            cfg.dataset_type,
            payer_norm.iloc[i],
            payee_fields["payee_key"].iloc[i],
            str(amount.iloc[i]) if pd.notna(amount.iloc[i]) else None,
            payment_date.iloc[i],
            str(nature.iloc[i]) if pd.notna(nature.iloc[i]) else None,
        ])
    record_id = pd.Series([stable_hash_hex(h) for h in hash_inputs], index=df.index, dtype="string")

    out = pd.DataFrame({
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
    })

    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = None

    return out[CANONICAL_COLS]


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _apply_scale_filter(ddf, cfg: CleanConfig):
    """Apply dataset scaling axis in Phase 1 only.

    Supported:
      - fraction: deterministic hash-based sampling (stable across runs for the same raw rows)
      - months: keep rows with payment_date month <= scale_value (best-effort using raw date columns)
    """
    if not cfg.scale_mode or cfg.scale_mode == "none" or cfg.scale_value in (None, "", 1, 1.0):
        return ddf, {"mode": "none", "value": None, "rows_kept": None}

    mode = str(cfg.scale_mode).lower().strip()

    try:
        import dask.dataframe as dd
    except Exception:
        dd = None

    # Best-effort raw date column candidates (before canonicalization)
    date_candidates = [
        "Date_of_Payment",
        "date_of_payment",
        "Payment_Date",
        "payment_date",
    ]

    if mode == "months":
        try:
            months = int(cfg.scale_value)
        except Exception as e:
            raise ValueError(
                f"scale_value must be an integer 1-12 for scale_mode=months. Got: {cfg.scale_value}") from e
        months = max(1, min(12, months))
        date_col = next((c for c in date_candidates if c in ddf.columns), None)
        if date_col is None:
            # Cannot apply month slicing without a date column; fall back to no scaling.
            return ddf, {"mode": "months", "value": months, "rows_kept": None, "note": "no_date_column_found"}
        dt = ddf[date_col]
        # Let Dask infer parsing; errors become NaT and are dropped by filter.
        if dd is None:
            return ddf, {"mode": "months", "value": months, "rows_kept": None, "note": "dask_not_available"}
        dt_parsed = dd.to_datetime(dt, errors="coerce", infer_datetime_format=True)
        ddf2 = ddf[dt_parsed.dt.month <= months]
        return ddf2, {"mode": "months", "value": months, "rows_kept": None, "date_col": date_col}

    if mode != "fraction":
        # Unknown mode -> no scaling
        return ddf, {"mode": mode, "value": cfg.scale_value, "rows_kept": None, "note": "unsupported_mode"}

    # fraction mode: deterministic hash filter
    frac = float(cfg.scale_value)
    if frac <= 0.0:
        return ddf.head(0), {"mode": "fraction", "value": frac, "rows_kept": 0}
    if frac >= 1.0:
        return ddf, {"mode": "fraction", "value": frac, "rows_kept": None}

    base = 10000
    threshold = int(frac * base)

    # Choose stable key columns: user-provided or first available in common identifiers
    common_keys = [
        "Record_ID",
        "record_id",
        "General_Payment_ID",
        "Research_Payment_ID",
        "Ownership_ID",
        "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
    ]
    key_cols = list(cfg.scale_key_cols) if cfg.scale_key_cols else [c for c in common_keys if c in ddf.columns]
    if not key_cols:
        # fallback: use a small set of columns (excluding source file)
        key_cols = [c for c in ddf.columns if c != "__source_file"][:3]

    def _hash_filter(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty:
            return pdf
        parts = []
        for c in key_cols:
            if c in pdf.columns:
                parts.append(pdf[c].fillna("").astype(str))
        if not parts:
            return pdf.iloc[0:0]
        key_series = parts[0]
        for s in parts[1:]:
            key_series = key_series.str.cat(s, sep="|")
        # deterministic sha1 -> modulo
        mask = []
        for v in key_series.tolist():
            h = hashlib.sha1(v.encode("utf-8", errors="ignore")).hexdigest()
            mask.append(int(h[:8], 16) % base < threshold)
        return pdf.loc[mask]

    meta = ddf._meta
    ddf2 = ddf.map_partitions(_hash_filter, meta=meta)
    return ddf2, {"mode": "fraction", "value": frac, "threshold": threshold, "base": base, "key_cols": key_cols,
                  "rows_kept": None}


def run(cfg: CleanConfig, config_path: Optional[str] = None) -> None:
    t_phase_start = time.perf_counter()
    timings: Dict[str, float] = {}
    out_dir = Path(cfg.output_dir)
    ensure_dir(str(out_dir))

    # Fingerprint + config snapshot
    if config_path:
        fingerprint = create_dataset_fingerprint(cfg, config_path)
        with open(out_dir / "dataset_fingerprint.json", "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)
        with open(config_path, "r", encoding="utf-8") as src, open(out_dir / "config_used.yaml", "w",
                                                                   encoding="utf-8") as dst:
            dst.write(src.read())

    clean_dir = out_dir / "payments_clean"
    rej_dir = out_dir / "payments_rejected"

    if cfg.write_format != "parquet":
        raise ValueError("Concurrent pipeline expects parquet output (write_format=parquet).")

    # Clear existing outputs safely (Dask parquet datasets are directories)
    if clean_dir.exists():
        shutil.rmtree(clean_dir, ignore_errors=True)
    if rej_dir.exists() and not cfg.skip_rejected:
        shutil.rmtree(rej_dir, ignore_errors=True)

    ensure_dir(str(clean_dir))
    if not cfg.skip_rejected:
        ensure_dir(str(rej_dir))

    # Dask path
    if cfg.use_dask:
        try:
            import dask
            import dask.dataframe as dd
        except Exception as e:
            raise ImportError(
                "Dask is required for concurrent CPU Phase 1. Install: pip install dask[dataframe]") from e

        t_read_build_start = time.perf_counter()

        # Build a single ddf across all files; include source file column if requested
        # We keep dtype=str to preserve determinism before parsing.
        ddf_list = []
        for fpath in cfg.input_files:
            ddf = dd.read_csv(
                fpath,
                dtype=str,
                blocksize=cfg.blocksize,
                assume_missing=True,
                encoding="utf-8",
            )
            if cfg.keep_source_file:
                ddf["__source_file"] = Path(fpath).name
            else:
                ddf["__source_file"] = ""
            ddf_list.append(ddf)

        ddf = dd.concat(ddf_list, axis=0, interleave_partitions=True)
        if cfg.dask_npartitions and cfg.dask_npartitions > 0:
            ddf = ddf.repartition(npartitions=int(cfg.dask_npartitions))

        timings["t_read_build_graph"] = round(time.perf_counter() - t_read_build_start, 4)

        # Optional dataset scaling axis (fraction or months). Applied before canonicalization.
        t_scale_start = time.perf_counter()
        ddf, scale_info = _apply_scale_filter(ddf, cfg)
        timings["t_scale_filter_graph"] = round(time.perf_counter() - t_scale_start, 4)

        t_canon_graph_start = time.perf_counter()
        # Apply canonicalization per partition (pandas)
        # We need meta to keep Dask happy.
        meta = make_canonical_meta(cfg)

        def _canon_part(pdf: pd.DataFrame) -> pd.DataFrame:
            source_file = ""
            if "__source_file" in pdf.columns and len(pdf):
                # mode() can be empty in weird partitions; be defensive
                try:
                    m = pdf["__source_file"].mode()
                    source_file = str(m.iloc[0]) if len(m) else ""
                except Exception:
                    source_file = ""
            return canonicalize_partition(
                pdf.drop(columns=["__source_file"], errors="ignore"),
                cfg,
                source_file=source_file,
            )

        canon = ddf.map_partitions(_canon_part, meta=meta)

        # Hard fail early if Dask schema got corrupted (this is what caused your KeyError).
        missing = [c for c in CANONICAL_COLS if c not in canon.columns]
        if missing:
            raise RuntimeError(
                "Canonical schema mismatch after map_partitions.\n"
                f"Missing columns: {missing}\n"
                f"canon.columns: {list(canon.columns)}\n"
                f"canon._meta columns: {list(getattr(canon, '_meta', pd.DataFrame()).columns)}"
            )

        valid = canon[canon["is_valid"] == True]
        rejected = canon[canon["is_valid"] == False]

        timings["t_canon_graph"] = round(time.perf_counter() - t_canon_graph_start, 4)

        # Sanitize partitions to avoid pyarrow failures on weird python objects
        valid = valid.map_partitions(sanitize_for_parquet, meta=make_canonical_meta(cfg))
        if not cfg.skip_rejected:
            rejected = rejected.map_partitions(sanitize_for_parquet, meta=make_canonical_meta(cfg))

        # Write datasets
        t_write_start = time.perf_counter()
        valid.to_parquet(str(clean_dir), engine="pyarrow", write_index=False)
        timings["t_write_parquet_valid"] = round(time.perf_counter() - t_write_start, 4)

        if not cfg.skip_rejected:
            t_write_rej_start = time.perf_counter()
            rejected.to_parquet(str(rej_dir), engine="pyarrow", write_index=False)
            timings["t_write_parquet_rejected"] = round(time.perf_counter() - t_write_rej_start, 4)

        # Stats (computed via reductions)
        t_stats_start = time.perf_counter()
        rows_in = int(ddf.shape[0].compute())
        rows_valid = int(valid.shape[0].compute())
        rows_rej = int(rejected.shape[0].compute()) if not cfg.skip_rejected else 0

        amt = dd.to_numeric(valid["amount_usd"], errors="coerce")
        amt_min = float(amt.min().compute()) if rows_valid else None
        amt_mean = float(amt.mean().compute()) if rows_valid else None
        amt_median = float(amt.quantile(0.5).compute()) if rows_valid else None
        amt_max = float(amt.max().compute()) if rows_valid else None

        # payee breakdown
        phys = int((valid["payee_type"] == "physician").sum().compute()) if rows_valid else 0
        hosp = int((valid["payee_type"] == "teaching_hospital").sum().compute()) if rows_valid else 0

        # missingness computed on correct subgroup (same semantics as your sequential version)
        vpdf = valid[["payee_type", "physician_npi", "physician_profile_id", "teaching_hospital_id"]]
        phys_missing_npi = int(
            vpdf[vpdf["payee_type"] == "physician"]["physician_npi"].isna().sum().compute()) if phys else 0
        phys_missing_prof = int(
            vpdf[vpdf["payee_type"] == "physician"]["physician_profile_id"].isna().sum().compute()) if phys else 0
        hosp_missing_id = int(vpdf[vpdf["payee_type"] == "teaching_hospital"][
                                  "teaching_hospital_id"].isna().sum().compute()) if hosp else 0

        timings["t_stats"] = round(time.perf_counter() - t_stats_start, 4)

        total_time = time.perf_counter() - t_phase_start

        report = {
            "dataset_type": cfg.dataset_type,
            "program_year": cfg.program_year,
            "rows_in": rows_in,
            "rows_valid": rows_valid,
            "rows_rejected": rows_rej,
            "valid_rate": float(rows_valid / rows_in) if rows_in else None,
            "amount_usd_stats": {
                "min": amt_min,
                "mean": amt_mean,
                "median": amt_median,
                "max": amt_max,
            },
            "payee_counts": {
                "physicians": phys,
                "teaching_hospitals": hosp,
            },
            "identifier_missingness": {
                "physician_npi_missing_count": phys_missing_npi,
                "physician_npi_missing_pct": (phys_missing_npi / phys * 100.0) if phys else None,
                "physician_profile_missing_count": phys_missing_prof,
                "physician_profile_missing_pct": (phys_missing_prof / phys * 100.0) if phys else None,
                "hospital_id_missing_count": hosp_missing_id,
                "hospital_id_missing_pct": (hosp_missing_id / hosp * 100.0) if hosp else None,
            },
            "timings_sec": {"total": round(total_time, 4), **timings},
            "scaling": {**scale_info, **sample_info},
            "reproducibility": {
                "normalization_version": NORMALIZATION_VERSION,
                "normalization_description": NORMALIZATION_DESCRIPTION,
                "hash_function": HASH_FUNCTION,
                "use_dask": cfg.use_dask,
                "blocksize": cfg.blocksize,
                "dask_npartitions": cfg.dask_npartitions,
                "scheduler": cfg.scheduler,
            },
            "outputs": {
                "payments_clean_dir": str(clean_dir),
                "payments_rejected_dir": str(rej_dir) if not cfg.skip_rejected else None,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logging.info("Wrote clean dataset: %s", clean_dir)
        if not cfg.skip_rejected:
            logging.info("Wrote rejected dataset: %s", rej_dir)
        logging.info("Wrote report: %s", out_dir / "cleaning_report.json")
        return

    raise ValueError("use_dask=false is not supported in this concurrent CPU script (intentionally).")


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

    phase_cfg = pipeline_cfg.get("phase1_clean", {})
    dataset_cfg = pipeline_cfg.get("dataset", {})
    inputs_cfg = pipeline_cfg.get("inputs", {})
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
    if raw_inputs is None or raw_inputs == []:
        raw_inputs = inputs_cfg.get("csv_glob", [])
    if isinstance(raw_inputs, str):
        raw_inputs = [raw_inputs]
    if not raw_inputs:
        raise ValueError("pipeline_cfg.inputs.csv_files is required for phase1_clean")

    input_files = []
    for f in list(raw_inputs):
        p = Path(f)
        if not p.is_absolute():
            p = config_dir / f
        input_files.append(str(p.resolve()))

    cfg = CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=input_files,
        output_dir=str(out_dir),
        use_dask=bool(phase_cfg.get("use_dask", True)),
        blocksize=str(phase_cfg.get("blocksize", "256MB")),
        dask_npartitions=int(phase_cfg.get("dask_npartitions", auto_parts)),
        scheduler=str(phase_cfg.get("scheduler", "threads")),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        keep_source_file=bool(phase_cfg.get("keep_source_file", True)),
        skip_rejected=bool(phase_cfg.get("skip_rejected", True)),
        # legacy scale params
        scale_mode=str(scale_cfg.get("mode", scale_cfg.get("scale_mode", "none"))),
        scale_value=scale_cfg.get("value", scale_cfg.get("scale_value", None)),
        scale_key_cols=scale_cfg.get("key_cols", None),
        # new sampling knob (disabled by default)
        scale_enabled=bool(scale_cfg.get("enabled", False)),
        scale_fraction=float(scale_cfg.get("fraction", 1.0)),
        scale_seed=int(scale_cfg.get("seed", 123)),
        scale_method=str(scale_cfg.get("method", "hash_record_id")),
        column_mapping=phase_cfg.get("column_mapping"),
        max_workers=max_workers,
    )

    cfg_snapshot = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "input_files": cfg.input_files,
        "output_dir": cfg.output_dir,
        "write_format": cfg.write_format,
        "keep_source_file": cfg.keep_source_file,
        "skip_rejected": cfg.skip_rejected,
        "scale_mode": cfg.scale_method,
        "scale_value": cfg.scale_fraction,
        "scale_seed": cfg.scale_seed,
        # legacy scale fields for compatibility
        "legacy_scale_mode": cfg.scale_mode,
        "legacy_scale_value": cfg.scale_value,
        "legacy_scale_key_cols": cfg.scale_key_cols,
        "use_dask": cfg.use_dask,
        "blocksize": cfg.blocksize,
        "dask_npartitions": cfg.dask_npartitions,
        "scheduler": cfg.scheduler,
        "column_mapping": cfg.column_mapping,
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
            "fingerprint": str(out_dir / "dataset_fingerprint.json"),
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
