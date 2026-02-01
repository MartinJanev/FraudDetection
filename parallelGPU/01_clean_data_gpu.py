#!/usr/bin/env python3
"""
Phase 1 (GPU single-device): Extract + Clean CMS Open Payments into canonical schema.

Parity note:
- record_id is computed with *exactly* the same SHA1 semantics as CPU version.
- Because per-row SHA1 is not available efficiently in cuDF, record_id is computed on CPU
  at the end of each chunk, then merged back into the cuDF output.

Outputs (same structure as CPU):
- payments_clean/part-xxxxx.parquet (valid rows)
- payments_rejected/part-xxxxx.parquet (invalid rows + drop_reason) [optional]
- cleaning_report.json (summary stats)
- manifests + dataset_fingerprint.json
"""

from __future__ import annotations

import argparse
import hashlib
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
from tqdm import tqdm

# GPU stack
import cudf

# ---------------------------
# Reproducibility Constants
# ---------------------------

NORMALIZATION_VERSION = "v1"
NORMALIZATION_DESCRIPTION = "punctuation-strip + uppercase + whitespace collapse"
HASH_FUNCTION = "sha1"
FINGERPRINT_MB = 10

# ---------------------------
# Logging
# ---------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class CleanConfig:
    dataset_type: str  # "general_payment" | "research_payment" | "ownership"
    program_year: int
    input_files: List[str]
    output_dir: str
    chunk_size: int = 500_000
    write_format: str = "parquet"  # "parquet" or "csv"
    keep_source_file: bool = True
    column_mapping: Optional[Dict[str, str]] = None
    max_workers: int = 1
    skip_rejected: bool = False
    # Optional: dataset scaling (Phase 1 only)
    scale_enabled: bool = False
    scale_fraction: float = 1.0
    scale_seed: int = 123
    scale_method: str = "hash_record_id"


def load_config(path: str) -> CleanConfig:
    cfg_path = Path(path).resolve()
    cfg_dir = cfg_path.parent

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Backward compatibility: allow either nested pipeline schema or flat schema
    dataset_cfg = raw.get("dataset", {}) or {}
    inputs_cfg = raw.get("inputs", {}) or {}
    output_cfg = raw.get("output", {}) or {}
    phase_cfg = raw.get("phase1_clean", {}) or {}

    # dataset_type: accept nested dataset.dataset_type or legacy raw.dataset_type
    dataset_type_raw = str(dataset_cfg.get("dataset_type", raw.get("dataset_type"))).strip().lower()
    type_map = {
        "general": "general_payment",
        "general_payment": "general_payment",
        "general_payments": "general_payment",
        "research": "research_payment",
        "research_payment": "research_payment",
        "research_payments": "research_payment",
        "ownership": "ownership",
        "ownership_payment": "ownership",
        "ownership_payments": "ownership",
    }
    dataset_type = type_map.get(dataset_type_raw, dataset_type_raw)

    # program_year: accept dataset.program_year or legacy raw.program_year
    program_year_raw = dataset_cfg.get("program_year", raw.get("program_year"))
    if program_year_raw is None:
        raise KeyError("Missing dataset.program_year (or legacy program_year) in config.")
    program_year = int(program_year_raw)

    # inputs: accept inputs.csv_files (or legacy input_files)
    csv_files = inputs_cfg.get("csv_files", raw.get("input_files"))
    if csv_files is None:
        raise KeyError("Missing inputs.csv_files (or legacy input_files) in config.")
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    input_files: List[str] = []
    for f in list(csv_files):
        p = Path(f)
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        input_files.append(str(p))

    # output directory root
    root_dir = output_cfg.get("root_dir", raw.get("output_dir", "output"))
    out_dir = Path(root_dir)
    if not out_dir.is_absolute():
        out_dir = (cfg_dir / out_dir).resolve()

    # phase1 settings
    chunk_size = int(phase_cfg.get("chunk_size", raw.get("chunk_size", 500_000)))
    write_format = str(phase_cfg.get("write_format", raw.get("write_format", "parquet")))
    keep_source_file = bool(phase_cfg.get("keep_source_file", raw.get("keep_source_file", True)))
    skip_rejected = bool(phase_cfg.get("skip_rejected", raw.get("skip_rejected", False)))

    # Optional keys if you later add them
    column_mapping = phase_cfg.get("column_mapping", raw.get("column_mapping"))
    max_workers = int(phase_cfg.get("max_workers", raw.get("max_workers", 1)))

    # Optional: dataset scaling (Phase 1 only)
    scale_cfg = dataset_cfg.get("scale", {}) or {}
    scale_enabled = bool(scale_cfg.get("enabled", False))
    scale_fraction = float(scale_cfg.get("fraction", 1.0))
    scale_seed = int(scale_cfg.get("seed", 123))
    scale_method = str(scale_cfg.get("method", "hash_record_id"))

    return CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=input_files,
        output_dir=str(out_dir),
        chunk_size=chunk_size,
        write_format=write_format,
        keep_source_file=keep_source_file,
        column_mapping=column_mapping,
        max_workers=max_workers,
        skip_rejected=skip_rejected,
        # scaling knob (disabled by default)
        scale_enabled=bool(scale_cfg.get("enabled", False)),
        scale_fraction=float(scale_cfg.get("fraction", 1.0)),
        scale_seed=int(scale_cfg.get("seed", 123)),
        scale_method=str(scale_cfg.get("method", "hash_record_id")),
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
            "chunk_size": config.chunk_size,
            "write_format": config.write_format,
            "keep_source_file": config.keep_source_file,
        },
    }

# ---------------------------
# Canonical schema + mappings (unchanged)
# ---------------------------

CANONICAL_COLS = [
    "record_id", "program_year", "dataset_type", "source_file",
    "payer_name_raw", "payer_name_norm", "payer_id", "payer_state", "payer_country",
    "payee_type", "payee_key",
    "physician_npi", "physician_profile_id",
    "physician_name_raw", "physician_name_norm",
    "physician_specialty", "physician_primary_type",
    "physician_state", "physician_zip5",
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

def get_column_mapping(df_cols: Iterable[str], dataset_type: str, custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    if custom_mapping:
        base_mapping = custom_mapping
    elif dataset_type in CMS_COLUMN_MAPPINGS:
        base_mapping = CMS_COLUMN_MAPPINGS[dataset_type]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be one of: {list(CMS_COLUMN_MAPPINGS.keys())}")

    actual_cols = set(df_cols)
    result: Dict[str, Optional[str]] = {}
    missing_cols = []

    for key, col_name in base_mapping.items():
        if col_name in actual_cols:
            result[key] = col_name
        else:
            result[key] = None
            if key in ["amount", "payer_name"]:
                missing_cols.append(col_name)

    if missing_cols:
        raise ValueError(
            f"Required columns missing from CSV file: {missing_cols}\n"
            f"Available columns: {sorted(actual_cols)}"
        )
    return result

# ---------------------------
# GPU vectorized normalization helpers
# ---------------------------

_PUNCT_RE = r"[.,;:()\[\]{}'\"`]+"

def _as_str_series(s: cudf.Series) -> cudf.Series:
    # normalize to string; keep nulls
    # (astype("str") will turn nulls into "<NA>" in some cases; so we guard with fillna(""))
    return s.astype("str")

def normalize_name_series(s: cudf.Series) -> cudf.Series:
    # Equivalent semantics to CPU normalize_name() but vectorized.
    # - strip
    # - empty -> null
    # - remove punctuation
    # - collapse whitespace
    # - upper
    s2 = _as_str_series(s)
    s2 = s2.str.strip()
    # Treat literal "nan"/"none" as missing (matches your CPU patterns)
    low = s2.str.lower()
    s2 = s2.mask((s2 == "") | (low == "nan") | (low == "none"), None)
    s2 = s2.str.replace(_PUNCT_RE, "", regex=True)
    s2 = s2.str.replace(r"\s+", " ", regex=True).str.strip()
    s2 = s2.mask(s2 == "", None)
    return s2.str.upper()

def normalize_zip5_series(z: cudf.Series) -> cudf.Series:
    z2 = _as_str_series(z)
    z2 = z2.str.replace(r"\D+", "", regex=True)
    z2 = z2.mask(z2.str.len() < 5, None)
    z2 = z2.mask(z2 == "", None)
    return z2.str.slice(0, 5)

def safe_float_series(x: cudf.Series) -> cudf.Series:
    s = _as_str_series(x).str.strip()
    low = s.str.lower()
    s = s.mask((s == "") | (low == "nan") | (low == "none"), None)
    # remove commas and currency symbols etc.
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.mask((s == "") | (s == "-") | (s == "."), None)
    return cudf.to_numeric(s, errors="coerce")

def parse_date_series(x: cudf.Series) -> cudf.Series:
    # CPU parse for robustness + pandas parity, then return ISO date strings to GPU.
    # Keeps nulls as nulls.
    s = x.to_pandas()  # pandas Series (object/strings)
    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})

    dt = pd.to_datetime(s, errors="coerce", utc=False)
    out = dt.dt.strftime("%Y-%m-%d")
    out = out.astype("string")  # keep <NA>
    return cudf.from_pandas(out)


# ---------------------------
# Payee fields (GPU vectorized, no Python loops)
# ---------------------------

def build_payee_fields_gpu(df: cudf.DataFrame, colmap: Dict[str, Optional[str]]) -> Dict[str, cudf.Series]:
    n = len(df)

    hosp_id_col = colmap.get("hospital_id")
    hosp_name_col = colmap.get("hospital_name")
    recipient_type_col = colmap.get("recipient_type")

    # Start with all False
    is_hospital = cudf.Series([False] * n)

    if hosp_id_col and hosp_id_col in df.columns:
        hid_orig = df[hosp_id_col]
        hid_str = _as_str_series(hid_orig).str.strip()
        hid_mask = hid_orig.notna() & (hid_str != "") & (hid_str.str.lower() != "nan")
        is_hospital = is_hospital | hid_mask

    if hosp_name_col and hosp_name_col in df.columns:
        hname_orig = df[hosp_name_col]
        hname_str = _as_str_series(hname_orig).str.strip()
        hname_mask = hname_orig.notna() & (hname_str != "") & (hname_str.str.lower() != "nan")
        is_hospital = is_hospital | hname_mask

    if recipient_type_col and recipient_type_col in df.columns:
        recip = _as_str_series(df[recipient_type_col]).str.upper()
        contains_hosp = recip.fillna("").str.contains("HOSPITAL", regex=False)
        is_hospital = is_hospital | contains_hosp

    # Hospital fields
    if hosp_id_col and hosp_id_col in df.columns:
        hosp_id_series = _as_str_series(df[hosp_id_col]).str.strip()
        low = hosp_id_series.str.lower()
        hosp_id_series = hosp_id_series.mask((hosp_id_series == "") | (low == "nan"), None)
    else:
        hosp_id_series = cudf.Series([None] * n)

    if hosp_name_col and hosp_name_col in df.columns:
        hosp_name_raw = _as_str_series(df[hosp_name_col]).str.strip()
        low = hosp_name_raw.str.lower()
        hosp_name_raw = hosp_name_raw.mask((hosp_name_raw == "") | (low == "nan"), None)
    else:
        hosp_name_raw = cudf.Series([None] * n)

    hosp_name_norm = normalize_name_series(hosp_name_raw)

    state_col = colmap.get("phys_state")
    zip_col = colmap.get("phys_zip")
    hosp_state = df[state_col] if state_col and state_col in df.columns else cudf.Series([None] * n)
    hosp_zip5 = normalize_zip5_series(df[zip_col]) if zip_col and zip_col in df.columns else cudf.Series([None] * n)

    # Physician fields
    npi_col = colmap.get("phys_npi")
    profile_col = colmap.get("phys_profile_id")

    if npi_col and npi_col in df.columns:
        npi_series = _as_str_series(df[npi_col])
        npi_clean = npi_series.str.replace(r"\D+", "", regex=True)
        low = npi_clean.str.lower()
        npi_clean = npi_clean.mask((npi_clean == "") | (low == "nan"), None)
    else:
        npi_clean = cudf.Series([None] * n)

    if profile_col and profile_col in df.columns:
        profile_clean = _as_str_series(df[profile_col]).str.strip()
        low = profile_clean.str.lower()
        profile_clean = profile_clean.mask((profile_clean == "") | (low == "nan"), None)
    else:
        profile_clean = cudf.Series([None] * n)

    first_col = colmap.get("phys_first")
    middle_col = colmap.get("phys_middle")
    last_col = colmap.get("phys_last")

    first = _as_str_series(df[first_col]).str.strip() if first_col and first_col in df.columns else cudf.Series([""] * n)
    middle = _as_str_series(df[middle_col]).str.strip() if middle_col and middle_col in df.columns else cudf.Series([""] * n)
    last = _as_str_series(df[last_col]).str.strip() if last_col and last_col in df.columns else cudf.Series([""] * n)

    # Remove literal nan tokens (matches CPU behavior)
    first = first.mask(first.str.lower() == "nan", "")
    middle = middle.mask(middle.str.lower() == "nan", "")
    last = last.mask(last.str.lower() == "nan", "")

    phys_name_raw = (first + " " + middle + " " + last).str.replace(r"\s+", " ", regex=True).str.strip()
    low = phys_name_raw.str.lower()
    phys_name_raw = phys_name_raw.mask((phys_name_raw == "") | (low == "nan"), None)
    phys_name_norm = normalize_name_series(phys_name_raw)

    specialty_col = colmap.get("phys_specialty")
    primary_type_col = colmap.get("phys_primary_type")

    phys_specialty = df[specialty_col] if specialty_col and specialty_col in df.columns else cudf.Series([None] * n)
    phys_primary_type = df[primary_type_col] if primary_type_col and primary_type_col in df.columns else cudf.Series([None] * n)
    phys_state = df[state_col] if state_col and state_col in df.columns else cudf.Series([None] * n)
    phys_zip5 = normalize_zip5_series(df[zip_col]) if zip_col and zip_col in df.columns else cudf.Series([None] * n)

    # payee_type
    payee_type = cudf.Series(["physician"] * n)
    payee_type = payee_type.where(~is_hospital, "teaching_hospital")

    # payee_key vectorized
    # Hospitals: HOSP_ID:<id> else HOSP_NAMEZIP:<sha1(hname|zip)>
    hosp_has_id = hosp_id_series.notna() & (_as_str_series(hosp_id_series).str.strip() != "")
    hosp_id_key = "HOSP_ID:" + _as_str_series(hosp_id_series)

    # For fallback hashes we compute on CPU for parity (sha1), so here we build the string parts:
    hosp_fallback_join = (
        _as_str_series(hosp_name_norm.fillna("")) + "|" + _as_str_series(hosp_zip5.fillna(""))
    )

    # Physicians: NPI > profile > hash(name|zip)
    npi_has = npi_clean.notna() & (_as_str_series(npi_clean).str.strip() != "")
    prof_has = profile_clean.notna() & (_as_str_series(profile_clean).str.strip() != "")

    phys_npi_key = "PHYS_NPI:" + _as_str_series(npi_clean)
    phys_prof_key = "PHYS_PROF:" + _as_str_series(profile_clean)
    phys_fallback_join = (
        _as_str_series(phys_name_norm.fillna("")) + "|" + _as_str_series(phys_zip5.fillna(""))
    )

    # Placeholder keys (we will replace fallback join strings with sha1 hashed keys on CPU)
    hosp_key = hosp_id_key.where(hosp_has_id, "HOSP_NAMEZIP:" + hosp_fallback_join)
    phys_key = phys_npi_key.where(npi_has, phys_prof_key.where(prof_has, "PHYS_NAMEZIP:" + phys_fallback_join))
    payee_key = hosp_key.where(is_hospital, phys_key)

    # Nullify subgroup-specific fields
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
        # for parity hashing of fallback keys
        "__is_hospital": is_hospital,
    }

# ---------------------------
# CPU SHA1 helpers for parity
# ---------------------------

def _sha1_hex_string(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _sha1_series_from_joined(joined: pd.Series) -> pd.Series:
    # joined is already a per-row string; compute sha1 per row (CPU).
    return joined.map(_sha1_hex_string)

# ---------------------------
# Canonicalization (GPU)
# ---------------------------

def canonicalize_chunk_gpu(df: cudf.DataFrame, cfg: CleanConfig, source_file: str) -> cudf.DataFrame:
    # Ensure 0..n-1 index
    df = df.reset_index(drop=True)

    colmap = get_column_mapping(df.columns, cfg.dataset_type, cfg.column_mapping)

    payer_name_col = colmap.get("payer_name")
    if not payer_name_col:
        raise ValueError(f"Required column 'payer_name' not found in dataset_type '{cfg.dataset_type}'")
    amount_col = colmap.get("amount")
    if not amount_col:
        raise ValueError(f"Required column 'amount' not found in dataset_type '{cfg.dataset_type}'")

    payer_raw = df[payer_name_col]
    payer_norm = normalize_name_series(payer_raw)

    amount_raw = df[amount_col]
    amount = safe_float_series(amount_raw).astype("float64")

    date_col = colmap.get("date")
    date_raw = df[date_col] if date_col else cudf.Series([None] * len(df))
    payment_date = parse_date_series(date_raw)

    # Optional
    nature_col = colmap.get("nature")
    form_col = colmap.get("form")
    context_col = colmap.get("context")
    product_col = colmap.get("product")
    flag_col = colmap.get("drug_device_flag")

    nature = df[nature_col] if nature_col else cudf.Series([None] * len(df))
    form = df[form_col] if form_col else cudf.Series([None] * len(df))
    context = df[context_col] if context_col else cudf.Series([None] * len(df))
    product = df[product_col] if product_col else cudf.Series([None] * len(df))
    flag = df[flag_col] if flag_col else cudf.Series([None] * len(df))

    payer_id_col = colmap.get("payer_id")
    payer_state_col = colmap.get("payer_state")
    payer_country_col = colmap.get("payer_country")

    payer_id = df[payer_id_col] if payer_id_col else cudf.Series([None] * len(df))
    payer_state = df[payer_state_col] if payer_state_col else cudf.Series([None] * len(df))
    payer_country = df[payer_country_col] if payer_country_col else cudf.Series([None] * len(df))

    payee_fields = build_payee_fields_gpu(df, colmap)

    # Product flag boolean
    flag_clean = _as_str_series(flag).str.strip().str.upper()
    is_product_related = (
        flag_clean.notna() &
        (flag_clean != "") &
        (flag_clean != "NAN") &
        (flag_clean != "NONE")
    )

    # Quarter from payment_date (YYYY-MM-DD)
    # month as int from ISO string "YYYY-MM-DD" -> take s[5:7]
    month_str = payment_date.str.slice(5, 7)
    month = cudf.to_numeric(month_str, errors="coerce")
    payment_quarter = ((month - 1) // 3 + 1).astype("int16")
    payment_quarter = payment_quarter.where(month.notna(), None)

    # Validity
    is_valid = (
        amount.notna() &
        (amount >= 0) &
        payer_norm.notna() &
        (payer_norm.str.len() > 0) &
        payee_fields["payee_key"].notna() &
        (payee_fields["payee_key"].str.len() > 0)
    )

    drop_reason = cudf.Series([None] * len(df))
    drop_reason = drop_reason.mask(amount.isna(), "missing_amount")
    drop_reason = drop_reason.mask(amount.notna() & (amount < 0), "negative_amount")
    drop_reason = drop_reason.mask(payer_norm.isna() | (payer_norm.str.len() == 0), "missing_payer")
    drop_reason = drop_reason.mask(
        payee_fields["payee_key"].isna() | (payee_fields["payee_key"].str.len() == 0),
        "missing_payee"
    )
    drop_reason = drop_reason.where(~is_valid, drop_reason)

    # --- Parity SHA1 for payee_key fallbacks (HOSP_NAMEZIP / PHYS_NAMEZIP) ---
    # We created payee_key that contains e.g. "HOSP_NAMEZIP:<name|zip>" or "PHYS_NAMEZIP:<name|zip>"
    # For parity, replace the "<name|zip>" part with sha1(name|zip).
    payee_key_gpu = payee_fields["payee_key"]

    # Bring only payee_key to CPU and rewrite fallback keys exactly
    payee_key_pd = payee_key_gpu.to_pandas()

    def _fix_payee_key_parity(k: Optional[str]) -> Optional[str]:
        if k is None:
            return None
        s = str(k)
        if s.startswith("HOSP_NAMEZIP:"):
            joined = s[len("HOSP_NAMEZIP:"):]
            h = _sha1_hex_string(joined)
            return "HOSP_NAMEZIP:" + h
        if s.startswith("PHYS_NAMEZIP:"):
            joined = s[len("PHYS_NAMEZIP:"):]
            h = _sha1_hex_string(joined)
            return "PHYS_NAMEZIP:" + h
        return s

    payee_key_pd = payee_key_pd.map(_fix_payee_key_parity)
    payee_key = cudf.from_pandas(payee_key_pd)

    # --- Parity SHA1 for record_id (CPU) ---
    # Build the exact same parts list as CPU:
    # [year, dataset_type, payer_norm, payee_key, amount, payment_date, nature]
    payer_norm_pd = payer_norm.to_pandas()
    amount_pd = amount.to_pandas()
    payment_date_pd = payment_date.to_pandas()
    nature_pd = nature.to_pandas()

    # Important: match CPU behavior: None stays None, amount None stays None, nature None stays None
    # joined string: "|".join("" if p is None else str(p) ...)
    year_str = str(cfg.program_year)
    dtype_str = str(cfg.dataset_type)

    # Build per-row joined strings in pandas
    # (This keeps parity with your stable_hash_hex() joining logic.)
    def _join_parts(i: int) -> str:
        parts = [
            year_str,
            dtype_str,
            payer_norm_pd.iat[i],
            payee_key_pd.iat[i],
            str(amount_pd.iat[i]) if pd.notna(amount_pd.iat[i]) else None,
            payment_date_pd.iat[i],
            str(nature_pd.iat[i]) if pd.notna(nature_pd.iat[i]) else None,
        ]
        return "|".join("" if p is None else str(p) for p in parts)

    joined = pd.Series((_join_parts(i) for i in range(len(df))), index=payer_norm_pd.index)
    record_id_pd = _sha1_series_from_joined(joined).astype("string")
    record_id = cudf.from_pandas(record_id_pd)

    out = cudf.DataFrame({
        "record_id": record_id,
        "program_year": np.int16(cfg.program_year),
        "dataset_type": cfg.dataset_type,
        "source_file": source_file if cfg.keep_source_file else None,

        "payer_name_raw": payer_raw,
        "payer_name_norm": payer_norm,
        "payer_id": payer_id,
        "payer_state": payer_state,
        "payer_country": payer_country,

        "payee_type": payee_fields["payee_type"],
        "payee_key": payee_key,

        "physician_npi": payee_fields["physician_npi"],
        "physician_profile_id": payee_fields["physician_profile_id"],
        "physician_name_raw": payee_fields["physician_name_raw"],
        "physician_name_norm": payee_fields["physician_name_norm"],
        "physician_specialty": payee_fields["physician_specialty"],
        "physician_primary_type": payee_fields["physician_primary_type"],
        "physician_state": payee_fields["physician_state"],
        "physician_zip5": payee_fields["physician_zip5"],

        "teaching_hospital_id": payee_fields["teaching_hospital_id"],
        "teaching_hospital_name_raw": payee_fields["teaching_hospital_name_raw"],
        "teaching_hospital_name_norm": payee_fields["teaching_hospital_name_norm"],
        "teaching_hospital_state": payee_fields["teaching_hospital_state"],
        "teaching_hospital_zip5": payee_fields["teaching_hospital_zip5"],

        "amount_usd": amount,
        "payment_date": payment_date,
        "payment_quarter": payment_quarter,

        "nature_of_payment": nature,
        "form_of_payment": form,
        "payment_context": context,
        "product_name": product,
        "associated_covered_drug_or_device_flag": flag,
        "is_product_related": is_product_related.astype("bool"),

        "is_valid": is_valid.astype("bool"),
        "drop_reason": drop_reason,
    })

    # Ensure all canonical cols exist
    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = None

    return out[CANONICAL_COLS]

# ---------------------------
# IO helpers
# ---------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def write_partitioned_parquet(df: cudf.DataFrame, output_dir: Path, part_num: int) -> None:
    part_file = output_dir / f"part-{part_num:05d}.parquet"
    df.to_parquet(str(part_file), index=False)

# ---------------------------
# Main run (chunked CSV via pandas -> cudf per chunk)
# ---------------------------

def run(cfg: CleanConfig, config_path: str = None) -> None:
    out_dir = Path(cfg.output_dir)
    ensure_dir(str(out_dir))

    if config_path:
        logging.info("Creating dataset fingerprint for reproducibility...")
        fingerprint = create_dataset_fingerprint(cfg, config_path)
        fingerprint_path = out_dir / "dataset_fingerprint.json"
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)
        logging.info("Wrote dataset fingerprint: %s", fingerprint_path)

        config_copy_path = out_dir / "config_used.yaml"
        with open(config_path, "r", encoding="utf-8") as src:
            with open(config_copy_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        logging.info("Saved config snapshot: %s", config_copy_path)

    clean_dir = out_dir / "payments_clean"
    rej_dir = out_dir / "payments_rejected"
    clean_path = out_dir / "payments_clean.csv"
    rej_path = out_dir / "payments_rejected.csv"

    if cfg.write_format == "parquet":
        ensure_dir(str(clean_dir))
        if not cfg.skip_rejected:
            ensure_dir(str(rej_dir))
        for part in clean_dir.glob("part-*.parquet"):
            part.unlink()
        if not cfg.skip_rejected:
            for part in rej_dir.glob("part-*.parquet"):
                part.unlink()
    else:
        if clean_path.exists():
            clean_path.unlink()
        if not cfg.skip_rejected and rej_path.exists():
            rej_path.unlink()

    report_path = out_dir / "cleaning_report.json"
    if report_path.exists():
        report_path.unlink()

    rows_in = 0
    rows_valid = 0
    rows_rejected = 0
    amounts = []
    part_num = 0

    physician_count = 0
    hospital_count = 0
    missing_npi = 0
    missing_profile = 0
    missing_hosp_id = 0

    manifest_clean = []
    manifest_rejected = []

    logging.info("Starting Phase 1 (GPU) clean. year=%s dataset_type=%s", cfg.program_year, cfg.dataset_type)
    t_phase_start = time.perf_counter()

    for fpath in cfg.input_files:
        fpath = str(fpath)
        logging.info("Reading file: %s", fpath)

        logging.info("Counting rows in %s...", Path(fpath).name)
        total_rows = sum(1 for _ in open(fpath, "r", encoding="utf-8")) - 1
        total_chunks = (total_rows + cfg.chunk_size - 1) // cfg.chunk_size
        logging.info("File has %d rows (%d chunks of size %d)", total_rows, total_chunks, cfg.chunk_size)

        reader = pd.read_csv(
            fpath,
            chunksize=cfg.chunk_size,
            low_memory=False,
            dtype=str,
            encoding="utf-8",
            sep=",",
            engine="c",
        )

        for chunk in tqdm(reader, total=total_chunks, desc=f"Processing {Path(fpath).name}", unit="chunk"):
            rows_in += len(chunk)

            # Transfer chunk to GPU
            gdf = cudf.from_pandas(chunk)

            canon = canonicalize_chunk_gpu(gdf, cfg, source_file=Path(fpath).name)

            valid = canon[canon["is_valid"]]
            rej = canon[~canon["is_valid"]]

            rows_valid += int(len(valid))
            rows_rejected += int(len(rej))

            if len(valid) > 0:
                amounts.append(valid["amount_usd"].to_pandas().to_numpy())

                # payee type counts
                payee_type_pd = valid["payee_type"].to_pandas()
                physician_count += int((payee_type_pd == "physician").sum())
                hospital_count += int((payee_type_pd == "teaching_hospital").sum())

                # missingness on subgroups
                phys_rows = valid[valid["payee_type"] == "physician"]
                hosp_rows = valid[valid["payee_type"] == "teaching_hospital"]

                missing_npi += int(phys_rows["physician_npi"].isna().sum())
                missing_profile += int(phys_rows["physician_profile_id"].isna().sum())
                missing_hosp_id += int(hosp_rows["teaching_hospital_id"].isna().sum())

            if cfg.write_format == "parquet":
                clean_rows = int(len(valid))
                rej_rows = int(len(rej))

                if clean_rows > 0:
                    write_partitioned_parquet(valid, clean_dir, part_num)
                    manifest_clean.append({
                        "part_file": f"part-{part_num:05d}.parquet",
                        "row_count": clean_rows,
                        "source_file": Path(fpath).name,
                    })

                if rej_rows > 0 and not cfg.skip_rejected:
                    write_partitioned_parquet(rej, rej_dir, part_num)
                    manifest_rejected.append({
                        "part_file": f"part-{part_num:05d}.parquet",
                        "row_count": rej_rows,
                        "source_file": Path(fpath).name,
                    })

                part_num += 1
            else:
                header = not clean_path.exists()
                if len(valid) > 0:
                    valid.to_pandas().to_csv(clean_path, mode="a", index=False, header=header)
                header2 = not rej_path.exists()
                if len(rej) > 0 and not cfg.skip_rejected:
                    rej.to_pandas().to_csv(rej_path, mode="a", index=False, header=header2)

    total_time = time.perf_counter() - t_phase_start

    if amounts:
        all_amt = np.concatenate(amounts)
        amt_stats = {
            "min": float(np.nanmin(all_amt)) if len(all_amt) else None,
            "mean": float(np.nanmean(all_amt)) if len(all_amt) else None,
            "median": float(np.nanmedian(all_amt)) if len(all_amt) else None,
            "max": float(np.nanmax(all_amt)) if len(all_amt) else None,
        }
    else:
        amt_stats = {"min": None, "mean": None, "median": None, "max": None}

    physician_missing_npi_pct = (missing_npi / physician_count * 100) if physician_count > 0 else None
    physician_missing_profile_pct = (missing_profile / physician_count * 100) if physician_count > 0 else None
    hospital_missing_id_pct = (missing_hosp_id / hospital_count * 100) if hospital_count > 0 else None

    report = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "rows_in": int(rows_in),
        "rows_valid": int(rows_valid),
        "rows_rejected": int(rows_rejected) if not cfg.skip_rejected else 0,
        "valid_rate": float(rows_valid / rows_in) if rows_in else None,
        "amount_usd_stats": amt_stats,
        "payee_counts": {
            "physicians": int(physician_count),
            "teaching_hospitals": int(hospital_count),
        },
        "identifier_missingness": {
            "physician_npi_missing_count": int(missing_npi),
            "physician_npi_missing_pct": float(physician_missing_npi_pct) if physician_missing_npi_pct is not None else None,
            "physician_profile_missing_count": int(missing_profile),
            "physician_profile_missing_pct": float(physician_missing_profile_pct) if physician_missing_profile_pct is not None else None,
            "hospital_id_missing_count": int(missing_hosp_id),
            "hospital_id_missing_pct": float(hospital_missing_id_pct) if hospital_missing_id_pct is not None else None,
        },
        "timings_sec": {"total": round(total_time, 4)},
        "reproducibility": {
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function": HASH_FUNCTION,
            "fingerprint_file": "dataset_fingerprint.json",
            "config_snapshot_file": "config_used.yaml",
        },
        "scaling": {"pre_canon": None, "post_record_id_sampling": sample_info},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if cfg.write_format == "parquet":
        manifest_clean_path = out_dir / "payments_clean_manifest.json"
        manifest_rejected_path = out_dir / "payments_rejected_manifest.json"

        with open(manifest_clean_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_parts": len(manifest_clean),
                "total_rows": rows_valid,
                "parts": manifest_clean,
            }, f, indent=2)

        if not cfg.skip_rejected:
            with open(manifest_rejected_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_parts": len(manifest_rejected),
                    "total_rows": rows_rejected,
                    "parts": manifest_rejected,
                }, f, indent=2)

            logging.info("Wrote manifest: %s", manifest_rejected_path)

    logging.info("Done.")
    if cfg.write_format == "parquet":
        logging.info("Wrote clean partitions: %s (%d parts)", clean_dir, part_num)
        if not cfg.skip_rejected:
            logging.info("Wrote rejected partitions: %s (%d parts)", rej_dir, part_num)
    else:
        logging.info("Wrote: %s", clean_path)
        if not cfg.skip_rejected:
            logging.info("Wrote: %s", rej_path)
    logging.info("Wrote: %s", report_path)

# Pipeline wrapper (kept compatible with your CPU script)

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
    input_candidates = list(raw_inputs or [])
    if not input_candidates:
        raise ValueError("pipeline_cfg.inputs.csv_files is required for phase1_clean")

    input_files = []
    for f in input_candidates:
        p = Path(f)
        if not p.is_absolute():
            p = config_dir / f
        input_files.append(str(p.resolve()))

    cfg = CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=input_files,
        output_dir=str(out_dir),
        chunk_size=int(phase_cfg.get("chunk_size", 500_000)),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        keep_source_file=bool(phase_cfg.get("keep_source_file", True)),
        column_mapping=phase_cfg.get("column_mapping"),
        max_workers=int(phase_cfg.get("max_workers", phase_cfg.get("num_workers", 1))),
        skip_rejected=bool(phase_cfg.get("skip_rejected", True)),
    )

    cfg_snapshot = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "input_files": cfg.input_files,
        "output_dir": cfg.output_dir,
        "chunk_size": cfg.chunk_size,
        "write_format": cfg.write_format,
        "keep_source_file": cfg.keep_source_file,
        "column_mapping": cfg.column_mapping,
        "scale_enabled": cfg.scale_enabled,
        "scale_fraction": cfg.scale_fraction,
        "scale_seed": cfg.scale_seed,
        "scale_method": cfg.scale_method,
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
            "rejected_dir": None if cfg.skip_rejected else str(out_dir / "payments_rejected"),
            "clean_manifest": str(out_dir / "payments_clean_manifest.json"),
            "rejected_manifest": None if cfg.skip_rejected else str(out_dir / "payments_rejected_manifest.json"),
            "report": str(out_dir / "cleaning_report.json"),
            "fingerprint": str(out_dir / "dataset_fingerprint.json"),
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config for Phase 1 cleaning.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")
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
