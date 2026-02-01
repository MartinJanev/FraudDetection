#!/usr/bin/env python3
"""
Phase 1 (Sequential): Extract + Clean CMS Open Payments into canonical schema.

Outputs:
- payments_clean.parquet (valid rows)
- payments_rejected.parquet (invalid rows + drop_reason)
- cleaning_report.json (summary stats)

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

# ---------------------------
# Reproducibility Constants
# ---------------------------

# Normalization version for deterministic name/string processing
NORMALIZATION_VERSION = "v1"
NORMALIZATION_DESCRIPTION = "punctuation-strip + uppercase + whitespace collapse"

# Hash function used for stable record IDs and payee keys
HASH_FUNCTION = "sha1"

# Number of MB to hash for input file fingerprinting (fast, not full file)
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
    chunk_size: int = 500_000  # adjust based on RAM
    write_format: str = "parquet"  # "parquet" or "csv"
    keep_source_file: bool = True
    column_mapping: Optional[Dict[str, str]] = None
    max_workers: int = 1
    skip_rejected: bool = False


def load_config(path: str) -> CleanConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return CleanConfig(
        dataset_type=str(raw["dataset_type"]),
        program_year=int(raw["program_year"]),
        input_files=list(raw["input_files"]),
        output_dir=str(raw["output_dir"]),
        chunk_size=int(raw.get("chunk_size", 500_000)),
        write_format=str(raw.get("write_format", "parquet")),
        keep_source_file=bool(raw.get("keep_source_file", True)),
        column_mapping=raw.get("column_mapping"),
        max_workers=int(raw.get("max_workers", 1)),
        skip_rejected=bool(raw.get("skip_rejected", False)),
    )


# ---------------------------
# Dataset Fingerprinting (for reproducibility)
# ---------------------------

def compute_file_fingerprint(filepath: str, hash_mb: int = FINGERPRINT_MB) -> Dict:
    """
    Compute a lightweight fingerprint for an input file:
    - File size (bytes)
    - Last modified timestamp
    - Fast hash of first N MB (not full file for performance)

    Args:
        filepath: Path to input file
        hash_mb: Number of megabytes to hash from start of file

    Returns:
        Dictionary with file fingerprint metadata
    """
    path = Path(filepath)

    if not path.exists():
        return {
            "file": str(path.name),
            "exists": False,
            "error": "File not found"
        }

    stat = path.stat()
    file_size = stat.st_size
    modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    # Hash first N MB for fast fingerprint (not full file)
    hash_obj = hashlib.sha1()
    bytes_to_hash = hash_mb * 1024 * 1024
    bytes_hashed = 0

    try:
        with open(filepath, 'rb') as f:
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
    """
    Create a complete fingerprint for the dataset processing run.
    Includes: input files, config snapshot, normalization version, hash function.

    Args:
        config: CleanConfig object
        config_path: Path to config YAML file

    Returns:
        Dictionary with complete dataset fingerprint
    """
    input_fingerprints = []
    for fpath in config.input_files:
        fingerprint = compute_file_fingerprint(fpath)
        input_fingerprints.append(fingerprint)

    # Load raw config for snapshot
    with open(config_path, 'r', encoding='utf-8') as f:
        config_snapshot = yaml.safe_load(f)

    fingerprint = {
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
        }
    }

    return fingerprint


# ---------------------------
# Normalization helpers
# ---------------------------

_PUNCT_RE = re.compile(r"[.,;:()\[\]{}'\"`]+")


def normalize_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    s = s.strip()
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
        # Some files use commas, currency symbols, etc.
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        s = s.replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        if s == "" or s == "-" or s == ".":
            return None
        val = float(s)
        return val
    except Exception:
        return None


def parse_date(x) -> Optional[str]:
    """
    Return ISO date string YYYY-MM-DD or None.
    """
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
# Column mapping
# ---------------------------
# Official CMS Open Payments column names as documented in CMS data dictionary

CANONICAL_COLS = [
    # identity/provenance
    "record_id", "program_year", "dataset_type", "source_file",

    # payer
    "payer_name_raw", "payer_name_norm", "payer_id", "payer_state", "payer_country",

    # payee (common)
    "payee_type", "payee_key",

    # physician (nullable)
    "physician_npi", "physician_profile_id",
    "physician_name_raw", "physician_name_norm",
    "physician_specialty", "physician_primary_type",
    "physician_state", "physician_zip5",

    # hospital (nullable)
    "teaching_hospital_id", "teaching_hospital_name_raw", "teaching_hospital_name_norm",
    "teaching_hospital_state", "teaching_hospital_zip5",

    # payment facts
    "amount_usd", "payment_date", "payment_quarter",
    "nature_of_payment", "form_of_payment", "payment_context",
    "product_name", "associated_covered_drug_or_device_flag", "is_product_related",

    # integrity
    "is_valid", "drop_reason",
]

# CMS Open Payments official column mappings by dataset type
# Source: CMS Open Payments Data Dictionary
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
        # Core fields
        "amount": "Total_Amount_Invested_USDollars",
        "payer_name": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        # Ownership uses "Physician_*" prefix, not "Covered_Recipient_*"
        "phys_npi": "Physician_NPI",
        "phys_profile_id": "Physician_Profile_ID",
        "phys_last": "Physician_Last_Name",
        "phys_first": "Physician_First_Name",
        "phys_middle": "Physician_Middle_Name",
        "phys_specialty": "Physician_Specialty",
        "phys_primary_type": "Physician_Primary_Type",
        "phys_state": "Recipient_State",
        "phys_zip": "Recipient_Zip_Code",
        # Note: Ownership data does NOT have:
        # - date (no Date_Ownership_Interest_Acquired in actual CSV)
        # - nature, form, context, product, drug_device_flag (not applicable to ownership)
        # - hospital_id, hospital_name, hospital_ccn (not in ownership data)
        # - recipient_type (not in ownership data)
    },
}


def get_column_mapping(df: pd.DataFrame, dataset_type: str, custom_mapping: Optional[Dict[str, str]] = None) -> Dict[
    str, Optional[str]]:
    """
    Get the column mapping for the dataset using official CMS column names.

    Args:
        df: DataFrame with actual column names
        dataset_type: Type of dataset (general_payment, research_payment, ownership)
        custom_mapping: Optional custom column mapping from config file

    Returns:
        Dict mapping internal keys to actual column names in the DataFrame

    Raises:
        ValueError: If dataset_type is not recognized or required columns are missing
    """
    if custom_mapping:
        # Use custom mapping from config if provided
        base_mapping = custom_mapping
    elif dataset_type in CMS_COLUMN_MAPPINGS:
        # Use official CMS mapping
        base_mapping = CMS_COLUMN_MAPPINGS[dataset_type]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be one of: {list(CMS_COLUMN_MAPPINGS.keys())}")

    # Verify that mapped columns exist in the actual DataFrame
    actual_cols = set(df.columns)
    result = {}
    missing_cols = []

    for key, col_name in base_mapping.items():
        if col_name in actual_cols:
            result[key] = col_name
        else:
            result[key] = None
            # Only track as missing if it's a critical column
            if key in ["amount", "payer_name"]:
                missing_cols.append(col_name)

    if missing_cols:
        raise ValueError(
            f"Required columns missing from CSV file: {missing_cols}\n"
            f"Available columns: {sorted(actual_cols)}"
        )

    return result


# ---------------------------
# Canonicalization
# ---------------------------

def build_payee_fields_vectorized(df: pd.DataFrame, colmap: Dict[str, Optional[str]]) -> Dict[str, pd.Series]:
    """
    Vectorized payee field construction. Much faster than row-wise iteration.

    Determines payee type based on:
    1. Teaching_Hospital_ID or Teaching_Hospital_Name presence
    2. Covered_Recipient_Type field (if available)
    3. Defaults to physician

    Returns dict of Series for all payee-related columns.
    """
    n = len(df)

    # Get columns (with safe defaults)
    hosp_id_col = colmap.get("hospital_id")
    hosp_name_col = colmap.get("hospital_name")
    recipient_type_col = colmap.get("recipient_type")

    # Determine if each row is a hospital
    # Priority: explicit hospital ID/name, then recipient_type field
    is_hospital = pd.Series([False] * n, index=df.index)

    # FIX 1: Exclude NaN strings and check nullness before casting
    # After astype(str), NaN becomes "nan" which is not empty, causing misclassification
    if hosp_id_col and hosp_id_col in df.columns:
        hosp_id_orig = df[hosp_id_col]
        hosp_id_mask = (
                hosp_id_orig.notna() &
                hosp_id_orig.astype(str).str.strip().ne("") &
                hosp_id_orig.astype(str).str.lower().ne("nan")
        )
        is_hospital |= hosp_id_mask

    if hosp_name_col and hosp_name_col in df.columns:
        hosp_name_orig = df[hosp_name_col]
        hosp_name_mask = (
                hosp_name_orig.notna() &
                hosp_name_orig.astype(str).str.strip().ne("") &
                hosp_name_orig.astype(str).str.lower().ne("nan")
        )
        is_hospital |= hosp_name_mask

    # Also check recipient_type for "Teaching Hospital" variants
    if recipient_type_col and recipient_type_col in df.columns:
        recip_type = df[recipient_type_col].astype(str).str.upper()
        is_hospital |= recip_type.str.contains("HOSPITAL", na=False)

    # Build hospital fields (vectorized)
    hosp_id_series = df[hosp_id_col].astype(
        str).str.strip() if hosp_id_col and hosp_id_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_id_series = hosp_id_series.replace("", None).replace("nan", None)

    hosp_name_raw = df[hosp_name_col].astype(
        str).str.strip() if hosp_name_col and hosp_name_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_name_raw = hosp_name_raw.replace("", None).replace("nan", None)
    hosp_name_norm = hosp_name_raw.map(lambda x: normalize_name(x) if x else None)

    # Hospital location uses recipient location fields
    state_col = colmap.get("phys_state")
    zip_col = colmap.get("phys_zip")
    hosp_state = df[state_col] if state_col and state_col in df.columns else pd.Series([None] * n, index=df.index)
    hosp_zip5 = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n,
                                                                                                    index=df.index)

    # Build physician fields (vectorized)
    npi_col = colmap.get("phys_npi")
    profile_col = colmap.get("phys_profile_id")

    # Clean NPI: strip non-digits
    npi_series = df[npi_col] if npi_col and npi_col in df.columns else pd.Series([None] * n, index=df.index)
    npi_clean = npi_series.astype(str).str.replace(r"\D+", "", regex=True)
    npi_clean = npi_clean.replace("", None).replace("nan", None)

    # Clean Profile ID
    profile_series = df[profile_col] if profile_col and profile_col in df.columns else pd.Series([None] * n,
                                                                                                 index=df.index)
    profile_clean = profile_series.astype(str).str.strip()
    profile_clean = profile_clean.replace("", None).replace("nan", None)

    # Build physician name (vectorized concatenation)
    first_col = colmap.get("phys_first")
    middle_col = colmap.get("phys_middle")
    last_col = colmap.get("phys_last")

    first = df[first_col].astype(str).str.strip() if first_col and first_col in df.columns else pd.Series([""] * n,
                                                                                                          index=df.index)
    middle = df[middle_col].astype(str).str.strip() if middle_col and middle_col in df.columns else pd.Series([""] * n,
                                                                                                              index=df.index)
    last = df[last_col].astype(str).str.strip() if last_col and last_col in df.columns else pd.Series([""] * n,
                                                                                                      index=df.index)

    # Concatenate name parts (handle NaN values from CSV)
    first = first.replace("nan", "").replace("NaN", "")
    middle = middle.replace("nan", "").replace("NaN", "")
    last = last.replace("nan", "").replace("NaN", "")

    phys_name_raw = (first + " " + middle + " " + last).str.replace(r"\s+", " ", regex=True).str.strip()
    phys_name_raw = phys_name_raw.replace("", None).replace("nan", None).replace("NaN", None)
    phys_name_norm = phys_name_raw.map(lambda x: normalize_name(x) if x else None)

    # Other physician fields
    specialty_col = colmap.get("phys_specialty")
    primary_type_col = colmap.get("phys_primary_type")

    phys_specialty = df[specialty_col] if specialty_col and specialty_col in df.columns else pd.Series([None] * n,
                                                                                                       index=df.index)
    phys_primary_type = df[primary_type_col] if primary_type_col and primary_type_col in df.columns else pd.Series(
        [None] * n, index=df.index)
    phys_state = df[state_col] if state_col and state_col in df.columns else pd.Series([None] * n, index=df.index)
    phys_zip5 = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n,
                                                                                                    index=df.index)

    # Build payee_key (priority: NPI > Profile > Hash)
    # This requires some row-wise logic for hash fallback, but most rows have NPI/Profile
    payee_key = pd.Series([None] * n, index=df.index, dtype=object)
    payee_type = pd.Series(["physician"] * n, index=df.index)

    # Hospitals
    payee_type = payee_type.where(~is_hospital, "teaching_hospital")

    # For hospitals: use hospital_id or hash(name+zip)
    for idx in df.index[is_hospital]:
        hid = hosp_id_series.loc[idx]
        # Check for actual None or pandas NA
        if pd.notna(hid) and str(hid).strip() and str(hid).lower() != 'nan':
            payee_key.loc[idx] = f"HOSP_ID:{hid}"
        else:
            hname = hosp_name_norm.loc[idx]
            hzip = hosp_zip5.loc[idx]
            payee_key.loc[
                idx] = f"HOSP_NAMEZIP:{stable_hash_hex([hname if pd.notna(hname) else None, hzip if pd.notna(hzip) else None])}"

    # For physicians: use NPI > Profile > hash(name+zip)
    for idx in df.index[~is_hospital]:
        npi = npi_clean.loc[idx]
        prof = profile_clean.loc[idx]

        # Check for valid NPI (not None, not empty, not 'nan' string)
        if pd.notna(npi) and str(npi).strip() and str(npi).lower() != 'nan':
            payee_key.loc[idx] = f"PHYS_NPI:{npi}"
        elif pd.notna(prof) and str(prof).strip() and str(prof).lower() != 'nan':
            payee_key.loc[idx] = f"PHYS_PROF:{prof}"
        else:
            pname = phys_name_norm.loc[idx]
            pzip = phys_zip5.loc[idx]
            payee_key.loc[
                idx] = f"PHYS_NAMEZIP:{stable_hash_hex([pname if pd.notna(pname) else None, pzip if pd.notna(pzip) else None])}"

    # Return all fields with appropriate nulls based on payee_type
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


def canonicalize_chunk(df: pd.DataFrame, cfg: CleanConfig, source_file: str) -> pd.DataFrame:
    """
    Transform raw CMS Open Payments data into canonical schema.
    Uses factual CMS column mappings - no assumptions.
    """
    # Reset index to avoid alignment issues with chunked reading
    df = df.reset_index(drop=True)

    colmap = get_column_mapping(df, cfg.dataset_type, cfg.column_mapping)

    # Extract fields using mapped column names (no fallback assumptions)
    payer_name_col = colmap.get("payer_name")
    if not payer_name_col:
        raise ValueError(f"Required column 'payer_name' not found in dataset_type '{cfg.dataset_type}'")

    amount_col = colmap.get("amount")
    if not amount_col:
        raise ValueError(f"Required column 'amount' not found in dataset_type '{cfg.dataset_type}'")

    payer_raw = df[payer_name_col].astype("string", errors="ignore")
    payer_norm = payer_raw.map(lambda x: normalize_name(x))

    amount_raw = df[amount_col]
    amount = amount_raw.map(safe_float).astype("float64")

    date_col = colmap.get("date")
    date_raw = df[date_col] if date_col else pd.Series([None] * len(df), index=df.index)
    payment_date = date_raw.map(parse_date)

    # Optional fields - may not exist in all dataset types
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

    # Build payee fields (vectorized for performance)
    payee_fields = build_payee_fields_vectorized(df, colmap)

    # Normalize product flag into boolean
    flag_clean = flag.astype(str).str.strip().str.upper()
    is_product_related = (
            flag_clean.notna() &
            (flag_clean != "") &
            (flag_clean != "NAN") &
            (flag_clean != "NONE")
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
            amount.notna() &
            (amount >= 0) &
            payer_norm.notna() &
            (payer_norm.astype(str).str.len() > 0) &
            payee_fields["payee_key"].notna() &
            (payee_fields["payee_key"].astype(str).str.len() > 0)
    )

    drop_reason = pd.Series([None] * len(df), dtype="string", index=df.index)
    drop_reason = drop_reason.mask(amount.isna(), "missing_amount")
    drop_reason = drop_reason.mask(amount.notna() & (amount < 0), "negative_amount")
    drop_reason = drop_reason.mask(payer_norm.isna() | (payer_norm.astype(str).str.len() == 0), "missing_payer")
    drop_reason = drop_reason.mask(
        payee_fields["payee_key"].isna() | (payee_fields["payee_key"].astype(str).str.len() == 0), "missing_payee")
    drop_reason = drop_reason.where(~is_valid, drop_reason)

    hash_inputs = []
    for i in range(len(payer_norm)):
        hash_input = [
            str(cfg.program_year),
            cfg.dataset_type,
            payer_norm.iloc[i],
            payee_fields["payee_key"].iloc[i],
            str(amount.iloc[i]) if pd.notna(amount.iloc[i]) else None,
            payment_date.iloc[i],
            str(nature.iloc[i]) if pd.notna(nature.iloc[i]) else None,
        ]
        hash_inputs.append(hash_input)

    record_id = pd.Series([stable_hash_hex(h) for h in hash_inputs], index=df.index, dtype="string")

    out = pd.DataFrame({
        "record_id": record_id,
        "program_year": np.int16(cfg.program_year),
        "dataset_type": cfg.dataset_type,
        "source_file": source_file if cfg.keep_source_file else None,

        "payer_name_raw": payer_raw,
        "payer_name_norm": payer_norm,
        "payer_id": payer_id.astype("string", errors="ignore"),
        "payer_state": payer_state.astype("string", errors="ignore"),
        "payer_country": payer_country.astype("string", errors="ignore"),

        "payee_type": payee_fields["payee_type"],
        "payee_key": payee_fields["payee_key"],

        "physician_npi": payee_fields["physician_npi"].astype("string", errors="ignore"),
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
        "payment_date": payment_date,
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

    # Ensure all canonical columns exist
    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = None

    return out[CANONICAL_COLS]


# ---------------------------
# IO helpers
# ---------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: str, fmt: str) -> None:
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def write_partitioned_parquet(df: pd.DataFrame, output_dir: Path, part_num: int) -> None:
    """
    Write a single parquet partition. Much faster than append for large datasets.
    Phase 2 can read all parts as a dataset.
    """
    part_file = output_dir / f"part-{part_num:05d}.parquet"
    df.to_parquet(part_file, index=False)


# ---------------------------
# Main
# ---------------------------

def run(cfg: CleanConfig, config_path: str = None) -> None:
    """
    Phase 1: Clean CMS Open Payments data into canonical schema.

    Args:
        cfg: CleanConfig with processing parameters
        config_path: Path to config file (for fingerprinting)
    """
    out_dir = Path(cfg.output_dir)
    ensure_dir(str(out_dir))

    # Create dataset fingerprint for reproducibility
    if config_path:
        logging.info("Creating dataset fingerprint for reproducibility...")
        fingerprint = create_dataset_fingerprint(cfg, config_path)
        fingerprint_path = out_dir / "dataset_fingerprint.json"
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)
        logging.info("Wrote dataset fingerprint: %s", fingerprint_path)

        config_copy_path = out_dir / "config_used.yaml"
        with open(config_path, 'r', encoding='utf-8') as src:
            with open(config_copy_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        logging.info("Saved config snapshot: %s", config_copy_path)

    # Use partitioned output for parquet (much faster)
    clean_dir = out_dir / "payments_clean"
    rej_dir = out_dir / "payments_rejected"
    clean_path = out_dir / "payments_clean.csv"
    rej_path = out_dir / "payments_rejected.csv"

    if cfg.write_format == "parquet":
        ensure_dir(str(clean_dir))
        if not cfg.skip_rejected:
            ensure_dir(str(rej_dir))
        # clear
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

    # Extended statistics for paper
    physician_count = 0
    hospital_count = 0
    missing_npi = 0
    missing_profile = 0
    missing_hosp_id = 0

    # FIX 3: Track manifest for part files (reproducibility + Phase 2 debugging)
    manifest_clean = []
    manifest_rejected = []

    logging.info("Starting Phase 1 clean. year=%s dataset_type=%s", cfg.program_year, cfg.dataset_type)

    t_phase_start = time.perf_counter()

    for fpath in cfg.input_files:
        fpath = str(fpath)
        logging.info("Reading file: %s", fpath)

        # Count total rows first for accurate progress bar
        logging.info("Counting rows in %s...", Path(fpath).name)
        total_rows = sum(1 for _ in open(fpath, 'r', encoding='utf-8')) - 1  # subtract header
        total_chunks = (total_rows + cfg.chunk_size - 1) // cfg.chunk_size
        logging.info("File has %d rows (%d chunks of size %d)", total_rows, total_chunks, cfg.chunk_size)

        # Robust CSV reading with explicit parameters
        reader = pd.read_csv(
            fpath,
            chunksize=cfg.chunk_size,
            low_memory=False,
            dtype=str,  # ingest as string first; we parse deterministically
            encoding="utf-8",
            sep=",",
            engine="c",
        )

        for chunk in tqdm(reader, total=total_chunks, desc=f"Processing {Path(fpath).name}", unit="chunk"):
            rows_in += len(chunk)
            canon = canonicalize_chunk(chunk, cfg, source_file=Path(fpath).name)

            valid = canon[canon["is_valid"]].copy()
            rej = canon[~canon["is_valid"]].copy()

            rows_valid += len(valid)
            rows_rejected += len(rej)

            if len(valid) > 0:
                amounts.append(valid["amount_usd"].to_numpy())

                # Track payee types
                physician_count += (valid["payee_type"] == "physician").sum()
                hospital_count += (valid["payee_type"] == "teaching_hospital").sum()

                # FIX 2: Track missing IDs on correct subgroup
                # Physician fields should only be checked on physician rows
                # Hospital fields should only be checked on hospital rows
                phys_rows = valid[valid["payee_type"] == "physician"]
                hosp_rows = valid[valid["payee_type"] == "teaching_hospital"]

                missing_npi += phys_rows["physician_npi"].isna().sum()
                missing_profile += phys_rows["physician_profile_id"].isna().sum()
                missing_hosp_id += hosp_rows["teaching_hospital_id"].isna().sum()

            # Write outputs incrementally
            if cfg.write_format == "parquet":
                # Track manifest entries for reproducibility
                clean_rows = len(valid)
                rej_rows = len(rej)

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
                # CSV append (write header only once)
                header = not clean_path.exists()
                if len(valid) > 0:
                    valid.to_csv(clean_path, mode="a", index=False, header=header)
                header2 = not rej_path.exists()
                if len(rej) > 0 and not cfg.skip_rejected:
                    rej.to_csv(rej_path, mode="a", index=False, header=header2)

    total_time = time.perf_counter() - t_phase_start

    # Summaries
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

    # Calculate missingness rates for key identifiers
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

        # Payee type breakdown
        "payee_counts": {
            "physicians": int(physician_count),
            "teaching_hospitals": int(hospital_count),
        },

        # Identifier completeness (for Dataset Quality Summary)
        "identifier_missingness": {
            "physician_npi_missing_count": int(missing_npi),
            "physician_npi_missing_pct": float(
                physician_missing_npi_pct) if physician_missing_npi_pct is not None else None,
            "physician_profile_missing_count": int(missing_profile),
            "physician_profile_missing_pct": float(
                physician_missing_profile_pct) if physician_missing_profile_pct is not None else None,
            "hospital_id_missing_count": int(missing_hosp_id),
            "hospital_id_missing_pct": float(hospital_missing_id_pct) if hospital_missing_id_pct is not None else None,
        },

        "timings_sec": {
            "total": round(total_time, 4),
        },

        "reproducibility": {
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function": HASH_FUNCTION,
            "fingerprint_file": "dataset_fingerprint.json",
            "config_snapshot_file": "config_used.yaml",
        },

        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # FIX 3: Write manifest files for part tracking
    # Save manifests in parent directory to avoid interfering with parquet reads
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

    # Persist minimal phase config snapshot for fingerprinting
    cfg_snapshot = {
        "dataset_type": cfg.dataset_type,
        "program_year": cfg.program_year,
        "input_files": cfg.input_files,
        "output_dir": cfg.output_dir,
        "chunk_size": cfg.chunk_size,
        "write_format": cfg.write_format,
        "keep_source_file": cfg.keep_source_file,
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

    # Allow passing bare config names by resolving against ./configs
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
