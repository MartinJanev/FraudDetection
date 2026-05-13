"""sequentialCPU/transforms.py

Row-level and chunk-level data transformation functions for Phase 1 cleaning.

Functions
---------
build_payee_fields_vectorized
    Derive all payee-related canonical columns from a raw chunk DataFrame in a
    single vectorised pass.  Physician and hospital payees are handled together
    without Python-level loops over rows (unlike the original sequential code).

canonicalize_chunk
    Full chunk → canonical schema transformation: resolves column mapping,
    parses amounts / dates, builds payee fields, computes record IDs and
    validity flags.

No Numba, no threading, no multiprocessing — pure pandas + Python stdlib.
Every hot path that the concurrentNumba pipeline accelerates with a JIT
kernel is implemented here as an equivalent pandas/NumPy expression, giving
identical *results* with single-core execution.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CleanConfig
from .normalizers import normalize_name, normalize_zip5, safe_float, parse_date, stable_hash_hex
from .schema import CANONICAL_COLS, get_column_mapping


# ---------------------------------------------------------------------------
# Payee field derivation (fully vectorised — no row-wise Python loops)
# ---------------------------------------------------------------------------

def build_payee_fields_vectorized(
    df: pd.DataFrame,
    colmap: Dict[str, Optional[str]],
) -> Dict[str, pd.Series]:
    """Derive all payee-related canonical columns from a raw chunk.

    A single vectorised pass determines whether each row represents a
    *physician* or a *teaching hospital* payee, then populates the
    corresponding identifier / name columns.

    Payee-key priority
    ------------------
    Teaching hospital:
      1. ``HOSP_ID:<id>``
      2. ``HOSP_NAMEZIP:<sha1(name_norm, zip5)>``

    Physician:
      1. ``PHYS_NPI:<npi>``
      2. ``PHYS_PROF:<profile_id>``
      3. ``PHYS_NAMEZIP:<sha1(name_norm, zip5)>``

    Parameters
    ----------
    df:
        Raw chunk (index must be a clean integer range after ``reset_index``).
    colmap:
        Logical→physical column mapping (from ``get_column_mapping``).

    Returns
    -------
    Dict of ``pd.Series`` keyed by canonical column name.
    """
    n = len(df)

    hosp_id_col        = colmap.get("hospital_id")
    hosp_name_col      = colmap.get("hospital_name")
    recipient_type_col = colmap.get("recipient_type")

    # ── Determine is_hospital mask ──────────────────────────────────────────
    is_hospital = pd.Series([False] * n, index=df.index)

    if hosp_id_col and hosp_id_col in df.columns:
        orig = df[hosp_id_col]
        is_hospital |= (
            orig.notna()
            & orig.astype(str).str.strip().ne("")
            & orig.astype(str).str.lower().ne("nan")
        )

    if hosp_name_col and hosp_name_col in df.columns:
        orig = df[hosp_name_col]
        is_hospital |= (
            orig.notna()
            & orig.astype(str).str.strip().ne("")
            & orig.astype(str).str.lower().ne("nan")
        )

    if recipient_type_col and recipient_type_col in df.columns:
        is_hospital |= (
            df[recipient_type_col].astype(str).str.upper().str.contains("HOSPITAL", na=False)
        )

    # ── Hospital identifier / name ──────────────────────────────────────────
    hosp_id_series = (
        df[hosp_id_col].astype(str).str.strip()
        if hosp_id_col and hosp_id_col in df.columns
        else pd.Series([None] * n, index=df.index)
    )
    hosp_id_series = hosp_id_series.replace("", None).replace("nan", None)

    hosp_name_raw = (
        df[hosp_name_col].astype(str).str.strip()
        if hosp_name_col and hosp_name_col in df.columns
        else pd.Series([None] * n, index=df.index)
    )
    hosp_name_raw  = hosp_name_raw.replace("", None).replace("nan", None)
    hosp_name_norm = hosp_name_raw.map(lambda x: normalize_name(x) if x else None)

    state_col = colmap.get("phys_state")
    zip_col   = colmap.get("phys_zip")
    hosp_state = (
        df[state_col] if state_col and state_col in df.columns
        else pd.Series([None] * n, index=df.index)
    )
    hosp_zip5 = (
        df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns
        else pd.Series([None] * n, index=df.index)
    )

    # ── Physician identifier / name ─────────────────────────────────────────
    npi_col     = colmap.get("phys_npi")
    profile_col = colmap.get("phys_profile_id")

    npi_series = (
        df[npi_col] if npi_col and npi_col in df.columns
        else pd.Series([None] * n, index=df.index)
    )
    npi_clean = npi_series.astype(str).str.replace(r"\D+", "", regex=True)
    npi_clean = npi_clean.replace("", None).replace("nan", None)

    profile_series = (
        df[profile_col] if profile_col and profile_col in df.columns
        else pd.Series([None] * n, index=df.index)
    )
    profile_clean = profile_series.astype(str).str.strip()
    profile_clean = profile_clean.replace("", None).replace("nan", None)

    first_col  = colmap.get("phys_first")
    middle_col = colmap.get("phys_middle")
    last_col   = colmap.get("phys_last")

    first  = df[first_col].astype(str).str.strip()  if first_col  and first_col  in df.columns else pd.Series([""] * n, index=df.index)
    middle = df[middle_col].astype(str).str.strip() if middle_col and middle_col in df.columns else pd.Series([""] * n, index=df.index)
    last   = df[last_col].astype(str).str.strip()   if last_col   and last_col   in df.columns else pd.Series([""] * n, index=df.index)

    first  = first.replace("nan", "").replace("NaN", "")
    middle = middle.replace("nan", "").replace("NaN", "")
    last   = last.replace("nan", "").replace("NaN", "")

    phys_name_raw  = (first + " " + middle + " " + last).str.replace(r"\s+", " ", regex=True).str.strip()
    phys_name_raw  = phys_name_raw.replace("", None).replace("nan", None).replace("NaN", None)
    phys_name_norm = phys_name_raw.map(lambda x: normalize_name(x) if x else None)

    specialty_col     = colmap.get("phys_specialty")
    primary_type_col  = colmap.get("phys_primary_type")
    phys_specialty    = df[specialty_col]    if specialty_col    and specialty_col    in df.columns else pd.Series([None] * n, index=df.index)
    phys_primary_type = df[primary_type_col] if primary_type_col and primary_type_col in df.columns else pd.Series([None] * n, index=df.index)
    phys_state        = df[state_col]        if state_col        and state_col        in df.columns else pd.Series([None] * n, index=df.index)
    phys_zip5         = df[zip_col].map(normalize_zip5) if zip_col and zip_col in df.columns else pd.Series([None] * n, index=df.index)

    # ── Build payee_key (fully vectorised) ──────────────────────────────────
    payee_key  = pd.Series([None] * n, index=df.index, dtype=object)
    payee_type = pd.Series(["physician"] * n, index=df.index)
    payee_type = payee_type.where(~is_hospital, "teaching_hospital")

    # Hospital keys
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
            hzip  = hosp_zip5.loc[hosp_hash_idx]
            payee_key.loc[hosp_hash_idx] = [
                f"HOSP_NAMEZIP:{stable_hash_hex([hn if pd.notna(hn) else None, hz if pd.notna(hz) else None])}"
                for hn, hz in zip(hname, hzip)
            ]

    # Physician keys
    phys_idx = df.index[~is_hospital]
    if len(phys_idx):
        npi_p  = npi_clean.loc[phys_idx]
        prof_p = profile_clean.loc[phys_idx]
        npi_mask = (
            npi_p.notna()
            & npi_p.astype(str).str.strip().ne("")
            & npi_p.astype(str).str.lower().ne("nan")
        )
        prof_mask = (
            prof_p.notna()
            & prof_p.astype(str).str.strip().ne("")
            & prof_p.astype(str).str.lower().ne("nan")
        )
        if npi_mask.any():
            payee_key.loc[phys_idx[npi_mask]] = (
                "PHYS_NPI:" + npi_p.loc[phys_idx[npi_mask]].astype(str)
            )
        prof_only_idx = phys_idx[~npi_mask & prof_mask]
        if len(prof_only_idx):
            payee_key.loc[prof_only_idx] = (
                "PHYS_PROF:" + prof_p.loc[prof_only_idx].astype(str)
            )
        hash_idx = phys_idx[~npi_mask & ~prof_mask]
        if len(hash_idx):
            pname = phys_name_norm.loc[hash_idx]
            pzip  = phys_zip5.loc[hash_idx]
            payee_key.loc[hash_idx] = [
                f"PHYS_NAMEZIP:{stable_hash_hex([pn if pd.notna(pn) else None, pz if pd.notna(pz) else None])}"
                for pn, pz in zip(pname, pzip)
            ]

    return {
        "payee_type":               payee_type,
        "payee_key":                payee_key,
        "physician_npi":            npi_clean.where(~is_hospital, None),
        "physician_profile_id":     profile_clean.where(~is_hospital, None),
        "physician_name_raw":       phys_name_raw.where(~is_hospital, None),
        "physician_name_norm":      phys_name_norm.where(~is_hospital, None),
        "physician_specialty":      phys_specialty.where(~is_hospital, None),
        "physician_primary_type":   phys_primary_type.where(~is_hospital, None),
        "physician_state":          phys_state.where(~is_hospital, None),
        "physician_zip5":           phys_zip5.where(~is_hospital, None),
        "teaching_hospital_id":         hosp_id_series.where(is_hospital, None),
        "teaching_hospital_name_raw":   hosp_name_raw.where(is_hospital, None),
        "teaching_hospital_name_norm":  hosp_name_norm.where(is_hospital, None),
        "teaching_hospital_state":      hosp_state.where(is_hospital, None),
        "teaching_hospital_zip5":       hosp_zip5.where(is_hospital, None),
    }


def _build_record_id_series(
    program_year: int,
    dataset_type: str,
    payer_norm: pd.Series,
    payee_key: pd.Series,
    amount: pd.Series,
    payment_date: pd.Series,
    nature: pd.Series,
) -> pd.Series:
    """Vectorised record-ID from concatenated key fields."""
    sep = "|"
    key = (
        str(program_year) + sep + dataset_type + sep
        + payer_norm.fillna("").astype(str) + sep
        + payee_key.fillna("").astype(str) + sep
        + amount.where(amount.notna(), "").astype(str) + sep
        + payment_date.fillna("").astype(str) + sep
        + nature.where(nature.notna(), "").astype(str)
    )
    return key.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()).astype("string[python]")


def _compute_validity_flags(
    amount: pd.Series,
    payer_norm: pd.Series,
    payee_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return (is_valid, drop_reason) based on core required fields."""
    is_valid = (
        amount.notna()
        & (amount >= 0)
        & payer_norm.notna()
        & (payer_norm.astype(str).str.len() > 0)
        & payee_key.notna()
        & (payee_key.astype(str).str.len() > 0)
    )

    drop_reason = pd.Series([None] * len(amount), dtype="string", index=amount.index)
    drop_reason = drop_reason.mask(amount.isna(), "missing_amount")
    drop_reason = drop_reason.mask(amount.notna() & (amount < 0), "negative_amount")
    drop_reason = drop_reason.mask(
        payer_norm.isna() | (payer_norm.astype(str).str.len() == 0), "missing_payer"
    )
    drop_reason = drop_reason.mask(
        payee_key.isna() | (payee_key.astype(str).str.len() == 0), "missing_payee"
    )
    drop_reason = drop_reason.where(~is_valid, drop_reason)
    return is_valid, drop_reason


# ---------------------------------------------------------------------------
# Full chunk → canonical schema transformation
# ---------------------------------------------------------------------------

def canonicalize_chunk(
    df: pd.DataFrame,
    cfg: CleanConfig,
    source_file: str,
) -> pd.DataFrame:
    """Transform a raw CSV chunk into the canonical schema.

    Steps
    -----
    1. Resolve logical→physical column mapping.
    2. Normalise payer name.
    3. Parse and validate amount (pure pandas/NumPy — no Numba).
    4. Parse payment date → ISO-8601; derive quarter.
    5. Build all payee fields (vectorised, no row-level Python loops).
    6. Compute validity flag and drop-reason.
    7. Compute SHA-1 ``record_id`` for each row.
    8. Assemble and return a DataFrame with exactly ``CANONICAL_COLS``.

    Parameters
    ----------
    df:
        Raw chunk (all columns as ``str`` dtype from ``pd.read_csv``).
    cfg:
        Pipeline configuration.
    source_file:
        Basename of the originating CSV (written to ``source_file`` column
        when ``cfg.keep_source_file`` is ``True``).
    """
    df = df.reset_index(drop=True)
    colmap = get_column_mapping(df, cfg.dataset_type, cfg.column_mapping)

    payer_name_col = colmap.get("payer_name")
    amount_col     = colmap.get("amount")
    if not payer_name_col:
        raise ValueError(f"Required column 'payer_name' not found for dataset_type '{cfg.dataset_type}'")
    if not amount_col:
        raise ValueError(f"Required column 'amount' not found for dataset_type '{cfg.dataset_type}'")

    payer_raw  = df[payer_name_col].astype("string", errors="ignore")
    payer_norm = payer_raw.map(normalize_name)

    amount_raw = df[amount_col]
    amount     = amount_raw.map(safe_float).astype("float64")

    date_col     = colmap.get("date")
    date_raw     = df[date_col] if date_col else pd.Series([None] * len(df), index=df.index)
    payment_date = date_raw.map(parse_date)

    nature_col  = colmap.get("nature")
    form_col    = colmap.get("form")
    context_col = colmap.get("context")
    product_col = colmap.get("product")
    flag_col    = colmap.get("drug_device_flag")

    nature  = df[nature_col]  if nature_col  else pd.Series([None] * len(df), index=df.index)
    form    = df[form_col]    if form_col    else pd.Series([None] * len(df), index=df.index)
    context = df[context_col] if context_col else pd.Series([None] * len(df), index=df.index)
    product = df[product_col] if product_col else pd.Series([None] * len(df), index=df.index)
    flag    = df[flag_col]    if flag_col    else pd.Series([None] * len(df), index=df.index)

    payer_id_col      = colmap.get("payer_id")
    payer_state_col   = colmap.get("payer_state")
    payer_country_col = colmap.get("payer_country")

    payer_id      = df[payer_id_col]      if payer_id_col      else pd.Series([None] * len(df), index=df.index)
    payer_state   = df[payer_state_col]   if payer_state_col   else pd.Series([None] * len(df), index=df.index)
    payer_country = df[payer_country_col] if payer_country_col else pd.Series([None] * len(df), index=df.index)

    payee_fields = build_payee_fields_vectorized(df, colmap)

    flag_clean = flag.astype(str).str.strip().str.upper()
    is_product_related = (
        flag_clean.notna()
        & (flag_clean != "")
        & (flag_clean != "NAN")
        & (flag_clean != "NONE")
    )

    def _quarter(d: Optional[str]) -> Optional[int]:
        if d is None:
            return None
        try:
            m = int(d.split("-")[1])
            return (m - 1) // 3 + 1
        except Exception:
            return None

    payment_quarter = payment_date.map(_quarter)

    is_valid, drop_reason = _compute_validity_flags(
        amount, payer_norm, payee_fields["payee_key"]
    )

    # SHA-1 record_id (deterministic, includes key payment fields)
    record_id = _build_record_id_series(
        cfg.program_year,
        cfg.dataset_type,
        payer_norm,
        payee_fields["payee_key"],
        amount,
        payment_date,
        nature,
    )

    out = pd.DataFrame({
        "record_id":       record_id,
        "program_year":    np.int16(cfg.program_year),
        "dataset_type":    cfg.dataset_type,
        "source_file":     source_file if cfg.keep_source_file else None,
        "payer_name_raw":  payer_raw,
        "payer_name_norm": payer_norm,
        "payer_id":        payer_id.astype("string", errors="ignore"),
        "payer_state":     payer_state.astype("string", errors="ignore"),
        "payer_country":   payer_country.astype("string", errors="ignore"),
        "payee_type":      payee_fields["payee_type"],
        "payee_key":       payee_fields["payee_key"],
        "physician_npi":            payee_fields["physician_npi"].astype("string", errors="ignore"),
        "physician_profile_id":     payee_fields["physician_profile_id"].astype("string", errors="ignore"),
        "physician_name_raw":       payee_fields["physician_name_raw"].astype("string", errors="ignore"),
        "physician_name_norm":      payee_fields["physician_name_norm"].astype("string", errors="ignore"),
        "physician_specialty":      payee_fields["physician_specialty"].astype("string", errors="ignore"),
        "physician_primary_type":   payee_fields["physician_primary_type"].astype("string", errors="ignore"),
        "physician_state":          payee_fields["physician_state"].astype("string", errors="ignore"),
        "physician_zip5":           payee_fields["physician_zip5"].astype("string", errors="ignore"),
        "teaching_hospital_id":         payee_fields["teaching_hospital_id"].astype("string", errors="ignore"),
        "teaching_hospital_name_raw":   payee_fields["teaching_hospital_name_raw"].astype("string", errors="ignore"),
        "teaching_hospital_name_norm":  payee_fields["teaching_hospital_name_norm"].astype("string", errors="ignore"),
        "teaching_hospital_state":      payee_fields["teaching_hospital_state"].astype("string", errors="ignore"),
        "teaching_hospital_zip5":       payee_fields["teaching_hospital_zip5"].astype("string", errors="ignore"),
        "amount_usd":        amount,
        "payment_date":      payment_date,
        "payment_quarter":   payment_quarter,
        "nature_of_payment": nature.astype("string", errors="ignore"),
        "form_of_payment":   form.astype("string", errors="ignore"),
        "payment_context":   context.astype("string", errors="ignore"),
        "product_name":      product.astype("string", errors="ignore"),
        "associated_covered_drug_or_device_flag": flag.astype("string", errors="ignore"),
        "is_product_related": is_product_related.astype(bool),
        "is_valid":    is_valid.astype(bool),
        "drop_reason": drop_reason.astype("string", errors="ignore"),
    })

    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = None

    return out[CANONICAL_COLS]

