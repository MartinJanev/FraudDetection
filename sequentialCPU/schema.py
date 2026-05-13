"""sequentialCPU/schema.py

CMS Open Payments canonical schema definition.

Responsibilities
----------------
CANONICAL_COLS          Ordered list of columns in every output dataset.
CMS_COLUMN_MAPPINGS     Logical field key → raw CMS CSV column name, per
                        dataset type (general_payment / research_payment /
                        ownership).
make_canonical_meta     Build an empty DataFrame with the correct dtypes —
                        used as the fallback when no rows survive cleaning.
get_column_mapping      Resolve the logical→physical column map for a chunk
                        DataFrame; raises early when required columns are absent.
sanitize_for_parquet    Coerce all columns to Parquet-compatible dtypes before
                        writing.

No Numba, no threading — pure pandas + stdlib.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CleanConfig

# String dtype used in canonical metadata tables.
STRING_DTYPE = "string[python]"

# ---------------------------------------------------------------------------
# Canonical output schema
# ---------------------------------------------------------------------------

CANONICAL_COLS: List[str] = [
    # identity / provenance
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

# ---------------------------------------------------------------------------
# CMS column mappings  (logical key → raw CSV column name)
# ---------------------------------------------------------------------------

CMS_COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {
    "general_payment": {
        "amount":        "Total_Amount_of_Payment_USDollars",
        "date":          "Date_of_Payment",
        "payer_name":    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id":      "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state":   "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "phys_npi":          "Covered_Recipient_NPI",
        "phys_profile_id":   "Covered_Recipient_Profile_ID",
        "phys_last":         "Covered_Recipient_Last_Name",
        "phys_first":        "Covered_Recipient_First_Name",
        "phys_middle":       "Covered_Recipient_Middle_Name",
        "phys_specialty":    "Covered_Recipient_Specialty_1",
        "phys_primary_type": "Covered_Recipient_Primary_Type_1",
        "phys_state":        "Recipient_State",
        "phys_zip":          "Recipient_Zip_Code",
        "hospital_id":       "Teaching_Hospital_ID",
        "hospital_name":     "Teaching_Hospital_Name",
        "hospital_ccn":      "Teaching_Hospital_CCN",
        "nature":            "Nature_of_Payment_or_Transfer_of_Value",
        "form":              "Form_of_Payment_or_Transfer_of_Value",
        "context":           "Contextual_Information",
        "product":           "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "drug_device_flag":  "Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "recipient_type":    "Covered_Recipient_Type",
    },
    "research_payment": {
        "amount":        "Total_Amount_of_Payment_USDollars",
        "date":          "Date_of_Payment",
        "payer_name":    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id":      "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state":   "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "phys_npi":          "Covered_Recipient_NPI",
        "phys_profile_id":   "Covered_Recipient_Profile_ID",
        "phys_last":         "Covered_Recipient_Last_Name",
        "phys_first":        "Covered_Recipient_First_Name",
        "phys_middle":       "Covered_Recipient_Middle_Name",
        "phys_specialty":    "Covered_Recipient_Specialty_1",
        "phys_primary_type": "Covered_Recipient_Primary_Type_1",
        "phys_state":        "Recipient_State",
        "phys_zip":          "Recipient_Zip_Code",
        "hospital_id":       "Teaching_Hospital_ID",
        "hospital_name":     "Teaching_Hospital_Name",
        "hospital_ccn":      "Teaching_Hospital_CCN",
        "nature":            "Nature_of_Payment_or_Transfer_of_Value",
        "form":              "Form_of_Payment_or_Transfer_of_Value",
        "context":           "Contextual_Information",
        "product":           "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "drug_device_flag":  "Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "recipient_type":    "Covered_Recipient_Type",
    },
    "ownership": {
        # Ownership uses "Physician_*" prefix, not "Covered_Recipient_*"
        "amount":        "Total_Amount_Invested_USDollars",
        "payer_name":    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        "payer_id":      "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
        "payer_state":   "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_State",
        "payer_country": "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "phys_npi":          "Physician_NPI",
        "phys_profile_id":   "Physician_Profile_ID",
        "phys_last":         "Physician_Last_Name",
        "phys_first":        "Physician_First_Name",
        "phys_middle":       "Physician_Middle_Name",
        "phys_specialty":    "Physician_Specialty",
        "phys_primary_type": "Physician_Primary_Type",
        "phys_state":        "Recipient_State",
        "phys_zip":          "Recipient_Zip_Code",
        # Ownership does NOT have: date, nature, form, context, product,
        # drug_device_flag, hospital_*, recipient_type
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canonical_meta(_: CleanConfig) -> pd.DataFrame:
    """Return an empty DataFrame whose dtypes match the canonical schema.

    Used as the fallback output when no records survive cleaning.
    """
    S = STRING_DTYPE
    dtypes = {
        "record_id":     S, "program_year": "Int16", "dataset_type": S, "source_file": S,
        "payer_name_raw": S, "payer_name_norm": S,
        "payer_id": S, "payer_state": S, "payer_country": S,
        "payee_type": S, "payee_key": S,
        "physician_npi": S, "physician_profile_id": S,
        "physician_name_raw": S, "physician_name_norm": S,
        "physician_specialty": S, "physician_primary_type": S,
        "physician_state": S, "physician_zip5": S,
        "teaching_hospital_id": S, "teaching_hospital_name_raw": S,
        "teaching_hospital_name_norm": S, "teaching_hospital_state": S,
        "teaching_hospital_zip5": S,
        "amount_usd": "float64", "payment_date": S, "payment_quarter": "Int8",
        "nature_of_payment": S, "form_of_payment": S, "payment_context": S,
        "product_name": S, "associated_covered_drug_or_device_flag": S,
        "is_product_related": "bool", "is_valid": "bool", "drop_reason": S,
    }
    data = {c: pd.Series([], dtype=dtypes.get(c, S)) for c in CANONICAL_COLS}
    return pd.DataFrame(data)


def get_column_mapping(
    df: pd.DataFrame,
    dataset_type: str,
    custom_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """Resolve the logical→physical column map for a chunk DataFrame.

    Parameters
    ----------
    df:
        The raw chunk read from CSV.
    dataset_type:
        One of ``"general_payment"``, ``"research_payment"``, ``"ownership"``.
    custom_mapping:
        Optional override mapping supplied via config.

    Returns
    -------
    Dict mapping logical key → actual CSV column name (or ``None`` when a
    non-required column is absent).

    Raises
    ------
    ValueError
        If *dataset_type* is unknown, or if required columns (``amount``,
        ``payer_name``) are absent from *df*.
    """
    base_mapping = custom_mapping if custom_mapping else CMS_COLUMN_MAPPINGS.get(dataset_type)
    if base_mapping is None:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. "
            f"Must be one of: {list(CMS_COLUMN_MAPPINGS.keys())}"
        )

    actual_cols = set(df.columns)
    result: Dict[str, Optional[str]] = {}
    missing_cols: List[str] = []

    for key, col_name in base_mapping.items():
        if col_name in actual_cols:
            result[key] = col_name
        else:
            result[key] = None
            if key in {"amount", "payer_name"}:
                missing_cols.append(col_name)

    if missing_cols:
        raise ValueError(
            f"Required columns missing from CSV: {missing_cols}\n"
            f"Available columns: {sorted(actual_cols)}"
        )
    return result


def sanitize_for_parquet(pdf: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to Parquet-compatible dtypes.

    Column-type rules
    -----------------
    * ``amount_usd``                    → float64
    * ``is_product_related``, ``is_valid`` → bool (NaN → False)
    * ``payment_quarter``               → Int8  (nullable integer)
    * ``program_year``                  → Int16 (nullable integer)
    * Everything else                   → ``string[python]`` (NaN → None)
    """
    if pdf is None or len(pdf) == 0:
        return pdf

    float_cols   = {"amount_usd"}
    bool_cols    = {"is_product_related", "is_valid"}
    int8_cols    = {"payment_quarter"}
    int16_cols   = {"program_year"}

    def _clean_cell(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return x if isinstance(x, str) else str(x)

    for c in pdf.columns:
        s = pdf[c]
        if c in float_cols:
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif c in bool_cols:
            if s.dtype.name == "boolean":
                pdf[c] = s.fillna(False).astype(bool)
            else:
                try:
                    pdf[c] = s.fillna(False).map(bool).astype(bool)
                except Exception:
                    pdf[c] = False
        elif c in int8_cols:
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("Int8")
        elif c in int16_cols:
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("Int16")
        else:
            pdf[c] = s.map(_clean_cell).astype("string[python]")

    return pdf

