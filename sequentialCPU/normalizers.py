"""sequentialCPU/normalizers.py

Stateless scalar normalisation helpers used throughout Phase 1 cleaning.

All functions accept a single scalar and return an Optional value.  They are
called via ``pandas.Series.map()`` on full columns inside
``canonicalize_chunk`` — no threading, no Numba, pure Python + stdlib.

Constants
---------
NORMALIZATION_VERSION       Bumped whenever the normalisation logic changes,
                            written into the cleaning report for reproducibility.
NORMALIZATION_DESCRIPTION   Human-readable summary of the current strategy.
HASH_FUNCTION               Algorithm name used by ``stable_hash_hex``.
FINGERPRINT_MB              Block size (MB) hashed for fast file fingerprinting.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Reproducibility constants
# ---------------------------------------------------------------------------

NORMALIZATION_VERSION = "v1"
NORMALIZATION_DESCRIPTION = "punctuation-strip + uppercase + whitespace collapse"
HASH_FUNCTION = "sha1"
FINGERPRINT_MB = 10

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[.,;:()\[\]{}'\"`]+")

# ---------------------------------------------------------------------------
# Public normalisation functions
# ---------------------------------------------------------------------------

def normalize_name(s: Optional[str]) -> Optional[str]:
    """Normalise a free-text entity name.

    Steps (in order):
    1. Strip leading / trailing whitespace.
    2. Remove punctuation chars defined by ``_PUNCT_RE``.
    3. Collapse internal whitespace runs to a single space.
    4. Convert to UPPER CASE.

    Returns ``None`` for empty / ``None`` inputs.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper()


def normalize_zip5(z: Optional[str]) -> Optional[str]:
    """Extract the first 5 digits from a raw ZIP code string.

    Returns ``None`` when fewer than 5 digits are present.
    """
    if z is None:
        return None
    z = re.sub(r"\D+", "", str(z))
    if len(z) < 5:
        return None
    return z[:5]


def safe_float(x) -> Optional[float]:
    """Parse an amount value to ``float``, tolerating currency formatting.

    Handles:
    * ``None`` / empty / ``"nan"`` / ``"none"``  → returns ``None``
    * Comma-separated thousands (``"1,234.56"`` → ``1234.56``)
    * Stray non-numeric characters stripped via regex

    Returns ``None`` on any parse failure.
    """
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
    """Parse a date string into ISO-8601 (``YYYY-MM-DD``) format.

    Supported input formats: ``%Y-%m-%d``, ``%m/%d/%Y``, ``%Y/%m/%d``.

    Returns ``None`` when parsing fails.
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
    """Produce a deterministic SHA-1 hex digest from an ordered sequence of strings.

    ``None`` values are treated as the empty string so the result is stable
    across calls regardless of optional field presence.

    Parameters
    ----------
    parts:
        Iterable of optional strings.  Order matters.

    Returns
    -------
    40-character lowercase hexadecimal string.
    """
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()

