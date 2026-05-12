"""sequentialCPU/fingerprint.py

Dataset fingerprinting for reproducibility.

Responsibilities
----------------
compute_file_fingerprint    Lightweight per-file metadata: size, mtime, SHA-1
                            of the first N MB (fast — does not read the whole
                            file).
create_dataset_fingerprint  Assemble a full-run fingerprint: input files +
                            config snapshot + normalization version, written
                            once at the start of Phase 1.

No Numba, no threading — pure Python + stdlib only.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

from normalizers import NORMALIZATION_VERSION, NORMALIZATION_DESCRIPTION, HASH_FUNCTION, FINGERPRINT_MB  # type: ignore[import]
from config import CleanConfig  # type: ignore[import]


# ---------------------------------------------------------------------------
# Per-file fingerprint
# ---------------------------------------------------------------------------

def compute_file_fingerprint(filepath: str, hash_mb: int = FINGERPRINT_MB) -> Dict:
    """Compute a lightweight fingerprint for a single input file.

    Captures:
    * File size in bytes and MB.
    * Last-modified timestamp (UTC ISO-8601).
    * SHA-1 of the first *hash_mb* megabytes (fast; not a full-file hash).

    Parameters
    ----------
    filepath:
        Path to the file to fingerprint.
    hash_mb:
        Number of MB to read for the partial hash (default ``FINGERPRINT_MB``).

    Returns
    -------
    Dict with keys: ``file``, ``absolute_path``, ``size_bytes``, ``size_mb``,
    ``modified_utc``, ``partial_hash``, ``hash_method``.
    On missing file: ``{"file": …, "exists": False, "error": …}``.
    """
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
                block = f.read(min(8192, bytes_to_hash - bytes_hashed))
                if not block:
                    break
                hash_obj.update(block)
                bytes_hashed += len(block)
        partial_hash = hash_obj.hexdigest()
    except Exception as exc:
        partial_hash = f"error: {exc}"

    return {
        "file":          str(path.name),
        "absolute_path": str(path.absolute()),
        "size_bytes":    file_size,
        "size_mb":       round(file_size / (1024 * 1024), 2),
        "modified_utc":  modified_time,
        "partial_hash":  partial_hash,
        "hash_method":   f"{HASH_FUNCTION}(first_{hash_mb}MB)",
    }


# ---------------------------------------------------------------------------
# Full-run fingerprint
# ---------------------------------------------------------------------------

def create_dataset_fingerprint(config: CleanConfig, config_path: str) -> Dict:
    """Assemble a complete run fingerprint from all input files and the config.

    Parameters
    ----------
    config:
        Resolved ``CleanConfig`` for this run.
    config_path:
        Path to the YAML config file (loaded for the ``config_snapshot`` field).

    Returns
    -------
    Dict suitable for serialisation to ``dataset_fingerprint.json``.
    """
    input_fingerprints: List[Dict] = [
        compute_file_fingerprint(fpath) for fpath in config.input_files
    ]

    with open(config_path, "r", encoding="utf-8") as f:
        config_snapshot = yaml.safe_load(f)

    return {
        "fingerprint_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "normalization_version":     NORMALIZATION_VERSION,
            "normalization_description": NORMALIZATION_DESCRIPTION,
            "hash_function":             HASH_FUNCTION,
        },
        "config_snapshot": config_snapshot,
        "input_files":     input_fingerprints,
        "processing_params": {
            "dataset_type":    config.dataset_type,
            "program_year":    config.program_year,
            "chunk_size":      config.chunk_size,
            "write_format":    config.write_format,
            "keep_source_file": config.keep_source_file,
        },
    }

