"""sequentialCPU/config.py

Configuration dataclass and YAML loader for Phase 1 (clean data) of the
sequential CPU pipeline.

Public API
----------
CleanConfig             Frozen dataclass holding every tunable parameter for
                        one pipeline run.
load_config(path)       Parse a YAML config file into a ``CleanConfig``.

Two YAML flavours are supported (auto-detected):

* **Pipeline-style** — top-level keys: ``dataset:``, ``inputs:``, ``output:``, …
* **Legacy flat**    — top-level scalars: ``dataset_type:``, ``input_files:``, …

Design notes
------------
* ``max_workers`` is accepted from config for interface parity with the
  concurrent pipelines but is **always enforced as 1** — the sequential
  pipeline intentionally runs on a single thread/process with no Numba.
* No Numba import anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# CleanConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CleanConfig:
    """All configuration required to run Phase 1 sequential cleaning.

    Parameters
    ----------
    dataset_type:
        One of ``"general_payment"``, ``"research_payment"``, ``"ownership"``.
    program_year:
        CMS programme year (e.g. ``2024``).
    input_files:
        Absolute paths to the source CSV files.
    output_dir:
        Root output directory for Phase 1 artefacts.
    chunk_size:
        Rows per ``pandas.read_csv`` chunk (default ``500 000``).
        Larger values use more RAM but fewer I/O round trips.
    write_format:
        ``"parquet"`` (default, partitioned) or ``"csv"`` (append mode).
    keep_source_file:
        Populate ``source_file`` column with the originating CSV filename.
    column_mapping:
        Optional config-supplied override of the CMS column name map.
    max_workers:
        Accepted for interface parity; **always 1** in this pipeline.
        Setting it higher has no effect — no threads or processes are spawned.
    skip_rejected:
        When ``True``, the rejected-rows output is not written to disk.
    sampling_fraction:
        Fraction of rows to keep after canonicalisation (``1.0`` = keep all).
        Rows are selected as the first X % of each chunk (head-slice), which
        is deterministic and does not require random state.
    sampling_seed:
        Unused by the head-slice strategy; retained for interface parity.
    """

    # Required
    dataset_type: str
    program_year: int
    input_files: List[str]
    output_dir: str

    # Processing
    chunk_size: int = 500_000
    write_format: str = "parquet"
    keep_source_file: bool = True
    column_mapping: Optional[Dict[str, str]] = None

    # Always 1 for sequential — accepted only for interface parity
    max_workers: int = 1

    # Output
    skip_rejected: bool = False

    # Sampling (head-slice, not random)
    sampling_fraction: float = 1.0
    sampling_seed: Optional[int] = None   # kept for parity; not used by head-slice


# ---------------------------------------------------------------------------
# Internal: pipeline-style YAML parser
# ---------------------------------------------------------------------------

def _parse_pipeline_style_config(raw: Dict, config_dir: Path) -> CleanConfig:
    """Parse the canonical pipeline-style YAML format into a ``CleanConfig``."""
    dataset = raw.get("dataset", {})
    payment_type = str(dataset.get("payment_type", "general")).lower()
    type_map = {
        "general":          "general_payment",
        "general_payment":  "general_payment",
        "research":         "research_payment",
        "research_payment": "research_payment",
        "ownership":        "ownership",
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
    output_dir = out_dir / "seq_cpu" / dataset_name / "phase1_clean"

    phase_cfg    = raw.get("phase1_clean", {}) or {}
    sampling_cfg = phase_cfg.get("sampling", {}) or {}

    return CleanConfig(
        dataset_type=dataset_type,
        program_year=program_year,
        input_files=resolved_inputs,
        output_dir=str(output_dir),
        chunk_size=int(phase_cfg.get("chunk_size", 500_000)),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        keep_source_file=bool(phase_cfg.get("keep_source_file", True)),
        column_mapping=phase_cfg.get("column_mapping"),
        max_workers=1,   # always single-threaded
        skip_rejected=bool(phase_cfg.get("skip_rejected", False)),
        sampling_fraction=float(sampling_cfg.get("fraction", 1.0)),
        sampling_seed=sampling_cfg.get("seed"),
    )


# ---------------------------------------------------------------------------
# Public: load_config
# ---------------------------------------------------------------------------

def load_config(path: str) -> CleanConfig:
    """Load a YAML config file and return a ``CleanConfig``.

    Detects the YAML flavour automatically:

    * **Pipeline-style** — when the file does **not** contain a top-level
      ``dataset_type`` key.
    * **Legacy flat**    — when ``dataset_type`` *is* a top-level key.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config_dir = Path(path).parent

    if "dataset_type" not in raw:
        return _parse_pipeline_style_config(raw, config_dir)

    # Legacy flat format
    def resolve_list(files: List[str]) -> List[str]:
        out: List[str] = []
        for fp in files:
            p = Path(fp)
            if not p.is_absolute():
                p = config_dir / p
            out.append(str(p.resolve()))
        return out

    sampling_cfg = raw.get("sampling", {}) or {}

    return CleanConfig(
        dataset_type=str(raw["dataset_type"]),
        program_year=int(raw["program_year"]),
        input_files=resolve_list(list(raw["input_files"])),
        output_dir=str(raw["output_dir"]),
        chunk_size=int(raw.get("chunk_size", 500_000)),
        write_format=str(raw.get("write_format", "parquet")),
        keep_source_file=bool(raw.get("keep_source_file", True)),
        column_mapping=raw.get("column_mapping"),
        max_workers=1,   # always single-threaded
        skip_rejected=bool(raw.get("skip_rejected", False)),
        sampling_fraction=float(sampling_cfg.get("fraction", 1.0)),
        sampling_seed=sampling_cfg.get("seed"),
    )

