"""concurrentNumba/config.py

Configuration dataclass and YAML loader for Phase 1 (clean data) of the
concurrent-Numba pipeline.

Public API
----------
CleanConfig             Frozen dataclass holding every tunable parameter for
                        one pipeline run.
load_config(path)       Parse a YAML config file into a ``CleanConfig``.

Two YAML flavours are supported:

* **Pipeline-style** (``dataset:``, ``inputs:``, ``output:``, … top-level
  keys) – the canonical project format.
* **Legacy flat** (``dataset_type:``, ``input_files:``, … top-level scalars).

``load_config`` detects the flavour automatically.
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
    """All configuration required to run Phase 1 cleaning.

    Parameters
    ----------
    dataset_type:
        One of ``"general_payment"``, ``"research_payment"``, ``"ownership"``.
    program_year:
        CMS programme year (e.g. ``2024``).
    input_files:
        Absolute paths to the source CSV files.
    output_dir:
        Root output directory for this pipeline run's Phase 1 artefacts.

    Numba / threading
    -----------------
    use_numba:
        Enable Numba JIT kernels (defaults to ``True``).
    chunk_size:
        Rows per pandas ``read_csv`` chunk (default ``250 000``).
    max_workers:
        Number of ``ThreadPoolExecutor`` workers.  ``None`` = CPU count.

    Output
    ------
    write_format:
        Only ``"parquet"`` is supported by this pipeline.
    keep_source_file:
        When ``True``, the ``source_file`` column is populated with the CSV
        filename.
    skip_rejected:
        When ``True``, the rejected-rows parquet is not written to disk.

    Sampling
    --------
    sampling_fraction:
        Fraction of rows to keep (``1.0`` = keep all).
    sampling_seed:
        Optional integer seed for deterministic sampling.
    sampling_method:
        ``"sha1"`` (sequential-equivalent) or ``"fast_hash"`` (Numba-accelerated).
    sampling_stage:
        ``"raw"`` (sample before canonicalisation) or ``"canonical"`` (after).

    Scaling
    -------
    scale_*:
        Reserved for dataset-size scaling experiments (not used in default runs).

    Stats
    -----
    compute_stats:
        When ``True``, amount statistics and payee counts are included in the
        cleaning report JSON.
    stats_compute_median:
        Compute median amount (expensive on large datasets).
    stats_median_method:
        ``"exact"`` (full sort) – only ``"exact"`` is implemented.
    stats_pandas_threshold:
        Unused; reserved for future optimisation.
    """

    # Required
    dataset_type: str
    program_year: int
    input_files: List[str]
    output_dir: str

    # Numba / threading
    use_numba: bool = True
    chunk_size: int = 250_000
    max_workers: Optional[int] = None

    # Output
    write_format: str = "parquet"
    keep_source_file: bool = True
    skip_rejected: bool = True

    # Sampling
    sampling_fraction: float = 1.0
    sampling_seed: Optional[int] = None
    sampling_method: str = "sha1"
    sampling_stage: str = "canonical"

    # Scaling (reserved)
    scale_mode: str = "none"
    scale_value: Optional[float] = None
    scale_key_cols: Optional[List[str]] = None
    scale_enabled: bool = False
    scale_fraction: float = 1.0
    scale_seed: int = 123
    scale_method: str = "hash_record_id"

    # Schema override
    column_mapping: Optional[Dict[str, str]] = None

    # Stats
    stats_compute_median: bool = False
    stats_median_method: str = "exact"
    stats_pandas_threshold: int = 0
    compute_stats: bool = True


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
    output_dir = out_dir / "concurrent_numba" / dataset_name / "phase1_clean"

    phase_cfg   = raw.get("phase1_clean", {})
    scale_cfg   = dataset.get("scale", {}) or {}
    sampling_cfg = (phase_cfg.get("sampling", {}) or {})
    stats_cfg   = (phase_cfg.get("stats", {}) or {})
    exec_cfg    = raw.get("execution", {}) or {}

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
        use_numba=bool(phase_cfg.get("use_numba", True)),
        chunk_size=int(phase_cfg.get("chunk_size", 250_000)),
        max_workers=max_workers,
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


# ---------------------------------------------------------------------------
# Public: load_config
# ---------------------------------------------------------------------------

def load_config(path: str) -> CleanConfig:
    """Load a YAML config file and return a ``CleanConfig``.

    Detects the YAML flavour automatically:

    * **Pipeline-style** – when the file does **not** contain a top-level
      ``dataset_type`` key.
    * **Legacy flat** – when ``dataset_type`` *is* a top-level key.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config_dir = Path(path).parent

    # Delegate to pipeline-style parser unless the file uses the legacy schema
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
    stats_cfg    = (raw.get("stats", {}) or {})

    return CleanConfig(
        dataset_type=str(raw["dataset_type"]),
        program_year=int(raw["program_year"]),
        input_files=resolve_list(list(raw["input_files"])),
        output_dir=(
            str((config_dir / raw["output_dir"]).resolve())
            if not Path(raw["output_dir"]).is_absolute()
            else str(Path(raw["output_dir"]).resolve())
        ),
        use_numba=bool(raw.get("use_numba", True)),
        chunk_size=int(raw.get("chunk_size", 250_000)),
        max_workers=raw.get("max_workers"),
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

