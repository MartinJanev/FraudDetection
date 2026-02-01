#!/usr/bin/env python3
"""
Phase 2 (GPU single-device): Graph Building (cuDF)

Reads canonical cleaned payments Parquet parts and builds a bipartite directed edge list:

    payer_node_id  ->  payee_key

Aggregates edges by (src, dst):
- w_total    = sum(amount_usd)
- n_payments = count(rows)
Optional (if date_col exists, ISO strings):
- min_date, max_date

Outputs:
- edges.parquet
- nodes.parquet
- graph_build_report.json
- graph_build_manifest.json

Design goals:
- Deterministic IDs (CPU/GPU semantic alignment)
- Two-stage aggregation to avoid OOM (tmp parts + final reduce)
- No fraud scoring, no graph algorithms here

Notes / fixes applied:
- Robust token cleaning ("None" and nulls handled consistently)
- Prevent cuDF value_counts() from producing a None-bucket by forcing node_type to non-null strings
- Avoid writing empty tmp parts silently; report per-part counters
- Keep tmp parts by default (set CLEAN_TMP=1 env var to delete them)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import cudf


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# Treat these as unusable identifiers / keys (case-insensitive)
BAD_TOKENS_UP = {"", "NONE", "NAN", "<NA>", "NULL", "N/A"}


def _to_str_clean(s: cudf.Series) -> cudf.Series:
    """Stringify, strip, and replace nulls with empty string."""
    # cuDF string ops need strings; keep it simple
    return s.fillna("").astype("str").str.strip()


def clean_id_series(s: cudf.Series) -> cudf.Series:
    """Normalize identifier strings and drop bad tokens -> empty string."""
    s2 = _to_str_clean(s)
    up = s2.str.upper()
    # cuDF isin prefers a list-like; ensure deterministic
    return s2.mask(up.isin(list(BAD_TOKENS_UP)), "")


@dataclass(frozen=True)
class GraphBuildConfig:
    input_clean_dir: str
    input_clean_manifest: Optional[str]
    output_graph_dir: str

    payer_id_col: str
    payer_name_norm_col: str
    payee_key_col: str
    amount_col: str
    date_col: Optional[str] = None

    min_edge_weight: float = 0.0

    tmp_dir_name: str = "_tmp_edge_parts"
    write_format: str = "parquet"
    output_dir_override: Optional[str] = None
    # force deterministic partitions scaling knob placeholder
    tmp_part_size: int = 0


def load_config(path: str) -> GraphBuildConfig:
    cfg_path = Path(path).resolve()
    cfg_dir = cfg_path.parent

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Pipeline-style sections (same as phase 1)
    output_cfg = raw.get("output", {}) or {}
    phase_cfg = raw.get("phase2_graph", {}) or {}

    # Output root directory (relative to config dir)
    out_root = Path(output_cfg.get("root_dir", "output"))
    if not out_root.is_absolute():
        out_root = (cfg_dir / out_root).resolve()

    # Phase1 outputs (input to phase2)
    input_clean_dir = out_root / "payments_clean"
    input_clean_manifest = out_root / "payments_clean_manifest.json"
    if not input_clean_manifest.exists():
        input_clean_manifest = None

    # Phase2 outputs go here by default
    output_graph_dir = out_root / "phase2_graph"
    output_graph_dir.mkdir(parents=True, exist_ok=True)

    # Column names / params from phase2_graph block
    payer_id_col = str(phase_cfg.get("payer_id_col", "payer_id"))
    payer_name_norm_col = str(phase_cfg.get("payer_name_norm_col", "payer_name_norm"))
    payee_key_col = str(phase_cfg.get("payee_key_col", "payee_key"))
    amount_col = str(phase_cfg.get("amount_col", "amount_usd"))
    date_col = str(phase_cfg.get("date_col")) if phase_cfg.get("date_col") else None
    min_edge_weight = float(phase_cfg.get("min_edge_weight", 0.0))

    tmp_dir_name = str(phase_cfg.get("tmp_dir_name", "_tmp_edge_parts"))
    write_format = str(phase_cfg.get("write_format", "parquet"))

    # Optional override
    output_dir_override = raw.get("output_dir") or raw.get("output_dir_override") or None
    if output_dir_override:
        od = Path(str(output_dir_override))
        if not od.is_absolute():
            od = (cfg_dir / od).resolve()
        output_dir_override = str(od)

    return GraphBuildConfig(
        input_clean_dir=str(input_clean_dir),
        input_clean_manifest=str(input_clean_manifest) if input_clean_manifest else None,
        output_graph_dir=str(output_graph_dir),
        payer_id_col=payer_id_col,
        payer_name_norm_col=payer_name_norm_col,
        payee_key_col=payee_key_col,
        amount_col=amount_col,
        date_col=date_col,
        min_edge_weight=min_edge_weight,
        tmp_dir_name=tmp_dir_name,
        write_format=write_format,
        output_dir_override=output_dir_override,
    )


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def list_parquet_parts(clean_dir: str) -> List[str]:
    p = Path(clean_dir)
    parts = sorted([str(x) for x in p.glob("part-*.parquet")])
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet found under {clean_dir}")
    return parts


def make_deterministic_node_ids_gpu(edges: "cudf.DataFrame") -> ("cudf.DataFrame", "cudf.DataFrame"):
    """
    Deterministically map src/dst (possibly strings) to int64 node IDs.

    Returns:
      edges_mapped: edges with src/dst int64
      nodes: [node_id:int64, node_key:<original>, node_type:str]
    """
    # unique node keys
    src_nodes = edges[["src"]].rename(columns={"src": "node_key"})
    dst_nodes = edges[["dst"]].rename(columns={"dst": "node_key"})
    all_nodes = cudf.concat([src_nodes, dst_nodes], ignore_index=True).drop_duplicates()

    # deterministic ordering => deterministic IDs
    # If node_key is string/object, sort_values gives stable order.
    all_nodes = all_nodes.sort_values("node_key").reset_index(drop=True)
    all_nodes["node_id"] = all_nodes.index.astype("int64")

    # node types (bipartite)
    # src-side are payers; dst-side are providers.
    src_keys = edges[["src"]].rename(columns={"src": "node_key"}).drop_duplicates()
    src_keys["node_type"] = "payer"
    dst_keys = edges[["dst"]].rename(columns={"dst": "node_key"}).drop_duplicates()
    dst_keys["node_type"] = "provider"

    node_types = cudf.concat([src_keys, dst_keys], ignore_index=True).drop_duplicates(subset=["node_key"], keep="first")

    nodes = all_nodes.merge(node_types, on="node_key", how="left")
    nodes = nodes[["node_id", "node_type", "node_key"]]

    # map edges
    mapped = edges.merge(nodes[["node_id", "node_key"]].rename(columns={"node_id": "src_id", "node_key": "src"}),
                         on="src", how="left")
    mapped = mapped.merge(nodes[["node_id", "node_key"]].rename(columns={"node_id": "dst_id", "node_key": "dst"}),
                          on="dst", how="left")

    # replace
    mapped = mapped.drop(columns=["src", "dst"]).rename(columns={"src_id": "src", "dst_id": "dst"})

    # enforce dtypes
    mapped["src"] = mapped["src"].astype("int64")
    mapped["dst"] = mapped["dst"].astype("int64")

    return mapped, nodes


def read_manifest_parts(manifest_path: str, clean_dir: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    parts = m.get("parts") if isinstance(m, dict) else m
    if not parts:
        raise ValueError(f"Manifest has no 'parts': {manifest_path}")

    out: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            part_file = part.get("part_file")
            if not part_file:
                continue
        else:
            part_file = str(part)

        part_path = Path(part_file)
        if not part_path.is_absolute():
            part_path = Path(clean_dir) / part_file
        out.append(str(part_path))

    if not out:
        raise ValueError(f"Manifest parts could not be resolved: {manifest_path}")

    return out


def payer_node_id_series_gpu(payer_id: cudf.Series, payer_name_norm: cudf.Series) -> cudf.Series:
    """Deterministic payer node ID.

    if payer_id present and non-empty -> PAYER_ID:<payer_id>
    else -> PAYER_NAME:<payer_name_norm>

    If payer_name_norm missing -> PAYER_NAME:UNKNOWN
    """
    pid = clean_id_series(payer_id)
    pname = clean_id_series(payer_name_norm)

    use_id = pid != ""

    out = cudf.Series([""], dtype="str").repeat(len(pid))

    out = out.where(~use_id, "PAYER_ID:" + pid)
    out = out.where(use_id, "PAYER_NAME:" + pname)

    # If pname empty -> UNKNOWN
    pname_is_empty = pname == ""
    out = out.where(~pname_is_empty, "PAYER_NAME:UNKNOWN")

    # Final cleanup (in case of any token weirdness)
    return clean_id_series(out)


def write_parquet_gpu(df: cudf.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def _safe_numeric_amount(s: cudf.Series) -> cudf.Series:
    """Parse amount to float64 with coerce-like behavior."""
    # cudf.to_numeric supports errors="coerce" for scalar-like in older versions;
    # in 24.10 it works on Series. Keep errors='coerce'.
    return cudf.to_numeric(s, errors="coerce").astype("float64")


def aggregate_part_gpu(
        part_path: str,
        cfg: GraphBuildConfig,
        part_idx: int,
        tmp_edges_dir: str
) -> Tuple[str, Dict[str, int]]:
    gdf = cudf.read_parquet(part_path)

    # IMPORTANT: prevent "Cannot align indices with non-unique values"
    # Always normalize to a fresh RangeIndex
    gdf = gdf.reset_index(drop=True)

    required = [cfg.payer_id_col, cfg.payer_name_norm_col, cfg.payee_key_col, cfg.amount_col]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing columns in {part_path}: {missing}")

    payer_id = gdf[cfg.payer_id_col].reset_index(drop=True)
    payer_name_norm = gdf[cfg.payer_name_norm_col].reset_index(drop=True)
    payee_key = gdf[cfg.payee_key_col].reset_index(drop=True)

    amount = cudf.to_numeric(gdf[cfg.amount_col], errors="coerce").astype("float64")
    amount = amount.reset_index(drop=True)

    src = payer_node_id_series_gpu(payer_id, payer_name_norm)
    src = clean_id_series(src).reset_index(drop=True)

    dst = clean_id_series(payee_key).reset_index(drop=True)

    # Build DataFrame column-by-column (avoids alignment on dict constructor)
    base = cudf.DataFrame()
    base["src"] = src
    base["dst"] = dst
    base["w"] = amount

    has_date = bool(cfg.date_col) and (cfg.date_col in gdf.columns)
    if has_date:
        d = gdf[cfg.date_col].fillna("").astype("str").str.strip()
        base["date"] = d.reset_index(drop=True)

    # filter invalid
    base = base[(base["dst"] != "") & (base["src"] != "") & base["w"].notna()]
    rows_used = int(len(base))

    # If nothing survives filtering, do not write an empty tmp file.
    if rows_used == 0:
        counters = {
            "rows_read": int(len(gdf)),
            "rows_used": 0,
            "tmp_edges": 0,
        }
        logging.warning("Part %s produced 0 usable rows after filtering.", Path(part_path).name)
        return None, counters

    if has_date:
        agg = base.groupby(["src", "dst"]).agg({"w": ["sum", "count"], "date": ["min", "max"]})
        agg.columns = ["w_total", "n_payments", "min_date", "max_date"]
        agg = agg.reset_index()
    else:
        agg = base.groupby(["src", "dst"]).agg({"w": ["sum", "count"]})
        agg.columns = ["w_total", "n_payments"]
        agg = agg.reset_index()

    if cfg.min_edge_weight > 0:
        agg = agg[agg["w_total"] >= cfg.min_edge_weight]

    out_path = str(Path(tmp_edges_dir) / f"tmp-edges-part-{part_idx:05d}.parquet")
    write_parquet_gpu(agg, out_path)

    counters = {
        "rows_read": int(len(gdf)),
        "rows_used": int(rows_used),
        "tmp_edges": int(len(agg)),
    }
    return out_path, counters


def final_reduce_gpu(tmp_parts: List[str], cfg: GraphBuildConfig) -> cudf.DataFrame:
    """Load tmp aggregated edge parts and reduce them into final edges on GPU."""

    if not tmp_parts:
        # Deterministic empty edges table (correct schema)
        cols = ["src", "dst", "w_total", "n_payments"]
        return cudf.DataFrame({c: cudf.Series([], dtype="str") for c in cols})

    dfs = [cudf.read_parquet(p) for p in tmp_parts]
    all_edges = cudf.concat(dfs, ignore_index=True)

    has_dates = ("min_date" in all_edges.columns) and ("max_date" in all_edges.columns)
    if has_dates:
        edges = (
            all_edges.groupby(["src", "dst"]).agg(
                {"w_total": "sum", "n_payments": "sum", "min_date": "min", "max_date": "max"}
            )
            .reset_index()
        )
    else:
        edges = all_edges.groupby(["src", "dst"]).agg({"w_total": "sum", "n_payments": "sum"}).reset_index()

    if cfg.min_edge_weight > 0:
        edges = edges[edges["w_total"] >= cfg.min_edge_weight]

    return edges


def build_nodes_gpu(edges: cudf.DataFrame, output_graph_dir: str) -> Tuple[str, Dict[str, int]]:
    """
    Build nodes table from edges (src + dst), with derived node_type.
    Avoids cuDF index alignment issues by normalizing indices and assigning columns explicitly.
    """
    # Pull src/dst, normalize strings, drop empties
    src_nodes = edges["src"].fillna("").astype("str").str.strip()
    dst_nodes = edges["dst"].fillna("").astype("str").str.strip()

    # concat + unique
    nid = cudf.concat([src_nodes, dst_nodes], ignore_index=True)
    nid = nid[nid != ""]
    nid = nid.drop_duplicates().reset_index(drop=True).astype("str")

    # classify
    is_payer = nid.str.startswith("PAYER_ID:") | nid.str.startswith("PAYER_NAME:")
    is_phys = nid.str.startswith("PHYS_")
    is_hosp = nid.str.startswith("HOSP_")

    node_type = cudf.Series(["unknown"] * len(nid), dtype="str").reset_index(drop=True)
    node_type = node_type.where(~is_hosp, "teaching_hospital")
    node_type = node_type.where(~is_phys, "physician")
    node_type = node_type.where(~is_payer, "payer")
    node_type = node_type.fillna("unknown").astype("str").reset_index(drop=True)

    # IMPORTANT: build DF via column assignment (no index alignment)
    nodes_df = cudf.DataFrame()
    nodes_df["node_id"] = nid
    nodes_df["node_type"] = node_type

    out_path = str(Path(output_graph_dir) / "nodes.parquet")
    nodes_df.to_parquet(out_path, index=False)

    vc = nodes_df["node_type"].value_counts(dropna=False).to_pandas().to_dict()
    return out_path, {
        "total_nodes": int(len(nodes_df)),
        "nodes_by_type": {str(k): int(v) for k, v in vc.items()},
    }


def run(cfg: GraphBuildConfig) -> None:
    t0 = time.perf_counter()

    output_dir = Path(cfg.output_dir_override) if cfg.output_dir_override else Path(cfg.output_graph_dir)
    ensure_dir(str(output_dir))

    tmp_edges_dir = output_dir / cfg.tmp_dir_name
    ensure_dir(str(tmp_edges_dir))

    if cfg.input_clean_manifest:
        parts = read_manifest_parts(cfg.input_clean_manifest, cfg.input_clean_dir)
    else:
        parts = list_parquet_parts(cfg.input_clean_dir)

    logging.info("Graph build input parts: %d", len(parts))
    logging.info("Output graph dir: %s", str(output_dir))

    t_read_agg_start = time.perf_counter()

    tmp_paths: List[str] = []
    totals = {"rows_read": 0, "rows_used": 0, "tmp_edges": 0}

    for idx, part in enumerate(parts):
        out_tmp, ctr = aggregate_part_gpu(part, cfg, idx, str(tmp_edges_dir))

        for k in totals:
            totals[k] += int(ctr.get(k, 0))

        if out_tmp:
            tmp_paths.append(out_tmp)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(parts):
            logging.info(
                "Processed %d/%d parts | rows_used=%d | tmp_edges=%d | tmp_parts=%d",
                idx + 1,
                len(parts),
                totals["rows_used"],
                totals["tmp_edges"],
                len(tmp_paths),
            )

    t_read_agg = time.perf_counter() - t_read_agg_start

    t_reduce_start = time.perf_counter()
    if not tmp_paths:
        # ensure edges/nodes still exist with correct schema; avoid crash in Phase 3
        edges = cudf.DataFrame({
            "src": cudf.Series([], dtype="int64"),
            "dst": cudf.Series([], dtype="int64"),
            "w_total": cudf.Series([], dtype="float64"),
            "n_payments": cudf.Series([], dtype="int64"),
        })
    else:
        edges = final_reduce_gpu(tmp_paths, cfg)
    t_reduce = time.perf_counter() - t_reduce_start

    t_id_start = time.perf_counter()
    edges, nodes = make_deterministic_node_ids_gpu(edges)
    t_id = time.perf_counter() - t_id_start

    edges_path = str(output_dir / "edges.parquet")
    t_write_start = time.perf_counter()
    write_parquet_gpu(edges, edges_path)
    t_write_edges = time.perf_counter() - t_write_start

    t_nodes_start = time.perf_counter()
    nodes_path = str(output_dir / "nodes.parquet")
    write_parquet_gpu(nodes[["node_id", "node_type"]], nodes_path)
    t_nodes = time.perf_counter() - t_nodes_start
    # Node stats (Phase-2-style)
    if len(edges) == 0:
        node_counts = {"total_nodes": 0, "nodes_by_type": {}}
    else:
        try:
            vc = nodes["node_type"].value_counts(dropna=False).to_pandas().to_dict()
            node_counts = {
                "total_nodes": int(len(nodes)),
                "nodes_by_type": {str(k): int(v) for k, v in vc.items()},
            }
        except Exception:
            node_counts = {"total_nodes": int(len(nodes)), "nodes_by_type": {}}

    # Edge stats
    if len(edges) > 0 and "w_total" in edges.columns:
        w = edges["w_total"].to_pandas().to_numpy()
        edge_stats = {
            "edges": int(len(edges)),
            "w_total_min": float(np.min(w)) if len(w) else None,
            "w_total_mean": float(np.mean(w)) if len(w) else None,
            "w_total_median": float(np.median(w)) if len(w) else None,
            "w_total_max": float(np.max(w)) if len(w) else None,
        }
    else:
        edge_stats = {"edges": int(len(edges)), "w_total_min": None, "w_total_mean": None, "w_total_median": None,
                      "w_total_max": None}

    total_time = time.perf_counter() - t0

    report = {
        "inputs": {
            "clean_dir": cfg.input_clean_dir,
            "manifest": cfg.input_clean_manifest,
            "parts": len(parts),
        },
        "outputs": {"edges_path": edges_path, "nodes_path": nodes_path},
        "counts": {**totals, **edge_stats, **(node_counts or {}), "tmp_parts_written": int(len(tmp_paths))},
        "timings_sec": {
            "read_and_local_aggregate": float(t_read_agg),
            "final_reduce": float(t_reduce),
            "assign_node_ids": float(t_id),
            "write_edges": float(t_write_edges),
            "build_nodes": float(t_nodes),
            "total": float(total_time),
        },
        "config": {
            "payer_id_col": cfg.payer_id_col,
            "payer_name_norm_col": cfg.payer_name_norm_col,
            "payee_key_col": cfg.payee_key_col,
            "amount_col": cfg.amount_col,
            "date_col": cfg.date_col,
            "min_edge_weight": cfg.min_edge_weight,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = str(output_dir / "graph_build_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    manifest = {
        "tmp_edge_parts_dir": str(tmp_edges_dir),
        "tmp_edge_parts": [Path(p).name for p in tmp_paths],
        "final_edges": "edges.parquet",
        "final_nodes": "nodes.parquet",
    }

    manifest_path = str(output_dir / "graph_build_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Optional cleanup: set CLEAN_TMP=1 to delete tmp edge parts
    if os.environ.get("CLEAN_TMP", "0") == "1":
        for p in Path(tmp_edges_dir).glob("tmp-edges-part-*.parquet"):
            try:
                p.unlink()
            except OSError:
                pass

    logging.info("Wrote edges: %s", edges_path)
    logging.info("Wrote nodes: %s", nodes_path)
    logging.info("Wrote report: %s", report_path)


def run_from_pipeline(
        pipeline_cfg: Dict,
        *,
        clean_dir: Path,
        clean_manifest: Optional[Path],
        out_dir: Path,
        approach: str,
        run_id: str,
        dataset_name: str,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_cfg = pipeline_cfg.get("phase2_graph", {})
    cfg = GraphBuildConfig(
        input_clean_dir=str(clean_dir),
        input_clean_manifest=str(clean_manifest) if clean_manifest else None,
        output_graph_dir=str(out_dir),
        payer_id_col=str(phase_cfg.get("payer_id_col", "payer_id")),
        payer_name_norm_col=str(phase_cfg.get("payer_name_norm_col", "payer_name_norm")),
        payee_key_col=str(phase_cfg.get("payee_key_col", "payee_key")),
        amount_col=str(phase_cfg.get("amount_col", "amount_usd")),
        date_col=str(phase_cfg.get("date_col")) if phase_cfg.get("date_col") else None,
        min_edge_weight=float(phase_cfg.get("min_edge_weight", 0.0)),
        tmp_dir_name=str(phase_cfg.get("tmp_dir_name", "_tmp_edge_parts")),
        write_format=str(phase_cfg.get("write_format", "parquet")),
        output_dir_override=str(out_dir),
    )

    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    run(cfg)
    wall = time.perf_counter() - t0
    end_ts = datetime.now(timezone.utc).isoformat()

    return {
        "phase": "phase2_graph",
        "dataset": dataset_name,
        "approach": approach,
        "run_id": run_id,
        "start_utc": start_ts,
        "end_utc": end_ts,
        "wall_time_seconds": round(wall, 4),
        "artifacts": {
            "edges": str(out_dir / "edges.parquet"),
            "nodes": str(out_dir / "nodes.parquet"),
            "manifest": str(out_dir / "graph_build_manifest.json"),
            "report": str(out_dir / "graph_build_report.json"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for GPU graph building.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).parent / "configs" / args.config
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    setup_logging(args.log_level)
    cfg = load_config(str(config_path))
    run(cfg)


if __name__ == "__main__":
    main()
