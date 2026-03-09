#!/usr/bin/env python3
"""Central CLI for GLOBE retrieval and sky segmentation pipeline."""

from __future__ import annotations

# Ensure float32 default for MPS compatibility (float64 not fully supported)
import torch
torch.set_default_dtype(torch.float32)

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load config.yaml from project root."""
    if not _CONFIG_PATH.exists():
        logger.warning("config.yaml not found; using defaults")
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(config: dict) -> argparse.Namespace:
    """Parse CLI: --retrieve, --process, and mutually exclusive search options."""
    parser = argparse.ArgumentParser(
        description="GLOBE retrieval and sky segmentation pipeline",
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Run retrieval step: fetch GLOBE observations",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Run processing step: SAM segmentation and sky index",
    )
    search = parser.add_mutually_exclusive_group(required=False)
    search.add_argument(
        "--targets",
        nargs="+",
        metavar="TARGET",
        help="List of teams/usernames (overrides config)",
    )
    search.add_argument(
        "--location",
        action="store_true",
        help="Use location-based search (requires --lat, --lon, --radius)",
    )
    parser.add_argument("--lat", type=float, help="Latitude (with --location)")
    parser.add_argument("--lon", type=float, help="Longitude (with --location)")
    parser.add_argument("--radius", type=float, metavar="KM", help="Radius in km (with --location)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, help="Retrieval output CSV path")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None and args.radius is not None:
        args.location = True

    return args


def merge_config_cli(config: dict, args: argparse.Namespace) -> dict:
    """Merge config with CLI; CLI overrides config."""
    retrieval = config.get("retrieval", {}) or {}
    targets_cfg = config.get("targets", {}) or {}
    loc = config.get("location", {}) or {}
    defaults = config.get("search_defaults", {}) or {}
    output_cfg = config.get("output", {}) or {}
    api = config.get("api", {}) or {}

    mode = "targets"
    if args.targets is not None:
        mode = "targets"
    elif args.location:
        mode = "location"
    else:
        mode = retrieval.get("mode", "targets")

    return {
        "mode": mode,
        "targets": args.targets if args.targets is not None else (
            (targets_cfg.get("teams") or []) + (targets_cfg.get("usernames") or [])
        ),
        "lat": args.lat if args.lat is not None else loc.get("lat"),
        "lon": args.lon if args.lon is not None else loc.get("lon"),
        "radius_km": args.radius if args.radius is not None else defaults.get("radius_km", 5),
        "start_date": args.start_date or defaults.get("start_date") or "2026-01-01",
        "end_date": (
            args.end_date if args.end_date is not None
            else (defaults.get("end_date") or date.today().isoformat())
        ),
        "output_dir": output_cfg.get("directory", "data"),
        "output_filename": output_cfg.get("filename") or "aggregated_globe_data.csv",
        "output_path": args.output,
        "base_url": api.get("base_url"),
        "timeout": api.get("timeout", 30),
    }


def run_retrieve(config: dict, merged: dict) -> int:
    """Execute retrieval step via api_client. Returns 0 on success, 1 on failure."""
    from src.api_client import fetch as api_fetch

    use_targets = merged["mode"] == "targets" and merged["targets"]
    use_location = (
        merged["mode"] == "location"
        and merged["lat"] is not None
        and merged["lon"] is not None
        and merged["radius_km"] is not None
    )

    if not use_targets and not use_location:
        logger.error("Provide --targets T1 T2 ... or --lat LAT --lon LON --radius KM")
        return 1

    out_path = merged["output_path"]
    if out_path is None:
        out_path = _PROJECT_ROOT / merged["output_dir"] / merged["output_filename"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if use_targets:
            df = api_fetch(
                mode="targets",
                targets=merged["targets"],
                start_date=merged["start_date"],
                end_date=merged["end_date"],
                base_url=merged.get("base_url"),
                timeout=merged["timeout"],
            )
        else:
            df = api_fetch(
                mode="location",
                lat=merged["lat"],
                lon=merged["lon"],
                radius_km=merged["radius_km"],
                start_date=merged["start_date"],
                end_date=merged["end_date"],
                base_url=merged.get("base_url"),
                timeout=merged["timeout"],
            )
    except requests.RequestException as e:
        logger.error("API unreachable: %s", e)
        return 1

    if df.empty:
        logger.warning("No data retrieved")
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return 0


def run_process(config: dict) -> int:
    """Execute segmentation step via sam_segmenter. Returns 0 on success, 1 on failure."""
    from src.sam_segmenter import run_segmentation

    sam_cfg = config.get("sam", {}) or {}
    seg_cfg = config.get("segmentation", {}) or {}
    output_cfg = config.get("output", {}) or {}

    input_csv = Path(seg_cfg.get("input_csv", "data/aggregated_globe_data.csv"))
    output_csv = Path(seg_cfg.get("output_csv", "data/segmented_metrics.csv"))
    output_masks_dir = Path(seg_cfg.get("output_masks_dir", "data/output_masks"))
    checkpoint = sam_cfg.get("checkpoint", "models/sam_vit_b_01ec64.pth")
    device = sam_cfg.get("device", "mps")

    return run_segmentation(
        input_csv=input_csv,
        output_csv=output_csv,
        output_masks_dir=output_masks_dir,
        checkpoint_path=checkpoint,
        device=device,
        project_root=_PROJECT_ROOT,
    )


def main() -> int:
    """Orchestrate pipeline based on --retrieve and --process flags."""
    config = load_config()
    args = parse_args(config)
    merged = merge_config_cli(config, args)

    if not args.retrieve and not args.process:
        logger.info("Use --retrieve and/or --process to run pipeline steps")
        return 0

    if args.retrieve:
        rc = run_retrieve(config, merged)
        if rc != 0:
            return rc

    if args.process:
        rc = run_process(config)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
