#!/usr/bin/env python3
"""GLOBE API CLI: fetch cloud/sky observations by targets or location."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.api_client import fetch as api_fetch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root (parent of this script)
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
    """Parse CLI with mutually exclusive targets vs location, CLI overrides config."""
    parser = argparse.ArgumentParser(description="Fetch GLOBE cloud/sky observations")
    search = parser.add_mutually_exclusive_group(required=False)
    search.add_argument(
        "--targets",
        nargs="+",
        metavar="TARGET",
        help="List of teams and/or usernames (e.g. Udea0311152 student_01)",
    )
    search.add_argument(
        "--location",
        action="store_true",
        help="Use location-based search (requires --lat, --lon, --radius)",
    )
    parser.add_argument("--lat", type=float, metavar="LAT", help="Latitude (with --location)")
    parser.add_argument("--lon", type=float, metavar="LON", help="Longitude (with --location)")
    parser.add_argument("--radius", type=float, metavar="KM", help="Radius in km (with --location)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="Output CSV path (default: data/aggregated_results.csv)",
    )
    args = parser.parse_args()

    # Infer --location when all three --lat, --lon, --radius are given
    if args.lat is not None and args.lon is not None and args.radius is not None:
        args.location = True

    return args


def merge_config_cli(config: dict, args: argparse.Namespace) -> dict:
    """Merge config with CLI; CLI overrides config."""
    out: dict = {}
    # Targets: CLI or config
    targets = config.get("targets", {}) or {}
    out["targets"] = args.targets if args.targets is not None else (
        (targets.get("teams") or []) + (targets.get("usernames") or [])
    )
    # Location: CLI or config
    loc = config.get("location", {}) or {}
    defaults = config.get("search_defaults", {}) or {}
    out["lat"] = args.lat if args.lat is not None else loc.get("lat")
    out["lon"] = args.lon if args.lon is not None else loc.get("lon")
    out["radius_km"] = args.radius if args.radius is not None else defaults.get("radius_km", 5)
    # Dates
    end_raw = args.end_date if args.end_date is not None else (defaults.get("end_date"))
    out["end_date"] = end_raw if end_raw is not None else date.today().isoformat()
    out["start_date"] = args.start_date or defaults.get("start_date") or "2026-01-01"
    # Output
    output_cfg = config.get("output", {}) or {}
    out["output_dir"] = output_cfg.get("directory", "data")
    out["output_filename"] = output_cfg.get("filename") or "aggregated_results.csv"
    out["output_path"] = args.output
    # API
    api = config.get("api", {}) or {}
    out["base_url"] = api.get("base_url")
    out["timeout"] = api.get("timeout", 30)
    return out


def run() -> int:
    """Main entry: load config, parse CLI, fetch, aggregate, save."""
    config = load_config()
    args = parse_args(config)
    merged = merge_config_cli(config, args)

    # Determine mode
    use_targets = args.targets is not None or (merged["targets"] and not args.location)
    use_location = args.location and merged["lat"] is not None and merged["lon"] is not None and merged["radius_km"] is not None

    if use_targets and use_location:
        logger.error("Cannot use both --targets and --location; choose one")
        return 1
    if not use_targets and not use_location:
        if merged["targets"]:
            use_targets = True
        elif merged["lat"] is not None and merged["lon"] is not None and merged["radius_km"] is not None:
            use_location = True
        else:
            logger.error("Provide --targets T1 T2 ... or --lat LAT --lon LON --radius KM (all three required)")
            return 1

    # Build output path
    if merged["output_path"] is not None:
        out_path = merged["output_path"]
    else:
        out_dir = Path(merged["output_dir"])
        out_path = out_dir / merged["output_filename"]

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
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return 1

    if df.empty:
        logger.warning("No data retrieved for the given criteria")
        df = pd.DataFrame()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
