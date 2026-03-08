"""GLOBE API client for cloud/sky observations retrieval."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
import requests

# Base URL for GLOBE measurement API (globeteams, userid, point/distance endpoints)
_GLOBE_BASE = "https://api.globe.gov/search/v1/measurement/protocol/measureddate/"


def _resolve_base_url(base_url: str | None) -> str:
    """Use config base_url unless it points to MARSS API (different endpoint structure)."""
    if base_url and "marss" not in base_url.lower():
        return base_url.rstrip("/") + "/"
    return _GLOBE_BASE


logger = logging.getLogger(__name__)


def _parse_geojson_features(data: dict) -> list[dict]:
    """Extract and flatten GeoJSON features into rows for DataFrame."""
    features = data.get("features", [])
    rows = []
    for f in features:
        props = dict(f.get("properties", {}))
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [0, 0])
        if len(coords) >= 2:
            props["longitude"] = coords[0]
            props["latitude"] = coords[1]
        rows.append(props)
    return rows


def _fetch_endpoint(
    url: str,
    params: dict,
    timeout: int,
) -> pd.DataFrame | None:
    """Fetch a single GLOBE API endpoint and return DataFrame or None on failure."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("API request failed for %s: %s", url, e)
        return None
    try:
        data = resp.json()
    except ValueError as e:
        logger.warning("Invalid JSON response: %s", e)
        return None
    rows = _parse_geojson_features(data)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch(
    mode: Literal["targets", "location"],
    targets: list[str] | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch GLOBE cloud/sky observations by targets (teams/usernames) or location.

    Returns an empty DataFrame on API failure or no data; never raises.
    """
    url_base = _resolve_base_url(base_url)
    params: dict = {
        "protocols": "sky_conditions",
        "startdate": start_date or "",
        "enddate": end_date or "",
        "geojson": "TRUE",
        "sample": "FALSE",
    }
    dfs: list[pd.DataFrame] = []

    if mode == "targets" and targets:
        for t in targets:
            t = t.strip()
            if not t:
                continue
            team_url = f"{url_base}globeteams/?teams={t}"
            user_url = f"{url_base}userid/?userid={t}"
            params_team = {**params, "globeteam": t}
            params_user = {**params}
            df = _fetch_endpoint(team_url, params_team, timeout)
            if df is None or df.empty:
                df = _fetch_endpoint(user_url, params_user, timeout)
            if df is not None and not df.empty:
                dfs.append(df)

    elif mode == "location" and lat is not None and lon is not None and radius_km is not None:
        loc_url = f"{url_base}point/distance/?lat={lat}&lon={lon}&distancekm={radius_km}"
        df = _fetch_endpoint(loc_url, params, timeout)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
