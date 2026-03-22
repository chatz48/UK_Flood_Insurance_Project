"""
Environment Agency Real-Time Flood Monitoring API
Pulls historical river level and flow readings from all active EA gauging stations.

API docs: https://environment.data.gov.uk/flood-monitoring/doc/reference
Free, no auth required. ~1,500 stations across England.

Data collected:
- Station metadata (location, river, catchment, datum)
- Historical water level readings (15-min intervals)
- Historical flow readings where available
- Flood warning thresholds per station
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://environment.data.gov.uk/flood-monitoring"
RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "ea_gauging"


def _scalar(v):
    """Return first element if v is a list, else v as-is."""
    return v[0] if isinstance(v, list) else v


def fetch_all_stations() -> pd.DataFrame:
    """Fetch metadata for all active EA gauging stations in England."""
    print("Fetching all EA gauging stations...")
    url = f"{BASE_URL}/id/stations"
    params = {
        "status": "Active",
        "_limit": 10000,
        "parameter": "level",  # water level stations
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    items = r.json()["items"]

    stations = []
    for s in items:
        stations.append({
            "station_reference": _scalar(s.get("stationReference")),
            "station_id": s.get("@id", "").split("/")[-1],
            "label": _scalar(s.get("label")),
            "river_name": _scalar(s.get("riverName")),
            "catchment_name": _scalar(s.get("catchmentName")),
            "town": _scalar(s.get("town")),
            "lat": _scalar(s.get("lat")),
            "lon": _scalar(s.get("long")),
            "easting": _scalar(s.get("easting")),
            "northing": _scalar(s.get("northing")),
            "datum": _scalar(s.get("datum")),
            "stage_scale": s.get("stageScale", {}).get("@id") if isinstance(s.get("stageScale"), dict) else None,
        })

    df = pd.DataFrame(stations)
    print(f"  Found {len(df)} active stations")
    return df


def fetch_station_thresholds(station_reference: str) -> dict:
    """Fetch flood warning thresholds for a given station."""
    url = f"{BASE_URL}/id/stations/{station_reference}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("items", {})
        stage = data.get("stageScale", {})
        return {
            "typical_range_high": stage.get("typicalRangeHigh"),
            "typical_range_low": stage.get("typicalRangeLow"),
            "max_on_record": stage.get("maxOnRecord", {}).get("value") if isinstance(stage.get("maxOnRecord"), dict) else None,
            "min_on_record": stage.get("minOnRecord", {}).get("value") if isinstance(stage.get("minOnRecord"), dict) else None,
        }
    except Exception:
        return {}


def fetch_historical_readings(station_reference: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical water level readings for a station.

    Args:
        station_reference: EA station reference (e.g. '531118')
        start_date: ISO date string 'YYYY-MM-DD'
        end_date: ISO date string 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: dateTime, value
    """
    url = f"{BASE_URL}/id/stations/{station_reference}/readings"
    params = {
        "startdate": start_date,
        "enddate": end_date,
        "_limit": 50000,
        "parameter": "level",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)[["dateTime", "value"]]
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["station_reference"] = station_reference
        return df.sort_values("dateTime")
    except Exception as e:
        print(f"    Warning: failed to fetch {station_reference}: {e}")
        return pd.DataFrame()


def fetch_current_flood_warnings() -> pd.DataFrame:
    """Fetch all current flood warnings and alerts across England."""
    print("Fetching current flood warnings...")
    url = f"{BASE_URL}/id/floods"
    params = {"_limit": 10000}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])

    warnings = []
    for w in items:
        warnings.append({
            "flood_area_id": w.get("floodAreaID"),
            "county": w.get("floodArea", {}).get("county") if isinstance(w.get("floodArea"), dict) else None,
            "description": w.get("description"),
            "severity": w.get("severity"),
            "severity_level": w.get("severityLevel"),
            "is_active": w.get("isTidal"),
            "time_raised": w.get("timeRaised"),
            "time_updated": w.get("timeMessageChanged"),
            "message": w.get("message"),
        })

    df = pd.DataFrame(warnings)
    print(f"  Found {len(df)} active flood warnings/alerts")
    return df


def run_full_pipeline(
    max_stations: int = None,
    fetch_readings: bool = False,
    readings_start: str = "2000-01-01",
    readings_end: str = "2024-12-31",
):
    """
    Main pipeline entry point. Downloads all EA gauging station data.

    Args:
        max_stations: Limit number of stations (None = all ~1500)
        fetch_readings: If True, also pull historical readings per station (slow)
        readings_start: Start date for historical readings
        readings_end: End date for historical readings
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Station metadata
    stations_df = fetch_all_stations()
    stations_path = RAW_DIR / "stations.parquet"
    stations_df.to_parquet(stations_path, index=False)
    print(f"  Saved stations to {stations_path}")

    # 2. Current flood warnings snapshot
    warnings_df = fetch_current_flood_warnings()
    warnings_path = RAW_DIR / "current_warnings.parquet"
    warnings_df.to_parquet(warnings_path, index=False)
    print(f"  Saved warnings to {warnings_path}")

    # 3. Optional: pull historical readings for each station
    if fetch_readings:
        stations_to_fetch = stations_df.head(max_stations) if max_stations else stations_df
        print(f"\nFetching historical readings for {len(stations_to_fetch)} stations...")
        print("  This will take a while — ~1 request per station with rate limiting")

        all_readings = []
        for _, row in tqdm(stations_to_fetch.iterrows(), total=len(stations_to_fetch)):
            ref = row["station_reference"]
            if not ref:
                continue
            df = fetch_historical_readings(ref, readings_start, readings_end)
            if not df.empty:
                all_readings.append(df)
            time.sleep(0.2)  # respect EA rate limits

        if all_readings:
            readings_df = pd.concat(all_readings, ignore_index=True)
            readings_path = RAW_DIR / "historical_readings.parquet"
            readings_df.to_parquet(readings_path, index=False)
            print(f"  Saved {len(readings_df):,} readings to {readings_path}")

    print("\nEA gauging pipeline complete.")
    return stations_df


if __name__ == "__main__":
    run_full_pipeline(fetch_readings=False)
