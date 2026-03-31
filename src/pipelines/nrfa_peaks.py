"""
National River Flow Archive (NRFA) Peak Flow Dataset
UKCEH — https://nrfa.ceh.ac.uk

This is the statistical backbone of UK flood frequency analysis.
Provides Annual Maximum (AMAX) and Peaks Over Threshold (POT) flow series
for ~1,500 gauging stations, many with 50-100+ years of records.

AMAX data: one peak flow per water year (Oct-Sep)
POT data: all flows exceeding a threshold, typically 3x per year on average

Both series are used to fit extreme value distributions (GEV, GPD)
to estimate return period flows — the core of the hazard layer.

API: https://nrfaapps.ceh.ac.uk/nrfa/ws/
"""

import requests
import pandas as pd
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

NRFA_API = "https://nrfaapps.ceh.ac.uk/nrfa/ws"
RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "nrfa"

# Rate-limit: max concurrent requests to NRFA API.
# NRFA docs say the service is "experimental" — be polite.
_NRFA_WORKERS = 20
_nrfa_semaphore = threading.Semaphore(_NRFA_WORKERS)


def _get(url: str, params: dict, timeout: int = 15) -> requests.Response:
    """Rate-limited GET wrapper for NRFA API calls."""
    with _nrfa_semaphore:
        r = requests.get(url, params=params, timeout=timeout)
        time.sleep(0.05)  # 50ms per slot — polite but fast
        return r


def fetch_station_catalogue() -> pd.DataFrame:
    """
    Fetch full NRFA station catalogue with metadata.
    Returns ~1,500 stations with location, catchment area, and data availability.
    """
    print("Fetching NRFA station catalogue...")
    url = f"{NRFA_API}/station-info"
    params = {
        "station": "*",
        "format": "json-object",
        "fields": "station-information,gdf-statistics,category",
    }
    r = _get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    stations = []
    for s in data.get("data", []):
        gr = s.get("grid-reference") or {}
        stations.append({
            "station_id": s.get("id"),
            "name": s.get("name"),
            "river": s.get("river"),
            "location": s.get("location"),
            "grid_ref": gr.get("ngr"),
            "easting": gr.get("easting"),
            "northing": gr.get("northing"),
            "catchment_area_km2": s.get("catchment-area"),
            "record_start": s.get("gdf-start-date"),
            "record_end": s.get("gdf-end-date"),
            "peak_flow_rejected": not s.get("nrfa-peak-flow", True),
            "sensitivity": s.get("sensitivity"),
            "feh_pooling_group": s.get("feh-pooling"),
        })

    df = pd.DataFrame(stations)
    print(f"  Found {len(df)} NRFA stations")
    return df


def fetch_amax_series(station_id: int) -> pd.DataFrame:
    """
    Fetch Annual Maximum (AMAX) flow series for a station.
    Each row is one water year's peak instantaneous flow in m³/s.

    This is the primary input for GEV distribution fitting.
    """
    url = f"{NRFA_API}/time-series"
    params = {
        "format": "json-object",
        "data-type": "amax-flow",
        "station": station_id,
    }
    try:
        r = _get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        stream = data.get("data-stream", [])
        if not stream:
            return pd.DataFrame()

        records = []
        for i in range(0, len(stream) - 1, 2):
            date_str = stream[i]
            flow = stream[i + 1]
            records.append({
                "station_id": station_id,
                "water_year": int(str(date_str)[:4]),
                "peak_flow_m3s": flow,
            })

        df = pd.DataFrame(records)
        df["peak_flow_m3s"] = pd.to_numeric(df["peak_flow_m3s"], errors="coerce")
        return df.dropna(subset=["peak_flow_m3s"])

    except Exception:
        return pd.DataFrame()


def fetch_pot_series(station_id: int) -> pd.DataFrame:
    """
    Fetch Peaks Over Threshold (POT) flow series for a station.
    Gives more data points than AMAX — useful for fitting GPD tails.
    """
    url = f"{NRFA_API}/time-series"
    params = {
        "format": "json-object",
        "data-type": "pot-flow",
        "station": station_id,
    }
    try:
        r = _get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        stream = data.get("data-stream", [])
        if not stream:
            return pd.DataFrame()

        records = []
        for i in range(0, len(stream) - 1, 2):
            records.append({
                "station_id": station_id,
                "date": stream[i],
                "peak_flow_m3s": stream[i + 1],
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["peak_flow_m3s"] = pd.to_numeric(df["peak_flow_m3s"], errors="coerce")
        return df.dropna(subset=["peak_flow_m3s"])

    except Exception:
        return pd.DataFrame()


def fetch_catchment_descriptors(station_id: int) -> dict:
    """
    Fetch FEH catchment descriptors for a station from the NRFA API.

    Uses json-object format with station-information + feh-descriptors fields.
    Fields are returned flat (not nested) in the API response.

    Available fields via this API:
      catchment-area → area (km²)
      bfihost        → base flow index from HOST soils
      farl           → flood attenuation by reservoirs and lakes (0–1)
      propwet        → proportion of time soils are wet (proxy for saar/wetness)

    Note: saar and urbext2000 are not directly available via the public NRFA API.
    These can be supplemented from the FEH Web Service (fee-based) or the
    UK Digital River Network.
    """
    url = f"{NRFA_API}/station-info"
    params = {
        "station": station_id,
        "format": "json-object",
        "fields": "station-information,feh-descriptors",
    }
    try:
        r = _get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        entries = data.get("data", [])
        entry = entries[0] if entries else {}

        return {
            "station_id": station_id,
            "area": entry.get("catchment-area"),
            "farl": entry.get("farl"),
            "bfihost": entry.get("bfihost"),
            "propwet": entry.get("propwet"),  # soil wetness proxy
            "saar": None,        # not available via public API
            "urbext2000": None,  # not available via public API
        }
    except Exception:
        return {
            "station_id": station_id,
            "area": None, "farl": None, "bfihost": None,
            "propwet": None, "saar": None, "urbext2000": None,
        }


def run_full_pipeline(
    max_stations: int = None,
    fetch_pot: bool = True,
    fetch_descriptors: bool = True,
    min_record_years: int = 10,
):
    """
    Main pipeline. Downloads AMAX (and optionally POT) for all NRFA stations.

    Args:
        max_stations: Limit for testing (None = all ~1500)
        fetch_pot: Also download POT series (more data, slower)
        fetch_descriptors: Also download FEH catchment descriptors
        min_record_years: Skip stations with fewer years of record
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Station catalogue
    catalogue = fetch_station_catalogue()
    catalogue_path = RAW_DIR / "station_catalogue.parquet"
    catalogue.to_parquet(catalogue_path, index=False)
    print(f"  Saved catalogue to {catalogue_path}")

    # Filter to usable stations
    usable = catalogue[
        (catalogue["peak_flow_rejected"] != True) &
        (catalogue["station_id"].notna())
    ].copy()

    # Estimate record length
    usable["record_start"] = pd.to_datetime(usable["record_start"], errors="coerce")
    usable["record_end"] = pd.to_datetime(usable["record_end"], errors="coerce")
    usable["record_years"] = (
        (usable["record_end"] - usable["record_start"]).dt.days / 365.25
    )
    usable = usable[usable["record_years"] >= min_record_years]

    if max_stations:
        usable = usable.head(max_stations)

    station_ids = [int(row["station_id"]) for _, row in usable.iterrows()]
    print(f"\nFetching AMAX series for {len(station_ids)} stations "
          f"(>= {min_record_years} years record, {_NRFA_WORKERS} parallel workers)...")

    # 2. AMAX series — parallel fetch
    all_amax = []
    with ThreadPoolExecutor(max_workers=_NRFA_WORKERS) as pool:
        futures = {pool.submit(fetch_amax_series, sid): sid for sid in station_ids}
        for future in tqdm(as_completed(futures), total=len(futures)):
            df = future.result()
            if not df.empty:
                all_amax.append(df)

    if all_amax:
        amax_df = pd.concat(all_amax, ignore_index=True)
        amax_path = RAW_DIR / "amax_all_stations.parquet"
        amax_df.to_parquet(amax_path, index=False)
        print(f"  Saved {len(amax_df):,} AMAX records ({amax_df['station_id'].nunique()} stations) to {amax_path}")

    # 3. POT series — parallel fetch
    if fetch_pot:
        print(f"\nFetching POT series ({_NRFA_WORKERS} workers)...")
        all_pot = []
        with ThreadPoolExecutor(max_workers=_NRFA_WORKERS) as pool:
            futures = {pool.submit(fetch_pot_series, sid): sid for sid in station_ids}
            for future in tqdm(as_completed(futures), total=len(futures)):
                df = future.result()
                if not df.empty:
                    all_pot.append(df)

        if all_pot:
            pot_df = pd.concat(all_pot, ignore_index=True)
            pot_path = RAW_DIR / "pot_all_stations.parquet"
            pot_df.to_parquet(pot_path, index=False)
            print(f"  Saved {len(pot_df):,} POT records to {pot_path}")

    # 4. Catchment descriptors — parallel fetch
    if fetch_descriptors:
        print(f"\nFetching FEH catchment descriptors ({_NRFA_WORKERS} workers)...")
        descriptors = []
        with ThreadPoolExecutor(max_workers=_NRFA_WORKERS) as pool:
            futures = {pool.submit(fetch_catchment_descriptors, sid): sid for sid in station_ids}
            for future in tqdm(as_completed(futures), total=len(futures)):
                descriptors.append(future.result())

        desc_df = pd.DataFrame(descriptors)
        desc_path = RAW_DIR / "catchment_descriptors.parquet"
        desc_df.to_parquet(desc_path, index=False)
        print(f"  Saved descriptors to {desc_path}")

    print("\nNRFA pipeline complete.")
    return catalogue


if __name__ == "__main__":
    # Quick test — first 50 stations
    run_full_pipeline(max_stations=50, fetch_pot=True, fetch_descriptors=True)
