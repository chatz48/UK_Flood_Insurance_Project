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
from pathlib import Path
from tqdm import tqdm

NRFA_API = "https://nrfaapps.ceh.ac.uk/nrfa/ws"
RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "nrfa"


def fetch_station_catalogue() -> pd.DataFrame:
    """
    Fetch full NRFA station catalogue with metadata.
    Returns ~1,500 stations with location, catchment area, and data availability.
    """
    print("Fetching NRFA station catalogue...")
    url = f"{NRFA_API}/station/list"
    params = {
        "format": "json",
        "fields": "id,name,river,location,grid-reference,catchment-area,waterbody-id,"
                  "peak-flow-rejected,pctd-rejected,record-length,gdf-start-date,"
                  "gdf-end-date,feh-pooling-group,sensitivity",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    stations = []
    for s in data.get("data", []):
        stations.append({
            "station_id": s.get("id"),
            "name": s.get("name"),
            "river": s.get("river"),
            "location": s.get("location"),
            "grid_ref": s.get("grid-reference", {}).get("ngr") if isinstance(s.get("grid-reference"), dict) else None,
            "easting": s.get("grid-reference", {}).get("easting") if isinstance(s.get("grid-reference"), dict) else None,
            "northing": s.get("grid-reference", {}).get("northing") if isinstance(s.get("grid-reference"), dict) else None,
            "catchment_area_km2": s.get("catchment-area"),
            "record_start": s.get("gdf-start-date"),
            "record_end": s.get("gdf-end-date"),
            "peak_flow_rejected": s.get("peak-flow-rejected"),
            "sensitivity": s.get("sensitivity"),
            "feh_pooling_group": s.get("feh-pooling-group"),
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
    url = f"{NRFA_API}/time-series/data"
    params = {
        "format": "json",
        "data-type": "amax-flow",
        "station": station_id,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        values = data.get("data", {}).get("values", [])
        if not values:
            return pd.DataFrame()

        records = []
        for entry in values:
            records.append({
                "station_id": station_id,
                "water_year": entry[0],
                "peak_flow_m3s": entry[1],
                "flag": entry[2] if len(entry) > 2 else None,
            })

        df = pd.DataFrame(records)
        df["peak_flow_m3s"] = pd.to_numeric(df["peak_flow_m3s"], errors="coerce")
        return df.dropna(subset=["peak_flow_m3s"])

    except Exception as e:
        return pd.DataFrame()


def fetch_pot_series(station_id: int) -> pd.DataFrame:
    """
    Fetch Peaks Over Threshold (POT) flow series for a station.
    Gives more data points than AMAX — useful for fitting GPD tails.
    """
    url = f"{NRFA_API}/time-series/data"
    params = {
        "format": "json",
        "data-type": "pot-flow",
        "station": station_id,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        values = data.get("data", {}).get("values", [])
        if not values:
            return pd.DataFrame()

        records = []
        for entry in values:
            records.append({
                "station_id": station_id,
                "date": entry[0],
                "peak_flow_m3s": entry[1],
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["peak_flow_m3s"] = pd.to_numeric(df["peak_flow_m3s"], errors="coerce")
        return df.dropna(subset=["peak_flow_m3s"])

    except Exception as e:
        return pd.DataFrame()


def fetch_catchment_descriptors(station_id: int) -> dict:
    """
    Fetch FEH (Flood Estimation Handbook) catchment descriptors.
    These are the physical catchment properties used in regional flood frequency.
    Key descriptors: AREA, SAAR (rainfall), BFIHOST (soil), FARL (lakes), FPEXT (floodplain)
    """
    url = f"{NRFA_API}/station/info"
    params = {
        "format": "json",
        "station": station_id,
        "fields": "catchment-area,feh-descriptors",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        descriptors = data.get("data", {}).get("feh-descriptors", {})
        descriptors["station_id"] = station_id
        return descriptors
    except Exception:
        return {"station_id": station_id}


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

    print(f"\nFetching AMAX series for {len(usable)} stations (>= {min_record_years} years record)...")

    # 2. AMAX series — one file per station, plus combined
    all_amax = []
    for _, row in tqdm(usable.iterrows(), total=len(usable)):
        sid = int(row["station_id"])
        df = fetch_amax_series(sid)
        if not df.empty:
            all_amax.append(df)
        time.sleep(0.1)

    if all_amax:
        amax_df = pd.concat(all_amax, ignore_index=True)
        amax_path = RAW_DIR / "amax_all_stations.parquet"
        amax_df.to_parquet(amax_path, index=False)
        print(f"  Saved {len(amax_df):,} AMAX records ({amax_df['station_id'].nunique()} stations) to {amax_path}")

    # 3. POT series
    if fetch_pot:
        print(f"\nFetching POT series...")
        all_pot = []
        for _, row in tqdm(usable.iterrows(), total=len(usable)):
            sid = int(row["station_id"])
            df = fetch_pot_series(sid)
            if not df.empty:
                all_pot.append(df)
            time.sleep(0.1)

        if all_pot:
            pot_df = pd.concat(all_pot, ignore_index=True)
            pot_path = RAW_DIR / "pot_all_stations.parquet"
            pot_df.to_parquet(pot_path, index=False)
            print(f"  Saved {len(pot_df):,} POT records to {pot_path}")

    # 4. Catchment descriptors
    if fetch_descriptors:
        print(f"\nFetching FEH catchment descriptors...")
        descriptors = []
        for _, row in tqdm(usable.iterrows(), total=len(usable)):
            sid = int(row["station_id"])
            d = fetch_catchment_descriptors(sid)
            descriptors.append(d)
            time.sleep(0.1)

        desc_df = pd.DataFrame(descriptors)
        desc_path = RAW_DIR / "catchment_descriptors.parquet"
        desc_df.to_parquet(desc_path, index=False)
        print(f"  Saved descriptors to {desc_path}")

    print("\nNRFA pipeline complete.")
    return catalogue


if __name__ == "__main__":
    # Quick test — first 50 stations
    run_full_pipeline(max_stations=50, fetch_pot=True, fetch_descriptors=True)
