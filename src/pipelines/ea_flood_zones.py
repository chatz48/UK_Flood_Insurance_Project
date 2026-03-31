"""
Environment Agency Flood Zone Shapefiles
Downloads Flood Zones 2 and 3 for England via the EA WFS service.

Flood Zone definitions (direct insurance relevance):
  Zone 1: < 0.1% annual probability (low risk)   — no loading
  Zone 2: 0.1–1.0% annual probability (medium)   — moderate loading
  Zone 3a: > 1.0% annual probability (high)       — significant loading / exclusion
  Zone 3b: Functional floodplain                  — typically excluded

Data source (2025+):
  The old individual Zone 2 / Zone 3 datasets on data.gov.uk were deprecated
  and removed in April 2025. The consolidated "Flood Map for Planning" dataset
  is now the authoritative source, accessible via WFS:
    Zone 2: https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-2/wfs
    Zone 3: https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-3/wfs

  The EA Real-Time Flood Monitoring API for flood warning areas remains operational:
    https://environment.data.gov.uk/flood-monitoring/id/floodAreas
"""

import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "ea_flood_zones"

# EA WFS endpoints — consolidated Flood Map for Planning (post-April 2025)
EA_WFS = {
    "zone_2": "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-2/wfs",
    "zone_3": "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-3/wfs",
}

# EA Flood Monitoring API for flood warning areas
EA_MONITORING_API = "https://environment.data.gov.uk/flood-monitoring"

# WFS page size — EA WFS returns up to 1000 features per request
WFS_PAGE_SIZE = 1000

# Parallel workers for polygon downloads
_POLYGON_WORKERS = 15
_polygon_semaphore = threading.Semaphore(_POLYGON_WORKERS)


def _wfs_get_feature_count(wfs_url: str, layer_name: str) -> int:
    """Return total feature count for a WFS layer (used for pagination)."""
    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer_name,
        "resultType": "hits",
        "outputFormat": "application/json",
    }
    try:
        r = requests.get(wfs_url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("totalFeatures", 0)
    except Exception as e:
        print(f"  Warning: could not get feature count: {e}")
        return 0


def _fetch_wfs_page(wfs_url: str, layer_name: str, start_index: int) -> list:
    """Fetch one page of WFS features as GeoJSON feature list."""
    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer_name,
        "outputFormat": "application/json",
        "count": WFS_PAGE_SIZE,
        "startIndex": start_index,
    }
    with _polygon_semaphore:
        try:
            r = requests.get(wfs_url, params=params, timeout=60)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception as e:
            print(f"  Warning: WFS page at offset {start_index} failed: {e}")
            return []


def fetch_flood_zone_wfs(zone_key: str, max_features: int = 50000) -> gpd.GeoDataFrame:
    """
    Download an EA flood zone layer via WFS, paginating until complete.

    Args:
        zone_key: "zone_2" or "zone_3"
        max_features: cap total features (large datasets can be 500k+ polygons)

    Returns GeoDataFrame with flood zone polygons, or empty GeoDataFrame on failure.
    """
    wfs_url = EA_WFS[zone_key]
    print(f"  Fetching {zone_key} via WFS: {wfs_url}")

    # Discover the layer name from GetCapabilities
    try:
        cap_r = requests.get(
            wfs_url,
            params={"SERVICE": "WFS", "REQUEST": "GetCapabilities"},
            timeout=30,
        )
        cap_r.raise_for_status()
        # Layer name is typically the first FeatureType Name in the XML
        import re
        names = re.findall(r"<Name>(.*?)</Name>", cap_r.text)
        layer_name = names[0] if names else zone_key
        print(f"    Layer name: {layer_name}")
    except Exception as e:
        print(f"  Warning: GetCapabilities failed ({e}), using default layer name")
        layer_name = zone_key

    total = _wfs_get_feature_count(wfs_url, layer_name)
    if total == 0:
        print(f"  Warning: no features found for {zone_key}")
        return gpd.GeoDataFrame()

    total = min(total, max_features)
    print(f"    {total:,} features to fetch ({(total // WFS_PAGE_SIZE) + 1} pages)...")

    # Parallel page fetches
    offsets = list(range(0, total, WFS_PAGE_SIZE))
    all_features = []

    with ThreadPoolExecutor(max_workers=_POLYGON_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_wfs_page, wfs_url, layer_name, offset): offset
            for offset in offsets
        }
        for future in as_completed(futures):
            all_features.extend(future.result())

    if not all_features:
        return gpd.GeoDataFrame()

    import json as _json
    fc = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.read_file(io.StringIO(_json.dumps(fc)))
    print(f"    Fetched {len(gdf):,} {zone_key} polygons")
    return gdf


def fetch_flood_warning_areas(max_areas: int = 5000) -> gpd.GeoDataFrame:
    """
    Fetch EA Flood Warning Areas via the EA Real-Time Flood Monitoring API.

    Downloads polygon GeoJSON for each area in parallel.
    These are flood alert zones — a practical proxy for fluvial flood risk
    and usable immediately without any API issues.
    """
    print(f"Fetching Flood Warning Areas (up to {max_areas})...")
    api_url = f"{EA_MONITORING_API}/id/floodAreas"

    # Collect all area metadata with pagination
    areas = []
    offset = 0
    batch = 500
    while offset < max_areas:
        try:
            r = requests.get(api_url, params={"_limit": batch, "_offset": offset}, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                break
            areas.extend(items)
            offset += batch
            if len(areas) >= max_areas:
                break
        except Exception as e:
            print(f"  Warning: metadata batch at offset {offset} failed: {e}")
            break

    areas = [a for a in areas if a.get("polygon")]
    print(f"  {len(areas)} areas with polygon URLs — fetching in parallel...")

    def _fetch_polygon(item: dict) -> list:
        with _polygon_semaphore:
            try:
                rp = requests.get(item["polygon"], timeout=15,
                                  headers={"Accept": "application/json"})
                if rp.status_code != 200:
                    return []
                feats = rp.json().get("features", [])
                for f in feats:
                    f["properties"].update({
                        "fwdCode": item.get("notation", ""),
                        "label": item.get("label", ""),
                        "county": item.get("county", ""),
                        "riverOrSea": item.get("riverOrSea", ""),
                    })
                return feats
            except Exception:
                return []

    all_features = []
    with ThreadPoolExecutor(max_workers=_POLYGON_WORKERS) as pool:
        futures = [pool.submit(_fetch_polygon, a) for a in areas]
        for i, future in enumerate(as_completed(futures)):
            all_features.extend(future.result())
            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{len(areas)} polygons done")

    if not all_features:
        print("  No flood warning area polygons retrieved")
        return gpd.GeoDataFrame()

    import json as _json
    gdf = gpd.read_file(io.StringIO(_json.dumps(
        {"type": "FeatureCollection", "features": all_features}
    )))
    print(f"  Fetched {len(gdf):,} flood warning area polygons")
    return gdf


def run_full_pipeline(
    fetch_zones: bool = True,
    fetch_warning_areas: bool = True,
    max_zone_features: int = 50000,
    max_warning_areas: int = 5000,
):
    """
    Download EA flood spatial datasets.

    fetch_zones: Download Flood Zone 2 and 3 polygons via WFS (large, slow)
    fetch_warning_areas: Download flood alert area polygons (faster, good proxy)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if fetch_warning_areas:
        warnings_gdf = fetch_flood_warning_areas(max_areas=max_warning_areas)
        if not warnings_gdf.empty:
            path = RAW_DIR / "flood_warning_areas.parquet"
            warnings_gdf.to_parquet(path, index=False)
            print(f"  Saved {len(warnings_gdf):,} flood warning areas → {path}")

    if fetch_zones:
        for zone_key in ("zone_2", "zone_3"):
            gdf = fetch_flood_zone_wfs(zone_key, max_features=max_zone_features)
            if not gdf.empty:
                path = RAW_DIR / f"flood_{zone_key}.parquet"
                gdf.to_parquet(path, index=False)
                print(f"  Saved {len(gdf):,} {zone_key} polygons → {path}")

    print("\nEA Flood Zones pipeline complete.")


if __name__ == "__main__":
    run_full_pipeline()
