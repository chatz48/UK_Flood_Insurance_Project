"""
Environment Agency Flood Zone Shapefiles
Downloads Flood Zones 1, 2, 3a and 3b for England.

Flood Zone definitions (direct insurance relevance):
  Zone 1: < 0.1% annual probability (low risk)   — no loading
  Zone 2: 0.1–1.0% annual probability (medium)   — moderate loading
  Zone 3a: > 1.0% annual probability (high)       — significant loading / exclusion
  Zone 3b: Functional floodplain                  — typically excluded

Also downloads:
  - Risk of Flooding from Rivers and Sea (RoFRS) — 5m grid, 4 risk bands
  - Historic Flood Outlines — recorded flood extents since records began
  - Flood Warning Areas — zones that receive EA alerts
  - Surface Water Flood Maps (SWFM) — pluvial flooding at 3 return periods

Data portal: https://www.data.gov.uk/dataset/flood-risk-zones
EA GeoServer: https://environment.data.gov.uk/arcgis/rest/services
"""

import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import zipfile
import io
import time

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "ea_flood_zones"

# EA ArcGIS REST API endpoints
EA_ARCGIS = "https://environment.data.gov.uk/arcgis/rest/services"

# WFS endpoints for vector data
EA_WFS = "https://environment.data.gov.uk/spatialdata"

# Direct download URLs for national flood zone datasets
FLOOD_ZONE_DOWNLOADS = {
    # Risk of Flooding from Rivers and Sea — 4-band risk, 5m resolution
    "rofrs_england": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/Risk_of_Flooding_from_Rivers_and_Sea/FeatureServer/0/query",

    # Flood Warning Areas
    "flood_warning_areas": "https://environment.data.gov.uk/spatialdata/flood-warning-areas/wfs",
}

# EA Open Data download page for shapefiles
EA_OPEN_DATA = {
    "flood_zones_3": "https://www.data.gov.uk/dataset/cf494c44-05ba-4324-abb7-a35c3bd7c3ef/flood-zone-3",
    "flood_zones_2_3": "https://www.data.gov.uk/dataset/3c7c7296-0cf2-4d8d-9e0f-5d4e2d0df97d/flood-risk-zones-england",
}


def fetch_rofrs_by_region(region_bbox: tuple, output_name: str) -> gpd.GeoDataFrame:
    """
    Fetch Risk of Flooding from Rivers and Sea for a bounding box.

    NOTE: The EA ArcGIS REST services at environment.data.gov.uk/arcgis currently
    return 500 errors for all FeatureServer query requests. The full statutory
    RoFRS dataset must be manually downloaded as a shapefile:
      https://www.data.gov.uk/dataset/2a6f4a16-31c7-4cf2-a843-ec80bc7e88af

    This function is retained as a stub and always returns an empty GeoDataFrame
    until the EA restores API access.

    Args:
        region_bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
        output_name: label for this region

    Risk bands returned:
        1 = High (>3.3% annual chance)
        2 = Medium (1–3.3%)
        3 = Low (0.1–1%)
        4 = Very Low (<0.1%)
    """
    print(f"  RoFRS for {output_name}: EA ArcGIS endpoint unavailable — returning empty")
    return gpd.GeoDataFrame()


def fetch_flood_warning_areas(max_areas: int = 2000) -> gpd.GeoDataFrame:
    """
    Fetch EA Flood Warning Areas via the EA Real Time Flood Monitoring API.

    Uses the /id/floodAreas endpoint which returns metadata + polygon URL per area.
    Downloads polygon GeoJSON for each area and combines into a GeoDataFrame.

    Note: Full national set is ~24,000 areas; max_areas limits for speed.
    These are flood alert zones — a practical proxy for fluvial flood risk areas
    where full statutory Flood Zone 2/3 shapefiles are unavailable via API.
    """
    print(f"Fetching Flood Warning Areas via EA Monitoring API (up to {max_areas})...")
    api_url = "https://environment.data.gov.uk/flood-monitoring/id/floodAreas"

    features = []
    offset = 0
    batch = 100

    while offset < max_areas:
        try:
            r = requests.get(api_url, params={"_limit": batch, "_offset": offset}, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                break
            for item in items:
                poly_url = item.get("polygon")
                if not poly_url:
                    continue
                try:
                    rp = requests.get(poly_url, timeout=15, headers={"Accept": "application/json"})
                    if rp.status_code == 200:
                        fc = rp.json()
                        for feat in fc.get("features", []):
                            feat["properties"].update({
                                "fwdCode": item.get("notation", ""),
                                "label": item.get("label", ""),
                                "county": item.get("county", ""),
                                "riverOrSea": item.get("riverOrSea", ""),
                            })
                            features.append(feat)
                except Exception:
                    continue
            offset += batch
            time.sleep(1.0)  # EA Monitoring API rate-limits at ~300 req/session
            if offset % 200 == 0:
                print(f"  ... {offset} areas processed, {len(features)} polygons")
        except Exception as e:
            print(f"  Warning: batch at offset {offset} failed: {e}")
            break

    if not features:
        print("  No flood warning area polygons retrieved")
        return gpd.GeoDataFrame()

    import json as _json
    fc = {"type": "FeatureCollection", "features": features}
    gdf = gpd.read_file(io.StringIO(_json.dumps(fc)))
    print(f"  Fetched {len(gdf)} flood warning area polygons")
    return gdf


def fetch_historic_flood_outlines() -> gpd.GeoDataFrame:
    """
    Fetch Historic Flood Map — recorded flood extents since records began.

    Note: The EA ArcGIS REST services at environment.data.gov.uk/arcgis have been
    restructured and the FeatureServer query endpoints currently return errors.
    This function is kept as a stub; data must be manually downloaded from:
      https://www.data.gov.uk/dataset/6df43a9d-1b06-4bdb-8c90-2f07f4daeb9c
    """
    print("Historic Flood Outlines: EA ArcGIS endpoint unavailable (manual download required)")
    print("  Download from: https://www.data.gov.uk/dataset/6df43a9d-1b06-4bdb-8c90-2f07f4daeb9c")
    return gpd.GeoDataFrame()


# Major UK cities bounding boxes for regional downloads
UK_REGIONS = {
    "yorkshire":       (-2.5, 53.3, -0.5, 54.2),
    "midlands":        (-2.5, 52.0, -0.5, 53.0),
    "london_thames":   (-0.6, 51.3,  0.3, 51.7),
    "south_west":      (-5.5, 49.9, -2.0, 51.5),
    "north_east":      (-2.5, 54.5, -0.5, 55.5),
    "east_anglia":     ( 0.0, 51.5,  2.0, 53.0),
    "north_west":      (-3.5, 53.2, -1.8, 54.0),
    "wales":           (-5.3, 51.3, -2.6, 53.5),
}


def run_full_pipeline(max_flood_warning_areas: int = 250):
    """
    Download EA flood spatial datasets.

    What works via API (March 2026):
      - EA Flood Monitoring API: flood warning area polygons (proxy for flood risk)

    What requires manual download (EA ArcGIS REST endpoints return 500/Invalid URL):
      - Flood Map for Planning (Flood Zones 2 / 3a / 3b statutory boundaries)
      - Risk of Flooding from Rivers and Sea (RoFRS) — full national shapefile
      - Historic Flood Outlines

    Manual download instructions are printed at the end.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Flood warning areas via EA Monitoring API (polygon GeoJSON per area)
    warnings_gdf = fetch_flood_warning_areas(max_areas=max_flood_warning_areas)
    if not warnings_gdf.empty:
        path = RAW_DIR / "flood_warning_areas.parquet"
        warnings_gdf.to_parquet(path, index=False)
        print(f"  Saved {len(warnings_gdf)} flood warning areas to {path}")

    # 2. RoFRS by region — EA ArcGIS API currently unavailable
    print("\nRoFRS zone download: EA ArcGIS services unavailable (see manual steps below)")

    # 3. Manual download instructions
    print("\n" + "=" * 65)
    print("MANUAL DOWNLOAD REQUIRED — EA ArcGIS REST API returning errors")
    print("=" * 65)
    print("\nStep 1: Flood Map for Planning — Flood Zone 3 (statutory)")
    print("  https://www.data.gov.uk/dataset/2a6f4a16-31c7-4cf2-a843-ec80bc7e88af")
    print("  → Download SHP or GeoPackage → place in:", RAW_DIR / "manual_downloads/")

    print("\nStep 2: Flood Map for Planning — Flood Zone 2")
    print("  https://www.data.gov.uk/dataset/c7f5dc53-6957-4c7a-9619-d6f0e78b7e7c")
    print("  → Download SHP or GeoPackage → place in:", RAW_DIR / "manual_downloads/")

    print("\nStep 3: Risk of Flooding from Rivers and Sea (RoFRS)")
    print("  https://www.data.gov.uk/dataset/2a6f4a16-31c7-4cf2-a843-ec80bc7e88af")
    print("  → Choose 'Postcodes in Areas at Risk' for tabular postcode join")

    print("\nOnce downloaded, load the SHP/GPKG in assign_flood_zones() in")
    print("src/exposure/portfolio.py by placing as:")
    print(f"  {RAW_DIR}/rofrs_zones_by_region.parquet")
    print("=" * 65)

    print("\nEA Flood Zones pipeline complete.")


if __name__ == "__main__":
    run_full_pipeline()
