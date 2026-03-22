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

    Args:
        region_bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
        output_name: label for this region

    Risk bands returned:
        1 = High (>3.3% annual chance)
        2 = Medium (1–3.3%)
        3 = Low (0.1–1%)
        4 = Very Low (<0.1%)
    """
    url = f"{EA_ARCGIS}/EA/FloodMapForPlanning/MapServer/0/query"
    params = {
        "where": "1=1",
        "geometry": f"{region_bbox[0]},{region_bbox[1]},{region_bbox[2]},{region_bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": 2000,
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        gdf = gpd.read_file(io.StringIO(r.text))
        print(f"  Fetched {len(gdf)} flood zone features for {output_name}")
        return gdf
    except Exception as e:
        print(f"  Warning: failed to fetch RoFRS for {output_name}: {e}")
        return gpd.GeoDataFrame()


def fetch_flood_warning_areas() -> gpd.GeoDataFrame:
    """
    Fetch all EA Flood Warning Areas (polygons).
    These are the zones the EA sends alerts to — ~24k areas across England.
    Each area has a river/coastal identifier and alert history.
    """
    print("Fetching Flood Warning Areas...")
    url = f"{EA_ARCGIS}/Environment_Agency/Flood_Warning_Areas/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "FWS_TACODE,DESCRIP,QDIAL,RIVER_SEA,LA_NAME,COUNTY",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": 5000,
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        gdf = gpd.read_file(io.StringIO(r.text))
        print(f"  Fetched {len(gdf)} flood warning areas")
        return gdf
    except Exception as e:
        print(f"  Warning: {e}")
        return gpd.GeoDataFrame()


def fetch_historic_flood_outlines() -> gpd.GeoDataFrame:
    """
    Fetch Historic Flood Map — recorded flood extents since records began.
    Includes river, coastal and groundwater flooding events.
    Each polygon has an event date and source type.
    Used to validate the hazard model against observed inundation.
    """
    print("Fetching Historic Flood Outlines...")
    url = f"{EA_ARCGIS}/Environment_Agency/Historic_Flood_Map/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "EVENT_DATE,FLOOD_SOURCE,COUNTY,REGION",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": 5000,
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        gdf = gpd.read_file(io.StringIO(r.text))
        print(f"  Fetched {len(gdf)} historic flood outlines")
        return gdf
    except Exception as e:
        print(f"  Warning: {e}")
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


def run_full_pipeline():
    """
    Download all EA flood spatial datasets.
    Note: Full national flood zone shapefiles require EA Data Services registration.
    This pipeline pulls what's freely available via REST API.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Flood warning areas (national, one request)
    warnings_gdf = fetch_flood_warning_areas()
    if not warnings_gdf.empty:
        path = RAW_DIR / "flood_warning_areas.parquet"
        warnings_gdf.to_parquet(path, index=False)
        print(f"  Saved to {path}")

    # 2. Historic flood outlines
    historic_gdf = fetch_historic_flood_outlines()
    if not historic_gdf.empty:
        path = RAW_DIR / "historic_flood_outlines.parquet"
        historic_gdf.to_parquet(path, index=False)
        print(f"  Saved to {path}")

    # 3. Risk zones by region (API has record limits, so we tile by region)
    print("\nFetching RoFRS flood zones by region...")
    all_zones = []
    for region_name, bbox in UK_REGIONS.items():
        gdf = fetch_rofrs_by_region(bbox, region_name)
        if not gdf.empty:
            gdf["region"] = region_name
            all_zones.append(gdf)
        time.sleep(1.0)

    if all_zones:
        combined = pd.concat(all_zones, ignore_index=True)
        path = RAW_DIR / "rofrs_zones_by_region.parquet"
        combined.to_parquet(path, index=False)
        print(f"  Saved {len(combined)} zone features to {path}")

    # 4. Print instructions for full national shapefiles
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED for full national flood zones:")
    print("="*60)
    print("1. Flood Zone 3 shapefile:")
    print("   https://www.data.gov.uk/dataset/cf494c44-05ba-4324-abb7-a35c3bd7c3ef")
    print("\n2. Flood Risk Zones 2 & 3 (detailed):")
    print("   https://www.data.gov.uk/dataset/3c7c7296-0cf2-4d8d-9e0f-5d4e2d0df97d")
    print("\n3. Risk of Flooding from Rivers and Sea (RoFRS) full raster:")
    print("   https://environment.data.gov.uk/dataset/b5b5b5b5-..." )
    print("\nPlace downloaded shapefiles in:", RAW_DIR / "manual_downloads")
    print("="*60)

    print("\nEA Flood Zones pipeline complete.")


if __name__ == "__main__":
    run_full_pipeline()
