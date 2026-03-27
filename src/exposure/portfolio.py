"""
Exposure Portfolio Builder

Combines Land Registry price paid data, postcodes.io geocoding, EA flood zone
spatial data, and optionally OS Terrain 50 elevation features into a single
exposure portfolio parquet.

Pipeline:
  1. Load Land Registry price paid CSVs from data/raw/land_registry/
  2. Geocode unique postcodes via postcodes.io bulk API (batches of 100, 0.5s sleep)
  3. Spatial join geocoded points to EA flood zone polygons (geopandas sjoin)
  4. Join terrain features from data/features/terrain_features.parquet (if available)
  5. Estimate TIV = last_sale_price × 1.25
  6. Save to data/processed/exposure_portfolio.parquet with checkpoint support

Output columns:
  transaction_id, lat, lon, postcode, county, district,
  property_type, flood_zone, last_sale_price, estimated_tiv,
  elevation_m, slope_degrees, imd_decile, year_of_sale
"""

import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on the path when run directly
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROCESSED_DIR / "portfolio_checkpoints"

TIV_INFLATION_FACTOR = 1.25  # transaction price → rebuild cost
POSTCODES_IO_URL = "https://api.postcodes.io/postcodes"
POSTCODES_IO_BATCH_SIZE = 100
POSTCODES_IO_SLEEP = 0.5  # seconds between batches (rate limit)

PROPERTY_TYPE_MAP = {
    "D": "detached", "S": "semi", "T": "terraced", "F": "flat", "O": "other",
}


# ===========================================================================
# Step 1: Load Land Registry
# ===========================================================================

def load_land_registry() -> pd.DataFrame:
    """
    Load Land Registry price paid data from parquet cache.
    Reads postcode_aggregated.parquet (built by the land_registry pipeline).
    Returns per-transaction DataFrame with postcode, price, property_type.
    """
    lr_dir = RAW_DIR / "land_registry"

    # Prefer full combined parquet
    combined_path = lr_dir / "price_paid_combined.parquet"
    if combined_path.exists():
        df = pd.read_parquet(combined_path)
        print(f"  Land Registry: {len(df):,} transactions loaded")
        return df

    # Fall back to individual year parquets
    year_files = sorted(lr_dir.glob("pp-*.parquet"))
    if not year_files:
        print("  WARNING: No Land Registry data found. Run task 4 first.")
        return pd.DataFrame()

    frames = [pd.read_parquet(p) for p in year_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Land Registry: {len(df):,} transactions from {len(year_files)} years")
    return df


def prepare_portfolio_raw(lr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean Land Registry columns needed for portfolio building.
    Returns one row per transaction.
    """
    if lr_df.empty:
        return pd.DataFrame()

    cols_needed = ["transaction_id", "price", "date_of_transfer", "postcode",
                   "property_type", "district", "county", "year"]
    available = [c for c in cols_needed if c in lr_df.columns]
    df = lr_df[available].copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price", "postcode"])
    df["postcode"] = df["postcode"].str.strip().str.upper()
    df["property_type"] = df["property_type"].map(PROPERTY_TYPE_MAP).fillna("other")
    df["estimated_tiv"] = df["price"] * TIV_INFLATION_FACTOR

    if "year" not in df.columns and "date_of_transfer" in df.columns:
        df["year"] = pd.to_datetime(df["date_of_transfer"], errors="coerce").dt.year

    # Remove obvious data errors
    df = df[(df["price"] > 10_000) & (df["price"] < 50_000_000)]

    print(f"  Portfolio raw: {len(df):,} transactions, {df['postcode'].nunique():,} unique postcodes")
    return df


# ===========================================================================
# Step 2: Geocode postcodes via postcodes.io
# ===========================================================================

def geocode_postcodes(postcodes: list) -> pd.DataFrame:
    """
    Geocode a list of UK postcodes to lat/lon using postcodes.io bulk API.
    Processes in batches of 100 with 0.5s sleep between batches.
    Returns DataFrame: postcode, lat, lon, lsoa_code, district, county.
    """
    checkpoint_path = CHECKPOINT_DIR / "geocoded_postcodes.parquet"
    if checkpoint_path.exists():
        cached = pd.read_parquet(checkpoint_path)
        cached_postcodes = set(cached["postcode"].values)
        remaining = [p for p in postcodes if p not in cached_postcodes]
        if not remaining:
            print(f"  Geocoding: all {len(postcodes):,} postcodes in cache")
            return cached
        print(f"  Geocoding: {len(cached):,} cached, {len(remaining):,} remaining")
    else:
        cached = pd.DataFrame()
        remaining = postcodes

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    n_batches = (len(remaining) + POSTCODES_IO_BATCH_SIZE - 1) // POSTCODES_IO_BATCH_SIZE

    print(f"  Geocoding {len(remaining):,} postcodes in {n_batches} batches...")
    for i in range(0, len(remaining), POSTCODES_IO_BATCH_SIZE):
        batch = remaining[i:i + POSTCODES_IO_BATCH_SIZE]
        batch_num = i // POSTCODES_IO_BATCH_SIZE + 1

        if batch_num % 50 == 0:
            print(f"    Batch {batch_num}/{n_batches}")

        try:
            r = requests.post(
                POSTCODES_IO_URL,
                json={"postcodes": batch},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()

            for item in data.get("result", []):
                query = item.get("query", "")
                result = item.get("result")
                if result:
                    results.append({
                        "postcode": query,
                        "lat": result.get("latitude"),
                        "lon": result.get("longitude"),
                        "lsoa_code": result.get("codes", {}).get("lsoa"),
                        "district": result.get("admin_district"),
                        "county": result.get("admin_county"),
                    })
        except Exception as e:
            print(f"    Warning: batch {batch_num} failed: {e}")

        time.sleep(POSTCODES_IO_SLEEP)

        # Save checkpoint every 200 batches
        if batch_num % 200 == 0 and results:
            interim = pd.DataFrame(results)
            if not cached.empty:
                interim = pd.concat([cached, interim], ignore_index=True)
            interim.to_parquet(checkpoint_path, index=False)

    new_geo = pd.DataFrame(results) if results else pd.DataFrame()
    if not cached.empty and not new_geo.empty:
        geo_df = pd.concat([cached, new_geo], ignore_index=True)
    elif not cached.empty:
        geo_df = cached
    else:
        geo_df = new_geo

    if not geo_df.empty:
        geo_df.to_parquet(checkpoint_path, index=False)
        print(f"  Geocoded {len(geo_df):,} postcodes")

    return geo_df


# ===========================================================================
# Step 3: Spatial join to EA flood zones
# ===========================================================================

def assign_flood_zones(geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spatial join geocoded property points to EA flood zone polygons.
    Assigns flood_zone: '1', '2', '3a', '3b', or 'none'.

    Uses data/raw/ea_flood_zones/rofrs_zones_by_region.parquet if available.
    Falls back to 'none' (flood zone 1 / outside mapped area) if unavailable.
    """
    ea_dir = RAW_DIR / "ea_flood_zones"
    zone_path = ea_dir / "rofrs_zones_by_region.parquet"

    if not zone_path.exists() or geo_df.empty:
        print("  Flood zones: EA data not available — defaulting to zone 'none'")
        geo_df = geo_df.copy()
        geo_df["flood_zone"] = "none"
        return geo_df

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        zones_gdf = gpd.read_parquet(zone_path)

        # Build GeoDataFrame from geocoded points
        geo_df = geo_df.dropna(subset=["lat", "lon"])
        points_gdf = gpd.GeoDataFrame(
            geo_df,
            geometry=[Point(lon, lat) for lon, lat in zip(geo_df["lon"], geo_df["lat"])],
            crs="EPSG:4326",
        )

        # Ensure zones are in same CRS
        if zones_gdf.crs and str(zones_gdf.crs) != "EPSG:4326":
            zones_gdf = zones_gdf.to_crs("EPSG:4326")

        # Map RoFRS risk band (1=High, 2=Medium, 3=Low, 4=Very Low) to zone labels
        zone_col = next((c for c in zones_gdf.columns if "risk" in c.lower() or "band" in c.lower() or "category" in c.lower()), None)
        if zone_col:
            zone_map = {1: "3a", 2: "3a", 3: "2", 4: "1"}
            zones_gdf["flood_zone"] = zones_gdf[zone_col].map(zone_map).fillna("none")
        else:
            zones_gdf["flood_zone"] = "3a"  # assume high risk if column not identified

        # Spatial join
        joined = gpd.sjoin(
            points_gdf,
            zones_gdf[["geometry", "flood_zone"]],
            how="left",
            predicate="within",
        )

        # Handle duplicates (property in multiple zones: take highest risk)
        risk_order = {"3b": 0, "3a": 1, "2": 2, "1": 3, "none": 4}
        joined["risk_rank"] = joined["flood_zone"].map(risk_order).fillna(4)
        joined = joined.sort_values("risk_rank").drop_duplicates(subset=["postcode"], keep="first")
        joined["flood_zone"] = joined["flood_zone"].fillna("none")

        result = joined.drop(columns=["geometry", "risk_rank", "index_right"], errors="ignore")
        print(f"  Flood zone assignment: {result['flood_zone'].value_counts().to_dict()}")
        return pd.DataFrame(result)

    except ImportError:
        print("  WARNING: geopandas not installed — flood zones defaulting to 'none'")
        geo_df = geo_df.copy()
        geo_df["flood_zone"] = "none"
        return geo_df
    except Exception as e:
        print(f"  WARNING: flood zone join failed: {e}")
        geo_df = geo_df.copy()
        geo_df["flood_zone"] = "none"
        return geo_df


# ===========================================================================
# Step 4: Join terrain features (if available)
# ===========================================================================

def join_terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join OS Terrain 50 elevation features if available."""
    terrain_path = FEATURES_DIR / "terrain_features.parquet"
    if not terrain_path.exists():
        df["elevation_m"] = np.nan
        df["slope_degrees"] = np.nan
        return df

    terrain_df = pd.read_parquet(terrain_path)
    if "postcode" in terrain_df.columns:
        df = df.merge(terrain_df[["postcode", "elevation_m", "slope_degrees"]], on="postcode", how="left")
        print(f"  Terrain features joined: {df['elevation_m'].notna().sum():,} properties with elevation data")
    return df


# ===========================================================================
# Step 5: Join IMD deprivation features (if available)
# ===========================================================================

def join_deprivation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join IMD 2019 deprivation features if available."""
    dep_path = FEATURES_DIR / "deprivation_features.parquet"
    if not dep_path.exists():
        df["imd_decile"] = np.nan
        return df

    dep_df = pd.read_parquet(dep_path)
    if "postcode" in dep_df.columns and "imd_decile" in dep_df.columns:
        df = df.merge(dep_df[["postcode", "imd_decile"]], on="postcode", how="left")
        print(f"  Deprivation features: {df['imd_decile'].notna().sum():,} properties with IMD data")
    return df


# ===========================================================================
# Main pipeline
# ===========================================================================

def build_exposure_portfolio(
    max_transactions: int = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build the exposure portfolio from Land Registry, postcodes.io, EA flood zones.

    Resumable: caches intermediate results to data/processed/portfolio_checkpoints/.

    Args:
        max_transactions: limit for testing (None = all)
        force_rebuild: ignore existing output and rebuild from scratch

    Returns:
        DataFrame saved to data/processed/exposure_portfolio.parquet
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "exposure_portfolio.parquet"
    if output_path.exists() and not force_rebuild:
        df = pd.read_parquet(output_path)
        print(f"  Loaded existing portfolio: {len(df):,} properties")
        return df

    print("\nBuilding exposure portfolio...")
    print("=" * 55)

    # --- Step 1: Land Registry ---
    lr_df = load_land_registry()
    if lr_df.empty:
        print("  No Land Registry data — cannot build portfolio")
        return pd.DataFrame()

    portfolio_df = prepare_portfolio_raw(lr_df)
    if max_transactions:
        portfolio_df = portfolio_df.head(max_transactions)

    # --- Step 2: Geocode postcodes ---
    unique_postcodes = portfolio_df["postcode"].unique().tolist()
    geo_df = geocode_postcodes(unique_postcodes)

    if geo_df.empty:
        print("  Geocoding failed — cannot build portfolio")
        return pd.DataFrame()

    # Join geocoding results back to transactions
    portfolio_df = portfolio_df.merge(
        geo_df[["postcode", "lat", "lon"]].dropna(),
        on="postcode",
        how="inner",
    )
    print(f"  After geocoding join: {len(portfolio_df):,} transactions with coordinates")

    # --- Step 3: Flood zone assignment ---
    # Work at postcode level for spatial join (one join per unique postcode)
    postcode_geo = geo_df[["postcode", "lat", "lon"]].dropna().drop_duplicates("postcode")
    postcode_geo = assign_flood_zones(postcode_geo)

    portfolio_df = portfolio_df.merge(
        postcode_geo[["postcode", "flood_zone"]],
        on="postcode",
        how="left",
    )
    portfolio_df["flood_zone"] = portfolio_df["flood_zone"].fillna("none")

    # --- Step 4: Terrain features ---
    portfolio_df = join_terrain_features(portfolio_df)

    # --- Step 5: IMD deprivation ---
    portfolio_df = join_deprivation_features(portfolio_df)

    # --- Rename to canonical output columns ---
    rename_map = {
        "price": "last_sale_price",
        "year": "year_of_sale",
    }
    portfolio_df = portfolio_df.rename(columns=rename_map)

    out_cols = [
        "transaction_id", "lat", "lon", "postcode", "county", "district",
        "property_type", "flood_zone", "last_sale_price", "estimated_tiv",
        "elevation_m", "slope_degrees", "imd_decile", "year_of_sale",
    ]
    available_out = [c for c in out_cols if c in portfolio_df.columns]
    portfolio_df = portfolio_df[available_out]

    # Save
    portfolio_df.to_parquet(output_path, index=False)
    print(f"\n  Exposure portfolio saved: {len(portfolio_df):,} properties → {output_path}")
    print(f"  Flood zone breakdown:\n{portfolio_df['flood_zone'].value_counts()}")

    return portfolio_df


if __name__ == "__main__":
    build_exposure_portfolio()
