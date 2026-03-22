"""
Index of Multiple Deprivation (IMD) 2019
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019

Why this matters for insurance cat modelling:
  - IMD Housing & Living Environment sub-indices correlate with:
      * Property age (older = less flood resilient, more damage per metre depth)
      * Housing quality (damp, poor maintenance = faster damage accumulation)
      * Rebuild costs (lower quality = lower £/sqft rebuild, but also lower resilience)
  - Flood risk + deprivation overlap is significant: Zone 3 areas often include
    older industrial/working-class housing (higher damage vulnerability)
  - Could shift damage curve calibration by ~10-20% for high-deprivation areas

Download: direct CSV, no auth, from gov.uk
"""

import requests
import pandas as pd
import io
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "deprivation"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

# LSOA-level IMD 2019 (most granular public data)
IMD_URL = (
    "https://assets.publishing.service.gov.uk/government/uploads/"
    "system/uploads/attachment_data/file/833970/"
    "File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx"
)

# Postcode to LSOA lookup (ONS open geography)
POSTCODE_LSOA_URL = (
    "https://opendata.camden.gov.uk/api/geospatial/tr8t-gqz7/"
    "rows.csv?method=export"
)

# Alternative postcode-LSOA lookup from ONS
ONS_LOOKUP_URL = (
    "https://geoportal.statistics.gov.uk/datasets/ons::postcode-to-output-area-to-lower-layer-super-output-area-to-middle-layer-super-output-area-to-local-authority-district-november-2022-lookup-in-the-uk-2/about"
)


def download_imd() -> pd.DataFrame:
    """Download IMD 2019 data at LSOA level."""
    print("Downloading IMD 2019...")
    try:
        r = requests.get(IMD_URL, timeout=120)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name=0)
        print(f"  Downloaded {len(df):,} LSOAs")
        return df
    except Exception as e:
        print(f"  IMD download failed: {e}")
        return pd.DataFrame()


def download_postcode_lsoa_lookup() -> pd.DataFrame:
    """
    Download ONS postcode to LSOA lookup.
    This maps every UK postcode to its LSOA code.
    """
    print("Downloading postcode-LSOA lookup...")

    # Try ONS direct CSV
    ONS_POSTCODE_URL = (
        "https://www.arcgis.com/sharing/rest/content/items/"
        "6a46e14a6c2441e3ab08c7b277335558/data"
    )
    try:
        r = requests.get(ONS_POSTCODE_URL, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), usecols=["pcds", "lsoa11cd"])
        df.columns = ["postcode", "lsoa_code"]
        df["postcode"] = df["postcode"].str.strip().str.upper()
        print(f"  Downloaded {len(df):,} postcode-LSOA mappings")
        return df
    except Exception as e:
        print(f"  Postcode lookup failed: {e}")
        print("  Alternative: download from geoportal.statistics.gov.uk")
        return pd.DataFrame()


def process_to_features(imd_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join IMD scores to postcodes and extract relevant features.

    Key columns from IMD:
      - Index of Multiple Deprivation Score (overall)
      - Income Deprivation Score
      - Housing and Services Sub-domain Score (living environment)
      - Living Environment Deprivation Score
    """
    if imd_df.empty or lookup_df.empty:
        return pd.DataFrame()

    # Normalise column names
    imd_df.columns = [str(c).strip() for c in imd_df.columns]

    # Find relevant columns
    lsoa_col = next((c for c in imd_df.columns if "lsoa" in c.lower() and "code" in c.lower()), None)
    imd_score_col = next((c for c in imd_df.columns if "index of multiple" in c.lower() and "score" in c.lower()), None)
    housing_col = next((c for c in imd_df.columns if "living environment" in c.lower() and "score" in c.lower()), None)
    income_col = next((c for c in imd_df.columns if "income" in c.lower() and "score" in c.lower()), None)

    if not lsoa_col:
        print(f"  Could not find LSOA column. Available: {list(imd_df.columns[:10])}")
        return pd.DataFrame()

    # Build LSOA feature dataframe
    feat_cols = {lsoa_col: "lsoa_code"}
    if imd_score_col:
        feat_cols[imd_score_col] = "imd_score"
    if housing_col:
        feat_cols[housing_col] = "living_env_score"
    if income_col:
        feat_cols[income_col] = "income_score"

    lsoa_features = imd_df[list(feat_cols.keys())].rename(columns=feat_cols)

    # Convert to decile (1=most deprived, 10=least)
    if "imd_score" in lsoa_features.columns:
        lsoa_features["imd_decile"] = pd.qcut(
            lsoa_features["imd_score"], q=10, labels=False
        ) + 1

    # Join to postcodes
    result = lookup_df.merge(lsoa_features, on="lsoa_code", how="left")

    # Drop LSOA code (not needed by train.py)
    result = result.drop(columns=["lsoa_code"], errors="ignore")

    # Normalise scores to 0-1
    for col in ["imd_score", "living_env_score", "income_score"]:
        if col in result.columns:
            max_val = result[col].max()
            if max_val > 0:
                result[col] = result[col] / max_val

    return result.dropna(subset=["postcode"])


def run_pipeline():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    imd_df = download_imd()
    lookup_df = download_postcode_lsoa_lookup()

    if imd_df.empty or lookup_df.empty:
        print("Required data not available. Feature not produced.")
        return pd.DataFrame()

    # Cache raw files
    if not imd_df.empty:
        imd_df.to_parquet(RAW_DIR / "imd_2019_lsoa.parquet", index=False)
    if not lookup_df.empty:
        lookup_df.to_parquet(RAW_DIR / "postcode_lsoa_lookup.parquet", index=False)

    features_df = process_to_features(imd_df, lookup_df)
    if features_df.empty:
        print("Feature processing failed.")
        return pd.DataFrame()

    out_path = FEATURES_DIR / "deprivation_features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Saved {len(features_df):,} postcodes to {out_path}")
    print(f"Columns: {list(features_df.columns)}")
    return features_df


if __name__ == "__main__":
    run_pipeline()
