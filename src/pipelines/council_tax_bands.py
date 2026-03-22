"""
VOA Council Tax Stock of Properties
https://www.gov.uk/government/statistics/council-tax-stock-of-properties

Why this matters for insurance cat modelling:
  - Council tax band is a direct proxy for property value
  - Unlike Land Registry (only tracks sales), this covers ALL properties
  - Band distribution per postcode tells us the property value MIX
    (Zone 3 postcodes often have lower-value Band A/B properties = lower TIV)
  - More accurate than Land Registry median for total insured value estimation

Download: no auth needed, direct CSV from VOA website
Updated: annually
"""

import requests
import pandas as pd
import io
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "council_tax"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

# VOA council tax stock data URL (check for latest year)
VOA_URL = "https://assets.publishing.service.gov.uk/media/council-tax-stock-of-properties-2023.csv"

# Approximate midpoint values per band (England, 2024)
# Used to compute a weighted average property value per postcode
BAND_VALUES = {
    "A": 52000,   # Up to £40k (1991 values, ~£85k modern)
    "B": 77000,
    "C": 96000,
    "D": 120000,
    "E": 175000,
    "F": 240000,
    "G": 320000,
    "H": 500000,  # Over £320k (1991), no upper limit
}


def download_voa_data() -> pd.DataFrame:
    """Download VOA council tax stock CSV."""
    print("Downloading VOA council tax data...")
    try:
        r = requests.get(VOA_URL, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        print(f"  Downloaded {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Try manually: gov.uk/government/statistics/council-tax-stock-of-properties")
        return pd.DataFrame()


def process_to_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VOA data to postcode level.
    Produces band distribution and weighted average property value.
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Column names vary by year — normalise
    raw_df.columns = [c.strip().lower().replace(" ", "_") for c in raw_df.columns]

    # Need: postcode (or LSOA), band A-H counts
    band_cols = [c for c in raw_df.columns if any(f"band_{b}" in c or f"_band{b}" in c for b in "ABCDEFGH")]

    if not band_cols:
        print("  Could not identify band columns. Columns available:", list(raw_df.columns)[:20])
        return pd.DataFrame()

    # Identify the geographic unit column
    geo_col = next((c for c in raw_df.columns if "postcode" in c or "lsoa" in c or "oa" in c), None)
    if not geo_col:
        print("  Could not identify geographic unit column")
        return pd.DataFrame()

    # Compute band percentages and weighted value
    total = raw_df[band_cols].sum(axis=1).clip(lower=1)

    features = pd.DataFrame()
    features["postcode"] = raw_df[geo_col]

    for band in "ABCDEFGH":
        matching = [c for c in band_cols if band.lower() in c.lower()]
        if matching:
            features[f"pct_band_{band}"] = raw_df[matching[0]] / total

    # Weighted average property value
    weighted_value = sum(
        features.get(f"pct_band_{band}", 0) * BAND_VALUES[band]
        for band in "ABCDEFGH"
    )
    features["weighted_avg_value"] = weighted_value
    features["total_properties"] = total.values

    return features.dropna(subset=["postcode"])


def run_pipeline():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = download_voa_data()
    if raw_df.empty:
        print("No data downloaded. Feature not produced.")
        return

    raw_path = RAW_DIR / "voa_council_tax.parquet"
    raw_df.to_parquet(raw_path, index=False)

    features_df = process_to_features(raw_df)
    if features_df.empty:
        print("Feature processing failed.")
        return

    out_path = FEATURES_DIR / "council_tax_features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Saved {len(features_df):,} postcodes to {out_path}")
    print(f"Columns: {list(features_df.columns)}")
    return features_df


if __name__ == "__main__":
    run_pipeline()
