"""
VOA Council Tax Stock of Properties
https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2024

Why this matters for insurance cat modelling:
  - Council tax band is a direct proxy for property value
  - Unlike Land Registry (only tracks sales), this covers ALL properties
  - Band distribution per local authority tells us the property value MIX
    (Zone 3 LAs often have lower-value Band A/B properties = lower TIV)
  - More accurate than Land Registry median for total insured value estimation

Data format: CTSOP1.0 — all dwellings by local authority, council tax band, 2024.
Geography: Local Authority (LAUA level). Joined to portfolio via district name.

Download: ZIP containing annual CSVs, no auth needed.
Updated: annually
"""

import io
import requests
import zipfile
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "council_tax"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

# CTSOP1.0 time-series ZIP (1993–2024), 2024 edition
VOA_ZIP_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "6685468cab5fc5929851b928/CTSOP1-0-1993-2024.zip"
)
VOA_CSV_NAME = "CTSOP1_0_2024_03_31.csv"  # single-year 2024 file inside ZIP

# Approximate midpoint rebuild values per band (England, 2024)
BAND_VALUES = {
    "A": 85_000,
    "B": 125_000,
    "C": 165_000,
    "D": 210_000,
    "E": 300_000,
    "F": 420_000,
    "G": 580_000,
    "H": 900_000,
}


def download_voa_data() -> pd.DataFrame:
    """
    Download VOA CTSOP1.0 ZIP and extract the 2024 single-year CSV.
    Returns raw DataFrame at Local Authority level.
    """
    cache_path = RAW_DIR / "voa_ctsop1_2024.parquet"
    if cache_path.exists():
        print(f"  VOA: loading from cache ({cache_path})")
        return pd.read_parquet(cache_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading VOA CTSOP1.0 ZIP (2024)...")
    try:
        r = requests.get(VOA_ZIP_URL, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(VOA_CSV_NAME))
        print(f"  Downloaded {len(df):,} local authority rows")
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        print(f"  Download failed: {e}")
        return pd.DataFrame()


def process_to_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VOA CTSOP1.0 to Local Authority level features.

    Produces band distribution percentages and weighted average property value
    per Local Authority District, for joining to exposure portfolio via district name.

    Output columns: district, pct_band_A..H, weighted_avg_value, total_properties
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Filter to LAUA rows only (exclude national/regional aggregates)
    la_df = raw_df[raw_df["geography"] == "LAUA"].copy()
    if la_df.empty:
        print("  No LAUA rows found")
        return pd.DataFrame()

    band_cols = [f"band_{b.lower()}" for b in "ABCDEFGH"]
    missing_cols = [c for c in band_cols if c not in la_df.columns]
    if missing_cols:
        print(f"  Missing band columns: {missing_cols}")
        print(f"  Available: {list(la_df.columns)}")
        return pd.DataFrame()

    # Replace '..' (suppressed) with 0
    for col in band_cols:
        la_df[col] = pd.to_numeric(la_df[col].replace("..", "0"), errors="coerce").fillna(0)

    total = la_df[band_cols].sum(axis=1).clip(lower=1)

    features = pd.DataFrame()
    features["district"] = la_df["area_name"].str.strip()

    for band in "ABCDEFGH":
        col = f"band_{band.lower()}"
        features[f"pct_band_{band}"] = la_df[col].values / total.values

    features["weighted_avg_value"] = sum(
        features[f"pct_band_{band}"] * BAND_VALUES[band]
        for band in "ABCDEFGH"
    )
    features["total_properties"] = total.values

    return features.dropna(subset=["district"])


def run_pipeline() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = download_voa_data()
    if raw_df.empty:
        print("  No VOA data — council tax features not produced.")
        return pd.DataFrame()

    features_df = process_to_features(raw_df)
    if features_df.empty:
        print("  Feature processing failed.")
        return pd.DataFrame()

    out_path = FEATURES_DIR / "council_tax_features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"  Saved {len(features_df):,} local authority rows → {out_path}")
    print(f"  Columns: {list(features_df.columns)}")
    print(f"  Weighted avg value range: "
          f"£{features_df['weighted_avg_value'].min():,.0f} – "
          f"£{features_df['weighted_avg_value'].max():,.0f}")
    return features_df


if __name__ == "__main__":
    run_pipeline()
