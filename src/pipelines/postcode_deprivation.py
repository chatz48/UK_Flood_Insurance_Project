"""
English Index of Multiple Deprivation 2019 — postcode-level features.

Source: MHCLG (Ministry of Housing, Communities & Local Government)
File 7: All IoD2019 Scores, Ranks, Deciles and Population Denominators

IMD adjusts the vulnerability curve: deprived areas have older housing stock
with less flood-resilient construction, fewer pre-flood mitigation measures
(raised electrics, flood barriers, resilience grant uptake), and higher damage
fractions at equivalent flood depth. Expected effect: increases losses in urban
flood-prone areas (Hull, Sheffield, York), improving calibration for Hull 2007
and Sheffield 2007 events.

This pipeline must not break if data sources are unavailable (returns empty
DataFrame and prints a warning).
"""

import requests
import pandas as pd
import io
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "imd"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

# File 7 from gov.uk: all scores, ranks, deciles and population denominators
IMD_URL = (
    "https://assets.publishing.service.gov.uk/government/uploads/system/"
    "uploads/attachment_data/file/845345/"
    "File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
)

# ONS postcode-to-LSOA lookup — tried in order until one succeeds.
# The ArcGIS sharing URLs change with each NSPL release; multiple sources improve resilience.
ONS_LOOKUP_URLS = [
    # ONS Open Geography Portal — NSPL (National Statistics Postcode Lookup) latest
    "https://www.arcgis.com/sharing/rest/content/items/6a46e14a6c2441e3ab08c7b277335558/data",
    # ONS GeoPortal direct NSPL Nov 2022
    "https://geoportal.statistics.gov.uk/datasets/ons::nspd-online-latest-centroids/about",
    # ONS bulk download via Open Data portal (zipped CSV)
    "https://www.arcgis.com/sharing/rest/content/items/a644dd04d18f4592b7d36705f93270d8/data",
]


def download_imd() -> pd.DataFrame:
    """
    Download and parse IMD 2019 (File 7) at LSOA level.
    Returns DataFrame with: lsoa_code, imd_decile, income_deprivation_score,
    living_env_score.
    """
    raw_path = RAW_DIR / "imd_2019.csv"
    if raw_path.exists():
        print(f"  IMD: loading from cache ({raw_path})")
        return _parse_imd_csv(raw_path.read_bytes())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading IMD 2019 (File 7) from gov.uk...")
    try:
        r = requests.get(IMD_URL, timeout=120)
        r.raise_for_status()
        raw_path.write_bytes(r.content)
        print(f"    Saved to {raw_path}")
        return _parse_imd_csv(r.content)
    except Exception as e:
        print(f"    Warning: IMD download failed: {e}")
        return pd.DataFrame()


def _parse_imd_csv(content: bytes) -> pd.DataFrame:
    """Parse IMD CSV, mapping verbose column headers to short internal names."""
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    except Exception as e:
        print(f"    Warning: CSV parse failed: {e}")
        return pd.DataFrame()

    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if "lsoa code" in cl:
            col_map[col] = "lsoa_code"
        elif (
            ("index of multiple deprivation" in cl or cl == "imd score")
            and "decile" in cl
            and "income" not in cl
            and "imd_decile" not in col_map.values()  # take only first match
        ):
            col_map[col] = "imd_decile"
        elif (
            "income score" in cl
            and "rate" in cl
            and "idaci" not in cl
            and "idaopi" not in cl
            and "income_deprivation_score" not in col_map.values()
        ):
            # Specifically "Income Score (rate)" — NOT IDACI or IDAOPI
            col_map[col] = "income_deprivation_score"
        elif (
            "living environment score" in cl
            and "living_env_score" not in col_map.values()
        ):
            col_map[col] = "living_env_score"

    df = df.rename(columns=col_map)
    needed = ["lsoa_code", "imd_decile", "income_deprivation_score", "living_env_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        # Try broader match for the decile column
        for col in df.columns:
            if "decile" in col.lower() and col not in col_map.values():
                df = df.rename(columns={col: "imd_decile"})
                break
        missing = [c for c in needed if c not in df.columns]

    if missing:
        print(f"    Warning: IMD columns not matched: {missing}")
        print(f"    Available columns (first 15): {list(df.columns[:15])}")
        return pd.DataFrame()

    result = df[needed].copy()
    result["lsoa_code"] = result["lsoa_code"].astype(str).str.strip()
    for col in ["imd_decile", "income_deprivation_score", "living_env_score"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    print(f"    Parsed {len(result):,} LSOA records")
    return result.dropna(subset=["lsoa_code"])


def download_postcode_lsoa_lookup() -> pd.DataFrame:
    """
    Download ONS postcode-to-LSOA lookup.
    Returns DataFrame with: postcode, lsoa_code.
    Returns empty DataFrame (without crashing) if unavailable.
    """
    lookup_path = RAW_DIR / "postcode_to_lsoa.parquet"
    if lookup_path.exists():
        print("  Postcode-LSOA lookup: loading from cache")
        return pd.read_parquet(lookup_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading ONS postcode-to-LSOA lookup (~150MB)...")

    content = None
    for url in ONS_LOOKUP_URLS:
        try:
            print(f"    Trying: {url}")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            content = r.content
            print(f"    Success ({len(content) / 1e6:.0f} MB)")
            break
        except Exception as e:
            print(f"    Failed: {e}")

    if content is None:
        print("    Warning: all ONS lookup URLs failed")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8", low_memory=False)

        # Column names differ across ONS releases — try known variants
        col_map = {}
        for col in df.columns:
            cl = col.strip().lower()
            if cl in ("pcd", "pcds", "postcode"):
                col_map[col] = "postcode"
            elif cl in ("lsoa11cd", "lsoa21cd", "lsoa11", "lsoa_code") or (
                "lsoa" in cl and ("cd" in cl or "code" in cl or "11" in cl)
            ):
                col_map[col] = "lsoa_code"

        df = df.rename(columns=col_map)
        if "postcode" not in df.columns or "lsoa_code" not in df.columns:
            print(f"    Warning: unexpected columns: {list(df.columns[:10])}")
            return pd.DataFrame()

        df = df[["postcode", "lsoa_code"]].dropna()
        df["postcode"] = df["postcode"].str.strip().str.upper()
        df["lsoa_code"] = df["lsoa_code"].str.strip()
        df.to_parquet(lookup_path, index=False)
        print(f"    Saved {len(df):,} postcode-LSOA mappings to cache")
        return df
    except Exception as e:
        print(f"    Warning: ONS lookup download failed: {e}")
        return pd.DataFrame()


def run_pipeline() -> pd.DataFrame:
    """
    Build postcode-level deprivation features from IMD 2019.

    If ONS postcode-LSOA lookup is unavailable, saves LSOA-level data to
    data/raw/imd/imd_lsoa_level.parquet for use in Task 7 portfolio building
    via the postcodes.io per-postcode LSOA lookup.

    Output: data/features/deprivation_features.parquet
    Columns: postcode, imd_decile, income_deprivation_score, living_env_score
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    imd_df = download_imd()
    if imd_df.empty:
        print("  IMD data unavailable — deprivation features skipped")
        return pd.DataFrame()

    print(f"  IMD: {len(imd_df):,} LSOAs")

    lookup_df = download_postcode_lsoa_lookup()
    if lookup_df.empty:
        # Save LSOA-level data for Task 7 join via postcodes.io
        lsoa_path = RAW_DIR / "imd_lsoa_level.parquet"
        imd_df.to_parquet(lsoa_path, index=False)
        print(f"  Saved LSOA-level IMD to {lsoa_path}")
        print("  NOTE: postcode join will be completed in Task 7 via postcodes.io API")
        return imd_df

    merged = lookup_df.merge(imd_df, on="lsoa_code", how="left")
    merged = merged.dropna(subset=["postcode"])

    # One postcode → one LSOA; median handles any edge cases
    features = (
        merged.groupby("postcode")[
            ["imd_decile", "income_deprivation_score", "living_env_score"]
        ]
        .median()
        .reset_index()
    )

    out_path = FEATURES_DIR / "deprivation_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"  Saved {len(features):,} postcode deprivation features → {out_path}")
    return features


if __name__ == "__main__":
    run_pipeline()
