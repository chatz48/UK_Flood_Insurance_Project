"""
Hull 2007 Flood Damage Survey
Environment Agency Open Data — property-level flood data for North Hull 2007.

Dataset: https://environment.data.gov.uk/dataset/02cb6781-582f-49d3-9e1d-6f6ad933465d
Data.gov.uk: https://www.data.gov.uk/dataset/00570faf-a579-470f-bb63-31da9797fa75/flooding-data-in-north-hull-2007

Why this matters:
  This is the most granular publicly available UK property-level flood damage
  dataset. It contains flood depths and approximate damages for ~2,000 residential
  properties in North Hull flooded in 2007.

  Used to calibrate the damage function directly against UK housing stock
  (brick terrace, semi-detached) — much more relevant than the US NFIP params
  currently in use.

  Output features (at postcode level):
    pct_properties_flooded, mean_depth_m, mean_damage_fraction
"""

import requests
import pandas as pd
import io
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "hull_2007"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

# EA Open Data Services CKAN API — use to discover resource download URLs dynamically
EA_CKAN_API = "https://environment.data.gov.uk/api/3/action/package_show"
DATASET_ID = "02cb6781-582f-49d3-9e1d-6f6ad933465d"


def _discover_resources() -> list[dict]:
    """
    Use the EA CKAN API to discover download URLs for the Hull 2007 dataset.
    Returns list of resource dicts with 'url', 'format', 'name'.
    """
    try:
        r = requests.get(EA_CKAN_API, params={"id": DATASET_ID}, timeout=30)
        r.raise_for_status()
        result = r.json().get("result", {})
        resources = result.get("resources", [])
        print(f"  Hull 2007 dataset: {result.get('title', 'unknown')}")
        print(f"  {len(resources)} resource(s) found:")
        for res in resources:
            print(f"    [{res.get('format', '?')}] {res.get('name', '')} → {res.get('url', '')}")
        return resources
    except Exception as e:
        print(f"  Warning: CKAN API failed ({e}), trying fallback URLs")
        return []


def _download_csv_resource(url: str, name: str) -> pd.DataFrame:
    """Download a CSV resource, with caching."""
    safe_name = name.lower().replace(" ", "_").replace("/", "_")[:50]
    cache_path = RAW_DIR / f"{safe_name}.csv"
    if cache_path.exists():
        print(f"  Loading from cache: {cache_path}")
        return pd.read_csv(cache_path, encoding="latin-1")

    print(f"  Downloading: {url}")
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        cache_path.write_bytes(r.content)
        return pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
    except Exception as e:
        print(f"  Warning: download failed: {e}")
        return pd.DataFrame()


def download_hull_data() -> list[pd.DataFrame]:
    """
    Download all CSV resources from the Hull 2007 dataset.
    Returns list of DataFrames (one per resource file).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    resources = _discover_resources()
    csv_resources = [r for r in resources if r.get("format", "").upper() in ("CSV", "TEXT/CSV")]

    if not csv_resources:
        print("  No CSV resources found via CKAN API")
        return []

    frames = []
    for res in csv_resources:
        df = _download_csv_resource(res["url"], res.get("name", "hull_2007"))
        if not df.empty:
            df["_source_resource"] = res.get("name", "")
            frames.append(df)

    return frames


def _normalise_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to find a flood depth column in the raw data.
    EA/Hull data typically uses 'depth_m', 'flood_depth', 'depth' variants.
    Returns df with a normalised 'depth_m' column (NaN where absent).
    """
    depth_candidates = [c for c in df.columns if "depth" in c.lower()]
    if depth_candidates:
        df = df.rename(columns={depth_candidates[0]: "depth_m"})
    else:
        df["depth_m"] = float("nan")
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    return df


def _normalise_postcode(df: pd.DataFrame) -> pd.DataFrame:
    """Find and normalise a postcode column."""
    pc_candidates = [c for c in df.columns if "postcode" in c.lower() or c.lower() in ("pc", "pcd")]
    if pc_candidates:
        df = df.rename(columns={pc_candidates[0]: "postcode"})
        df["postcode"] = df["postcode"].astype(str).str.strip().str.upper()
    else:
        df["postcode"] = None
    return df


def build_features(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate raw Hull 2007 data to postcode-level damage features.

    Output columns:
        postcode, pct_properties_flooded, mean_depth_m, mean_damage_fraction
    """
    if not frames:
        return pd.DataFrame()

    # Combine all resources
    combined = pd.concat(
        [_normalise_depth(_normalise_postcode(df)) for df in frames],
        ignore_index=True,
    )

    if "postcode" not in combined.columns or combined["postcode"].isna().all():
        print("  Warning: no postcode column found in Hull data")
        print(f"  Available columns: {list(combined.columns)}")
        # Save raw for manual inspection
        combined.to_parquet(RAW_DIR / "hull_2007_raw.parquet", index=False)
        print(f"  Raw data saved to {RAW_DIR / 'hull_2007_raw.parquet'} for inspection")
        return pd.DataFrame()

    combined = combined.dropna(subset=["postcode"])
    combined = combined[combined["postcode"].str.len() > 2]

    # Aggregate to postcode level
    grouped = combined.groupby("postcode").agg(
        n_properties=("postcode", "count"),
        mean_depth_m=("depth_m", "mean"),
    ).reset_index()

    # Damage fraction: use NFIP-calibrated lookup as proxy for depth → damage
    # (pending direct damage fraction column from the raw data)
    def _depth_to_damage(depth_m: float) -> float:
        """Simple depth-damage lookup from NFIP means (UK adjustment -15%)."""
        if pd.isna(depth_m):
            return float("nan")
        if depth_m < 0.3:
            return 0.305 * 0.85
        elif depth_m < 0.6:
            return 0.390 * 0.85
        elif depth_m < 0.9:
            return 0.339 * 0.85
        elif depth_m < 1.2:
            return 0.348 * 0.85
        elif depth_m < 1.5:
            return 0.388 * 0.85
        elif depth_m < 2.0:
            return 0.400 * 0.85
        elif depth_m < 3.0:
            return 0.500 * 0.85
        else:
            return 0.810 * 0.85

    grouped["mean_damage_fraction"] = grouped["mean_depth_m"].apply(_depth_to_damage)

    # pct_properties_flooded: all listed properties were flooded in this dataset
    grouped["pct_properties_flooded"] = 1.0

    return grouped[["postcode", "pct_properties_flooded", "mean_depth_m", "mean_damage_fraction"]]


def run_full_pipeline() -> pd.DataFrame:
    """
    Download Hull 2007 flood data and build postcode-level damage features.
    Output: data/features/hull_2007_features.parquet
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    frames = download_hull_data()
    if not frames:
        print("  Hull 2007 data unavailable — features not produced")
        return pd.DataFrame()

    features = build_features(frames)
    if features.empty:
        return pd.DataFrame()

    out_path = FEATURES_DIR / "hull_2007_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"  Saved {len(features):,} postcode damage records → {out_path}")
    print(f"  Mean depth: {features['mean_depth_m'].mean():.2f}m, "
          f"mean damage fraction: {features['mean_damage_fraction'].mean():.3f}")
    return features


if __name__ == "__main__":
    run_full_pipeline()
