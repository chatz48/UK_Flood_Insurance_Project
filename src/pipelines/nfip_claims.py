"""
FEMA NFIP Redacted Claims Dataset — Wing et al. (2020) beta vulnerability fitting.

Source: FEMA OpenFEMA
URL: https://www.fema.gov/about/reports-and-data/openfema/v2/FimaNfipClaimsV2.parquet
Size: ~200MB. No registration required.

Purpose:
  Fit beta distributions to damage ratio (claim / coverage) per flood depth bin.
  Following Wing et al. (2020, Nature Communications) "Estimates of present and
  future flood risk in the conterminous United States".

IMPORTANT CAVEAT: NFIP data is US-specific. Transferability to UK:
  - The distribution *shape* (bimodal at 0 and 1, skewed by depth) is the
    transferable insight — UK damage ratios likely have similar characteristics.
  - The absolute depth thresholds differ: UK houses have higher floor levels
    relative to external ground level than US slab-on-grade construction.
  - Always present Wing beta alongside DEFRA FD2320 as a sensitivity check,
    not as a replacement for UK-calibrated data.

Output: Wing et al. beta parameters cached to data/raw/nfip_claims/wing_beta_params.json
"""

import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "nfip_claims"
PARAMS_PATH = RAW_DIR / "wing_beta_params.json"

NFIP_URL = "https://www.fema.gov/about/reports-and-data/openfema/v2/FimaNfipClaimsV2.parquet"

# Depth bins in metres (0.3m = 1 foot intervals as per Wing et al.)
DEPTH_BINS_M = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5, float("inf")]
DEPTH_BIN_LABELS = [
    "0-0.3m", "0.3-0.6m", "0.6-0.9m", "0.9-1.2m", "1.2-1.5m",
    "1.5-1.8m", "1.8-2.1m", "2.1-2.4m", "2.4-2.7m", "2.7-3.0m",
    "3.0-3.5m", "3.5m+",
]

# Residential occupancy types in NFIP (1=single family, 2=2-4 family, 11=condo)
RESIDENTIAL_TYPES = [1, 2, 11]


def download_nfip() -> pd.DataFrame:
    """
    Download NFIP claims parquet. ~200MB. Cached after first download.
    Returns only the columns needed for Wing beta fitting.
    """
    raw_path = RAW_DIR / "FimaNfipClaimsV2.parquet"
    if raw_path.exists():
        print(f"  NFIP: loading from cache ({raw_path})")
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading NFIP claims (~200MB)...")
        try:
            r = requests.get(NFIP_URL, timeout=600, stream=True)
            r.raise_for_status()
            with open(raw_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"    Saved to {raw_path}")
        except Exception as e:
            print(f"    Warning: NFIP download failed: {e}")
            return pd.DataFrame()

    needed_cols = [
        "waterdepth",
        "amountpaidonbuildingclaim",
        "totalbuildinginsuranccoverage",
        "occupancytype",
    ]
    try:
        df = pd.read_parquet(raw_path, columns=needed_cols)
        print(f"    Loaded {len(df):,} NFIP claims")
        return df
    except Exception as e:
        print(f"    Warning: failed to read parquet: {e}")
        # Try without column filtering (column names may vary)
        try:
            df = pd.read_parquet(raw_path)
            print(f"    Loaded {len(df):,} NFIP claims (all columns)")
            return df
        except Exception as e2:
            print(f"    Error: {e2}")
            return pd.DataFrame()


def prepare_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and prepare NFIP claims for beta fitting.

    Filters:
      - Residential occupancy only (types 1, 2, 11)
      - Positive water depth
      - Positive building claim amount
      - Positive building coverage (avoid division by zero)

    Converts:
      - waterdepth: inches → metres (× 0.0254)
      - damage_ratio: claim / coverage, clipped to [0, 1]
    """
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Find column names (may vary across NFIP data vintages)
    depth_col = next(
        (c for c in df.columns if "waterdepth" in c or "water_depth" in c), None
    )
    claim_col = next(
        (c for c in df.columns if "amountpaid" in c and "building" in c), None
    )
    coverage_col = next(
        (c for c in df.columns if "totalbuildinginsurance" in c or "buildinginsurancc" in c), None
    )
    occ_col = next(
        (c for c in df.columns if "occupancy" in c), None
    )

    if not all([depth_col, claim_col, coverage_col]):
        print(f"    Warning: expected columns not found. Available: {list(df.columns[:15])}")
        return pd.DataFrame()

    df = df.copy()
    df["depth_m"] = pd.to_numeric(df[depth_col], errors="coerce") * 0.0254  # inches → metres
    df["claim"] = pd.to_numeric(df[claim_col], errors="coerce")
    df["coverage"] = pd.to_numeric(df[coverage_col], errors="coerce")

    # Filter to residential with positive depth and valid claim
    mask = (
        (df["depth_m"] > 0)
        & (df["claim"] > 0)
        & (df["coverage"] > 0)
    )
    if occ_col:
        df["occ"] = pd.to_numeric(df[occ_col], errors="coerce")
        mask &= df["occ"].isin(RESIDENTIAL_TYPES)

    df = df[mask].copy()
    df["damage_ratio"] = (df["claim"] / df["coverage"]).clip(0, 1)

    print(f"    {len(df):,} residential claims with valid depth and damage ratio")
    return df[["depth_m", "damage_ratio"]]


def fit_beta_per_bin(claims_df: pd.DataFrame) -> dict:
    """
    Fit scipy beta distribution to damage ratios within each depth bin.

    For each bin, stores: alpha, beta, n_claims, mean, std.
    floc=0 and fscale=1 fix the support to [0, 1].

    Returns dict keyed by depth bin label.
    """
    claims_df = claims_df.copy()
    claims_df["depth_bin"] = pd.cut(
        claims_df["depth_m"],
        bins=DEPTH_BINS_M,
        labels=DEPTH_BIN_LABELS,
        right=True,
    )

    params = {}
    for label in DEPTH_BIN_LABELS:
        ratios = claims_df[claims_df["depth_bin"] == label]["damage_ratio"].dropna().values

        # Need at least 30 claims for a reliable fit
        if len(ratios) < 30:
            params[label] = {
                "alpha": None, "beta_param": None, "n_claims": int(len(ratios)),
                "mean": float(np.mean(ratios)) if len(ratios) > 0 else None,
                "std": float(np.std(ratios)) if len(ratios) > 0 else None,
            }
            continue

        try:
            # Clip away exact 0 and 1 (beta undefined at boundaries)
            ratios_clipped = np.clip(ratios, 1e-6, 1 - 1e-6)
            a, b, _, _ = stats.beta.fit(ratios_clipped, floc=0, fscale=1)
            params[label] = {
                "alpha": float(a),
                "beta_param": float(b),
                "n_claims": int(len(ratios)),
                "mean": float(np.mean(ratios)),
                "std": float(np.std(ratios)),
            }
        except Exception as e:
            params[label] = {
                "alpha": None, "beta_param": None, "n_claims": int(len(ratios)),
                "mean": float(np.mean(ratios)) if len(ratios) > 0 else None,
                "std": float(np.std(ratios)) if len(ratios) > 0 else None,
                "fit_error": str(e),
            }

    fitted = sum(1 for v in params.values() if v["alpha"] is not None)
    print(f"    Beta distributions fitted for {fitted}/{len(DEPTH_BIN_LABELS)} depth bins")
    return params


def run_pipeline() -> dict:
    """
    Download NFIP claims, filter to residential, fit Wing beta distributions.
    Saves parameters to data/raw/nfip_claims/wing_beta_params.json.
    Returns the params dict (also read by damage_functions.py at import time).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if PARAMS_PATH.exists():
        print(f"  Wing beta params: loading from cache ({PARAMS_PATH})")
        with open(PARAMS_PATH) as f:
            return json.load(f)

    raw_df = download_nfip()
    if raw_df.empty:
        print("  NFIP data unavailable — Wing beta params not fitted")
        return {}

    claims_df = prepare_claims(raw_df)
    if claims_df.empty:
        print("  Claims preparation failed — Wing beta params not fitted")
        return {}

    params = fit_beta_per_bin(claims_df)
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved Wing beta params to {PARAMS_PATH}")
    return params


if __name__ == "__main__":
    run_pipeline()
