"""
Thames Flood Realistic Disaster Scenario (RDS)
Lloyd's benchmark: £6.2bn total industry insured loss (Oxford to Teddington, 194km²)

This module estimates the residential-only gross loss for the Thames RDS,
using the calibrated DEFRA FD2320 damage function with depth_offset=0.15m.

IMPORTANT SCOPE NOTE:
  This model covers the residential insured book only (~40-50% of the total
  insured market). The Lloyd's £6.2bn benchmark is industry-wide (residential
  + commercial + infrastructure + contents). The plausible residential-only
  range consistent with the Lloyd's figure is £2-3bn.

Methodology:
  1. Filter exposure to Thames corridor (lon -1.35 to 0.15, lat 51.4 to 51.8)
  2. Filter to EA flood zones 2, 3a, 3b
  3. Assign flood depth per property:
       - If OS Terrain 50 elevation available: depth = 100yr flood level - elevation
       - Fallback: zone-based proxies (zone 2 → 0.3m, 3a → 0.7m, 3b → 1.2m)
  4. Monte Carlo sampling for depth and damage uncertainty (±0.3m depth, ±10% damage)
  5. Apply calibrated DEFRA FD2320 with depth_offset=0.15m and contamination/duration
  6. Output: gross loss central estimate, P10, P90, comparison to Lloyd's benchmark

References:
  Lloyd's Realistic Disaster Scenarios (2023 specification)
  Environment Agency: Thames Tidal Flood Scenario (2022)
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# Thames RDS geographic bounds (Oxford to Teddington)
THAMES_LON_MIN, THAMES_LON_MAX = -1.35, 0.15
THAMES_LAT_MIN, THAMES_LAT_MAX = 51.4, 51.8
THAMES_FLOOD_ZONES = ["2", "3a", "3b"]

# Zone-based fallback depths (metres above floor level) and uncertainty (±σ)
ZONE_DEPTH_PROXIES = {
    "2":  (0.30, 0.20),
    "3a": (0.70, 0.25),
    "3b": (1.20, 0.30),
}

# DEFRA FD2320 calibrated parameters (from best model: a320f43)
DAMAGE_DEPTH_OFFSET = 0.15
CONTAMINATION_RATE = 0.50
MEAN_FLOOD_DURATION_DAYS = 3.5

# Lloyd's RDS benchmark (industry-wide: residential + commercial + infrastructure)
LLOYDS_BENCHMARK_GBP = 6.2e9
RESIDENTIAL_MARKET_SHARE = 0.45  # ~40-50% of total insured market is residential

N_MONTE_CARLO = 10_000
SIMULATION_SEED = 42


def _load_exposure(required_cols: list = None) -> pd.DataFrame:
    """
    Load exposure portfolio. Falls back to a synthetic Thames portfolio if
    data/processed/exposure_portfolio.parquet is not yet built (Task 7).
    """
    portfolio_path = PROCESSED_DIR / "exposure_portfolio.parquet"
    if portfolio_path.exists():
        df = pd.read_parquet(portfolio_path)
        print(f"  Loaded {len(df):,} properties from exposure portfolio")
        return df

    print("  WARNING: exposure_portfolio.parquet not found (Task 7 not yet run)")
    print("  Using synthetic Thames exposure for demonstration")
    return _synthetic_thames_exposure()


def _synthetic_thames_exposure() -> pd.DataFrame:
    """
    Synthetic Thames corridor exposure for demonstration.
    Based on ~180,000 residential properties in EA flood zones 2/3 along Thames.
    """
    np.random.seed(SIMULATION_SEED)
    n = 5000  # representative sample

    # Mix of zones matching approximate EA Zone 2/3 area fractions
    zones = np.random.choice(["2", "3a", "3b"], p=[0.40, 0.45, 0.15], size=n)
    property_types = np.random.choice(["terraced", "semi", "detached", "flat"],
                                       p=[0.35, 0.30, 0.15, 0.20], size=n)

    # London-Thames property values (~£400k median, higher for Thames-side)
    values = np.random.lognormal(mean=12.9, sigma=0.5, size=n)
    tiv = values * 1.25  # rebuild cost uplift

    lons = np.random.uniform(THAMES_LON_MIN, THAMES_LON_MAX, n)
    lats = np.random.uniform(THAMES_LAT_MIN, THAMES_LAT_MAX, n)

    return pd.DataFrame({
        "lat": lats, "lon": lons,
        "flood_zone": zones,
        "property_type": property_types,
        "estimated_tiv": tiv,
        "postcode": [f"THAMES{i:05d}" for i in range(n)],
    })


def _assign_flood_depths(exposure_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Assign flood depths using OS Terrain 50 elevation if available,
    otherwise apply zone-based depth proxies with uncertainty.

    Terrain fallback: zone depth proxy ± uniform uncertainty.
    """
    df = exposure_df.copy()

    # Check for elevation data (from Task 2 OS Terrain 50)
    if "elevation_m" in df.columns and df["elevation_m"].notna().any():
        # Thames 100-year flood level: approx 6.5m AOD at Teddington
        # Use simplified linear tidal gradient: 6.5m at lon=−0.31, 4.0m at lon=−1.35
        lon_vals = df["lon"].clip(THAMES_LON_MIN, THAMES_LON_MAX)
        tidal_fraction = (lon_vals - THAMES_LON_MIN) / (THAMES_LON_MAX - THAMES_LON_MIN)
        flood_level_aod = 4.0 + (6.5 - 4.0) * tidal_fraction
        df["flood_depth_m"] = (flood_level_aod - df["elevation_m"]).clip(lower=0)
        depth_source = "terrain_elevation"
    else:
        # Fallback: zone-based proxies with uncertainty
        depths = np.zeros(len(df))
        for zone, (mean_d, std_d) in ZONE_DEPTH_PROXIES.items():
            mask = df["flood_zone"] == zone
            n_zone = mask.sum()
            if n_zone > 0:
                depths[mask.values] = np.maximum(
                    0,
                    rng.normal(mean_d, std_d, n_zone)
                )
        df["flood_depth_m"] = depths
        depth_source = "zone_proxy_fallback"

    print(f"  Depth assignment method: {depth_source}")
    return df


def _compute_damage_fraction(depth_m: float, property_type: str) -> float:
    """Apply calibrated DEFRA FD2320 with contamination and duration adjustments."""
    from src.vulnerability.damage_functions import defra_fd2320_residential

    adjusted_depth = max(0, depth_m + DAMAGE_DEPTH_OFFSET)
    base = defra_fd2320_residential(adjusted_depth, property_type)

    contamination_adj = 1 + CONTAMINATION_RATE * 0.25
    if MEAN_FLOOD_DURATION_DAYS < 3:
        duration_adj = 1.15
    elif MEAN_FLOOD_DURATION_DAYS < 7:
        duration_adj = 1.30
    else:
        duration_adj = 1.50

    return min(base * contamination_adj * duration_adj, 1.0)


def run_thames_rds_scenario(n_mc: int = N_MONTE_CARLO, seed: int = SIMULATION_SEED) -> dict:
    """
    Run the Thames Flood RDS scenario with Monte Carlo uncertainty quantification.

    Returns:
        dict with:
          gross_loss_central_gbp  — P50 gross residential loss
          gross_loss_p10_gbp      — P10 (optimistic) loss
          gross_loss_p90_gbp      — P90 (pessimistic) loss
          tiv_exposed_gbp         — total insured value in flood zones 2/3
          loss_ratio              — gross_loss / tiv_exposed
          lloyds_comparison       — dict with benchmark comparison and note
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("Thames RDS Scenario")
    print("=" * 55)

    # --- Step 1: Filter to Thames corridor ---
    exposure_df = _load_exposure()

    thames_mask = (
        exposure_df["lon"].between(THAMES_LON_MIN, THAMES_LON_MAX) &
        exposure_df["lat"].between(THAMES_LAT_MIN, THAMES_LAT_MAX) &
        exposure_df["flood_zone"].isin(THAMES_FLOOD_ZONES)
    )
    thames_df = exposure_df[thames_mask].copy()

    if thames_df.empty:
        print("  WARNING: No properties in Thames RDS zone. Check exposure portfolio.")
        return {}

    total_tiv = thames_df["estimated_tiv"].sum()
    print(f"  Thames corridor properties: {len(thames_df):,}")
    print(f"  Flood zones: {thames_df['flood_zone'].value_counts().to_dict()}")
    print(f"  TIV exposed: £{total_tiv/1e9:.2f}bn")

    # --- Step 2: Assign flood depths ---
    thames_df = _assign_flood_depths(thames_df, rng)

    # --- Step 3: Monte Carlo loss sampling ---
    losses = np.zeros(n_mc)
    n_props = len(thames_df)

    prop_types = (
        thames_df["property_type"].values
        if "property_type" in thames_df.columns
        else np.array(["terraced"] * n_props)
    )

    # Pre-compute base damage fractions for each property
    base_damages = np.array([
        _compute_damage_fraction(row["flood_depth_m"], pt)
        for (_, row), pt in zip(thames_df.iterrows(), prop_types)
    ])
    tiv_arr = thames_df["estimated_tiv"].values

    for i in range(n_mc):
        # Add depth uncertainty: ±10% depth perturbation
        depth_perturb = rng.uniform(0.90, 1.10, n_props)
        perturbed_depths = thames_df["flood_depth_m"].values * depth_perturb

        # Recompute damage with perturbed depths
        damages = np.array([
            _compute_damage_fraction(d, p)
            for d, p in zip(perturbed_depths, prop_types)
        ])

        # Add damage uncertainty: ±5% damage fraction noise
        damages *= rng.uniform(0.95, 1.05, n_props)
        damages = np.clip(damages, 0, 1)

        losses[i] = np.sum(tiv_arr * damages)

    # --- Step 4: Summary statistics ---
    p10 = float(np.percentile(losses, 10))
    p50 = float(np.percentile(losses, 50))
    p90 = float(np.percentile(losses, 90))
    loss_ratio = p50 / total_tiv if total_tiv > 0 else 0

    # --- Step 5: Lloyd's benchmark comparison ---
    lloyds_residential_equiv = LLOYDS_BENCHMARK_GBP * RESIDENTIAL_MARKET_SHARE

    result = {
        "gross_loss_central_gbp": p50,
        "gross_loss_p10_gbp": p10,
        "gross_loss_p90_gbp": p90,
        "tiv_exposed_gbp": float(total_tiv),
        "loss_ratio": float(loss_ratio),
        "n_properties": int(len(thames_df)),
        "lloyds_comparison": {
            "lloyds_total_industry_gbp": LLOYDS_BENCHMARK_GBP,
            "lloyds_residential_equivalent_gbp": lloyds_residential_equiv,
            "our_residential_central_gbp": p50,
            "our_as_pct_of_lloyds_residential": p50 / lloyds_residential_equiv * 100,
            "note": (
                "This model covers the residential insured book only (~40-50% of total "
                "insured market). Lloyd's £6.2bn benchmark is industry-wide (residential "
                "+ commercial + infrastructure + contents). Residential-only target range: £2-3bn."
            ),
        },
        "methodology": {
            "damage_function": "DEFRA FD2320",
            "depth_offset_m": DAMAGE_DEPTH_OFFSET,
            "contamination_rate": CONTAMINATION_RATE,
            "mean_flood_duration_days": MEAN_FLOOD_DURATION_DAYS,
            "depth_source": "zone_proxy_fallback" if "elevation_m" not in thames_df.columns else "terrain_elevation",
            "n_monte_carlo": n_mc,
        },
    }

    # Save result
    out_path = RESULTS_DIR / "thames_rds_scenario.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n{'='*55}")
    print("THAMES RDS RESULTS")
    print(f"{'='*55}")
    print(f"  Gross loss P10:  £{p10/1e9:.2f}bn")
    print(f"  Gross loss P50:  £{p50/1e9:.2f}bn  ← central estimate")
    print(f"  Gross loss P90:  £{p90/1e9:.2f}bn")
    print(f"  Loss ratio:      {loss_ratio:.1%}")
    print(f"\n  Lloyd's comparison:")
    print(f"    Industry total (Lloyd's benchmark): £{LLOYDS_BENCHMARK_GBP/1e9:.1f}bn")
    print(f"    Residential equiv (~{RESIDENTIAL_MARKET_SHARE:.0%} of market): £{lloyds_residential_equiv/1e9:.1f}bn")
    print(f"    Our central estimate:               £{p50/1e9:.2f}bn")
    print(f"    Target range (residential-only):    £2-3bn")
    print(f"  Saved to: {out_path}")
    print(f"{'='*55}")

    return result


if __name__ == "__main__":
    run_thames_rds_scenario()
