"""
Feature Registry — Plug-in data sources for train.py

Any file dropped into data/features/ as a parquet with a 'postcode' or
'station_id' column is automatically available to train.py.

The agent can:
  1. Write a new pipeline script in src/pipelines/
  2. Run it to produce a parquet in data/features/
  3. Register it here with a name + join key
  4. Add the feature name to ACTIVE_FEATURES in train.py
  5. val_score tells it whether the new data helped

This means the agent can add entirely new data sources without
rewriting train.py's core structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"


# ===========================================================================
# Feature catalogue — the agent adds entries here as it discovers new sources
# ===========================================================================

FEATURE_CATALOGUE = {
    # --- Already implemented ---

    "nrfa_amax": {
        "description": "NRFA annual maximum flows — core flood frequency input",
        "join_key": "station_id",
        "path": "nrfa_amax_features.parquet",
        "columns": ["station_id", "mean_amax", "cv_amax", "l_skewness", "q_T100_gev"],
        "source": "NRFA API — nrfa.ceh.ac.uk",
        "status": "active",
    },

    "ea_flood_zones": {
        "description": "EA flood zone assignment per postcode (Zone 1/2/3)",
        "join_key": "postcode",
        "path": "ea_flood_zone_features.parquet",
        "columns": ["postcode", "flood_zone", "zone_3_area_pct", "in_zone_3a", "in_zone_3b"],
        "source": "Environment Agency — data.gov.uk",
        "status": "active",
    },

    "land_registry": {
        "description": "Property values and type mix per postcode",
        "join_key": "postcode",
        "path": "land_registry_features.parquet",
        "columns": ["postcode", "median_price", "pct_terraced", "pct_new_build", "transaction_count"],
        "source": "HM Land Registry Price Paid",
        "status": "active",
    },

    # --- Candidate sources for the agent to explore ---

    "ukcp18_climate": {
        "description": "UKCP18 climate change projections — future flood frequency uplift factors",
        "join_key": "grid_cell",
        "path": "ukcp18_features.parquet",
        "columns": ["grid_cell", "rcp85_2050_uplift", "rcp45_2050_uplift", "precip_change_pct"],
        "source": "Met Office UKCP18 — metoffice.gov.uk/research/approach/collaboration/ukcp",
        "status": "candidate",
        "fetch_notes": "Download RCM 12km projections from CEDA archive. Needs free account.",
    },

    "os_addressbase": {
        "description": "OS AddressBase — precise property count and type per postcode",
        "join_key": "postcode",
        "path": "os_addressbase_features.parquet",
        "columns": ["postcode", "property_count", "residential_count", "commercial_count", "build_year_median"],
        "source": "Ordnance Survey AddressBase Premium — Free under PSGA for research",
        "status": "candidate",
        "fetch_notes": "Apply at: https://www.ordnancesurvey.co.uk/business-government/tools-support/psga-member-site",
    },

    "council_tax_bands": {
        "description": "VOA council tax band distribution per postcode — property value proxy",
        "join_key": "postcode",
        "path": "council_tax_features.parquet",
        "columns": ["postcode", "pct_band_A", "pct_band_D", "pct_band_H", "median_band_value"],
        "source": "Valuation Office Agency — voa.gov.uk/corporate/datasets/council-tax-stock-of-properties.html",
        "status": "candidate",
        "fetch_notes": "Direct CSV download, no auth needed. Updated annually.",
    },

    "ea_gauge_statistics": {
        "description": "EA real-time gauging station statistics — flood exceedance frequency",
        "join_key": "station_id",
        "path": "ea_gauge_features.parquet",
        "columns": ["station_id", "days_above_alert", "days_above_warning", "peak_percentile_95"],
        "source": "Environment Agency Flood Monitoring API — environment.data.gov.uk/flood-monitoring",
        "status": "candidate",
        "fetch_notes": "Pull historical readings and compute exceedance statistics per station.",
    },

    "postcode_deprivation": {
        "description": "IMD 2019 deprivation — property age/quality/flood resilience proxy",
        "join_key": "postcode",
        "path": "deprivation_features.parquet",
        "columns": ["postcode", "imd_decile", "income_deprivation_score", "living_env_score"],
        "source": "MHCLG — gov.uk/government/statistics/english-indices-of-deprivation-2019",
        "status": "candidate",
        "fetch_notes": "Run: python src/pipelines/postcode_deprivation.py",
    },

    "insurance_market_penetration": {
        "description": "Flood Re insurance penetration rate by postcode area",
        "join_key": "postcode_area",
        "path": "insurance_penetration_features.parquet",
        "columns": ["postcode_area", "penetration_rate", "cession_rate"],
        "source": "Flood Re annual report — floodre.co.uk/industry/reports",
        "status": "candidate",
        "fetch_notes": "Only available at postcode area level (e.g. 'YO', 'CA') in annual reports. Manual extraction needed.",
    },

    "ceh_soil_wetness": {
        "description": "CEH Soil Wetness Index — antecedent soil moisture affects flood generation",
        "join_key": "grid_1km",
        "path": "soil_wetness_features.parquet",
        "columns": ["grid_1km", "mean_swi", "winter_swi", "antecedent_wet_days"],
        "source": "UKCEH — Environmental Information Platform (EIP)",
        "status": "candidate",
        "fetch_notes": "Access via UKCEH EIP: eidc.ceh.ac.uk. Soil Wetness Index (SWI) dataset.",
    },

    "historic_flood_claims_hull": {
        "description": "Hull 2007 flood event — EA open data, property-level flood depths and damage",
        "join_key": "postcode",
        "path": "hull_2007_features.parquet",
        "columns": ["postcode", "pct_properties_flooded", "mean_depth_m", "mean_damage_fraction"],
        "source": "Environment Agency Open Data — environment.data.gov.uk/dataset/02cb6781-582f-49d3-9e1d-6f6ad933465d",
        "status": "active",
        "fetch_notes": "Run: python src/pipelines/hull_2007.py — uses EA CKAN API to discover download URLs automatically.",
    },

    "noaa_hurdat2": {
        "description": "Atlantic hurricane tracks — not directly applicable but useful for coastal storm surge",
        "join_key": "grid_cell",
        "path": "hurdat2_features.parquet",
        "columns": ["grid_cell", "storm_count_50yr", "max_wind_speed"],
        "source": "NOAA HURDAT2 — nhc.noaa.gov/data/#hurdat",
        "status": "low_priority",
        "fetch_notes": "More relevant for Scottish west coast and Northern Ireland. Less so for inland England.",
    },

    "era5_precipitation": {
        "description": "ERA5 reanalysis precipitation — 80 years of hourly rainfall for return period estimation",
        "join_key": "grid_cell",
        "path": "era5_precip_features.parquet",
        "columns": ["grid_cell", "mean_annual_precip", "p99_daily_precip", "wet_day_frequency"],
        "source": "Copernicus Climate Data Store — cds.climate.copernicus.eu",
        "status": "candidate",
        "fetch_notes": "Register free at cds.climate.copernicus.eu. Use cdsapi Python library. ~80GB for UK full resolution.",
    },
}


def load_feature(feature_name: str, join_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Load a feature file from data/features/.

    Args:
        feature_name: key from FEATURE_CATALOGUE
        join_df: if provided, left-join the feature onto this DataFrame

    Returns:
        Feature DataFrame, or join_df with features merged in
    """
    if feature_name not in FEATURE_CATALOGUE:
        raise ValueError(f"Unknown feature: {feature_name}. Add it to FEATURE_CATALOGUE first.")

    spec = FEATURE_CATALOGUE[feature_name]
    path = FEATURES_DIR / spec["path"]

    if not path.exists():
        print(f"  Feature '{feature_name}' not yet downloaded. Status: {spec['status']}")
        print(f"  Source: {spec['source']}")
        print(f"  Notes: {spec.get('fetch_notes', 'No notes')}")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    if join_df is not None and not df.empty:
        join_key = spec["join_key"]
        if join_key in join_df.columns and join_key in df.columns:
            merged = join_df.merge(df[spec["columns"]], on=join_key, how="left")
            return merged

    return df


def list_available_features() -> pd.DataFrame:
    """Print what's downloaded vs what's candidate."""
    rows = []
    for name, spec in FEATURE_CATALOGUE.items():
        path = FEATURES_DIR / spec["path"]
        rows.append({
            "feature": name,
            "status": spec["status"],
            "downloaded": path.exists(),
            "join_key": spec["join_key"],
            "description": spec["description"][:60],
        })
    return pd.DataFrame(rows)


def load_active_features(
    postcode_df: pd.DataFrame,
    active_features: list,
) -> pd.DataFrame:
    """
    Load and merge all active features onto a postcode DataFrame.
    Called from train.py with the ACTIVE_FEATURES list.
    """
    result = postcode_df.copy()
    for feature_name in active_features:
        spec = FEATURE_CATALOGUE.get(feature_name, {})
        path = FEATURES_DIR / spec.get("path", "")
        if not path.exists():
            continue
        feat_df = pd.read_parquet(path)
        join_key = spec.get("join_key")
        cols_to_add = [c for c in spec.get("columns", []) if c != join_key and c not in result.columns]
        if join_key and join_key in result.columns and join_key in feat_df.columns and cols_to_add:
            result = result.merge(
                feat_df[[join_key] + cols_to_add],
                on=join_key,
                how="left",
            )
            print(f"  Loaded feature '{feature_name}': {len(cols_to_add)} columns added")
    return result


if __name__ == "__main__":
    print("Feature Registry Status:")
    print(list_available_features().to_string(index=False))
