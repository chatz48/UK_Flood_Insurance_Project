"""
TIV Accumulation Table and Choropleth Map

Aggregates the exposure portfolio by flood zone, property type, and county
to produce:
  - outputs/results/tiv_by_zone.parquet
  - outputs/plots/tiv_accumulation.html (Plotly choropleth by local authority district)

The choropleth colours each LAD by its share of TIV sitting in flood zone 3a/3b,
which is the key risk concentration metric for reinsurance exposure management.
"""

import sys
import numpy as np
import pandas as pd
import requests
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# ONS Local Authority Districts GeoJSON (Dec 2023, simplified boundaries)
LAD_GEOJSON_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/"
    "FeatureServer/0/query?where=1%3D1&outFields=LAD23CD,LAD23NM"
    "&returnGeometry=true&f=geojson&resultRecordCount=400"
)


def load_portfolio() -> pd.DataFrame:
    """Load exposure portfolio. Returns synthetic data if not yet built."""
    portfolio_path = PROCESSED_DIR / "exposure_portfolio.parquet"
    if portfolio_path.exists():
        df = pd.read_parquet(portfolio_path)
        print(f"  Loaded portfolio: {len(df):,} properties")
        return df

    print("  WARNING: exposure_portfolio.parquet not found — using synthetic data")
    return _synthetic_portfolio()


def _synthetic_portfolio() -> pd.DataFrame:
    """
    Synthetic portfolio for demonstration.
    Represents ~500k properties across England with realistic zone distribution.
    """
    np.random.seed(42)
    n = 10_000

    counties = [
        "Yorkshire", "Cumbria", "Somerset", "Hull", "Gloucestershire",
        "Oxfordshire", "Surrey", "Kent", "Norfolk", "Lancashire",
    ]
    # Rough zone proportions: 70% zone 1/none, 20% zone 2, 7% zone 3a, 3% zone 3b
    zones = np.random.choice(["none", "2", "3a", "3b"],
                              p=[0.70, 0.20, 0.07, 0.03], size=n)
    prop_types = np.random.choice(["terraced", "semi", "detached", "flat"],
                                   p=[0.30, 0.30, 0.20, 0.20], size=n)
    counties_arr = np.random.choice(counties, size=n)

    # Property values vary by county and flood zone (higher value near water)
    base_values = np.random.lognormal(mean=12.3, sigma=0.5, size=n)
    # Zone 3 properties slightly cheaper on average (flood discount)
    zone_adj = np.where(zones == "3b", 0.85, np.where(zones == "3a", 0.90, 1.0))
    tiv = base_values * zone_adj * 1.25

    # London-like districts for realism
    districts = np.random.choice([
        "Leeds", "York", "Carlisle", "Taunton", "Gloucester", "Oxford",
        "Guildford", "Maidstone", "Norwich", "Preston", "Hull", "Sheffield",
    ], size=n)

    return pd.DataFrame({
        "property_type": prop_types,
        "flood_zone": zones,
        "county": counties_arr,
        "district": districts,
        "estimated_tiv": tiv,
        "last_sale_price": base_values,
    })


def build_tiv_tables(portfolio_df: pd.DataFrame) -> dict:
    """
    Aggregate TIV by flood zone × property type and flood zone × county.
    Returns dict of DataFrames.
    """
    tables = {}

    # 1. By flood zone × property type
    zone_type = (
        portfolio_df.groupby(["flood_zone", "property_type"])["estimated_tiv"]
        .agg(tiv_sum="sum", count="count", mean_tiv="mean")
        .reset_index()
    )
    zone_type["tiv_gbpm"] = zone_type["tiv_sum"] / 1e6
    tables["zone_x_property_type"] = zone_type

    # 2. By flood zone × county
    zone_county = (
        portfolio_df.groupby(["flood_zone", "county"])["estimated_tiv"]
        .agg(tiv_sum="sum", count="count")
        .reset_index()
    )
    zone_county["tiv_gbpm"] = zone_county["tiv_sum"] / 1e6
    tables["zone_x_county"] = zone_county

    # 3. Top 20 districts by Zone 3 TIV
    zone3_district = (
        portfolio_df[portfolio_df["flood_zone"].isin(["3a", "3b"])]
        .groupby("district")["estimated_tiv"]
        .sum()
        .reset_index()
        .sort_values("estimated_tiv", ascending=False)
        .head(20)
    )
    zone3_district["tiv_gbpm"] = zone3_district["estimated_tiv"] / 1e6
    tables["top20_zone3_districts"] = zone3_district

    return tables


def print_tiv_summary(tables: dict):
    """Print a readable summary of the TIV accumulation."""
    print("\n" + "=" * 55)
    print("TIV ACCUMULATION SUMMARY")
    print("=" * 55)

    zp = tables["zone_x_property_type"].groupby("flood_zone")["tiv_sum"].sum().reset_index()
    total_tiv = zp["tiv_sum"].sum()

    for _, row in zp.iterrows():
        pct = row["tiv_sum"] / total_tiv * 100
        print(f"  Zone {row['flood_zone']:<5}: £{row['tiv_sum']/1e9:>6.1f}bn  ({pct:.1f}%)")
    print(f"  {'TOTAL':<6}: £{total_tiv/1e9:>6.1f}bn")

    print("\n  Top 5 zone-3 districts by TIV:")
    for _, row in tables["top20_zone3_districts"].head(5).iterrows():
        print(f"    {row['district']:<20}: £{row['tiv_gbpm']:>8,.0f}m")
    print("=" * 55)


def build_choropleth(portfolio_df: pd.DataFrame) -> None:
    """
    Build Plotly choropleth map: % of district TIV in flood zone 3a/3b.
    Saves to outputs/plots/tiv_accumulation.html.
    """
    try:
        import plotly.express as px
    except ImportError:
        print("  plotly not installed — skipping choropleth")
        return

    # Compute zone-3 TIV share per district
    total_by_district = (
        portfolio_df.groupby("district")["estimated_tiv"].sum().reset_index()
        .rename(columns={"estimated_tiv": "total_tiv"})
    )
    zone3_by_district = (
        portfolio_df[portfolio_df["flood_zone"].isin(["3a", "3b"])]
        .groupby("district")["estimated_tiv"].sum().reset_index()
        .rename(columns={"estimated_tiv": "zone3_tiv"})
    )
    district_stats = total_by_district.merge(zone3_by_district, on="district", how="left")
    district_stats["zone3_tiv"] = district_stats["zone3_tiv"].fillna(0)
    district_stats["zone3_pct"] = district_stats["zone3_tiv"] / district_stats["total_tiv"] * 100
    district_stats["total_tiv_gbpm"] = district_stats["total_tiv"] / 1e6

    # Try to download LAD GeoJSON for choropleth
    geojson = None
    geojson_path = PLOTS_DIR / "lad_geojson.json"
    if geojson_path.exists():
        with open(geojson_path) as f:
            geojson = json.load(f)
    else:
        try:
            r = requests.get(LAD_GEOJSON_URL, timeout=60)
            r.raise_for_status()
            geojson = r.json()
            with open(geojson_path, "w") as f:
                json.dump(geojson, f)
            print(f"  LAD GeoJSON downloaded ({len(geojson.get('features',[]))} LADs)")
        except Exception as e:
            print(f"  LAD GeoJSON not available: {e}")

    if geojson:
        # Join district stats to LAD names
        lad_names = {f["properties"].get("LAD23NM", ""): f["properties"].get("LAD23CD", "")
                     for f in geojson.get("features", [])}
        district_stats["lad_code"] = district_stats["district"].map(lad_names)

        fig = px.choropleth(
            district_stats.dropna(subset=["lad_code"]),
            geojson=geojson,
            locations="lad_code",
            featureidkey="properties.LAD23CD",
            color="zone3_pct",
            hover_name="district",
            hover_data={"zone3_pct": ":.1f", "total_tiv_gbpm": ":,.0f"},
            color_continuous_scale="Reds",
            range_color=[0, 20],
            title="% of District TIV in EA Flood Zone 3a/3b",
            labels={"zone3_pct": "Zone 3 TIV %", "total_tiv_gbpm": "Total TIV (£m)"},
        )
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            projection_type="mercator",
        )
    else:
        # Fallback: bar chart of top districts
        top_districts = district_stats.sort_values("zone3_pct", ascending=False).head(30)
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=top_districts["district"],
            y=top_districts["zone3_pct"],
            marker_color="firebrick",
            text=top_districts["zone3_pct"].round(1).astype(str) + "%",
            textposition="auto",
        ))
        fig.update_layout(
            title="Top Districts by % TIV in Flood Zone 3a/3b",
            xaxis_title="District",
            yaxis_title="Zone 3 TIV %",
            xaxis_tickangle=-45,
        )

    fig.update_layout(template="plotly_white", height=600)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "tiv_accumulation.html"
    fig.write_html(str(out_path))
    print(f"  Choropleth saved → {out_path}")


def run_tiv_accumulation() -> dict:
    """
    Main entry point: build TIV tables and choropleth from the exposure portfolio.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_df = load_portfolio()

    tables = build_tiv_tables(portfolio_df)
    print_tiv_summary(tables)

    # Save parquet outputs
    tables["zone_x_property_type"].to_parquet(RESULTS_DIR / "tiv_by_zone.parquet", index=False)
    tables["zone_x_county"].to_parquet(RESULTS_DIR / "tiv_by_county.parquet", index=False)
    tables["top20_zone3_districts"].to_parquet(RESULTS_DIR / "tiv_top20_zone3_districts.parquet", index=False)
    print(f"  TIV tables saved to {RESULTS_DIR}")

    # Build choropleth
    build_choropleth(portfolio_df)

    return tables


if __name__ == "__main__":
    run_tiv_accumulation()
