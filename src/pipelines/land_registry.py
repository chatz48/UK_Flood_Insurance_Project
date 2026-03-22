"""
HM Land Registry Price Paid Data
https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

Full transaction-level property sale data for England & Wales since 1995.
~27 million records. Updated monthly. Free, Open Government Licence.

Columns:
  transaction_id, price, date_of_transfer, postcode, property_type,
  old_new (new build?), duration (freehold/leasehold), paon (building number),
  saon (flat/unit), street, locality, town_city, district, county,
  ppd_category (A=standard, B=additional)

Used in this project:
  - Property values as exposure (what's at risk financially)
  - Property type distribution per postcode (residential mix)
  - New build flag (newer builds have better flood resilience requirements)
  - Transaction count as proxy for number of properties per postcode
"""

import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import io

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "land_registry"

# Land Registry full download URLs
LR_BASE = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"

COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode",
    "property_type", "old_new", "duration", "paon", "saon",
    "street", "locality", "town_city", "district", "county",
    "ppd_category", "record_status",
]

PROPERTY_TYPE_MAP = {
    "D": "Detached",
    "S": "Semi-detached",
    "T": "Terraced",
    "F": "Flat/Maisonette",
    "O": "Other",
}


def download_yearly_file(year: int) -> pd.DataFrame:
    """
    Download Land Registry Price Paid data for a full year.
    Files are ~150-300MB each (millions of rows).
    """
    url = f"{LR_BASE}/pp-{year}.csv"
    print(f"  Downloading Land Registry {year}...")
    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        content = r.content
        df = pd.read_csv(
            io.BytesIO(content),
            header=None,
            names=COLUMNS,
            dtype=str,
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce")
        df["year"] = year
        print(f"    {year}: {len(df):,} transactions")
        return df
    except Exception as e:
        print(f"    Warning: failed to download {year}: {e}")
        return pd.DataFrame()


def download_complete_dataset() -> pd.DataFrame:
    """
    Download the complete Land Registry dataset (all years combined).
    Warning: This is a ~4GB download.
    """
    url = f"{LR_BASE}/pp-complete.csv"
    print("Downloading complete Land Registry dataset (may take 10-20 min)...")
    r = requests.get(url, timeout=1800, stream=True)
    r.raise_for_status()

    chunks = []
    for chunk in tqdm(r.iter_content(chunk_size=10 * 1024 * 1024), desc="Downloading"):
        if chunk:
            chunks.append(chunk)

    content = b"".join(chunks)
    df = pd.read_csv(io.BytesIO(content), header=None, names=COLUMNS, dtype=str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce")
    return df


def aggregate_by_postcode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions to postcode level for flood risk modelling.

    Output per postcode:
      - median_price: central estimate of property value
      - mean_price: mean property value
      - transaction_count: proxy for number of properties
      - property_type_mix: % detached, semi, terraced, flat
      - pct_new_build: proportion of new builds
      - latest_year: most recent transaction year
    """
    # Property type dummies
    df["prop_type_name"] = df["property_type"].map(PROPERTY_TYPE_MAP)

    type_counts = (
        df.groupby(["postcode", "prop_type_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    type_totals = type_counts.drop("postcode", axis=1).sum(axis=1)
    for col in ["Detached", "Semi-detached", "Terraced", "Flat/Maisonette", "Other"]:
        if col in type_counts.columns:
            type_counts[f"pct_{col.lower().replace('/', '_').replace('-', '_').replace(' ', '_')}"] = (
                type_counts[col] / type_totals
            )

    # Price aggregates
    price_stats = df.groupby("postcode").agg(
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        transaction_count=("price", "count"),
        pct_new_build=("old_new", lambda x: (x == "Y").mean()),
        latest_year=("year", "max"),
    ).reset_index()

    result = price_stats.merge(type_counts[
        ["postcode"] + [c for c in type_counts.columns if c.startswith("pct_")]
    ], on="postcode", how="left")

    return result


def run_full_pipeline(
    years: list = None,
    use_complete: bool = False,
    postcode_prefix_filter: str = None,
):
    """
    Download and process Land Registry data.

    Args:
        years: List of years to download (e.g. [2015,2016,...,2024]).
               None defaults to 2010-2024.
        use_complete: Download complete historical dataset (all years, ~4GB)
        postcode_prefix_filter: Only keep postcodes starting with this
                                (e.g. 'YO' for York, 'CA' for Cumbria)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if years is None:
        years = list(range(2010, 2025))

    all_frames = []

    if use_complete:
        df = download_complete_dataset()
        all_frames.append(df)
    else:
        for year in years:
            path = RAW_DIR / f"pp-{year}.parquet"
            if path.exists():
                print(f"  {year}: already downloaded, loading from cache")
                df = pd.read_parquet(path)
            else:
                df = download_yearly_file(year)
                if not df.empty:
                    df.to_parquet(path, index=False)
            if not df.empty:
                all_frames.append(df)

    if not all_frames:
        print("No data downloaded.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)

    if postcode_prefix_filter:
        full_df = full_df[full_df["postcode"].str.startswith(postcode_prefix_filter, na=False)]
        print(f"  Filtered to postcodes starting with '{postcode_prefix_filter}': {len(full_df):,} rows")

    # Save raw combined
    raw_path = RAW_DIR / "price_paid_combined.parquet"
    full_df.to_parquet(raw_path, index=False)
    print(f"\nSaved {len(full_df):,} transactions to {raw_path}")

    # Aggregate to postcode level
    print("Aggregating to postcode level...")
    postcode_df = aggregate_by_postcode(full_df)
    postcode_path = RAW_DIR / "postcode_aggregated.parquet"
    postcode_df.to_parquet(postcode_path, index=False)
    print(f"Saved {len(postcode_df):,} postcodes to {postcode_path}")

    print("\nLand Registry pipeline complete.")
    return postcode_df


if __name__ == "__main__":
    # Start with flood-risk-heavy regions: Yorkshire (YO), Cumbria (CA),
    # Somerset (TA), Hull (HU), Gloucester (GL)
    run_full_pipeline(years=list(range(2015, 2025)))
