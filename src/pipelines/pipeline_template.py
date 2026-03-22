"""
NEW DATA SOURCE PIPELINE TEMPLATE
==================================
The agent copies and adapts this file when it discovers a new useful dataset.

Instructions for the agent:
  1. Copy this file to src/pipelines/<source_name>.py
  2. Fill in the fetch logic
  3. Save output to data/features/<feature_name>.parquet
  4. Add an entry to src/features/feature_registry.py FEATURE_CATALOGUE
  5. Add the feature name to ACTIVE_FEATURES in train.py
  6. Run python train.py — if val_score improves, keep; else discard

The output parquet MUST have:
  - A join key column: 'postcode', 'station_id', or 'grid_cell'
  - Only columns listed in the FEATURE_CATALOGUE entry
  - Numeric values only (strings should be encoded or dropped)
  - No NaN in the join key column
"""

import requests
import pandas as pd
from pathlib import Path

FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def fetch_data() -> pd.DataFrame:
    """
    TODO: Replace with actual data fetching logic.
    Must return a DataFrame with a join key column + feature columns.
    """
    raise NotImplementedError("Agent: implement this fetch function")


def process_to_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Transform raw data into model-ready features.
    All values should be numeric. Normalise if needed.
    """
    raise NotImplementedError("Agent: implement this processing function")


def run_pipeline():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching data...")
    raw_df = fetch_data()

    print("Processing features...")
    features_df = process_to_features(raw_df)

    output_path = FEATURES_DIR / "CHANGE_ME.parquet"
    features_df.to_parquet(output_path, index=False)
    print(f"Saved {len(features_df)} rows to {output_path}")
    print(f"Columns: {list(features_df.columns)}")
    return features_df


if __name__ == "__main__":
    run_pipeline()
