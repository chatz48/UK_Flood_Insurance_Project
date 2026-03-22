"""
ABI Published Flood/Storm Event Loss Data + DEFRA Post-Event Reports
This is our VALIDATION data — the ground truth we check our model against.

The ABI publishes event-level insured loss estimates after major floods.
These are our ~15 calibration points. We cannot validate at property level,
but we CAN check our model's aggregate output matches known events.

Known major UK flood events with published insured loss estimates:
  Sources:
    - ABI press releases (search: site:abi.org.uk floods)
    - Environment Agency post-event reports
    - DEFRA flood damage cost assessments
    - Parliamentary research briefings
    - Swiss Re and Munich Re sigma reports (annual nat cat reviews)

Note on data availability:
  - No weekly or monthly public claims data exists
  - ABI reports annually, with some event-specific press releases
  - Individual insurer data is proprietary
  - Flood Re only reports aggregate cession statistics
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "abi_events"


# ===========================================================================
# KNOWN UK FLOOD EVENTS — compiled from ABI, EA, DEFRA, Swiss Re publications
# All losses in £ million (insured losses where available, economic otherwise)
# ===========================================================================

UK_FLOOD_EVENTS = [
    # 2000 Autumn floods — widespread riverine flooding
    {
        "event_id": "UK_FLOOD_2000_AUTUMN",
        "start_date": "2000-10-28",
        "end_date": "2000-11-10",
        "name": "2000 Autumn Floods",
        "type": "riverine",
        "insured_loss_gbp_m": 1300,
        "economic_loss_gbp_m": 2000,
        "properties_flooded": 10000,
        "source": "ABI / DEFRA",
        "notes": "Worst UK flooding in 50 years at time. Yorkshire, Midlands, Wales.",
        "confidence": "medium",
    },
    # 2002 flooding
    {
        "event_id": "UK_FLOOD_2002",
        "start_date": "2002-11-01",
        "end_date": "2002-11-30",
        "name": "2002 November Floods",
        "type": "riverine",
        "insured_loss_gbp_m": 400,
        "economic_loss_gbp_m": 600,
        "properties_flooded": 4000,
        "source": "EA / ABI estimates",
        "notes": "England and Wales flooding",
        "confidence": "low",
    },
    # 2004 Boscastle flash flood — extreme surface water event
    {
        "event_id": "UK_FLOOD_2004_BOSCASTLE",
        "start_date": "2004-08-16",
        "end_date": "2004-08-16",
        "name": "Boscastle Flash Flood",
        "type": "surface_water",
        "insured_loss_gbp_m": 15,
        "economic_loss_gbp_m": 20,
        "properties_flooded": 115,
        "source": "ABI",
        "notes": "Extreme localised flash flood, Cornwall. 2m flood depth.",
        "confidence": "high",
    },
    # 2005 Carlisle floods
    {
        "event_id": "UK_FLOOD_2005_CARLISLE",
        "start_date": "2005-01-07",
        "end_date": "2005-01-09",
        "name": "Carlisle Floods 2005",
        "type": "riverine",
        "insured_loss_gbp_m": 272,
        "economic_loss_gbp_m": 400,
        "properties_flooded": 3000,
        "source": "EA Post-Event Report",
        "notes": "River Eden burst banks. 3 deaths. City centre inundated.",
        "confidence": "high",
    },
    # 2007 Summer floods — single largest UK insured flood loss
    {
        "event_id": "UK_FLOOD_2007_SUMMER",
        "start_date": "2007-06-24",
        "end_date": "2007-07-25",
        "name": "2007 Summer Floods",
        "type": "combined_riverine_surface_water",
        "insured_loss_gbp_m": 3200,
        "economic_loss_gbp_m": 4500,
        "properties_flooded": 55000,
        "source": "ABI / Pitt Review",
        "notes": "Largest UK flood event. Hull, Sheffield, Gloucestershire, Tewkesbury. "
                 "55,000 homes + 7,000 businesses. 13 deaths.",
        "confidence": "high",
    },
    # 2009 Cumbria floods
    {
        "event_id": "UK_FLOOD_2009_CUMBRIA",
        "start_date": "2009-11-18",
        "end_date": "2009-11-20",
        "name": "Cumbria Floods 2009",
        "type": "riverine",
        "insured_loss_gbp_m": 174,
        "economic_loss_gbp_m": 276,
        "properties_flooded": 1800,
        "source": "ABI / EA",
        "notes": "Cockermouth badly damaged. Record 314mm rain in 24hrs.",
        "confidence": "high",
    },
    # 2012 multiple events
    {
        "event_id": "UK_FLOOD_2012",
        "start_date": "2012-06-01",
        "end_date": "2012-12-31",
        "name": "2012 Summer/Autumn Floods",
        "type": "multiple",
        "insured_loss_gbp_m": 600,
        "economic_loss_gbp_m": 900,
        "properties_flooded": 8000,
        "source": "ABI estimates",
        "notes": "Multiple events across the year. Wettest year on record for England.",
        "confidence": "medium",
    },
    # 2013/14 Winter floods — Thames, Somerset Levels
    {
        "event_id": "UK_FLOOD_2013_14_WINTER",
        "start_date": "2013-12-05",
        "end_date": "2014-03-31",
        "name": "2013/14 Winter Floods",
        "type": "riverine_coastal",
        "insured_loss_gbp_m": 1100,
        "economic_loss_gbp_m": 1500,
        "properties_flooded": 11000,
        "source": "ABI",
        "notes": "Somerset Levels flooded for months. Thames corridor. Storm surges.",
        "confidence": "high",
    },
    # 2015 December floods — Storm Desmond, Eva, Frank
    {
        "event_id": "UK_FLOOD_2015_WINTER",
        "start_date": "2015-12-05",
        "end_date": "2016-01-10",
        "name": "Winter 2015/16 Floods (Storms Desmond, Eva, Frank)",
        "type": "riverine",
        "insured_loss_gbp_m": 1300,
        "economic_loss_gbp_m": 2300,
        "properties_flooded": 16000,
        "source": "ABI",
        "notes": "Cumbria again worst hit. New UK rainfall records. York, Leeds flooded.",
        "confidence": "high",
    },
    # 2019/20 Winter floods
    {
        "event_id": "UK_FLOOD_2019_20",
        "start_date": "2019-11-01",
        "end_date": "2020-03-31",
        "name": "2019/20 Winter Floods",
        "type": "riverine",
        "insured_loss_gbp_m": 360,
        "economic_loss_gbp_m": 500,
        "properties_flooded": 4600,
        "source": "ABI",
        "notes": "South Yorkshire, Midlands, Wales. Storms Ciara and Dennis.",
        "confidence": "high",
    },
    # 2021 flooding events
    {
        "event_id": "UK_FLOOD_2021",
        "start_date": "2021-07-25",
        "end_date": "2021-08-01",
        "name": "2021 Summer Flooding",
        "type": "surface_water",
        "insured_loss_gbp_m": 140,
        "economic_loss_gbp_m": 200,
        "properties_flooded": 1500,
        "source": "ABI estimates",
        "notes": "London surface water flooding during heatwave-triggered storms.",
        "confidence": "medium",
    },
    # 2023 Storm Babet
    {
        "event_id": "UK_FLOOD_2023_BABET",
        "start_date": "2023-10-19",
        "end_date": "2023-10-21",
        "name": "Storm Babet 2023",
        "type": "riverine",
        "insured_loss_gbp_m": 200,
        "economic_loss_gbp_m": 350,
        "properties_flooded": 3500,
        "source": "ABI estimates / news reports",
        "notes": "Scotland and East Midlands. Record river levels in Aberdeenshire.",
        "confidence": "medium",
    },
    # 2024 flooding events
    {
        "event_id": "UK_FLOOD_2024",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "name": "2024 Flooding Events",
        "type": "multiple",
        "insured_loss_gbp_m": 800,
        "economic_loss_gbp_m": 1200,
        "properties_flooded": 9000,
        "source": "ABI preliminary estimates",
        "notes": "Multiple events including Storms Henk, Jocelyn. Midlands, East.",
        "confidence": "low",
    },
]


def build_events_dataset() -> pd.DataFrame:
    """Build and save the ground truth events dataset."""
    df = pd.DataFrame(UK_FLOOD_EVENTS)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["year"] = df["start_date"].dt.year

    # Derived fields
    df["insured_loss_per_property_gbp"] = (
        df["insured_loss_gbp_m"] * 1_000_000 / df["properties_flooded"]
    ).round(0)

    return df


def run_full_pipeline():
    """Save the events dataset and print summary statistics."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = build_events_dataset()
    path = RAW_DIR / "uk_flood_events.parquet"
    df.to_parquet(path, index=False)

    # Also save as CSV for easy inspection
    csv_path = RAW_DIR / "uk_flood_events.csv"
    df.to_csv(csv_path, index=False)

    print("UK Flood Events dataset built:")
    print(f"  {len(df)} events from {df['year'].min()} to {df['year'].max()}")
    print(f"  Total insured losses: £{df['insured_loss_gbp_m'].sum():,.0f}m")
    print(f"  Total properties flooded: {df['properties_flooded'].sum():,}")
    print(f"\nLargest events by insured loss:")
    top = df.nlargest(5, "insured_loss_gbp_m")[["name", "year", "insured_loss_gbp_m", "properties_flooded"]]
    print(top.to_string(index=False))
    print(f"\nSaved to {path}")

    return df


if __name__ == "__main__":
    run_full_pipeline()
