"""
run_pipelines.py — Download all data for the UK Flood Insurance Cat Model.

Run this first before train.py.
Each pipeline can be run independently if one fails.

Estimated times:
  abi_events:   < 1 second (hardcoded compiled data)
  nrfa:         ~10-20 min (1,500 stations, API calls with rate limiting)
  ea_gauging:   ~2 min (metadata only), hours (with historical readings)
  ea_zones:     ~5 min (regional tiles)
  land_registry: ~30 min per year downloaded (large files)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_abi_events():
    print("\n[1/5] Building ABI flood events validation dataset...")
    from src.pipelines.abi_events import run_full_pipeline
    df = run_full_pipeline()
    print(f"  Done: {len(df)} validation events")


def run_nrfa(max_stations=None, quick=False):
    print("\n[2/5] Downloading NRFA peak flow data...")
    from src.pipelines.nrfa_peaks import run_full_pipeline
    run_full_pipeline(
        max_stations=50 if quick else max_stations,
        fetch_pot=not quick,
        fetch_descriptors=not quick,
        min_record_years=10,
    )


def run_ea_gauging(fetch_readings=False):
    print("\n[3/5] Downloading EA gauging station metadata...")
    from src.pipelines.ea_gauging import run_full_pipeline
    run_full_pipeline(fetch_readings=fetch_readings)


def run_ea_zones():
    print("\n[4/5] Downloading EA flood zone spatial data...")
    from src.pipelines.ea_flood_zones import run_full_pipeline
    run_full_pipeline()


def run_land_registry(years=None, quick=False):
    print("\n[5/5] Downloading Land Registry price paid data...")
    from src.pipelines.land_registry import run_full_pipeline
    if quick:
        # Quick mode: only 2022-2023 for testing
        run_full_pipeline(years=[2022, 2023])
    else:
        run_full_pipeline(years=years or list(range(2015, 2025)))


def main():
    parser = argparse.ArgumentParser(description="Download all data for UK flood cat model")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: small samples for testing pipelines work")
    parser.add_argument("--skip-land-registry", action="store_true",
                        help="Skip Land Registry download (large files)")
    parser.add_argument("--skip-ea-zones", action="store_true",
                        help="Skip EA flood zone spatial download")
    parser.add_argument("--nrfa-only", action="store_true",
                        help="Only download NRFA data (the most important source)")
    args = parser.parse_args()

    print("=" * 55)
    print("UK FLOOD INSURANCE — DATA PIPELINE")
    print("=" * 55)

    # ABI events always run (no downloads needed)
    run_abi_events()

    if args.nrfa_only:
        run_nrfa(quick=args.quick)
        return

    run_nrfa(quick=args.quick)
    run_ea_gauging(fetch_readings=False)

    if not args.skip_ea_zones:
        run_ea_zones()

    if not args.skip_land_registry:
        run_land_registry(quick=args.quick)

    print("\n" + "=" * 55)
    print("All pipelines complete. Now run: python train.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
