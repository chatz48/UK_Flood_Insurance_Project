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

Use --parallel to run all downloads simultaneously (recommended).
"""

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_abi_events():
    from src.pipelines.abi_events import run_full_pipeline
    df = run_full_pipeline()
    return f"{len(df)} validation events"


def run_nrfa(max_stations=None, quick=False):
    from src.pipelines.nrfa_peaks import run_full_pipeline
    run_full_pipeline(
        max_stations=50 if quick else max_stations,  # type: ignore[arg-type]
        fetch_pot=not quick,
        fetch_descriptors=not quick,
        min_record_years=10,
    )
    return "done"


def run_ea_gauging(fetch_readings=False):
    from src.pipelines.ea_gauging import run_full_pipeline
    run_full_pipeline(fetch_readings=fetch_readings)
    return "done"


def run_ea_zones():
    from src.pipelines.ea_flood_zones import run_full_pipeline
    run_full_pipeline()
    return "done"


def run_land_registry(years=None, quick=False):
    from src.pipelines.land_registry import run_full_pipeline
    if quick:
        run_full_pipeline(years=[2022, 2023])
    else:
        run_full_pipeline(years=years or list(range(2015, 2025)))
    return "done"


def run_council_tax():
    from src.pipelines.council_tax_bands import run_pipeline
    run_pipeline()
    return "done"


def run_postcode_deprivation():
    from src.pipelines.postcode_deprivation import run_pipeline
    run_pipeline()
    return "done"


def run_hull_2007():
    from src.pipelines.hull_2007 import run_full_pipeline
    run_full_pipeline()
    return "done"


def main():
    parser = argparse.ArgumentParser(description="Download all data for UK flood cat model")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: small samples for testing pipelines work")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all downloads simultaneously (faster, recommended)")
    parser.add_argument("--skip-land-registry", action="store_true",
                        help="Skip Land Registry download (large files)")
    parser.add_argument("--skip-ea-zones", action="store_true",
                        help="Skip EA flood zone spatial download (API currently down)")
    parser.add_argument("--nrfa-only", action="store_true",
                        help="Only download NRFA data (the most important source)")
    args = parser.parse_args()

    print("=" * 55)
    print("UK FLOOD INSURANCE — DATA PIPELINE")
    print("=" * 55)

    if args.nrfa_only:
        print("\nRunning NRFA pipeline only...")
        run_nrfa(quick=args.quick)
        print("Done.")
        return

    # Build task list
    tasks = {
        "abi_events":          lambda: run_abi_events(),
        "nrfa":                lambda: run_nrfa(quick=args.quick),
        "ea_gauging":          lambda: run_ea_gauging(fetch_readings=False),
        "council_tax_bands":   lambda: run_council_tax(),
        "postcode_deprivation":lambda: run_postcode_deprivation(),
        "hull_2007":           lambda: run_hull_2007(),
    }
    if not args.skip_ea_zones:
        tasks["ea_zones"] = lambda: run_ea_zones()
    if not args.skip_land_registry:
        tasks["land_registry"] = lambda: run_land_registry(quick=args.quick)

    if args.parallel:
        print(f"\nRunning {len(tasks)} pipelines in parallel...\n")
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    print(f"  [OK]   {name}: {result}")
                    results[name] = "ok"
                except Exception as e:
                    print(f"  [FAIL] {name}: {e}")
                    traceback.print_exc()
                    results[name] = "failed"

        print("\n" + "=" * 55)
        ok = [k for k, v in results.items() if v == "ok"]
        failed = [k for k, v in results.items() if v == "failed"]
        if ok:
            print(f"Completed: {', '.join(ok)}")
        if failed:
            print(f"Failed:    {', '.join(failed)}")
    else:
        print("\nRunning pipelines sequentially (use --parallel to run simultaneously)...")
        for name, fn in tasks.items():
            print(f"\n  Running {name}...")
            try:
                result = fn()
                print(f"  [OK] {name}: {result}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")

    print("\n" + "=" * 55)
    print("Pipelines complete. Now run: python train.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
