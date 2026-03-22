"""
Initialize results.tsv for an autoresearch run.
Run this once before starting a new experiment branch.
"""
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.tsv"

header = "commit\tval_score\tstatus\tdescription\n"
if not RESULTS_PATH.exists():
    RESULTS_PATH.write_text(header)
    print(f"Created {RESULTS_PATH}")
else:
    print(f"Already exists: {RESULTS_PATH}")
    print("Delete it manually if you want to start fresh.")
