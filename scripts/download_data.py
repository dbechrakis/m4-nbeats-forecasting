"""Download the pinned M4 files and verify their SHA-256.

    python scripts/download_data.py                 # Hourly, Daily and Weekly
    python scripts/download_data.py --frequency Weekly
"""

import argparse
from pathlib import Path

from deepforecast.data import FREQUENCIES, download


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=list(FREQUENCIES), action="append")
    for frequency in parser.parse_args().frequency or list(FREQUENCIES):
        download(frequency, ROOT / "data" / "raw")
        print(f"{frequency}: verified")
