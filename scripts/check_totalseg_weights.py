"""
Print TotalSegmentator nnU-Net weight readiness for oppoCT lumbar segmentation.

Usage (from repo root, with venv active):
  python scripts/check_totalseg_weights.py
  python scripts/check_totalseg_weights.py --fast

Exit code 1 if any required folder is missing for the selected mode (with/without vertebrae_body).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.totalseg_local_weights import (  # noqa: E402
    missing_weight_dirs,
    weight_readiness_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Check fast-total weights (297 + 298) instead of full five-part total (291–295 + 298).",
    )
    args = parser.parse_args()

    print(weight_readiness_report(fast=args.fast))
    print()

    problems = []
    for inc_body, label in ((True, "with vertebrae_body"), (False, "without vertebrae_body (skip env)")):
        miss = missing_weight_dirs(fast=args.fast, include_vertebrae_body=inc_body)
        if miss:
            problems.append(f"{label}: missing {', '.join(miss)}")

    if problems:
        print("Summary — fix before OPPOCT_OFFLINE_SEGMENTATION=1:")
        for p in problems:
            print(" ", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
