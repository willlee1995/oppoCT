#!/usr/bin/env python
"""Copy TotalSegmentator license into totalsegmentator_home/ for offline packaging."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install TotalSegmentator license into totalsegmentator_home/config.json"
    )
    parser.add_argument(
        "-l",
        "--license",
        type=str,
        help="License number (aca_ + 14 chars). If omitted, copies from ~/.totalsegmentator/config.json",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Write config without calling totalseg_set_license (offline only).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dest_dir = root / "totalsegmentator_home"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / "config.json"

    if args.license:
        if not args.license.startswith("aca_") or len(args.license) != 18:
            print("ERROR: license must start with aca_ and be exactly 18 characters.", file=sys.stderr)
            return 1
        if args.skip_validation:
            config = {
                "totalseg_id": "totalseg_OPPOCTPKG",
                "send_usage_stats": False,
                "prediction_counter": 0,
                "license_number": args.license,
            }
            dest.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
            print(f"Wrote {dest}")
            return 0
        import subprocess

        cmd = [sys.executable, "-m", "totalsegmentator.bin.totalseg_set_license", "-l", args.license]
        if subprocess.run(cmd).returncode != 0:
            return 1
        src = Path.home() / ".totalsegmentator" / "config.json"
        if not src.is_file():
            print(f"ERROR: expected config at {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, dest)
        print(f"Copied validated license config to {dest}")
        return 0

    src = Path.home() / ".totalsegmentator" / "config.json"
    if not src.is_file():
        print(
            f"ERROR: no --license and no {src}. Run: totalseg_set_license -l aca_...",
            file=sys.stderr,
        )
        return 1
    shutil.copy2(src, dest)
    print(f"Copied {src} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
