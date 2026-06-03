#!/usr/bin/env python
"""Download TotalSegmentator weights needed by the oppoCT lumbar workflow."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


LUMBAR_VERTEBRAE = [
    "vertebrae_T11",
    "vertebrae_T12",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TotalSegmentator model weights for offline PyInstaller builds."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("totalsegmentator_weights"),
        help="Directory to store downloaded weights.",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="cpu",
        help="Device used for the small dummy inference.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Do not run full-resolution total (291–295); only fast total (297) and other tasks.",
    )
    parser.add_argument(
        "--skip-fast",
        action="store_true",
        help="Do not run fast total (297); only full-resolution total (291–295) and other tasks.",
    )
    args = parser.parse_args()

    weights_dir = args.weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOTALSEG_WEIGHTS_PATH"] = str(weights_dir)

    from src.totalseg_runtime import apply_totalseg_runtime_paths

    apply_totalseg_runtime_paths()

    from totalsegmentator.python_api import totalsegmentator

    work_dir = Path("weight_download_work")
    dummy_path = work_dir / "dummy_ct.nii.gz"
    total_out = work_dir / "total_out"
    body_out = work_dir / "vertebrae_body.nii.gz"
    work_dir.mkdir(exist_ok=True)

    logger.info("Using weights directory: %s", weights_dir)
    nib.save(nib.Nifti1Image(np.zeros((32, 32, 32), dtype=np.float32), np.eye(4)), dummy_path)

    try:
        if not args.skip_full:
            logger.info("Triggering full-resolution total task weights (291–295 + 298)...")
            totalsegmentator(
                input=dummy_path,
                output=total_out,
                task="total",
                roi_subset=LUMBAR_VERTEBRAE,
                fast=False,
                device=args.device,
                preview=False,
            )

        if not args.skip_fast:
            fast_out = work_dir / "total_out_fast"
            logger.info("Triggering fast total task weights (Dataset297 + 298)...")
            totalsegmentator(
                input=dummy_path,
                output=fast_out,
                task="total",
                roi_subset=LUMBAR_VERTEBRAE,
                fast=True,
                device=args.device,
                preview=False,
            )

        logger.info("Triggering vertebrae_body task weights...")
        try:
            totalsegmentator(
                input=dummy_path,
                output=body_out,
                task="vertebrae_body",
                fast=False,
                device=args.device,
                preview=False,
                ml=True,
            )
        except BaseException as exc:
            logger.warning("vertebrae_body dummy inference did not complete: %s", exc)
            logger.warning("If this task is licensed in your environment, rerun after resolving the issue.")

        datasets = sorted(p.name for p in weights_dir.iterdir() if p.is_dir() and p.name.startswith("Dataset"))
        logger.info("Found %d Dataset directories in %s", len(datasets), weights_dir)
        for dataset in datasets:
            logger.info("  %s", dataset)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
