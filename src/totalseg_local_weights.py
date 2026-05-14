"""
TotalSegmentator nnU-Net weight directories required for oppoCT lumbar segmentation.

TotalSegmentator downloads from the network when expected folders are missing under
``TOTALSEG_WEIGHTS_PATH`` (if set) or ``<home>/.totalsegmentator/nnunet/results`` (default).

This module lists those folders so you can prefetch them and optionally block runs
when anything is missing (``OPPOCT_OFFLINE_SEGMENTATION=1``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

# Folder names after unzip — must match totalsegmentator.libs.download_pretrained_weights
# for the task_ids used by segment_lumbar_vertebrae() + totalsegmentator(..., roi_subset=...).

DIR_TOTAL_FULL_1 = "Dataset291_TotalSegmentator_part1_organs_1559subj"
DIR_TOTAL_FULL_2 = "Dataset292_TotalSegmentator_part2_vertebrae_1532subj"
DIR_TOTAL_FULL_3 = "Dataset293_TotalSegmentator_part3_cardiac_1559subj"
DIR_TOTAL_FULL_4 = "Dataset294_TotalSegmentator_part4_muscles_1559subj"
DIR_TOTAL_FULL_5 = "Dataset295_TotalSegmentator_part5_ribs_1559subj"
DIR_TOTAL_FAST_3MM = "Dataset297_TotalSegmentator_total_3mm_1559subj"
DIR_TOTAL_ROI_CROP_6MM = "Dataset298_TotalSegmentator_total_6mm_1559subj"
DIR_VERTEBRAE_BODY_COMMERCIAL = "Dataset305_vertebrae_discs_1559subj"

FULL_RES_TOTAL_DIRS = (
    DIR_TOTAL_FULL_1,
    DIR_TOTAL_FULL_2,
    DIR_TOTAL_FULL_3,
    DIR_TOTAL_FULL_4,
    DIR_TOTAL_FULL_5,
)


def _weights_base() -> Path:
    from totalsegmentator.config import get_weights_dir

    return Path(get_weights_dir())


def skip_vertebrae_body_seg() -> bool:
    return os.environ.get("OPPOCT_SKIP_VERTEBRAE_BODY_SEG", "").lower() in ("1", "true", "yes")


def offline_segmentation_enforced() -> bool:
    return os.environ.get("OPPOCT_OFFLINE_SEGMENTATION", "").lower() in ("1", "true", "yes")


def required_relative_weight_dirs(*, fast: bool, include_vertebrae_body: bool) -> Tuple[str, ...]:
    """
    Dataset folder names that must exist so TotalSegmentator will not hit the network
    for oppoCT's lumbar pipeline (total + roi_subset + optional vertebrae_body).
    """
    crop_for_roi = (DIR_TOTAL_ROI_CROP_6MM,)
    if fast:
        main = (DIR_TOTAL_FAST_3MM,)
    else:
        main = FULL_RES_TOTAL_DIRS
    body = (DIR_VERTEBRAE_BODY_COMMERCIAL,) if include_vertebrae_body else ()
    return main + crop_for_roi + body


def missing_weight_dirs(*, fast: bool, include_vertebrae_body: bool) -> List[str]:
    """Return relative folder names that are absent under the nnU-Net weights directory."""
    base = _weights_base()
    missing: List[str] = []
    for name in required_relative_weight_dirs(
        fast=fast, include_vertebrae_body=include_vertebrae_body
    ):
        if not (base / name).is_dir():
            missing.append(name)
    return missing


def enforce_local_weights_if_configured(*, fast: bool, verbose: bool = True) -> None:
    """
    If OPPOCT_OFFLINE_SEGMENTATION is set, raise before TotalSegmentator runs when weights
    are missing (prevents automatic downloads).
    """
    if not offline_segmentation_enforced():
        return
    import logging

    include_body = not skip_vertebrae_body_seg()
    missing = missing_weight_dirs(fast=fast, include_vertebrae_body=include_body)
    if not missing:
        return
    base = _weights_base()
    lines = "\n  ".join(missing)
    hint = (
        f"Expected nnU-Net weight folders under:\n  {base}\n\n"
        f"Missing ({len(missing)}):\n  {lines}\n\n"
        "Prefetch: see TotalSegmentator releases / `totalsegmentator.download_pretrained_weights` "
        "task_ids — full-quality total uses 291–295; fast total uses 297; "
        "roi_subset (used by oppoCT) also loads the 6mm crop model (298). "
        "vertebrae_body uses commercial Dataset305 (license + download).\n\n"
        "To run without the commercial vertebrae_body step (no Dataset305), set "
        "OPPOCT_SKIP_VERTEBRAE_BODY_SEG=1 (L1 trabecular core from body intersection may be skipped)."
    )
    if verbose:
        logging.error(hint)
    raise RuntimeError(
        "Offline segmentation: required TotalSegmentator weights are missing. "
        "Set OPPOCT_OFFLINE_SEGMENTATION=0 to allow automatic downloads, or install weights. "
        f"Details: {base}"
    )


def weight_readiness_report(*, fast: bool) -> str:
    """Human-readable summary for diagnostics or scripts."""
    base = _weights_base()
    lines = [f"nnU-Net weights directory: {base}", ""]
    for label, inc in (
        ("With vertebrae_body (commercial)", True),
        ("Without vertebrae_body (OPPOCT_SKIP_VERTEBRAE_BODY_SEG=1)", False),
    ):
        miss = missing_weight_dirs(fast=fast, include_vertebrae_body=inc)
        lines.append(label + ":")
        if miss:
            lines.append("  MISSING: " + ", ".join(miss))
        else:
            lines.append("  OK (all folders present)")
        lines.append("")
    return "\n".join(lines).rstrip()
