"""
TotalSegmentator runtime paths for oppoCT (dev tree and PyInstaller bundle).

Sets ``TOTALSEG_HOME_DIR`` when ``totalsegmentator_home/config.json`` exists (license)
and ``TOTALSEG_WEIGHTS_PATH`` when ``totalsegmentator_weights/`` exists.
Call before importing TotalSegmentator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APPLIED = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bundle_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    return project_root()


def apply_totalseg_runtime_paths(*, force: bool = False) -> None:
    """Apply bundled TotalSegmentator home/weights env vars (idempotent)."""
    global _APPLIED
    if _APPLIED and not force:
        return

    base = _bundle_base_dir()
    home = base / "totalsegmentator_home"
    if (home / "config.json").is_file():
        os.environ.setdefault("TOTALSEG_HOME_DIR", str(home))

    weights = base / "totalsegmentator_weights"
    if weights.is_dir():
        os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", str(weights))

    _APPLIED = True


def bundled_license_config_path() -> Path | None:
    """Path to bundled config.json if present."""
    path = _bundle_base_dir() / "totalsegmentator_home" / "config.json"
    return path if path.is_file() else None
