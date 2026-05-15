"""Record / verify native-stack versions for exe-only vs full PyInstaller onedir builds."""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

FINGERPRINT_FILENAME = ".oppoct_batch_bundle_fingerprint.json"

# Wheels with compiled extensions in _internal; mismatch with embedded PYZ causes ImportError at runtime.
_TRACKED_PACKAGES = (
    "numpy",
    "matplotlib",
    "torch",
    "scipy",
    "pandas",
    "nibabel",
    "pydicom",
    "SimpleITK",
)


def _interpreter_tag() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = ""
    return out


def record_batch_gui_bundle_fingerprint(package_dir: Path) -> None:
    payload = {
        "interpreter": _interpreter_tag(),
        "packages": _package_versions(),
    }
    (package_dir / FINGERPRINT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def verify_batch_gui_bundle_fingerprint(package_dir: Path) -> tuple[bool, list[str]]:
    path = package_dir / FINGERPRINT_FILENAME
    if not path.is_file():
        return False, [
            f"Missing {FINGERPRINT_FILENAME} under {package_dir}. "
            "Run a full build once: python build_batch_gui.py",
        ]

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, [f"Invalid fingerprint JSON ({e}); run: python build_batch_gui.py"]

    errs: list[str] = []
    cur_interp = _interpreter_tag()
    stored_interp = stored.get("interpreter", "")
    if stored_interp != cur_interp:
        errs.append(
            f"Interpreter mismatch: bundle was built with Python {stored_interp}, "
            f"this venv is {cur_interp}. Run: python build_batch_gui.py",
        )

    cur_pkgs = _package_versions()
    stored_pkgs = stored.get("packages") if isinstance(stored.get("packages"), dict) else {}
    for name in _TRACKED_PACKAGES:
        a, b = stored_pkgs.get(name, ""), cur_pkgs.get(name, "")
        if a != b:
            errs.append(
                f"Package {name!r} version mismatch: _internal has {a!r}, "
                f"current venv has {b!r}. Run: python build_batch_gui.py "
                "(exe-only cannot refresh native extensions in _internal).",
            )

    return not errs, errs
