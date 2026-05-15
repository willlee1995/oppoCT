#!/usr/bin/env python
"""
Build the oppoCT batch QC GUI with PyInstaller.

This script invokes ``python -m PyInstaller`` against ``build_windows_batch_gui.spec``.
Run it with the **same interpreter** you use for development (ideally the project venv)
so the frozen bundle matches your installed stack (torch, TotalSegmentator, etc.).

Environment setup (.venv)
-------------------------
From the repository root (``oppoCT/``):

1. Create a virtual environment::

       python -m venv .venv

2. Activate it (Windows PowerShell)::

       .\\.venv\\Scripts\\Activate.ps1

   Or cmd.exe::

       .venv\\Scripts\\activate.bat

3. Install runtime dependencies (see ``requirements.txt`` / your usual install path), e.g.::

       python -m pip install -r requirements.txt

4. Install PyInstaller into **this** venv::

       python -m pip install pyinstaller

   If ``pip`` is missing (``No module named pip``), bootstrap it then retry::

       python -m ensurepip --upgrade
       python -m pip install pyinstaller

5. Optional but recommended for offline executables: download model weights so
   ``totalsegmentator_weights/`` exists (see ``download_weights_lumbar.py`` and the
   warning this script prints if the folder is absent).

Build
-----
Always run from the project root so paths in the ``.spec`` file resolve correctly
(``Path.cwd()`` in the spec matches the working directory).

Usage::

    python build_batch_gui.py
    python build_batch_gui.py --clean

After a full build, pure workflow iterations can use ``python build_batch_gui_exe_only.py``
(see that script’s docstring). Output goes under ``dist/`` — typically ``dist/oppoCT-Batch-QC-Package/`` with
``oppoCT-Batch-QC.exe`` inside.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build oppoCT-Batch-QC with PyInstaller.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Pass PyInstaller --clean (wipe cache before build).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    spec_path = project_root / "build_windows_batch_gui.spec"

    if not spec_path.is_file():
        print(f"ERROR: Spec not found: {spec_path}", file=sys.stderr)
        return 1

    weights_dir = project_root / "totalsegmentator_weights"
    if not weights_dir.is_dir():
        print(
            "WARNING: totalsegmentator_weights/ is missing. "
            "Offline builds will fail at runtime. Run:\n"
            "  python download_weights_lumbar.py --weights-dir totalsegmentator_weights --device cpu\n",
            file=sys.stderr,
        )

    license_config = project_root / "totalsegmentator_home" / "config.json"
    if not license_config.is_file():
        print(
            "WARNING: totalsegmentator_home/config.json is missing. "
            "vertebrae_body (commercial) will fail without a license. Run:\n"
            "  python scripts/setup_totalseg_license.py -l aca_...\n"
            "  or: totalseg_set_license -l aca_... then python scripts/setup_totalseg_license.py\n",
            file=sys.stderr,
        )

    # Use sys.executable so the build uses the active venv / interpreter, not a random PyInstaller on PATH.
    cmd = [sys.executable, "-m", "PyInstaller", str(spec_path), "-y"]
    if args.clean:
        cmd.append("--clean")

    print("Running:", " ".join(cmd))
    print("cwd:", project_root)
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(
            "PyInstaller failed. In this venv run: python -m pip install pyinstaller\n"
            "If pip is missing: python -m ensurepip --upgrade",
            file=sys.stderr,
        )
        return result.returncode

    from bundle_build_guard import record_batch_gui_bundle_fingerprint

    package_dir = project_root / "dist" / "oppoCT-Batch-QC-Package"
    if package_dir.is_dir():
        record_batch_gui_bundle_fingerprint(package_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
