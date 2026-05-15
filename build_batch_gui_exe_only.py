#!/usr/bin/env python
"""
Rebuild ``oppoCT-Batch-QC.exe`` against an existing onedir layout (reuse ``_internal``).

Requires a prior **full** build from ``build_batch_gui.py`` so
``dist/oppoCT-Batch-QC-Package/_internal/`` already exists. This script runs
PyInstaller with ``--exe-only`` (see ``build_windows_batch_gui.spec``), which
skips ``COLLECT``, then copies the executable from the spec workpath
(``build/build_windows_batch_gui/``) into the existing package folder.

Use a **full** build when native deps, datas, weights, hidden imports, or
Python/PyInstaller versions change — otherwise the frozen app can break at runtime.

Environment: same as ``build_batch_gui.py`` (project venv, PyInstaller installed).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_NAME = "oppoCT-Batch-QC-Package"
EXE_NAME = "oppoCT-Batch-QC.exe"
WORK_SUBDIR = "build_windows_batch_gui"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Batch QC exe only; reuse existing dist package _internal.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Pass PyInstaller --clean (wipe cache before build).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    spec_path = project_root / "build_windows_batch_gui.spec"
    dist_dir = project_root / "dist"
    package_dir = dist_dir / PACKAGE_NAME
    internal_dir = package_dir / "_internal"
    work_exe = project_root / "build" / WORK_SUBDIR / EXE_NAME
    dest_exe = package_dir / EXE_NAME

    if not spec_path.is_file():
        print(f"ERROR: Spec not found: {spec_path}", file=sys.stderr)
        return 1

    if not package_dir.is_dir() or not internal_dir.is_dir():
        print(
            "ERROR: Run a full build first so the onedir package exists:\n"
            f"  Expected: {internal_dir}\n"
            "  python build_batch_gui.py\n",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_path),
        "-y",
    ]
    if args.clean:
        cmd.append("--clean")
    cmd.extend(["--", "--exe-only"])

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

    if not work_exe.is_file():
        print(f"ERROR: PyInstaller did not produce: {work_exe}", file=sys.stderr)
        return 1

    shutil.copy2(work_exe, dest_exe)
    print(f"Copied: {work_exe} -> {dest_exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
