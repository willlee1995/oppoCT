# -*- mode: python ; coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Trailing arguments after ``pyinstaller ... --`` are forwarded here; PyInstaller sets
# ``sys.argv`` to ``[spec_path, ...spec_args]`` (the ``--`` itself is not included).
_spec_argv = sys.argv[1:]
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--exe-only",
    action="store_true",
    help="Build the EXE (PKG/PYZ) only; skip COLLECT for pairing with an existing onedir package.",
)
_spec_opts, _spec_unknown = _parser.parse_known_args(_spec_argv)
exe_only = _spec_opts.exe_only

block_cipher = None
project_root = Path.cwd()

datas = []
weights_dir = project_root / "totalsegmentator_weights"
if weights_dir.exists():
    datas.append((str(weights_dir), "totalsegmentator_weights"))

readme_path = project_root / "README.md"
if readme_path.exists():
    datas.append((str(readme_path), "."))

datas += collect_data_files("totalsegmentator", include_py_files=False)
datas += collect_data_files("nnunetv2", include_py_files=False)

hidden_imports = [
    "batch_verification",
    "verify_segmentation",
    "scripts.gui_batch_verification",
    "src",
    "src.csv_exporter",
    "src.dicom_processor",
    "src.patient_manager",
    "src.pipeline",
    "src.segmentator",
    "src.statistics",
    "src.visualizer",
    "matplotlib.backends.backend_tkagg",
    "nibabel",
    "numpy",
    "pandas",
    "pydicom",
    "PIL",
    "scipy",
    "SimpleITK",
    "tkinter",
    "torch",
    "torchvision",
    "torchaudio",
    "totalsegmentator",
    "nnunetv2",
]

hidden_imports += collect_submodules("nnunetv2")
hidden_imports += collect_submodules("totalsegmentator")

a = Analysis(
    ["run_batch_gui.py"],
    pathex=[str(project_root), str(project_root / "scripts")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="oppoCT-Batch-QC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if not exe_only:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="oppoCT-Batch-QC-Package",
    )
