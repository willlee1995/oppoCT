
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nibabel as nib
import numpy as np
from src.pipeline import find_patient_folders
from src.dicom_processor import dicom_to_nifti
import tempfile
import shutil

def check_file(path, label="File"):
    p = Path(path)
    if not p.exists():
        print(f"{label} not found: {p}")
        return None

    img = nib.load(str(p))
    header = img.header
    zooms = header.get_zooms()
    shape = img.shape
    print(f"=== {label} ===")
    print(f"Path: {p}")
    print(f"Shape: {shape}")
    print(f"Spacing: {zooms}")
    return zooms, shape

# Find the processed case
processed_dir = Path("processed-one") # From the user's last command
output_dir = Path("out-one")

patient_folders = find_patient_folders(processed_dir)
if not patient_folders:
    print("No patient folders found in processed-one")
    exit(1)

patient_folder = patient_folders[0]
patient_id = patient_folder.parent.name # Roughly guessing ID structure S0089178/...
# Actually let's use the folder name structure from logs: processed\S0089178\Abdomen0_GIB_Flash_BT_Multiphase Adult\C PV 10.8
# ID is S0089178.
# Output is out-one\S0089178\C PV 10.8\segmentations

print(f"Checking Patient Folder: {patient_folder}")

# 1. Check Original by converting one series to NIfTI temporarily
temp_dir = Path(tempfile.mkdtemp())
try:
    temp_nifti = temp_dir / "temp_ct.nii.gz"
    print("Converting DICOM to NIfTI for inspection...")
    dicom_to_nifti(patient_folder, temp_nifti)
    check_file(temp_nifti, "Original CT (Converted)")

finally:
    shutil.rmtree(temp_dir)

# 2. Check Segmentation
# Try to find the output folder.
# We'll search recursively in out-one for .nii.gz
seg_files = list(output_dir.rglob("vertebrae_L5.nii.gz"))
if seg_files:
    check_file(seg_files[0], "Segmentation (L5)")
else:
    print("No segmentation found in out-one")
