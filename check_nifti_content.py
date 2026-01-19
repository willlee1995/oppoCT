import nibabel as nib
import numpy as np
from pathlib import Path
import sys

def check_file(path):
    print(f"Checking {path}...")
    try:
        img = nib.load(path)
        data = img.get_fdata()
        unique = np.unique(data)
        print(f"  Shape: {data.shape}")
        print(f"  Unique values: {unique}")
        print(f"  Non-zero count: {np.count_nonzero(data)}")
        print(f"  Affine:\n{img.affine}")
    except Exception as e:
        print(f"  Error reading file: {e}")

folder = Path("out/test_run_8/ANNOY/segmentations")
if not folder.exists():
    print(f"Folder not found: {folder}")
    sys.exit(1)

# Check one segmentation file
files = list(folder.glob("*.nii.gz"))
if not files:
    print("No segmentation files found.")
else:
    # Check L1 or just the first few
    for p in files[:3]:
        check_file(p)
