
import nibabel as nib
from pathlib import Path
import sys

def check_spacing(path):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return

    img = nib.load(str(p))
    header = img.header
    zooms = header.get_zooms()
    shape = img.shape
    print(f"File: {p.name}")
    print(f"  Shape: {shape}")
    print(f"  Spacing: {zooms}")
    print("-" * 30)

# Check specific files from the previous run
base_dir = Path("out-one")
scan_dir = list(base_dir.glob("*/*"))[0] # Just grab the first inner folder
seg_dir = scan_dir / "segmentations"

print(f"Checking in: {scan_dir}")

# Find any segmentation file
for f in seg_dir.glob("*.nii.gz"):
    check_spacing(f)

# Also check if we can find the temp nifti if it exists, or one of the inputs
# Ideally we'd check the reference CT but it might be in the DICOM folder.
# Let's try to find an input nifti if generated, otherwise skip.
