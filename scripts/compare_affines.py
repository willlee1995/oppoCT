
import sys
import nibabel as nib
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from dicom_processor import dicom_to_nifti

def compare_affines():
    # 1. Generate NIfTI from DICOM using our pipeline logic
    dicom_dir = Path("data/Plain5mmSTD")
    temp_nifti_path = Path("debug_pipeline_converted.nii.gz")
    
    print(f"Converting DICOM from {dicom_dir} to {temp_nifti_path}...")
    dicom_to_nifti(dicom_dir, temp_nifti_path)
    
    my_img = nib.load(str(temp_nifti_path))
    my_affine = my_img.affine
    
    print("\nPipeline Generated Affine:")
    print(my_affine)
    
    # 2. Load Reference Segmentation (assuming it shares space with the Reference Input Image)
    # Note: The segmentation mask SHOULD have the same affine as the input image used to generate it.
    ref_seg_path = Path("seg/vertebrae_L1.nii.gz")
    if not ref_seg_path.exists():
        print("Reference segmentation not found.")
        return

    ref_img = nib.load(str(ref_seg_path))
    ref_affine = ref_img.affine
    
    print("\nReference TotalSegmentator Output Affine:")
    print(ref_affine)
    
    # Compare
    diff = np.abs(my_affine - ref_affine)
    print("\nDifference:")
    print(diff)
    
    if np.allclose(my_affine, ref_affine, atol=1e-3):
        print("\nAffines MATCH closely.")
    else:
        print("\nAffines DO NOT MATCH.")

if __name__ == "__main__":
    compare_affines()
