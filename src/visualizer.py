"""
Visualization Module

Generates preview images showing segmented vertebrae overlaid on original CT images.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Define vertebrae and their colors for visualization
VERTEBRAE_COLORS = {
    'vertebrae_T11': 'purple',
    'vertebrae_T12': 'pink',
    'vertebrae_L1': 'red',
    'vertebrae_L2': 'orange',
    'vertebrae_L3': 'yellow',
    'vertebrae_L4': 'green',
    'vertebrae_L5': 'blue',
    'vertebrae_T11_body': 'purple',
    'vertebrae_T12_body': 'pink',
    'vertebrae_L1_body': 'red',
    'vertebrae_L2_body': 'orange',
    'vertebrae_L3_body': 'yellow',
    'vertebrae_L4_body': 'green',
    'vertebrae_L5_body': 'blue'
}

# Define the list of vertebrae to process
LUMBAR_VERTEBRAE = [
    'vertebrae_T11',
    'vertebrae_T12',
    'vertebrae_L1',
    'vertebrae_L2',
    'vertebrae_L3',
    'vertebrae_L4',
    'vertebrae_L5'
]

def find_representative_slices(ct_volume: np.ndarray, masks: Dict[str, np.ndarray], num_slices: int = 3) -> List[int]:
    """
    Find representative slices that contain the most vertebral content.
    
    Args:
        ct_volume: 3D CT volume
        masks: Dictionary of vertebra_name -> mask array
        num_slices: Number of slices to return
        
    Returns:
        List of slice indices
    """
    # Count non-zero pixels per slice across all masks
    slice_counts = np.zeros(ct_volume.shape[2])
    
    for mask in masks.values():
        # Ensure mask matches CT depth
        if len(mask.shape) == 3 and mask.shape[2] == ct_volume.shape[2]:
            # Sum non-zero pixels for each slice
            # axis=(0, 1) sums over height and width, leaving depth
            slice_counts += np.sum(mask > 0, axis=(0, 1))
            
    # Find slices with maximum content
    # Get indices of slices sorted by count (descending)
    sorted_indices = np.argsort(slice_counts)[::-1]
    
    # Filter out slices with zero content
    valid_indices = [idx for idx in sorted_indices if slice_counts[idx] > 0]
    
    if not valid_indices:
        return []
    
    # Select top N slices, but try to space them out if possible
    # For now, just take the top N unique slices
    selected_slices = sorted(valid_indices[:num_slices])
    
    return [int(s) for s in selected_slices]

def create_preview(
    ct_image: np.ndarray,
    segmentation_masks: Dict[str, np.ndarray],
    output_path: Path,
    window_level: int = 40,
    window_width: int = 400
) -> None:
    """
    Generate a preview image with segmentation overlays.
    """
    # Find the slice with the most segmentation content
    best_slice = 0
    max_content = 0
    
    # Combine all masks to find best slice
    combined_mask = np.zeros_like(ct_image, dtype=bool)
    for mask in segmentation_masks.values():
        if mask.shape == ct_image.shape:
            combined_mask = combined_mask | (mask > 0)
            
    # Sum content per slice
    slice_content = np.sum(combined_mask, axis=(0, 1))
    best_slice = np.argmax(slice_content)
    
    if slice_content[best_slice] == 0:
        # No segmentation found, use middle slice
        best_slice = ct_image.shape[2] // 2
        
    # Extract slice
    ct_slice = ct_image[:, :, best_slice]
    
    # Windowing
    ct_min = window_level - window_width / 2
    ct_max = window_level + window_width / 2
    ct_display = np.clip(ct_slice, ct_min, ct_max)
    ct_display = (ct_display - ct_min) / (ct_max - ct_min)
    
    # Create plot
    plt.figure(figsize=(10, 10))
    plt.imshow(ct_display, cmap='gray', origin='lower')
    
    # Overlay masks
    for name, mask in segmentation_masks.items():
        if mask.shape == ct_image.shape:
            mask_slice = mask[:, :, best_slice]
            if np.any(mask_slice):
                color = VERTEBRAE_COLORS.get(name, 'red')
                plt.contour(mask_slice, levels=[0.5], colors=[color], linewidths=2)
                
    plt.title(f"Preview - Slice {best_slice}")
    plt.axis('off')
    
    # Save
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def create_patient_preview(
    patient_id: str,
    ct_image_path: Path,
    segmentation_dir: Path,
    output_dir: Path,
    window_level: int = 40,
    window_width: int = 400
) -> Path:
    """
    Create preview image for a patient.
    
    Args:
        patient_id: Patient identifier
        ct_image_path: Path to CT NIfTI file
        segmentation_dir: Directory containing segmentation masks
        output_dir: Output directory for preview image
        window_level: Window level for CT display
        window_width: Window width for CT display
        
    Returns:
        Path to saved preview image
    """
    # Load CT image
    ct_nifti = nib.load(str(ct_image_path))
    ct_image = ct_nifti.get_fdata()
    
    # Load segmentation masks
    segmentation_masks = {}
    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            try:
                mask_nifti = nib.load(str(mask_path))
                
                # Check if resampling is needed
                # Check shape and affine (with some tolerance for affine)
                shape_match = mask_nifti.shape == ct_nifti.shape
                affine_match = np.allclose(mask_nifti.affine, ct_nifti.affine, atol=1e-3)
                
                if shape_match and affine_match:
                    mask = mask_nifti.get_fdata()
                else:
                    logging.info(f"Resampling {vertebra} mask to match CT space...")
                    resampled_nifti = resample_from_to(mask_nifti, ct_nifti, order=0)
                    mask = resampled_nifti.get_fdata()
                
                segmentation_masks[vertebra] = mask
                
            except Exception as e:
                logging.warning(f"Failed to load/resample mask {mask_path}: {e}")
    
    # Create preview
    output_path = output_dir / f"{patient_id}_preview.png"
    create_preview(
        ct_image=ct_image,
        segmentation_masks=segmentation_masks,
        output_path=output_path,
        window_level=window_level,
        window_width=window_width
    )
    
    return output_path
