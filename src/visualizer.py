"""
Visualization Module

Generates preview images showing segmented vertebrae overlaid on original CT images.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Optional, Dict

# Lumbar vertebrae labels
LUMBAR_VERTEBRAE = [
    'vertebrae_L1',
    'vertebrae_L2',
    'vertebrae_L3',
    'vertebrae_L4',
    'vertebrae_L5'
]

# Color map for different vertebrae
VERTEBRAE_COLORS = {
    'vertebrae_L1': 'red',
    'vertebrae_L2': 'orange',
    'vertebrae_L3': 'yellow',
    'vertebrae_L4': 'green',
    'vertebrae_L5': 'blue'
}


def find_representative_slices(
    ct_image: np.ndarray,
    segmentation_masks: Dict[str, np.ndarray],
    num_slices: int = 3
) -> List[int]:
    """
    Find representative slices that contain the most vertebra content.
    
    Args:
        ct_image: 3D CT image array
        segmentation_masks: Dictionary of vertebra_name -> mask array
        num_slices: Number of slices to return
        
    Returns:
        List of slice indices
    """
    # Sum all masks to find slices with most content
    combined_mask = np.zeros_like(ct_image, dtype=bool)
    for mask in segmentation_masks.values():
        combined_mask |= (mask > 0)
    
    # Sum along z-axis to find slices with most content
    slice_sums = np.sum(combined_mask, axis=(0, 1))
    
    # Get indices of slices with most content
    if np.sum(slice_sums) == 0:
        # No segmentation found, return middle slices
        return [ct_image.shape[2] // 2]
    
    top_indices = np.argsort(slice_sums)[-num_slices:]
    return sorted(top_indices.tolist())


def create_preview(
    ct_image: np.ndarray,
    segmentation_masks: Dict[str, np.ndarray],
    output_path: Path,
    slice_indices: Optional[List[int]] = None,
    window_level: int = 40,
    window_width: int = 400
) -> None:
    """
    Create preview image showing segmented vertebrae overlaid on CT.
    
    Args:
        ct_image: 3D CT image array
        segmentation_masks: Dictionary of vertebra_name -> mask array
        output_path: Path to save preview image
        slice_indices: List of slice indices to display. If None, finds representative slices.
        window_level: Window level for CT display
        window_width: Window width for CT display
    """
    # Find representative slices if not provided
    if slice_indices is None:
        slice_indices = find_representative_slices(ct_image, segmentation_masks, num_slices=3)
    
    num_slices = len(slice_indices)
    fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 5))
    
    if num_slices == 1:
        axes = [axes]
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        
        # Extract slice
        ct_slice = ct_image[:, :, slice_idx]
        
        # Apply windowing
        ct_min = window_level - window_width / 2
        ct_max = window_level + window_width / 2
        ct_slice_display = np.clip(ct_slice, ct_min, ct_max)
        ct_slice_display = (ct_slice_display - ct_min) / (ct_max - ct_min)
        
        # Display CT slice
        ax.imshow(ct_slice_display, cmap='gray', origin='lower')
        
        # Overlay segmentation masks
        for vertebra_name, mask in segmentation_masks.items():
            if mask.shape != ct_image.shape:
                # Resize mask if needed
                min_shape = tuple(min(m, c) for m, c in zip(mask.shape, ct_image.shape))
                mask_slice = mask[:min_shape[0], :min_shape[1], slice_idx] if slice_idx < min_shape[2] else np.zeros((min_shape[0], min_shape[1]))
            else:
                mask_slice = mask[:, :, slice_idx]
            
            if np.any(mask_slice > 0):
                color = VERTEBRAE_COLORS.get(vertebra_name, 'cyan')
                ax.contour(mask_slice, levels=[0.5], colors=color, linewidths=2, alpha=0.8)
        
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            mask_nifti = nib.load(str(mask_path))
            mask = mask_nifti.get_fdata()
            
            # Ensure mask matches CT shape
            if mask.shape == ct_image.shape:
                segmentation_masks[vertebra] = mask
            else:
                print(f"Warning: Mask shape {mask.shape} doesn't match CT shape {ct_image.shape} for {vertebra}")
                # Try to match by cropping
                min_shape = tuple(min(m, c) for m, c in zip(mask.shape, ct_image.shape))
                mask_cropped = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
                segmentation_masks[vertebra] = mask_cropped
    
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



