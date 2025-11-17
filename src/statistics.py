"""
Statistics Module

Calculates HU intensity statistics (mean, volume) for lumbar vertebrae
from segmentation masks.
"""

import json
import logging
import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path
from typing import Dict, Optional

# Lumbar vertebrae labels
LUMBAR_VERTEBRAE = [
    'vertebrae_L1',
    'vertebrae_L2',
    'vertebrae_L3',
    'vertebrae_L4',
    'vertebrae_L5'
]


def calculate_vertebra_statistics(
    ct_image: np.ndarray,
    mask: np.ndarray,
    voxel_volume_mm3: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate volume and mean intensity for a single vertebra.
    
    Args:
        ct_image: 3D numpy array of CT image (HU values)
        mask: 3D numpy array of segmentation mask (binary or label)
        voxel_volume_mm3: Volume of a single voxel in mm³. If None, uses voxel count.
        
    Returns:
        Dictionary with 'volume' and 'intensity' (mean HU)
    """
    # Ensure mask is boolean
    mask_binary = mask > 0
    
    # Count voxels
    voxel_count = np.sum(mask_binary)
    
    # Calculate volume
    if voxel_volume_mm3 is not None:
        volume = voxel_count * voxel_volume_mm3
    else:
        # Use voxel count as volume (will need proper spacing for real volume)
        volume = float(voxel_count)
    
    # Extract intensity values within mask
    intensities = ct_image[mask_binary]
    
    # Calculate mean intensity
    if len(intensities) > 0:
        mean_intensity = float(np.mean(intensities))
    else:
        mean_intensity = 0.0
        volume = 0.0
    
    return {
        'volume': volume,
        'intensity': mean_intensity
    }


def calculate_patient_statistics(
    patient_id: str,
    ct_image_path: Path,
    segmentation_dir: Path,
    voxel_spacing: Optional[list] = None
) -> Dict:
    """
    Calculate statistics for all lumbar vertebrae for a patient.
    
    Args:
        patient_id: Patient identifier
        ct_image_path: Path to original CT NIfTI file
        segmentation_dir: Directory containing segmentation masks
        voxel_spacing: Voxel spacing [z, y, x] in mm. If None, uses voxel count.
        
    Returns:
        Dictionary with patient_id and statistics for each vertebra
    """
    # Load CT image
    ct_nifti = nib.load(str(ct_image_path))
    ct_image = ct_nifti.get_fdata()
    
    # Calculate voxel volume if spacing provided
    voxel_volume_mm3 = None
    if voxel_spacing:
        voxel_volume_mm3 = np.prod(voxel_spacing)
    
    # Initialize statistics dictionary
    statistics = {
        'patient_id': patient_id
    }
    
    # Process each lumbar vertebra
    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        
        if mask_path.exists():
            # Load mask
            mask_nifti = nib.load(str(mask_path))
            mask = mask_nifti.get_fdata()
            
            # Ensure mask and CT have same shape
            if mask.shape != ct_image.shape:
                # TotalSegmentator resamples images internally, so masks are in resampled space
                # Instead of resampling masks, resample CT to match mask space
                # This preserves the mask shape as-is and works in TotalSegmentator's space
                logging.info(f"Mask shape {mask.shape} doesn't match CT shape {ct_image.shape} for {vertebra}. Resampling CT to match mask space...")
                
                # Get spacings
                mask_spacing = mask_nifti.header.get_zooms()[:3]
                ct_spacing = ct_nifti.header.get_zooms()[:3]
                
                # Calculate resampling factors (CT to mask space)
                zoom_factors = [c / m for c, m in zip(ct_spacing, mask_spacing)]
                
                # Resample CT to mask space using order=1 (linear interpolation for CT HU values)
                ct_resampled = ndimage.zoom(ct_image, zoom_factors, order=1, mode='nearest')
                
                # If resampled CT is still different size, crop/pad to match mask
                if ct_resampled.shape != mask.shape:
                    # Calculate padding or cropping needed to match mask shape
                    target_shape = mask.shape
                    pad_before = []
                    pad_after = []
                    slices = []
                    
                    for i, (target_dim, resampled_dim) in enumerate(zip(target_shape, ct_resampled.shape)):
                        if resampled_dim < target_dim:
                            # Need padding
                            diff = target_dim - resampled_dim
                            pad_before.append(diff // 2)
                            pad_after.append(diff - diff // 2)
                            slices.append(slice(None))
                        else:
                            # Need cropping
                            pad_before.append(0)
                            pad_after.append(0)
                            diff = resampled_dim - target_dim
                            start = diff // 2
                            slices.append(slice(start, start + target_dim))
                    
                    # Apply cropping first
                    ct_resampled = ct_resampled[tuple(slices)]
                    
                    # Apply padding if needed
                    if any(pb > 0 or pa > 0 for pb, pa in zip(pad_before, pad_after)):
                        ct_resampled = np.pad(ct_resampled, 
                                              list(zip(pad_before, pad_after)),
                                              mode='constant', constant_values=0)
                
                # Use resampled CT and original mask (no mask resampling)
                ct_slice = ct_resampled
                # Update voxel spacing to mask spacing for volume calculation
                if voxel_volume_mm3 is not None:
                    mask_voxel_volume = np.prod(mask_spacing)
                    voxel_volume_mm3 = mask_voxel_volume
            else:
                ct_slice = ct_image
            
            # Calculate statistics
            stats = calculate_vertebra_statistics(ct_slice, mask, voxel_volume_mm3)
            statistics[vertebra] = stats
        else:
            # No segmentation found
            statistics[vertebra] = {
                'volume': 0.0,
                'intensity': 0.0
            }
    
    return statistics


def save_statistics_json(statistics: Dict, output_path: Path) -> None:
    """
    Save statistics to JSON file matching existing format.
    
    Args:
        statistics: Statistics dictionary
        output_path: Path to output JSON file
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=4)


def load_statistics_json(json_path: Path) -> Dict:
    """
    Load statistics from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Statistics dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)



