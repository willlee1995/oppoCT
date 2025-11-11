"""
Statistics Module

Calculates HU intensity statistics (mean, volume) for lumbar vertebrae
from segmentation masks.
"""

import json
import numpy as np
import nibabel as nib
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
                # Resample mask to match CT (simple approach - may need proper resampling)
                print(f"Warning: Mask shape {mask.shape} doesn't match CT shape {ct_image.shape} for {vertebra}")
                # Try to match by cropping/padding
                min_shape = tuple(min(m, c) for m, c in zip(mask.shape, ct_image.shape))
                mask = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
                ct_slice = ct_image[:min_shape[0], :min_shape[1], :min_shape[2]]
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



