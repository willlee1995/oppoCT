"""
Segmentation Module

Interfaces with TotalSegmentator Python API to segment lumbar vertebrae.
"""

from pathlib import Path
from typing import List, Optional
import logging

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    raise ImportError(
        "TotalSegmentator is not installed. Please install it with: pip install totalsegmentator"
    )

# Lumbar vertebrae labels
LUMBAR_VERTEBRAE = [
    'vertebrae_L1',
    'vertebrae_L2',
    'vertebrae_L3',
    'vertebrae_L4',
    'vertebrae_L5'
]


def segment_lumbar_vertebrae(
    nifti_path: Path,
    output_dir: Path,
    fast: bool = False,
    device: str = 'gpu',
    verbose: bool = True
) -> Path:
    """
    Segment lumbar vertebrae (L1-L5) from CT image using TotalSegmentator.
    
    Args:
        nifti_path: Path to input NIfTI file
        output_dir: Directory to save segmentation masks
        fast: Use fast mode (lower quality, faster processing)
        device: Device to use ('gpu' or 'cpu')
        verbose: Print progress messages
        
    Returns:
        Path to output directory containing segmentation masks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        logging.info(f"Segmenting lumbar vertebrae from {nifti_path}")
        logging.info(f"Output directory: {output_dir}")
    
    try:
        # Call TotalSegmentator API with ROI subset for lumbar vertebrae only
        totalsegmentator(
            input=str(nifti_path),
            output=str(output_dir),
            roi_subset=LUMBAR_VERTEBRAE,
            fast=fast,
            device=device,
            verbose=verbose
        )
        
        if verbose:
            logging.info(f"Segmentation completed. Masks saved to {output_dir}")
        
        return output_dir
        
    except Exception as e:
        error_msg = f"Error during segmentation: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def verify_segmentation_output(output_dir: Path) -> List[str]:
    """
    Verify that segmentation output contains expected lumbar vertebrae masks.
    
    Args:
        output_dir: Directory containing segmentation masks
        
    Returns:
        List of found vertebra labels
    """
    found_vertebrae = []
    
    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = output_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            found_vertebrae.append(vertebra)
    
    return found_vertebrae


def process_patient_batch(
    patient_nifti_paths: List[tuple],
    output_base_dir: Path,
    fast: bool = False,
    device: str = 'gpu'
) -> dict:
    """
    Process multiple patients in batch.
    
    Args:
        patient_nifti_paths: List of tuples (patient_id, nifti_path)
        output_base_dir: Base output directory
        fast: Use fast mode
        device: Device to use
        
    Returns:
        Dictionary mapping patient_id to output directory
    """
    results = {}
    
    for patient_id, nifti_path in patient_nifti_paths:
        patient_output_dir = output_base_dir / patient_id / 'segmentations'
        
        try:
            segment_lumbar_vertebrae(
                nifti_path=Path(nifti_path),
                output_dir=patient_output_dir,
                fast=fast,
                device=device,
                verbose=True
            )
            results[patient_id] = patient_output_dir
        except Exception as e:
            logging.error(f"Failed to segment patient {patient_id}: {e}")
            results[patient_id] = None
    
    return results



