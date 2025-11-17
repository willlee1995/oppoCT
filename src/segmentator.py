"""
Segmentation Module

Interfaces with TotalSegmentator Python API to segment lumbar vertebrae.
"""

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import nibabel as nib
import shutil
import platform

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
    
    # Determine if preview can be enabled (Linux/WSL supports it, Windows native doesn't)
    enable_preview = platform.system() != 'Windows'
    
    try:
        # Try with roi_subset first (more memory efficient)
        # If it fails with empty crops, fall back to full image segmentation
        try:
            if verbose:
                logging.info("Attempting segmentation with ROI subset for lumbar vertebrae...")
            totalsegmentator(
                input=str(nifti_path),
                output=str(output_dir),
                roi_subset=LUMBAR_VERTEBRAE,
                fast=fast,
                device=device,
                verbose=verbose,
                preview=enable_preview  # Enable preview on Linux/WSL, disable on Windows native
            )
            
            # Check if we got any non-empty masks
            found_vertebrae = verify_segmentation_output(output_dir)
            if len(found_vertebrae) == 0:
                if verbose:
                    logging.warning("ROI subset produced empty masks. Falling back to full image segmentation...")
                raise ValueError("Empty masks from ROI subset")
            else:
                if verbose:
                    logging.info(f"Successfully segmented {len(found_vertebrae)} vertebrae using ROI subset")
        except (ValueError, RuntimeError) as roi_error:
            # Fall back to full image segmentation if ROI subset fails
            if verbose:
                logging.info("Falling back to full image segmentation (this uses more memory)...")
                if "memory" in str(roi_error).lower() or "out of memory" in str(roi_error).lower():
                    logging.warning("Memory error detected. Consider using --fast flag for lower memory usage.")
            
            # Clear output directory before retry
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use fast mode on fallback to reduce memory usage (unless already specified)
            use_fast = True  # Force fast mode on fallback for memory efficiency
            
            totalsegmentator(
                input=str(nifti_path),
                output=str(output_dir),
                fast=use_fast,
                device=device,
                verbose=verbose,
                preview=enable_preview  # Enable preview on Linux/WSL, disable on Windows native
            )
        
        # Clean up any non-vertebrae files that might exist
        _cleanup_non_vertebrae_files(output_dir, verbose)
        
        if verbose:
            logging.info(f"Segmentation completed. Masks saved to {output_dir}")
        
        return output_dir
        
    except Exception as e:
        error_msg = f"Error during segmentation: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def _cleanup_non_vertebrae_files(output_dir: Path, verbose: bool = True) -> None:
    """
    Remove any segmentation files that are not lumbar vertebrae.
    
    Args:
        output_dir: Directory containing segmentation masks
        verbose: Print cleanup messages
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    
    removed_count = 0
    for file_path in output_dir.glob("*.nii*"):
        # Check if file is a lumbar vertebra (handle both .nii.gz and .nii extensions)
        is_vertebra = False
        for vertebra in LUMBAR_VERTEBRAE:
            if file_path.name.startswith(vertebra) and file_path.name.endswith(('.nii.gz', '.nii')):
                is_vertebra = True
                break
        
        if not is_vertebra:
            try:
                file_path.unlink()
                removed_count += 1
                if verbose:
                    logging.debug(f"Removed non-vertebrae file: {file_path.name}")
            except Exception as e:
                if verbose:
                    logging.warning(f"Failed to remove {file_path.name}: {e}")
    
    if verbose and removed_count > 0:
        logging.info(f"Cleaned up {removed_count} non-vertebrae segmentation files")


def verify_segmentation_output(output_dir: Path) -> List[str]:
    """
    Verify that segmentation output contains expected lumbar vertebrae masks.
    Checks both file existence and that masks are non-empty.
    
    Args:
        output_dir: Directory containing segmentation masks
        
    Returns:
        List of found vertebra labels with non-empty masks
    """
    found_vertebrae = []
    
    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = output_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            try:
                # Check if mask is non-empty
                mask_nifti = nib.load(str(mask_path))
                mask = mask_nifti.get_fdata()
                if np.sum(mask > 0) > 0:
                    found_vertebrae.append(vertebra)
            except Exception as e:
                logging.warning(f"Error reading mask {mask_path}: {e}")
    
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



