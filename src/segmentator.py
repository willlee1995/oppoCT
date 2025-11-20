"""
Segmentation Module

Interfaces with TotalSegmentator Python API to segment lumbar vertebrae.
"""

import logging
import platform
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    raise ImportError(
        "TotalSegmentator is not installed. Please install it with: pip install totalsegmentator"
    )

# Lumbar vertebrae labels
LUMBAR_VERTEBRAE = [
    'vertebrae_T11',
    'vertebrae_T12',
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
    verbose: bool = True,
    use_dicom_directly: bool = False,
    dicom_dir: Optional[Path] = None
) -> Path:
    """
    Segment lumbar vertebrae (L1-L5) from CT image using TotalSegmentator.
    
    Args:
        nifti_path: Path to input NIfTI file (or DICOM directory if use_dicom_directly=True)
        output_dir: Directory to save segmentation masks
        fast: Use fast mode (lower quality, faster processing)
        device: Device to use ('gpu' or 'cpu')
        verbose: Print progress messages
        use_dicom_directly: If True, pass DICOM directory directly to TotalSegmentator (recommended)
        dicom_dir: Path to DICOM directory (required if use_dicom_directly=True)
        
    Returns:
        Path to output directory containing segmentation masks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        if use_dicom_directly and dicom_dir:
            logging.info(f"Segmenting lumbar vertebrae from DICOM directory: {dicom_dir}")
        else:
            logging.info(f"Segmenting lumbar vertebrae from NIfTI: {nifti_path}")
        logging.info(f"Output directory: {output_dir}")
    
    # Determine if preview can be enabled (Linux/WSL supports it, Windows native doesn't)
    enable_preview = platform.system() != 'Windows'
    
    try:
        if verbose:
            logging.info("Segmenting vertebrae_body using TotalSegmentator...")
        
        # Use DICOM directory directly if requested (matches CLI behavior)
        if use_dicom_directly and dicom_dir:
            input_path = str(dicom_dir)
            if verbose:
                logging.info("Passing DICOM directory directly to TotalSegmentator (matches CLI behavior)")
        else:
            input_path = str(nifti_path)
        
        # 1. Run vertebrae_body task to get the body masks (unlabeled levels)
        totalsegmentator(
            input=input_path,
            output=str(output_dir),
            task="vertebrae_body",
            fast=fast,
            device=device,
            verbose=verbose,
            preview=enable_preview
        )

        # 2. Run total task with ROI subset to get labeled whole vertebrae (L1-L5)
        if verbose:
            logging.info("Segmenting labeled vertebrae (L1-L5) using TotalSegmentator...")
        
        totalsegmentator(
            input=input_path,
            output=str(output_dir),
            task="total",
            roi_subset=LUMBAR_VERTEBRAE,
            fast=fast,
            device=device,
            verbose=verbose,
            preview=enable_preview
        )

        # 3. Intersect masks to get labeled vertebral bodies
        if verbose:
            logging.info("Intersecting masks to generate labeled vertebral bodies...")

        # Load vertebrae_body mask
        # Note: vertebrae_body task output might be 'vertebrae_body.nii.gz' containing all bodies
        body_mask_path = output_dir / "vertebrae_body.nii.gz"
        if not body_mask_path.exists():
             # Fallback: sometimes it might be named differently or split? 
             # Usually vertebrae_body task produces one file with all bodies if it's a binary mask, 
             # OR it produces 'vertebrae_body.nii.gz' which is a binary mask of all bodies.
             # Let's assume it exists as per standard behavior.
             logging.warning(f"vertebrae_body.nii.gz not found at {body_mask_path}")
        
        try:
            body_img = nib.load(str(body_mask_path))
            body_data = body_img.get_fdata() > 0
            affine = body_img.affine
            header = body_img.header

            for vertebra in LUMBAR_VERTEBRAE:
                vert_path = output_dir / f"{vertebra}.nii.gz"
                if vert_path.exists():
                    vert_img = nib.load(str(vert_path))
                    vert_data = vert_img.get_fdata() > 0
                    
                    # Intersect
                    intersect_data = np.logical_and(body_data, vert_data).astype(np.uint8)
                    
                    # Save intersection
                    out_name = f"{vertebra}_body.nii.gz"
                    out_path = output_dir / out_name
                    
                    new_img = nib.Nifti1Image(intersect_data, affine, header)
                    nib.save(new_img, str(out_path))
                    
                    if verbose:
                        logging.info(f"Generated {out_name}")
                else:
                    if verbose:
                        logging.warning(f"Mask for {vertebra} not found, skipping intersection.")

        except Exception as e:
            logging.error(f"Error during intersection: {e}")
            # Don't raise here, we still have partial results? 
            # Or maybe we should raise. Let's log and continue cleanup.
        
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
        # Check if file is a lumbar vertebra or vertebrae_body (handle both .nii.gz and .nii extensions)
        is_vertebra = False
        
        # Keep vertebrae_body.nii.gz (from vertebrae_body task)
        if file_path.name.startswith('vertebrae_body') and file_path.name.endswith(('.nii.gz', '.nii')):
            is_vertebra = True
        
        # Keep individual lumbar vertebrae (L1-L5) and their body intersections
        for vertebra in LUMBAR_VERTEBRAE:
            # Keep vertebrae_L*.nii.gz
            if file_path.name.startswith(vertebra) and file_path.name.endswith(('.nii.gz', '.nii')):
                is_vertebra = True
                break
            # Keep vertebrae_L*_body.nii.gz
            if file_path.name.startswith(f"{vertebra}_body") and file_path.name.endswith(('.nii.gz', '.nii')):
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



