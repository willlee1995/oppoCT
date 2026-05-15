"""
Segmentation Module

Interfaces with TotalSegmentator Python API to segment lumbar vertebrae.
"""

import gc
import logging
import platform
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

from .totalseg_local_weights import (
    enforce_local_weights_if_configured,
    skip_vertebrae_body_seg,
)
from .totalseg_runtime import apply_totalseg_runtime_paths

apply_totalseg_runtime_paths()

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


def _load_nifti(path: Path) -> nib.Nifti1Image:
    """Load NIfTI without mmap so Windows does not keep volume files locked."""
    try:
        return nib.load(str(path), mmap=False)
    except TypeError:
        return nib.load(str(path))


def generate_trabecular_core(output_dir: Path, verbose: bool = True) -> None:
    """
    Generate trabecular core mask for L1 body by eroding 2.5mm.

    Args:
        output_dir: Directory containing segmentation masks
        verbose: Print progress messages
    """
    l1_body_path = output_dir / "vertebrae_L1_body.nii.gz"
    if not l1_body_path.exists():
        if verbose:
            logging.warning("vertebrae_L1_body.nii.gz not found, skipping trabecular core generation.")
        return

    try:
        img = _load_nifti(l1_body_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        del img

        # Get voxel spacing to calculate erosion distance in pixels
        zooms = header.get_zooms()
        # Use only first 3 dimensions (x, y, z)
        sampling = zooms[:3]

        if verbose:
            logging.info(f"Generating L1 trabecular core (erosion 2.5mm, spacing: {sampling})")

        # Use distance transform for accurate metric erosion
        # Calculate distance from background (0)
        dt = ndimage.distance_transform_edt(data, sampling=sampling)

        # Create core mask (keep pixels > 2.5mm from boundary)
        core_mask = dt > 2.5
        core_mask = core_mask.astype(np.uint8)

        # Save result
        out_name = "vertebrae_L1_body_trabecular_core.nii.gz"
        out_path = output_dir / out_name

        new_img = nib.Nifti1Image(core_mask, affine, header)
        nib.save(new_img, str(out_path))

        if verbose:
            logging.info(f"Generated {out_name}")

    except Exception as e:
        logging.error(f"Error generating trabecular core: {e}")


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

    # Check device availability
    import torch
    if device == 'gpu' and not torch.cuda.is_available():
        logging.warning("GPU requested but verify `torch.cuda.is_available()` is False. TotalSegmentator will likely fall back to CPU or fail.")
    elif device == 'gpu':
        try:
             logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except:
             pass

    # Determine if preview can be enabled (Linux/WSL supports it, Windows native doesn't)
    enable_preview = platform.system() != 'Windows'

    enforce_local_weights_if_configured(fast=fast, verbose=verbose)

    try:
        # Use DICOM directory directly if requested (matches CLI behavior)
        if use_dicom_directly and dicom_dir:
            input_path = str(dicom_dir)
            if verbose:
                logging.info("Passing DICOM directory directly to TotalSegmentator (matches CLI behavior)")
        else:
            input_path = str(nifti_path)

        body_mask_path = output_dir / "vertebrae_body.nii.gz"
        if skip_vertebrae_body_seg():
            if verbose:
                logging.info(
                    "Skipping vertebrae_body (OPPOCT_SKIP_VERTEBRAE_BODY_SEG); "
                    "labeled body intersections may be incomplete."
                )
        else:
            if verbose:
                logging.info("Segmenting vertebrae_body using TotalSegmentator...")

            # 1. Run vertebrae_body task to get the body masks (unlabeled levels)
            # This will produce 'vertebrae_body.nii.gz'
            # IMPORTANT: When ml=True, output must be a file path, not a directory
            try:
                totalsegmentator(
                    input=input_path,
                    output=str(body_mask_path),
                    task="vertebrae_body",
                    fast=False,  # vertebrae_body task does not support fast mode
                    device=device,
                    verbose=verbose,
                    preview=enable_preview,
                    ml=True  # Save as multilabel to get a single file (merged or single class)
                )
            except SystemExit:
                logging.warning("TotalSegmentator 'vertebrae_body' task failed (likely due to missing license).")
                logging.warning("Proceeding without separate body masks (trabecular core analysis may be skipped).")
                # Ensure we don't leave a partial file
                if body_mask_path.exists():
                    # check if it's empty or valid?
                    pass
            except Exception as e:
                logging.warning(f"TotalSegmentator 'vertebrae_body' task failed: {e}")
                logging.warning("Proceeding without separate body masks.")

        # 2. Run total task with ROI subset to get labeled whole vertebrae (L1-L5)
        # This will produce 'vertebrae_L1.nii.gz', 'vertebrae_L2.nii.gz', etc.
        # These are the "untempered" whole vertebrae masks.
        if verbose:
            logging.info(f"Segmenting labeled vertebrae (L1-L5) using TotalSegmentator... (fast={fast})")

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
            body_img = _load_nifti(body_mask_path)
            body_data = body_img.get_fdata() > 0
            affine = body_img.affine.copy()
            header = body_img.header.copy()
            del body_img

            for vertebra in LUMBAR_VERTEBRAE:
                vert_path = output_dir / f"{vertebra}.nii.gz"
                if vert_path.exists():
                    vert_img = _load_nifti(vert_path)
                    vert_data = vert_img.get_fdata() > 0
                    del vert_img

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

        # Generate trabecular core for L1
        generate_trabecular_core(output_dir, verbose)

        # Ensure file handles from nibabel are released before Windows unlinks in cleanup.
        gc.collect()
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
                mask_nifti = _load_nifti(mask_path)
                mask = mask_nifti.get_fdata()
                del mask_nifti
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



