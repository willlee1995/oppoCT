"""
DICOM Processing Module

Converts DICOM series to NIfTI format and extracts patient identifiers.
"""

from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom

from .patient_manager import get_patient_metadata, normalize_patient_id


def load_dicom_series(dicom_folder: Path) -> Tuple[np.ndarray, dict]:
    """
    Load DICOM series from folder and convert to numpy array.
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        
    Returns:
        Tuple of (image_array, metadata_dict)
        - image_array: 3D numpy array of CT image
        - metadata_dict: Dictionary with spacing, origin, direction, etc.
    """
    # Find all DICOM files
    dicom_files = []
    
    # Check root directory
    dicom_files.extend(dicom_folder.glob('*.dcm'))
    dicom_files.extend(dicom_folder.glob('*.DCM'))
    
    # Check subdirectories recursively
    for subdir in dicom_folder.rglob('*'):
        if subdir.is_file():
            if subdir.suffix.lower() in ['.dcm', '.dicom']:
                dicom_files.append(subdir)
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    # Read DICOM files and sort by slice location
    slices = []
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dicom_file))
            slices.append(ds)
        except Exception as e:
            print(f"Warning: Could not read {dicom_file}: {e}")
            continue
    
    if not slices:
        raise ValueError(f"No valid DICOM files found in {dicom_folder}")
    
    # Sort slices by ImagePositionPatient[2] (z-coordinate) or SliceLocation
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') and x.ImagePositionPatient else 
                   (float(x.SliceLocation) if hasattr(x, 'SliceLocation') and x.SliceLocation else 0))
    except:
        # Fallback: sort by filename
        slices.sort(key=lambda x: x.filename)
    
    # Get pixel data and spacing
    pixel_spacing = slices[0].PixelSpacing if hasattr(slices[0], 'PixelSpacing') else [1.0, 1.0]
    slice_thickness = slices[0].SliceThickness if hasattr(slices[0], 'SliceThickness') else 1.0
    
    # Get image orientation and position
    image_orientation = None
    image_position = None
    if hasattr(slices[0], 'ImageOrientationPatient'):
        image_orientation = slices[0].ImageOrientationPatient
    if hasattr(slices[0], 'ImagePositionPatient'):
        image_position = slices[0].ImagePositionPatient
    
    # Get image dimensions from first slice
    first_slice = slices[0]
    rows = first_slice.Rows
    cols = first_slice.Columns
    num_slices = len(slices)
    
    # Initialize volume with correct shape: (Columns, Rows, Slices) -> (X, Y, Z)
    # This matches NIfTI standard and standard affine construction
    volume = np.zeros((cols, rows, num_slices), dtype=np.float32)
    
    # Load slices into volume
    for i, slice_ds in enumerate(slices):
        pixel_array = slice_ds.pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept if present
        if hasattr(slice_ds, 'RescaleSlope') and hasattr(slice_ds, 'RescaleIntercept'):
            pixel_array = pixel_array * slice_ds.RescaleSlope + slice_ds.RescaleIntercept
        
        # Assign to volume: (cols, rows, slice_index) - Transpose needed because DICOM is (Row, Col)
        volume[:, :, i] = pixel_array.T
    
    # Create metadata dictionary
    metadata = {
        'spacing': [float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1])],
        'origin': list(image_position) if image_position else [0.0, 0.0, 0.0],
        'direction': list(image_orientation) + [0.0, 0.0, 1.0] if image_orientation else None,
        'affine': None  # Will be set when creating NIfTI
    }
    
    return volume, metadata


def dicom_to_nifti(dicom_folder: Path, output_path: Optional[Path] = None) -> Tuple[nib.Nifti1Image, str]:
    """
    Convert DICOM series to NIfTI format.
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        output_path: Optional path to save NIfTI file. If None, returns image object only.
        
    Returns:
        Tuple of (nifti_image, patient_id)
    """
    # Load DICOM series
    volume, metadata = load_dicom_series(dicom_folder)
    
    # Extract patient ID
    patient_metadata = get_patient_metadata(dicom_folder)
    patient_id = normalize_patient_id(patient_metadata['patient_id'] or dicom_folder.name)
    
    # Create affine matrix
    spacing = metadata['spacing']
    origin = metadata['origin']
    direction = metadata['direction']
    
    affine = np.eye(4)
    
    if direction and len(direction) >= 6:
        # Direction cosines from ImageOrientationPatient
        # IOP: [Rx, Ry, Rz, Cx, Cy, Cz]
        # X-axis (Cols) corresponds to first triplet
        # Y-axis (Rows) corresponds to second triplet
        
        # Volume has shape (Cols, Rows, Slices) because we used pixel_array.T
        # spacing: [z, y, x] (from above metadata creation)
        # spacing[2] is X spacing (cols)
        # spacing[1] is Y spacing (rows)
        # spacing[0] is Z spacing (slices)
        
        rx, ry, rz = direction[0], direction[1], direction[2]
        cx, cy, cz = direction[3], direction[4], direction[5]
        
        # Calculate Z direction (slice normal) using cross product
        # Cross product of X and Y gives Z
        # Note: DICOM uses LCS (Left Handed)? No, RCS usually.
        # But let's stick to simple vector math.
        # r = vector along row (increasing column index) -> My volume dim 0 (Cols)
        # c = vector along column (increasing row index) -> My volume dim 1 (Rows)
        
        norm_r = np.array([rx, ry, rz])
        norm_c = np.array([cx, cy, cz])
        norm_s = np.cross(norm_r, norm_c)
        
        # Set affine rotation part (scaled by spacing)
        # Column 0: X axis (Cols) -> norm_r * spacing_x
        affine[0:3, 0] = norm_r * spacing[2]
        
        # Column 1: Y axis (Rows) -> norm_c * spacing_y
        affine[0:3, 1] = norm_c * spacing[1]
        
        # Column 2: Z axis (Slices) -> norm_s * spacing_z
        # Note: This assumes equal slice spacing and no gantry tilt issues for Z
        affine[0:3, 2] = norm_s * spacing[0]
        
        # Set affine translation (Origin) - ImagePositionPatient
        affine[0:3, 3] = origin
        
    else:
        # Fallback to simple identity scaling
        affine[0, 0] = spacing[2]  # x spacing
        affine[1, 1] = spacing[1]  # y spacing
        affine[2, 2] = spacing[0]  # z spacing

    # Set translation (Origin)
    affine[0, 3] = origin[0] if len(origin) > 0 else 0
    affine[1, 3] = origin[1] if len(origin) > 1 else 0
    affine[2, 3] = origin[2] if len(origin) > 2 else 0
    
    # Convert from DICOM LPS to NIfTI RAS coordinate system
    # Negate X and Y rows (0 and 1) to flip Left->Right and Posterior->Anterior
    # This transforms the affine matrix to map current voxel indices to RAS space
    affine[0, :] = -affine[0, :]
    affine[1, :] = -affine[1, :]
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Ensure canonical orientation (RAS+) limits orientation issues with TotalSegmentator
    nifti_img = nib.as_closest_canonical(nifti_img)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nifti_img, str(output_path))
    
    return nifti_img, patient_id


def extract_patient_id(dicom_folder: Path) -> str:
    """
    Extract patient ID from DICOM folder.
    
    Args:
        dicom_folder: Path to DICOM folder
        
    Returns:
        Normalized patient ID
    """
    metadata = get_patient_metadata(dicom_folder)
    patient_id = metadata['patient_id'] or dicom_folder.name
    return normalize_patient_id(patient_id)



