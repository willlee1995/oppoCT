"""
DICOM Processing Module

Converts DICOM series to NIfTI format and extracts patient identifiers.
"""

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
from pydicom.tag import Tag

from .patient_manager import get_patient_metadata, normalize_patient_id

# Series / study identifiers only — avoids parsing large sequences or unrelated groups.
_SERIES_HEADER_TAGS: Tuple[Tag, ...] = (
    Tag(0x0008, 0x0060),  # Modality
    Tag(0x0020, 0x000E),  # SeriesInstanceUID
    Tag(0x0008, 0x103E),  # SeriesDescription
    Tag(0x0020, 0x000D),  # StudyInstanceUID
)


def _read_dicom_series_header_tags(fp: Path):
    """
    Load only tags needed to group instances by series. Falls back to a full meta read
    if a selective read fails (some encoders dislike ``specific_tags``).
    """
    try:
        return pydicom.dcmread(
            str(fp),
            stop_before_pixels=True,
            specific_tags=list(_SERIES_HEADER_TAGS),
        )
    except Exception:
        try:
            return pydicom.dcmread(str(fp), stop_before_pixels=True)
        except Exception:
            return None


def iter_dicom_file_paths(dicom_folder: Path) -> List[Path]:
    """Return sorted unique paths to DICOM instances under ``dicom_folder`` (root + recursive)."""
    paths = list(iter_dicom_file_paths_streaming(dicom_folder))
    return sorted(set(paths))


def iter_dicom_file_paths_streaming(dicom_folder: Path) -> Iterator[Path]:
    """Yield paths to DICOM instances (``os.walk``); avoids building a full path list before reads."""
    if not dicom_folder.is_dir():
        return
    for root, _dirs, files in os.walk(dicom_folder):
        root_path = Path(root)
        for name in files:
            lower = name.lower()
            if lower.endswith(".dcm") or lower.endswith(".dicom"):
                yield root_path / name


def single_ct_series_fields_if_unique(dicom_folder: Path) -> Optional[Tuple[str, str]]:
    """
    If ``dicom_folder`` contains exactly one CT series (by SeriesInstanceUID), return
    ``(series_instance_uid, series_description)`` from a CT instance.

    Stops reading files as soon as two distinct CT series UIDs are seen (cannot autofill).
    """
    if not dicom_folder.is_dir():
        return None
    ct_uid_to_desc: Dict[str, str] = {}
    for fp in iter_dicom_file_paths_streaming(dicom_folder):
        ds = _read_dicom_series_header_tags(fp)
        if ds is None:
            continue
        mod = str(getattr(ds, "Modality", "") or "").strip().upper()
        if mod != "CT":
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if uid in ct_uid_to_desc:
            # Same series as already seen — no effect on single-CT autofill.
            continue
        desc = str(getattr(ds, "SeriesDescription", "") or "").strip()
        if len(ct_uid_to_desc) >= 1:
            return None
        ct_uid_to_desc[uid] = desc
    if len(ct_uid_to_desc) == 1:
        uid, desc = next(iter(ct_uid_to_desc.items()))
        return uid, desc
    return None


def enumerate_dicom_series(dicom_folder: Path) -> List[Dict[str, object]]:
    """
    Group DICOM instances under ``dicom_folder`` by SeriesInstanceUID.

    Returns one row per series with human-readable fields for UI / CSV.
    Files without SeriesInstanceUID are grouped under an empty UID string.
    """
    by_uid: Dict[str, Dict[str, object]] = {}
    for fp in iter_dicom_file_paths_streaming(dicom_folder):
        ds = _read_dicom_series_header_tags(fp)
        if ds is None:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if uid in by_uid:
            by_uid[uid]["num_instances"] = int(by_uid[uid]["num_instances"]) + 1
            continue
        by_uid[uid] = {
            "series_instance_uid": uid,
            "series_description": str(getattr(ds, "SeriesDescription", "") or "").strip(),
            "modality": str(getattr(ds, "Modality", "") or "").strip(),
            "num_instances": 1,
            "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "") or "").strip(),
        }

    rows: List[Dict[str, object]] = []
    for uid in sorted(by_uid.keys(), key=lambda x: (x == "", x)):
        rows.append(by_uid[uid])
    return rows


def load_dicom_series(
    dicom_folder: Path, series_instance_uid: Optional[str] = None
) -> Tuple[np.ndarray, dict]:
    """
    Load DICOM series from folder and convert to numpy array.
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        series_instance_uid: If set, only instances with this SeriesInstanceUID are loaded.
            None or empty string keeps the legacy behavior (all instances in the tree).
        
    Returns:
        Tuple of (image_array, metadata_dict)
        - image_array: 3D numpy array of CT image
        - metadata_dict: Dictionary with spacing, origin, direction, etc.
    """
    dicom_files = iter_dicom_file_paths(dicom_folder)

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

    filter_uid = (series_instance_uid or "").strip()
    if filter_uid:
        filtered = []
        for ds in slices:
            suid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
            if suid == filter_uid:
                filtered.append(ds)
        if not filtered:
            raise ValueError(
                f"No slices with SeriesInstanceUID={filter_uid!r} under {dicom_folder}"
            )
        slices = filtered

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


def dicom_to_nifti(
    dicom_folder: Path,
    output_path: Optional[Path] = None,
    series_instance_uid: Optional[str] = None,
) -> Tuple[nib.Nifti1Image, str]:
    """
    Convert DICOM series to NIfTI format.
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        output_path: Optional path to save NIfTI file. If None, returns image object only.
        series_instance_uid: Optional SeriesInstanceUID to restrict which instances are stacked.
        
    Returns:
        Tuple of (nifti_image, patient_id)
    """
    # Load DICOM series
    volume, metadata = load_dicom_series(dicom_folder, series_instance_uid=series_instance_uid)
    
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



