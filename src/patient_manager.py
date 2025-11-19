"""
Patient Management Module

Handles patient identification, metadata extraction, and output directory organization
for batch processing of multiple patients.
"""

import re
from pathlib import Path
from typing import Dict, Optional

import pydicom


def normalize_patient_id(patient_id: str) -> str:
    """
    Normalize patient ID for consistent matching with DEXA data.
    
    Args:
        patient_id: Raw patient ID from DICOM
        
    Returns:
        Normalized patient ID (uppercase, spaces removed)
    """
    if not patient_id:
        return "UNKNOWN"
    
    # Remove spaces and convert to uppercase
    normalized = re.sub(r'\s+', '', str(patient_id).upper())
    
    # Remove special characters except alphanumeric and underscores
    normalized = re.sub(r'[^A-Z0-9_]', '', normalized)
    
    return normalized if normalized else "UNKNOWN"


def get_patient_metadata(dicom_folder: Path) -> Dict[str, Optional[str]]:
    """
    Extract patient metadata from DICOM files in a folder.
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        
    Returns:
        Dictionary with patient metadata:
        - patient_id: Patient ID
        - patient_name: Patient name
        - accession_number: Accession number
        - study_date: Study date
    """
    metadata = {
        'patient_id': None,
        'patient_name': None,
        'accession_number': None,
        'study_date': None
    }
    
    # Find first DICOM file in folder
    dicom_files = list(dicom_folder.glob('*.dcm')) + list(dicom_folder.glob('*.DCM'))
    
    # Also check subdirectories
    if not dicom_files:
        for subdir in dicom_folder.iterdir():
            if subdir.is_dir():
                dicom_files.extend(subdir.glob('*.dcm'))
                dicom_files.extend(subdir.glob('*.DCM'))
                if dicom_files:
                    break
    
    if not dicom_files:
        return metadata
    
    try:
        # Read first DICOM file to get metadata
        ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
        
        # Extract patient ID
        if hasattr(ds, 'PatientID') and ds.PatientID:
            metadata['patient_id'] = str(ds.PatientID).strip()
        
        # Extract patient name
        if hasattr(ds, 'PatientName'):
            patient_name = ds.PatientName
            if patient_name:
                if hasattr(patient_name, 'family_name'):
                    # PersonName object
                    name_parts = []
                    if patient_name.family_name:
                        name_parts.append(str(patient_name.family_name))
                    if patient_name.given_name:
                        name_parts.append(str(patient_name.given_name))
                    metadata['patient_name'] = ' '.join(name_parts) if name_parts else None
                else:
                    # String
                    metadata['patient_name'] = str(patient_name).strip()
        
        # Extract accession number
        if hasattr(ds, 'AccessionNumber') and ds.AccessionNumber:
            metadata['accession_number'] = str(ds.AccessionNumber).strip()
        
        # Extract study date
        if hasattr(ds, 'StudyDate') and ds.StudyDate:
            metadata['study_date'] = str(ds.StudyDate).strip()
            
    except Exception as e:
        print(f"Warning: Could not read DICOM metadata from {dicom_files[0]}: {e}")
    
    return metadata


def create_patient_output_dir(base_output_dir: Path, patient_id: str) -> Path:
    """
    Create patient-specific output directory structure.
    
    Args:
        base_output_dir: Base output directory
        patient_id: Patient identifier
        
    Returns:
        Path to patient-specific output directory
    """
    # Normalize patient ID for directory name
    normalized_id = normalize_patient_id(patient_id)
    
    # Create patient directory
    patient_dir = base_output_dir / normalized_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Create segmentations subdirectory
    segmentations_dir = patient_dir / 'segmentations'
    segmentations_dir.mkdir(exist_ok=True)
    
    return patient_dir


def extract_patient_id_from_folder(dicom_folder: Path) -> str:
    """
    Extract patient ID from DICOM folder.
    Falls back to folder name if DICOM metadata is unavailable.
    
    Args:
        dicom_folder: Path to DICOM folder
        
    Returns:
        Patient ID string
    """
    metadata = get_patient_metadata(dicom_folder)
    
    # Prefer PatientID from DICOM
    if metadata['patient_id']:
        return normalize_patient_id(metadata['patient_id'])
    
    # Fall back to folder name
    folder_name = dicom_folder.name
    return normalize_patient_id(folder_name)



