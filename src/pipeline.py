"""
Main Pipeline Module

Orchestrates batch processing of multiple patients through the entire workflow:
DICOM -> NIfTI -> Segmentation -> Statistics -> Visualization -> CSV Export
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import shutil

from .dicom_processor import dicom_to_nifti, extract_patient_id
from .patient_manager import create_patient_output_dir, get_patient_metadata
from .segmentator import segment_lumbar_vertebrae, verify_segmentation_output
from .statistics import calculate_patient_statistics, save_statistics_json
from .visualizer import create_patient_preview
from .csv_exporter import export_patient_to_csv, export_batch_to_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def process_single_patient(
    dicom_folder: Path,
    output_base_dir: Path,
    temp_dir: Optional[Path] = None,
    fast_segmentation: bool = False,
    device: str = 'gpu',
    keep_temp_files: bool = False
) -> Dict:
    """
    Process a single patient through the entire pipeline.
    
    Args:
        dicom_folder: Path to patient's DICOM folder
        output_base_dir: Base output directory
        temp_dir: Temporary directory for intermediate files
        fast_segmentation: Use fast segmentation mode
        device: Device for segmentation ('gpu' or 'cpu')
        keep_temp_files: Keep temporary NIfTI files
        
    Returns:
        Dictionary with patient_id and processing status/results
    """
    patient_id = None
    result = {
        'patient_id': None,
        'status': 'failed',
        'error': None
    }
    
    try:
        # Step 1: Extract patient ID
        logging.info(f"Processing patient from {dicom_folder}")
        patient_id = extract_patient_id(dicom_folder)
        result['patient_id'] = patient_id
        logging.info(f"Patient ID: {patient_id}")
        
        # Step 2: Create patient output directory
        patient_output_dir = create_patient_output_dir(output_base_dir, patient_id)
        segmentations_dir = patient_output_dir / 'segmentations'
        
        # Step 3: Convert DICOM to NIfTI
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        
        temp_nifti_path = temp_dir / f"{patient_id}_temp.nii.gz"
        logging.info(f"Converting DICOM to NIfTI: {temp_nifti_path}")
        
        nifti_img, extracted_pid = dicom_to_nifti(dicom_folder, temp_nifti_path)
        if extracted_pid != patient_id:
            logging.warning(f"Patient ID mismatch: extracted {extracted_pid}, using {patient_id}")
        
        # Get voxel spacing for volume calculation
        voxel_spacing = None
        if hasattr(nifti_img, 'header'):
            spacing = nifti_img.header.get_zooms()
            if len(spacing) >= 3:
                voxel_spacing = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        
        # Step 4: Segment lumbar vertebrae
        logging.info(f"Segmenting lumbar vertebrae...")
        segment_lumbar_vertebrae(
            nifti_path=temp_nifti_path,
            output_dir=segmentations_dir,
            fast=fast_segmentation,
            device=device,
            verbose=True
        )
        
        # Verify segmentation output
        found_vertebrae = verify_segmentation_output(segmentations_dir)
        logging.info(f"Found {len(found_vertebrae)} vertebrae: {found_vertebrae}")
        
        # Step 5: Calculate statistics
        logging.info("Calculating statistics...")
        statistics = calculate_patient_statistics(
            patient_id=patient_id,
            ct_image_path=temp_nifti_path,
            segmentation_dir=segmentations_dir,
            voxel_spacing=voxel_spacing
        )
        
        # Save statistics JSON
        stats_json_path = patient_output_dir / 'statistics.json'
        save_statistics_json(statistics, stats_json_path)
        logging.info(f"Statistics saved to {stats_json_path}")
        
        # Step 6: Generate preview image
        logging.info("Generating preview image...")
        preview_path = create_patient_preview(
            patient_id=patient_id,
            ct_image_path=temp_nifti_path,
            segmentation_dir=segmentations_dir,
            output_dir=patient_output_dir
        )
        logging.info(f"Preview saved to {preview_path}")
        
        # Step 7: Export to CSV
        logging.info("Exporting to CSV...")
        csv_path = patient_output_dir / 'statistics.csv'
        export_patient_to_csv(patient_id, statistics, csv_path)
        logging.info(f"CSV saved to {csv_path}")
        
        # Cleanup temporary files
        if not keep_temp_files and temp_nifti_path.exists():
            temp_nifti_path.unlink()
        
        result['status'] = 'success'
        result['output_dir'] = str(patient_output_dir)
        result['statistics'] = statistics
        
        logging.info(f"Successfully processed patient {patient_id}")
        
    except Exception as e:
        error_msg = f"Error processing patient {patient_id or 'UNKNOWN'}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        result['error'] = error_msg
        result['status'] = 'failed'
    
    return result


def find_patient_folders(input_path: Path) -> List[Path]:
    """
    Find patient DICOM folders in input directory.
    
    Args:
        input_path: Input path (file or directory)
        
    Returns:
        List of patient folder paths
    """
    patient_folders = []
    
    if input_path.is_file():
        # Single file - use parent directory
        patient_folders.append(input_path.parent)
    elif input_path.is_dir():
        # Check if directory contains DICOM files directly
        dicom_files = list(input_path.glob('*.dcm')) + list(input_path.glob('*.DCM'))
        
        if dicom_files:
            # Directory contains DICOM files - treat as single patient
            patient_folders.append(input_path)
        else:
            # Directory contains subdirectories - treat each as a patient
            for subdir in input_path.iterdir():
                if subdir.is_dir():
                    # Check if subdirectory contains DICOM files
                    subdir_dicom = list(subdir.glob('*.dcm')) + list(subdir.glob('*.DCM'))
                    if subdir_dicom:
                        patient_folders.append(subdir)
    
    return patient_folders


def process_batch(
    input_path: Path,
    output_dir: Path,
    fast_segmentation: bool = False,
    device: str = 'gpu',
    keep_temp_files: bool = False
) -> Dict:
    """
    Process batch of patients.
    
    Args:
        input_path: Path to input directory containing patient folders, or single patient folder
        output_dir: Base output directory
        fast_segmentation: Use fast segmentation mode
        device: Device for segmentation ('gpu' or 'cpu')
        keep_temp_files: Keep temporary NIfTI files
        
    Returns:
        Dictionary with batch processing results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find patient folders
    patient_folders = find_patient_folders(input_path)
    
    if not patient_folders:
        raise ValueError(f"No patient folders found in {input_path}")
    
    logging.info(f"Found {len(patient_folders)} patient folder(s) to process")
    
    # Create temporary directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix='lumbar_spine_pipeline_'))
    
    # Process each patient
    all_results = []
    all_statistics = []
    
    for i, patient_folder in enumerate(patient_folders, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing patient {i}/{len(patient_folders)}: {patient_folder.name}")
        logging.info(f"{'='*60}")
        
        result = process_single_patient(
            dicom_folder=patient_folder,
            output_base_dir=output_dir,
            temp_dir=temp_dir,
            fast_segmentation=fast_segmentation,
            device=device,
            keep_temp_files=keep_temp_files
        )
        
        all_results.append(result)
        
        if result['status'] == 'success' and 'statistics' in result:
            all_statistics.append(result['statistics'])
    
    # Generate consolidated CSV
    if all_statistics:
        logging.info("\nGenerating consolidated batch CSV...")
        batch_csv_path = output_dir / 'batch_statistics.csv'
        export_batch_to_csv(all_statistics, batch_csv_path)
        logging.info(f"Batch CSV saved to {batch_csv_path}")
    
    # Cleanup temporary directory
    if not keep_temp_files and temp_dir.exists():
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")
    
    # Summary
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = len(all_results) - successful
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Batch processing complete!")
    logging.info(f"Successfully processed: {successful}/{len(all_results)}")
    logging.info(f"Failed: {failed}/{len(all_results)}")
    logging.info(f"{'='*60}")
    
    return {
        'total_patients': len(all_results),
        'successful': successful,
        'failed': failed,
        'results': all_results,
        'output_dir': str(output_dir)
    }



