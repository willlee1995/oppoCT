"""
Main Pipeline Module

Orchestrates batch processing of multiple patients through the entire workflow:
DICOM -> NIfTI -> Segmentation -> Statistics -> Visualization -> CSV Export
"""

import logging
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .csv_exporter import export_batch_to_csv, export_patient_to_csv
from .dicom_processor import (
    dicom_to_nifti,
    extract_patient_id,
    get_study_instance_uid_for_grouping,
    iter_dicom_file_paths_streaming,
)
from .patient_manager import create_patient_output_dir
from .segmentator import segment_lumbar_vertebrae, verify_segmentation_output
from .statistics import calculate_patient_statistics, save_statistics_json
from .visualizer import create_patient_preview

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
    keep_temp_files: bool = False,
    forced_study_id: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
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
        forced_study_id: Optional forced study ID (e.g., with suffix) to use for output folder
        series_instance_uid: If set, only this DICOM series is converted and passed to segmentation
            (TotalSegmentator runs on the derived NIfTI for reproducible alignment).
        
    Returns:
        Dictionary with patient_id and processing status/results
    """
    patient_id = None
    start_time = time.perf_counter()
    result = {
        'patient_id': None,
        'status': 'failed',
        'error': None,
        'duration_seconds': None
    }
    
    try:
        series_filter = (series_instance_uid or "").strip() or None

        # Step 1: Extract patient ID
        logging.info(f"Processing patient from {dicom_folder}")
        patient_id = extract_patient_id(dicom_folder)
        result['patient_id'] = patient_id
        logging.info(f"Patient ID: {patient_id}")
        
        # Step 2: Create patient output directory
        # Use provided forced_study_id or fallback to folder name
        study_folder_name = forced_study_id if forced_study_id else dicom_folder.name
        patient_output_dir = create_patient_output_dir(output_base_dir, patient_id, study_id=study_folder_name)
        segmentations_dir = patient_output_dir / 'segmentations'
        
        # Step 3: Convert DICOM to NIfTI
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        
        temp_nifti_path = temp_dir / f"{patient_id}_temp.nii.gz"
        logging.info(f"Converting DICOM to NIfTI: {temp_nifti_path}")
        
        nifti_img, extracted_pid = dicom_to_nifti(
            dicom_folder, temp_nifti_path, series_instance_uid=series_filter
        )
        if extracted_pid != patient_id:
            logging.warning(f"Patient ID mismatch: extracted {extracted_pid}, using {patient_id}")
        
        # Get voxel spacing for volume calculation
        voxel_spacing = None
        if hasattr(nifti_img, 'header'):
            spacing = nifti_img.header.get_zooms()
            if len(spacing) >= 3:
                voxel_spacing = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        
        # Step 4: Segment lumbar vertebrae
        # We process directly from DICOM to ensure unmodified TotalSegmentator results.
        # The NIfTI conversion above is kept for statistics and visualization reference,
        # and we've ensured it uses canonical orientation to match TotalSegmentator's output.
        logging.info("Segmenting lumbar vertebrae...")
        # When a series UID is fixed, segment from the converted NIfTI so the stack matches QC.
        use_dicom_directly = series_filter is None
        segment_lumbar_vertebrae(
            nifti_path=temp_nifti_path,
            output_dir=segmentations_dir,
            fast=fast_segmentation,
            device=device,
            verbose=True,
            use_dicom_directly=use_dicom_directly,
            dicom_dir=dicom_folder if use_dicom_directly else None,
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
    finally:
        duration = time.perf_counter() - start_time
        result['duration_seconds'] = duration
        logging.info(
            "Total runtime for %s: %.2f seconds",
            result.get('patient_id') or 'UNKNOWN',
            duration
        )
    
    return result


def _find_patient_folders_legacy_leaf_dirs(input_path: Path) -> List[Path]:
    """One folder per directory that directly contains a .dcm file (old behaviour)."""
    patient_folders: List[Path] = []
    if input_path.is_file():
        patient_folders.append(input_path.parent)
    elif input_path.is_dir():
        for root, _, files in os.walk(input_path):
            if any(f.lower().endswith((".dcm", ".dicom")) for f in files):
                patient_folders.append(Path(root))
    return sorted(set(patient_folders))


def _common_root_for_dicom_paths(paths: List[Path]) -> Path:
    """Smallest directory tree root that contains all given DICOM file paths."""
    if not paths:
        raise ValueError("paths must be non-empty")
    if len(paths) == 1:
        return paths[0].parent
    try:
        common = os.path.commonpath([str(p.resolve()) for p in paths])
    except ValueError:
        return paths[0].parent
    p = Path(common)
    if p.is_file():
        return p.parent
    return p


def find_patient_folders(input_path: Path) -> List[Path]:
    """
    Find DICOM case roots under ``input_path``.

    Files are grouped by **StudyInstanceUID** (from a light header read). Each group
    becomes one case folder: the common ancestor directory of all instances in that
    study. This avoids treating every series subfolder as a separate study when
    series live under one study tree.

    Falls back to the legacy rule (one row per directory that directly contains
    ``.dcm`` files) if no readable DICOM headers are found.

    Args:
        input_path: Input path (file or directory)

    Returns:
        Sorted list of folder paths to pass as ``dicom_folder`` for each case.
    """
    if input_path.is_file():
        return sorted({input_path.parent})
    if not input_path.is_dir():
        return []

    by_study: Dict[str, List[Path]] = defaultdict(list)
    for fp in iter_dicom_file_paths_streaming(input_path):
        suid = get_study_instance_uid_for_grouping(fp)
        key = suid if suid else f"__NO_STUDY_UID__:{fp.parent.resolve()}"
        by_study[key].append(fp)

    if not by_study:
        return _find_patient_folders_legacy_leaf_dirs(input_path)

    roots: List[Path] = []
    for paths in by_study.values():
        try:
            roots.append(_common_root_for_dicom_paths(paths))
        except ValueError:
            continue

    if not roots:
        return _find_patient_folders_legacy_leaf_dirs(input_path)

    out = sorted(set(roots))
    logging.info(
        "find_patient_folders: grouped DICOM under %s into %d case root(s) by StudyInstanceUID",
        input_path,
        len(out),
    )
    return out


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
    
    # Track study folder usage to handle duplicated folder names
    # Key: (patient_id, study_folder_name), Value: count
    study_counts: Dict[str, int] = {}
    
    for i, patient_folder in enumerate(patient_folders, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing patient {i}/{len(patient_folders)}: {patient_folder.name}")
        logging.info(f"{'='*60}")
        
        # Pre-calculate unique study ID for this batch run
        try:
            # We need patient_id before processing to track duplicates
            # This mimics what process_single_patient does
            temp_pid = extract_patient_id(patient_folder)
            study_name = patient_folder.name
            
            # Construct key for tracking
            key = f"{temp_pid}_{study_name}"
            count = study_counts.get(key, 0)
            study_counts[key] = count + 1
            
            # Generate suffix if needed (first one is clean, subsequent get _1, _2)
            forced_study_id = study_name
            if count > 0:
                forced_study_id = f"{study_name}_{count}"
                logging.info(f"Duplicate study folder detected for {temp_pid}/{study_name}. Using suffix: {forced_study_id}")
                
        except Exception:
            # If extraction fails here, let process_single_patient handle errors
            forced_study_id = None
        
        result = process_single_patient(
            dicom_folder=patient_folder,
            output_base_dir=output_dir,
            temp_dir=temp_dir,
            fast_segmentation=fast_segmentation,
            device=device,
            keep_temp_files=keep_temp_files,
            forced_study_id=forced_study_id
        )
        
        all_results.append(result)
        duration = result.get('duration_seconds')
        if duration is not None:
            minutes, seconds = divmod(duration, 60)
            logging.info(
                "Patient %s runtime: %dm %.1fs (%.2fs total)",
                result.get('patient_id') or 'UNKNOWN',
                int(minutes),
                seconds,
                duration
            )
        
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
    logging.info("Batch processing complete!")
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



