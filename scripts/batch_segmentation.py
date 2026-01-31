"""
Batch Segmentation Script

Runs the segmentation phase ONLY for all patient cases.
This script is designed to run overnight without requiring user interaction.

The companion script, `batch_verification.py`, is used to interactively verify
the segmentation results afterwards.

For best performance, set NUMEXPR_MAX_THREADS to limit CPU usage, e.g. in PowerShell:
$env:NUMEXPR_MAX_THREADS="20"

Usage:
    python scripts/batch_segmentation.py data/processed out/test_batch --device gpu
"""

print("Initializing Batch Segmentation Pipeline...", flush=True)

import argparse
import logging
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

# Set up logging early to capture imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Imports starting...")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.patient_manager import get_patient_metadata, create_patient_output_dir
from src.pipeline import find_patient_folders, process_single_patient

logger.info("All imports complete.")

# Lumbar vertebrae labels (for checking existing segmentations)
LUMBAR_VERTEBRAE = [
    'vertebrae_T11',
    'vertebrae_T12',
    'vertebrae_L1',
    'vertebrae_L2',
    'vertebrae_L3',
    'vertebrae_L4',
    'vertebrae_L5'
]


class ProgressWindow:
    """Tkinter window to show segmentation progress."""
    def __init__(self, total_cases: int, title: str = "Segmentation Progress"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x450")

        self.total = total_cases
        self.current = 0

        # Configure layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Progress info
        self.lbl_status = ttk.Label(main_frame, text="Preparing...", font=('Helvetica', 10))
        self.lbl_status.pack(anchor=tk.W, pady=(0, 5))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=total_cases)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Log area
        lbl_log = ttk.Label(main_frame, text="Log:", font=('Helvetica', 9, 'bold'))
        lbl_log.pack(anchor=tk.W)

        self.log_text = tk.Text(main_frame, height=15, width=70, state=tk.NORMAL)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scrollbar.set

        self.root.update()

    def update_progress(self, current: int, message: str):
        self.current = current
        self.progress_var.set(current)
        self.lbl_status.config(text=f"Processing {current}/{self.total} - {message}")
        self.log(f"[{current}/{self.total}] {message}")
        self.root.update()

    def log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def close(self):
        self.root.destroy()


def check_segmentations_exist(segmentation_dir: Path) -> bool:
    """
    Check if segmentation files already exist in the directory.

    Args:
        segmentation_dir: Directory to check

    Returns:
        True if segmentations exist, False otherwise
    """
    if not segmentation_dir.exists():
        return False

    # Check for vertebrae_body.nii.gz (from vertebrae_body task)
    vertebrae_body_path = segmentation_dir / "vertebrae_body.nii.gz"
    if vertebrae_body_path.exists():
        return True

    # Check for individual vertebrae masks (L1-L5) and bodies
    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            return True

        # Check for body mask
        body_path = segmentation_dir / f"{vertebra}_body.nii.gz"
        if body_path.exists():
            return True

    return False


def process_case_for_segmentation(
    dicom_folder: Path,
    output_base_dir: Path,
    temp_dir: Optional[Path] = None,
    fast_segmentation: bool = False,
    device: str = 'gpu',
    skip_if_exists: bool = True,
    forced_study_id: Optional[str] = None
) -> Tuple[Path, str, Optional[str], bool]:
    """
    Process a single case through the pipeline to get segmentation results.
    Skips processing if segmentations already exist (if skip_if_exists=True).

    Returns:
        Tuple of (segmentation_dir, patient_id, exam_date, was_skipped)
    """
    try:
        # Extract metadata first
        metadata = get_patient_metadata(dicom_folder)
        patient_id = metadata['patient_id'] or dicom_folder.name
        exam_date = metadata['study_date']

        # Get segmentation directory
        study_folder_name = forced_study_id if forced_study_id else dicom_folder.name
        patient_output_dir = create_patient_output_dir(output_base_dir, patient_id, study_id=study_folder_name)
        segmentation_dir = patient_output_dir / 'segmentations'

        # Check if segmentations already exist
        if skip_if_exists and check_segmentations_exist(segmentation_dir):
            logger.info(f"Segmentations already exist for {patient_id}. Skipping processing.")
            logger.info(f"Using existing segmentations from: {segmentation_dir}")
            return segmentation_dir, patient_id, exam_date, True

        # Process through pipeline
        logger.info(f"Processing {patient_id} through segmentation pipeline...")
        result = process_single_patient(
            dicom_folder=dicom_folder,
            output_base_dir=output_base_dir,
            temp_dir=temp_dir,
            fast_segmentation=fast_segmentation,
            device=device,
            keep_temp_files=True,  # Keep temp files for verification
            forced_study_id=study_folder_name
        )

        if result['status'] != 'success':
            raise Exception(f"Pipeline processing failed: {result.get('error', 'Unknown error')}")

        return segmentation_dir, patient_id, exam_date, False

    except Exception as e:
        logger.error(f"Error processing case {dicom_folder}: {e}")
        raise


def run_batch_segmentation(
    input_path: Path,
    output_base_dir: Path,
    fast_segmentation: bool = False,
    device: str = 'gpu'
):
    """
    Run the batch segmentation pipeline (Phase 1 only).

    1. Process ALL cases to generate segmentations (showing progress window).
    """
    import tempfile

    patient_folders = find_patient_folders(input_path)
    if not patient_folders:
        raise ValueError(f"No patient folders found in {input_path}")

    logger.info(f"Found {len(patient_folders)} patient folder(s) to segment")

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix='segmentation_temp_'))

    # Track study ID uniqueness
    study_counts: Dict[str, int] = {}

    # Launch Progress Window
    progress_window = ProgressWindow(len(patient_folders), title="Batch Segmentation Progress")

    logger.info("\n=== STARTING BATCH SEGMENTATION ===")

    successful_cases = 0
    failed_cases = 0

    for i, patient_folder in enumerate(patient_folders, 1):
        try:
            logger.info(f"Batch Processing {i}/{len(patient_folders)}: {patient_folder.name}")
            progress_window.update_progress(i, f"Processing {patient_folder.name}")

            # Pre-calculate unique study ID for this batch run
            try:
                temp_metadata = get_patient_metadata(patient_folder)
                temp_pid = temp_metadata.get('patient_id') or patient_folder.name
                study_name = patient_folder.name

                key = f"{temp_pid}_{study_name}"
                count = study_counts.get(key, 0)
                study_counts[key] = count + 1

                forced_study_id = study_name
                if count > 0:
                    forced_study_id = f"{study_name}_{count}"
                    msg = f"Duplicate study folder for {temp_pid}. Using suffix: {forced_study_id}"
                    logger.info(msg)
                    progress_window.log(msg)

            except Exception as e:
                logger.warning(f"Failed to pre-calculate suffix: {e}")
                forced_study_id = None

            logger.info(f"Calling process_case_for_segmentation for {patient_folder.name}...")
            # Process case
            segmentation_dir, patient_id, exam_date, was_skipped = process_case_for_segmentation(
                dicom_folder=patient_folder,
                output_base_dir=output_base_dir,
                temp_dir=temp_dir,
                fast_segmentation=fast_segmentation,
                device=device,
                skip_if_exists=True,
                forced_study_id=forced_study_id
            )
            logger.info(f"Returned from process_case_for_segmentation for {patient_id}. Skipped: {was_skipped}")

            success_msg = f"Case {patient_id} {'(Existing)' if was_skipped else '(New Segment)'}"
            progress_window.log(success_msg)
            successful_cases += 1

        except BaseException as e:
            # Catch BaseException to trap SystemExit and KeyboardInterrupt
            err_msg = f"CRITICAL ERROR processing {patient_folder.name}: {e} (Type: {type(e).__name__})"
            logger.error(err_msg, exc_info=True)
            progress_window.log("ERROR: " + err_msg)
            failed_cases += 1

            # Re-raise KeyboardInterrupt to allow user to abort
            if isinstance(e, KeyboardInterrupt):
                raise

    progress_window.log(f"\nSegmentation Complete. Success: {successful_cases}, Failed: {failed_cases}")
    progress_window.log("Closing progress window in 3 seconds...")
    progress_window.root.update()
    time.sleep(3)
    progress_window.close()

    logger.info("\n=== SEGMENTATION COMPLETE ===")
    logger.info(f"Successfully processed {successful_cases}/{len(patient_folders)} cases")
    logger.info(f"Failed: {failed_cases}")
    logger.info(f"Output saved to: {output_base_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch segmentation pipeline: Segment all patients (for overnight runs)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input_path', type=Path, help='Input directory containing patient DICOM folders')
    parser.add_argument('output_dir', type=Path, help='Output directory where segmentations will be saved')

    parser.add_argument('--fast', action='store_true', help='Use fast segmentation mode')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu', help='Device (default: gpu)')

    args = parser.parse_args()

    if not args.input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_path}")
        sys.exit(1)

    try:
        run_batch_segmentation(
            input_path=args.input_path,
            output_base_dir=args.output_dir,
            fast_segmentation=args.fast,
            device=args.device
        )
    except KeyboardInterrupt:
        logger.info("\nBatch segmentation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
