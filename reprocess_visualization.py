
import logging
from pathlib import Path
import json
import shutil
import sys

# Add src to path
sys.path.append('src')

from src.dicom_processor import dicom_to_nifti
from src.csv_exporter import export_patient_to_csv
from src.visualizer import create_patient_preview
from src.statistics import calculate_patient_statistics

# Setup
dicom_path = Path("Plain5mmSTD")
output_base = Path("out/test_run_9/ANNOY")
segmentation_dir = output_base / "segmentations"
temp_nifti = Path("temp_reprocess.nii.gz")
patient_id = "ANNOY"

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Reprocessing Run 9 Output ---")
    
    # 1. Re-create NIfTI (using the fixed affine logic)
    print("1. Converting DICOM to NIfTI...")
    dicom_to_nifti(dicom_path, temp_nifti)
    
    # 2. Update CSV with vertebrae_body
    print("2. Updating Statistics CSV...")
    # Load existing JSON
    json_path = output_base / "statistics.json"
    with open(json_path, 'r') as f:
        stats = json.load(f)
    
    # Export to CSV (using updated list in csv_exporter.py)
    csv_path = output_base / "statistics.csv"
    export_patient_to_csv(patient_id, stats, csv_path)
    print(f"   CSV saved to {csv_path}")

    # 3. Update Preview with vertebrae_body color
    print("3. Updating Preview Image...")
    preview_path = create_patient_preview(
        patient_id,
        temp_nifti,
        segmentation_dir,
        output_base
    )
    print(f"   Preview saved to {preview_path}")

    # Cleanup
    if temp_nifti.exists():
        temp_nifti.unlink()
    
    print("--- Done ---")

if __name__ == "__main__":
    main()
