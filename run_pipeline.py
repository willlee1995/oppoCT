#!/usr/bin/env python
"""
CLI Entry Point for Lumbar Spine CT Segmentation Pipeline

Usage:
    python run_pipeline.py <input_path> <output_dir> [options]
"""

import argparse
import sys
from pathlib import Path
import logging

from src.pipeline import process_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Lumbar Spine CT Segmentation Pipeline - Batch process CT DICOM folders to segment lumbar vertebrae and calculate HU intensity statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single patient folder
  python run_pipeline.py /path/to/patient_dicom /path/to/output
  
  # Process batch of patients
  python run_pipeline.py /path/to/patients_directory /path/to/output
  
  # Use CPU instead of GPU
  python run_pipeline.py /path/to/input /path/to/output --device cpu
  
  # Use fast segmentation mode
  python run_pipeline.py /path/to/input /path/to/output --fast
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input DICOM folder (single patient) or directory containing patient folders (batch mode)'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to output directory where results will be saved'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast segmentation mode (lower quality, faster processing)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['gpu', 'cpu'],
        default='gpu',
        help='Device to use for segmentation (default: gpu)'
    )
    
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary NIfTI files after processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Validate output directory (will be created if doesn't exist)
    output_dir = Path(args.output_dir)
    
    try:
        # Run batch processing
        results = process_batch(
            input_path=input_path,
            output_dir=output_dir,
            fast_segmentation=args.fast,
            device=args.device,
            keep_temp_files=args.keep_temp
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total patients: {results['total_patients']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Output directory: {results['output_dir']}")
        print("="*60)
        
        # Print failed patients if any
        failed_patients = [r for r in results['results'] if r['status'] == 'failed']
        if failed_patients:
            print("\nFailed patients:")
            for failed in failed_patients:
                print(f"  - {failed['patient_id']}: {failed.get('error', 'Unknown error')}")
        
        # Exit with error code if any failures
        if results['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()



