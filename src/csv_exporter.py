"""
CSV Export Module

Converts statistics JSON to CSV format for easy analysis and DEXA matching.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Lumbar vertebrae labels
LUMBAR_VERTEBRAE = [
    'vertebrae_L1',
    'vertebrae_L2',
    'vertebrae_L3',
    'vertebrae_L4',
    'vertebrae_L5'
]


def export_patient_to_csv(
    patient_id: str,
    statistics_dict: Dict,
    output_path: Path
) -> None:
    """
    Convert single patient statistics JSON to CSV format.
    
    Args:
        patient_id: Patient identifier
        statistics_dict: Statistics dictionary (may include patient_id or not)
        output_path: Path to output CSV file
    """
    # Extract patient_id from dict if present, otherwise use provided
    pid = statistics_dict.get('patient_id', patient_id)
    
    # Prepare data for DataFrame
    rows = []
    for vertebra in LUMBAR_VERTEBRAE:
        if vertebra in statistics_dict:
            stats = statistics_dict[vertebra]
            rows.append({
                'patient_id': pid,
                'vertebra': vertebra,
                'volume': stats.get('volume', 0.0),
                'intensity': stats.get('intensity', 0.0)
            })
        else:
            rows.append({
                'patient_id': pid,
                'vertebra': vertebra,
                'volume': 0.0,
                'intensity': 0.0
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def export_batch_to_csv(
    all_patient_statistics: List[Dict],
    output_path: Path
) -> None:
    """
    Create consolidated CSV file for all patients.
    
    Args:
        all_patient_statistics: List of statistics dictionaries (one per patient)
        output_path: Path to output CSV file
    """
    all_rows = []
    
    for patient_stats in all_patient_statistics:
        # Extract patient_id
        patient_id = patient_stats.get('patient_id', 'UNKNOWN')
        
        # Add rows for each vertebra
        for vertebra in LUMBAR_VERTEBRAE:
            if vertebra in patient_stats:
                stats = patient_stats[vertebra]
                all_rows.append({
                    'patient_id': patient_id,
                    'vertebra': vertebra,
                    'volume': stats.get('volume', 0.0),
                    'intensity': stats.get('intensity', 0.0)
                })
            else:
                all_rows.append({
                    'patient_id': patient_id,
                    'vertebra': vertebra,
                    'volume': 0.0,
                    'intensity': 0.0
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(all_rows)
    
    # Sort by patient_id and vertebra for easier reading
    df = df.sort_values(['patient_id', 'vertebra'])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_statistics_from_json(json_path: Path) -> Dict:
    """
    Load statistics from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Statistics dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def export_json_to_csv(
    json_path: Path,
    csv_path: Path,
    patient_id: Optional[str] = None
) -> None:
    """
    Convert statistics JSON file to CSV file.
    
    Args:
        json_path: Path to input JSON file
        csv_path: Path to output CSV file
        patient_id: Optional patient ID (if not in JSON)
    """
    stats = load_statistics_from_json(json_path)
    
    # Use patient_id from JSON or provided
    pid = stats.get('patient_id', patient_id or 'UNKNOWN')
    
    export_patient_to_csv(pid, stats, csv_path)

