# Lumbar Spine CT Segmentation Pipeline

A platform-agnostic Python pipeline for batch processing abdominal CT DICOM folders, segmenting lumbar vertebrae (L1-L5) using TotalSegmentator, and calculating Hounsfield Unit (HU) intensity statistics for correlation with DEXA BMD measurements.

## Features

- **Batch Processing**: Process multiple patients' CT scans in a single run
- **Patient Organization**: Automatically organizes output by patient ID extracted from DICOM metadata
- **Lumbar Vertebrae Segmentation**: Segments only L1-L5 vertebrae using TotalSegmentator
- **HU Intensity Statistics**: Calculates mean HU intensity and volume for each vertebra
- **Visualization**: Generates preview images showing segmented vertebrae overlaid on CT slices
- **Multiple Output Formats**: Exports results in both JSON and CSV formats
- **DEXA Matching**: CSV output includes patient_id column for easy correlation with DEXA data
- **Platform Agnostic**: Works on Windows, Linux, and macOS

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (install from [uv installation guide](https://github.com/astral-sh/uv#installation))
- CUDA-capable GPU (recommended) or CPU for TotalSegmentator
- At least 8GB RAM (16GB+ recommended)

### Install Dependencies

1. Install uv (if not already installed):
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  
.venv\Scripts\activate # Windows
uv pip install -r requirements.txt
```
Alternatively, use uv's project management (recommended):
```bash
uv sync  # Creates venv and installs dependencies automatically
```

3. TotalSegmentator will download its model weights automatically on first use.

## Usage

### Basic Usage

Process a single patient folder:
```bash
python run_pipeline.py /path/to/patient_dicom_folder /path/to/output_directory
```

Process a batch of patients (directory containing multiple patient folders):
```bash
python run_pipeline.py /path/to/patients_directory /path/to/output_directory
```

### Command Line Options

```
positional arguments:
  input_path            Path to input DICOM folder (single patient) or directory containing patient folders (batch mode)
  output_dir           Path to output directory where results will be saved

optional arguments:
  --fast               Use fast segmentation mode (lower quality, faster processing)
  --device {gpu,cpu}   Device to use for segmentation (default: gpu)
  --keep-temp          Keep temporary NIfTI files after processing
  --verbose            Enable verbose logging
  -h, --help           Show help message
```

### Examples

**Process single patient with GPU:**
```bash
python run_pipeline.py ./data/patient001 ./output
```

**Process batch with CPU (if GPU unavailable):**
```bash
python run_pipeline.py ./data/patients ./output --device cpu
```

**Process batch with fast mode:**
```bash
python run_pipeline.py ./data/patients ./output --fast
```

**Process with verbose logging:**
```bash
python run_pipeline.py ./data/patients ./output --verbose
```

## Input Format

The pipeline accepts DICOM folders in the following structures:

**Single Patient:**
```
patient_folder/
├── slice001.dcm
├── slice002.dcm
├── ...
└── sliceN.dcm
```

**Batch Processing:**
```
patients_directory/
├── patient001/
│   ├── slice001.dcm
│   └── ...
├── patient002/
│   ├── slice001.dcm
│   └── ...
└── ...
```

The pipeline automatically:
- Detects DICOM files (.dcm, .DCM extensions)
- Extracts patient ID from DICOM metadata (PatientID tag)
- Falls back to folder name if DICOM metadata unavailable
- Normalizes patient IDs for consistent matching

## Output Structure

For each patient, the pipeline creates:

```
output_directory/
├── PATIENT001/
│   ├── statistics.json          # Patient-specific statistics (JSON)
│   ├── statistics.csv            # Patient-specific statistics (CSV)
│   ├── PATIENT001_preview.png    # Segmentation preview image
│   └── segmentations/             # Segmentation masks
│       ├── vertebrae_L1.nii.gz
│       ├── vertebrae_L2.nii.gz
│       ├── vertebrae_L3.nii.gz
│       ├── vertebrae_L4.nii.gz
│       └── vertebrae_L5.nii.gz
├── PATIENT002/
│   └── ...
└── batch_statistics.csv           # Consolidated CSV for all patients
```

### Output Formats

**Per-Patient statistics.json:**
```json
{
    "patient_id": "PATIENT001",
    "vertebrae_L1": {
        "volume": 39447.0,
        "intensity": 341.75975
    },
    "vertebrae_L2": {
        "volume": 43821.0,
        "intensity": 390.55206
    },
    ...
}
```

**Consolidated batch_statistics.csv:**
```csv
patient_id,vertebra,volume,intensity
PATIENT001,vertebrae_L1,39447.0,341.75975
PATIENT001,vertebrae_L2,43821.0,390.55206
PATIENT002,vertebrae_L1,42000.0,350.25
PATIENT002,vertebrae_L2,45000.0,395.50
...
```

## DEXA Matching

The CSV output includes a `patient_id` column that can be used to match CT HU values with DEXA BMD measurements. Patient IDs are normalized (uppercase, spaces removed) for consistent matching.

**Example matching in Python:**
```python
import pandas as pd

# Load CT statistics
ct_stats = pd.read_csv('output/batch_statistics.csv')

# Load DEXA data
dexa_data = pd.read_csv('dexa_data.csv')

# Match on patient_id
merged = pd.merge(
    ct_stats[ct_stats['vertebra'] == 'vertebrae_L1'],
    dexa_data,
    on='patient_id',
    how='inner'
)

# Correlate HU intensity with BMD
correlation = merged['intensity'].corr(merged['BMD_L1'])
```

## Platform Compatibility

The pipeline is designed to be platform-agnostic and has been tested on:
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (10.15+)

All file paths use `pathlib.Path` for cross-platform compatibility.

## Troubleshooting

### Common Issues

**1. TotalSegmentator not found:**
```bash
uv pip install totalsegmentator
```

**1a. uv not found:**
- Install uv following the [official installation guide](https://github.com/astral-sh/uv#installation)
- Or use traditional pip: `pip install -r requirements.txt`

**2. CUDA/GPU errors - PyTorch not detecting GPU:**

If `torch.cuda.is_available()` returns `False` even though `nvidia-smi` works:

1. **Check your CUDA version:**
   ```bash
   nvcc --version  # If installed, shows CUDA version
   nvidia-smi      # Shows driver version and supported CUDA version
   ```

2. **Reinstall PyTorch with CUDA support:**
   ```bash
   # For CUDA 11.8 (most common)
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 12.4
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Verify GPU detection:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

4. **If still not working:**
   - Use `--device cpu` flag as a temporary workaround
   - Check that your GPU compute capability is supported (CUDA 11.8+ requires compute capability 3.5+)
   - Ensure NVIDIA drivers are up to date
   - Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific CUDA version

**3. DICOM files not found:**
- Ensure DICOM files have .dcm or .DCM extension
- Check that input path is correct
- Pipeline searches subdirectories recursively

**4. Memory errors:**
- Use `--fast` flag for lower memory usage
- Process fewer patients at once
- Ensure sufficient RAM available

**5. Patient ID extraction issues:**
- Check DICOM metadata contains PatientID tag
- Pipeline falls back to folder name if metadata unavailable
- Patient IDs are normalized automatically

**6. Shape resizing/resampling:**
- TotalSegmentator internally resamples CT images to a standard resolution (typically 1.5mm isotropic) for its neural network model
- The segmentation masks output by TotalSegmentator have the same shape as the resampled input, not the original CT
- **The pipeline preserves mask shapes as-is** - instead of resampling masks, it resamples the CT image to match the mask space
- This ensures masks remain unchanged while allowing accurate HU intensity statistics calculation
- Statistics are calculated in TotalSegmentator's resampled space, which is appropriate for the segmentation results
- If you see info messages about shape mismatches, this is normal - CT is resampled to match mask space, masks are never resampled

## Performance

- **GPU mode**: ~2-5 minutes per patient (depending on CT size)
- **CPU mode**: ~10-20 minutes per patient
- **Fast mode**: ~30-50% faster but lower segmentation quality

## Citation

If you use TotalSegmentator in your research, please cite:
```
Wasserthal, J., et al. "TotalSegmentator: robust segmentation of 104 anatomical structures in CT images." arXiv preprint arXiv:2208.05868 (2022).
```

## License

This pipeline is provided as-is for research purposes. Please refer to individual dependency licenses:
- TotalSegmentator: Apache 2.0
- pydicom: MIT
- nibabel: MIT
- Other dependencies: See respective licenses

## Support

For issues related to:
- **TotalSegmentator**: See [TotalSegmentator GitHub](https://github.com/wasserth/TotalSegmentator)
- **This pipeline**: Check logs for error messages, use `--verbose` flag for detailed output

## Version

Current version: 1.0.0



