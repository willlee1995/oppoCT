# Segmentation Verification Guide

This guide explains how to verify that TotalSegmentator correctly identified vertebrae (L1-L5) in your CT scans.

## Quick Start

### View a Single Vertebra (Interactive)

```bash
python verify_segmentation.py --vertebra L1
```

This will open an **interactive viewer** with:
- A slider to navigate through CT slices
- The segmentation mask overlaid on the CT in red (for L1)
- Labels showing which vertebra is present in each slice

### View All Lumbar Vertebrae

```bash
python verify_segmentation.py --vertebra all
```

This shows all 5 lumbar vertebrae (L1-L5) with different colors:
- **L1**: Red
- **L2**: Orange  
- **L3**: Yellow
- **L4**: Green
- **L5**: Blue

### Generate Static Verification Images

If you want to save images instead of viewing interactively:

```bash
python verify_segmentation.py --vertebra L1 --mode static
```

This creates a PNG file in `verification_output/` with multiple representative slices.

### Both Interactive + Static

```bash
python verify_segmentation.py --vertebra all --mode both
```

## Advanced Options

### Custom Window Settings

For bone visualization:
```bash
python verify_segmentation.py --vertebra L1 --window-level 400 --window-width 1800
```

For soft tissue:
```bash
python verify_segmentation.py --vertebra L1 --window-level 40 --window-width 400
```

### Custom Paths

```bash
python verify_segmentation.py \
    --vertebra L1 \
    --ct-dir "path/to/dicom/files" \
    --seg-dir "path/to/segmentations" \
    --output "path/to/output"
```

## How to Verify Correctness

When viewing the overlay:

1. **Check Position**: 
   - L1 is the first lumbar vertebra below the ribcage
   - L5 is the last lumbar vertebra above the sacrum
   - Count from top to bottom: L1, L2, L3, L4, L5

2. **Check Coverage**:
   - The colored overlay should cover the entire vertebral body
   - Should include both the body and posterior elements (if segmented)

3. **Check Continuity**:
   - Use the slider to scroll through slices
   - The mask should appear continuous through multiple slices
   - No gaps or unexpected jumps

4. **Check Boundaries**:
   - The contour (colored line) should align with the vertebra edges
   - Should not extend into surrounding structures

## Common Issues

### ✅ Issue: Mask doesn't align with CT (FIXED)
**Status**: This issue has been fixed! The script now automatically detects and corrects orientation mismatches between the CT and segmentation masks.

If you still see alignment issues:
- The mask and CT may have different resolutions (this is rare but possible)
- Check the log output for warnings about shape mismatches

### Issue: No mask visible
**Possible causes**:
- The mask file doesn't exist (check `segmentations/` directory)
- You're on a slice that doesn't contain the vertebra (use the slider to navigate)
- The segmentation failed (check TotalSegmentator logs)

### Issue: Wrong vertebra labeled
This is what you're trying to verify! If L1 appears to be labeled as L2 or another vertebra:
1. Check the anatomy carefully (count from ribs downward)
2. Verify with medical expertise
3. Consider re-running TotalSegmentator or using manual correction

## Tips

1. **Start with "all" mode**: `--vertebra all` lets you see all vertebrae at once, making it easier to verify the overall labeling

2. **Use keyboard shortcuts**: In interactive mode, you can click and drag the slider, or click on the slider bar to jump to a slice

3. **Save verification images**: Use `--mode both` to get both interactive viewing and saved images for documentation

4. **Adjust windowing**: If you can't see the vertebrae clearly, adjust `--window-level` and `--window-width` values

## Example Workflow

```bash
# Step 1: Quick check of all lumbar vertebrae
python verify_segmentation.py --vertebra all

# Step 2: If something looks wrong with L1, examine it in detail
python verify_segmentation.py --vertebra L1

# Step 3: Generate documentation images
python verify_segmentation.py --vertebra all --mode static
```

## Integration with Pipeline

If you're running the full pipeline, the verification can be added as a step:

```python
from verify_segmentation import create_static_verification
import numpy as np

# After segmentation...
create_static_verification(
    ct_volume=ct_data,
    masks={"vertebrae_L1": l1_mask},
    output_path=Path("verification_output/patient_123_L1.png")
)
```

