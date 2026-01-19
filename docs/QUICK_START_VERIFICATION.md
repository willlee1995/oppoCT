# Quick Start: Segmentation Verification

## 🎯 What This Tool Does

Overlays TotalSegmentator's vertebrae segmentation masks on your original CT scan so you can **visually verify** that L1-L5 are correctly identified.

## 🚀 Quick Commands

### View All Lumbar Vertebrae (Recommended First Step)

```bash
python verify_segmentation.py --vertebra all
```

This opens an **interactive viewer** with:
- 🎚️ Slider to navigate through CT slices
- 🎨 Color-coded vertebrae:
  - **Red** = L1
  - **Orange** = L2
  - **Yellow** = L3
  - **Green** = L4
  - **Blue** = L5

### Focus on a Specific Vertebra (e.g., L1)

```bash
python verify_segmentation.py --vertebra L1
```

### Generate Static Images for Documentation

```bash
python verify_segmentation.py --vertebra all --mode static
```

Output saved to: `verification_output/verification_all.png`

### Both Interactive + Save Images

```bash
python verify_segmentation.py --vertebra all --mode both
```

## 📊 What to Look For

### ✅ Correct Segmentation
- Colored overlay covers the **entire vertebral body**
- Contour edges align with **bone boundaries**
- Labeling follows **cranio-caudal order** (L1 at top, L5 at bottom)

### ❌ Incorrect Segmentation
- Overlay extends into soft tissue or other organs
- Missing coverage of vertebral body
- Wrong order (e.g., L2 labeled as L1)

## 🔧 Advanced Options

### Adjust Window Settings for Better Visualization

**For bone viewing:**
```bash
python verify_segmentation.py --vertebra all --window-level 400 --window-width 1800
```

**For soft tissue (default):**
```bash
python verify_segmentation.py --vertebra all --window-level 40 --window-width 400
```

### Custom Input/Output Paths

```bash
python verify_segmentation.py \
    --vertebra all \
    --ct-dir "path/to/dicom/folder" \
    --seg-dir "path/to/segmentations" \
    --output "path/to/save/images"
```

## 📝 Anatomical Reference

**L1**: First lumbar vertebra
- Below the last rib (T12)
- Roughly at the level of the transpyloric plane

**L2**: Second lumbar vertebra
- Renal hilum level

**L3**: Third lumbar vertebra  
- Typically at umbilicus level

**L4**: Fourth lumbar vertebra
- At the level of the iliac crest

**L5**: Fifth lumbar vertebra
- Just above the sacrum
- Most inferior lumbar vertebra

## ✅ Orientation Fix Applied

The tool automatically corrects orientation mismatches between:
- CT DICOM data (typically RAS orientation)
- TotalSegmentator masks (typically LAS orientation)

No manual intervention needed! The masks will properly overlay on the anatomy.

## 🎓 Example Workflow

1. **Initial Check**: `python verify_segmentation.py --vertebra all`
2. **Verify each vertebra** using the slider
3. **If something looks wrong**: Focus on that vertebra  
   `python verify_segmentation.py --vertebra L1`
4. **Document results**: Add `--mode static` to save images

## 📧 Integration with Pipeline

You can also import the verification functions into your pipeline:

```python
from verify_segmentation import create_static_verification
import nibabel as nib

# Load your data
ct_nifti = nib.load("ct.nii.gz")
l1_mask = nib.load("segmentations/vertebrae_L1.nii.gz")

# Create verification image
create_static_verification(
    ct_volume=ct_nifti.get_fdata(),
    masks={"vertebrae_L1": l1_mask.get_fdata()},
    output_path="verification.png"
)
```

---

**Need more help?** See `VERIFICATION_GUIDE.md` for detailed instructions.

