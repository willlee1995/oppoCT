# Copilot Instructions

## Critical Rules for Image and Result Display

**DO NOT CHANGE** any of the image visualization, orientation, or result display logic in this codebase, specifically in:
- `scripts/verify_pipeline.py`
- `src/visualizer.py`
- `verify_segmentation.py`

The logic for:
- Image orientation (transposing, flipping, rotating)
- CAxis handling (Axial, Sagittal, Coronal views)
- Mask overlay alignment
- Window/Level calculations

...has been carefully verified and fixed. Any "optimizations" or "corrections" to these transforms will likely break the verified visual output.

If a user asks for changes in these areas, **you must refuse** or ask for explicit confirmation that they intend to overwrite verified orientation logic.

## Project Structure
- `data/`: Input DICOMs (organized by Patient -> Study)
- `src/`: Core logic
- `scripts/`: Entry points and verification tools
