"""
Interactive Segmentation Verification Tool

This script allows you to visualize segmentation masks overlaid on the original CT scan
to verify that structures like L1 vertebrae are correctly identified.

Usage:
    python verify_segmentation.py --vertebra L1
    python verify_segmentation.py --vertebra L1 --mode multi  # Show all lumbar vertebrae
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
from matplotlib.widgets import Slider
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    ornt_transform,
)
from nibabel.processing import resample_from_to

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Color map for different vertebrae
VERTEBRAE_COLORS = {
    'vertebrae_T11': '#800080',  # Purple
    'vertebrae_T12': '#FFC0CB',  # Pink
    'vertebrae_L1': '#FF0000',  # Red
    'vertebrae_L2': '#FF8C00',  # Dark Orange
    'vertebrae_L3': '#FFD700',  # Gold
    'vertebrae_L4': '#00FF00',  # Lime
    'vertebrae_L5': '#0000FF',  # Blue
    'vertebrae_T11_body': '#4B0082',  # Indigo
    'vertebrae_T12_body': '#DB7093',  # Pale Violet Red
    'vertebrae_L1_body': '#800000',  # Dark Red
    'vertebrae_L2_body': '#8B4500',  # Saddle Brown
    'vertebrae_L3_body': '#B8860B',  # Dark Goldenrod
    'vertebrae_L4_body': '#006400',  # Dark Green
    'vertebrae_L5_body': '#00008B',  # Dark Blue
}

LUMBAR_VERTEBRAE = ['vertebrae_T11', 'vertebrae_T12', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5']
LUMBAR_BODIES = [f"{v}_body" for v in LUMBAR_VERTEBRAE]


def load_dicom_series(dicom_dir: Path) -> nib.Nifti1Image:
    """
    Load DICOM series and return 3D volume with proper orientation.
    
    Args:
        dicom_dir: Directory containing DICOM files
        
    Returns:
        Tuple of (3D numpy array with CT data, nibabel image object)
    """
    logger.info(f"Loading DICOM series from {dicom_dir}")
    
    # Get all DICOM files
    dicom_files = sorted(list(dicom_dir.glob("*.dcm")))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
    # Load first slice to get metadata
    first_slice = pydicom.dcmread(str(dicom_files[0]))
    
    # Initialize volume
    img_shape = (first_slice.Rows, first_slice.Columns, len(dicom_files))
    volume = np.zeros(img_shape, dtype=np.float32)
    
    # Load all slices
    slice_locations = []
    for i, dcm_file in enumerate(dicom_files):
        ds = pydicom.dcmread(str(dcm_file))
        volume[:, :, i] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        if hasattr(ds, 'ImagePositionPatient'):
            slice_locations.append(float(ds.ImagePositionPatient[2]))
    
    # Sort by slice location if available
    if slice_locations:
        sorted_indices = np.argsort(slice_locations)
        volume = volume[:, :, sorted_indices]
    
    # Create affine matrix for proper orientation
    try:
        img_position = np.array(first_slice.ImagePositionPatient)
        img_orientation = np.array(first_slice.ImageOrientationPatient)
        pixel_spacing = np.array(first_slice.PixelSpacing)
        
        if hasattr(first_slice, 'SliceThickness'):
            slice_spacing = float(first_slice.SliceThickness)
        else:
            slice_spacing = 1.0
        
        # Build affine transformation matrix
        row_vec = img_orientation[:3] * pixel_spacing[0]
        col_vec = img_orientation[3:] * pixel_spacing[1]
        slice_vec = np.cross(row_vec, col_vec)
        slice_vec = slice_vec / np.linalg.norm(slice_vec) * slice_spacing
        
        affine = np.eye(4)
        affine[:3, 0] = row_vec
        affine[:3, 1] = col_vec
        affine[:3, 2] = slice_vec
        affine[:3, 3] = img_position
        
    except (AttributeError, ValueError):
        # Default affine if metadata is missing
        affine = np.eye(4)
    
    # Create NIfTI image for proper orientation handling
    nifti_img = nib.Nifti1Image(volume, affine)
    
    logger.info(f"Loaded DICOM volume with shape: {volume.shape}")
    logger.info(f"Original orientation: {nib.aff2axcodes(affine)}")
    
    return nifti_img


def load_totalseg_ct(seg_dir: Path) -> nib.Nifti1Image:
    """
    Load the CT volume that TotalSegmentator used/resampled to.
    Using this for visualization guarantees masks and CT share
    exactly the same grid and affine.
    """
    ct_path = seg_dir / "vertebrae_body.nii.gz"
    if not ct_path.exists():
        # Fallback: try a generic ct.nii.gz if present
        alt_path = seg_dir / "ct.nii.gz"
        if alt_path.exists():
            ct_path = alt_path
        else:
            raise FileNotFoundError(
                f"Could not find a TotalSegmentator CT volume in {seg_dir} "
                "(expected vertebrae_body.nii.gz or ct.nii.gz)."
            )

    logger.info(f"Loading TotalSegmentator CT from {ct_path}")
    ct_img = nib.load(str(ct_path))
    logger.info(f"TotalSegmentator CT shape: {ct_img.shape}")
    logger.info(f"TotalSegmentator CT orientation: {nib.aff2axcodes(ct_img.affine)}")
    return ct_img


def load_segmentation_mask(mask_path: Path) -> nib.Nifti1Image:
    """
    Load segmentation mask from NIfTI file.
    
    Args:
        mask_path: Path to segmentation mask file
        
    Returns:
        NIfTI image object with mask data
    """
    logger.info(f"Loading segmentation mask from {mask_path}")
    
    if not mask_path.exists():
        raise ValueError(f"Mask file not found: {mask_path}")
    
    nifti = nib.load(str(mask_path))
    
    logger.info(f"Loaded mask with shape: {nifti.shape}")
    logger.info(f"Mask orientation: {nib.aff2axcodes(nifti.affine)}")
    
    return nifti


def reorient_to_canonical(img: nib.Nifti1Image, target_orientation='RAS') -> nib.Nifti1Image:
    """
    Reorient image to canonical orientation.
    
    Args:
        img: NIfTI image to reorient
        target_orientation: Target orientation code (default: RAS)
        
    Returns:
        Reoriented NIfTI image
    """
    current_orient = nib.aff2axcodes(img.affine)
    logger.info(f"Reorienting from {current_orient} to {target_orientation}")
    
    # Reorient to target orientation
    reoriented = nib.as_closest_canonical(img)
    
    return reoriented


def reorient_image_to_axcodes(
    img: nib.Nifti1Image, target_axcodes: tuple[str, str, str]
) -> nib.Nifti1Image:
    """
    Reorient image data so that it has the specified orientation code triplet.
    
    Args:
        img: Source NIfTI image
        target_axcodes: Target orientation (e.g., ('R','A','S'))
        
    Returns:
        Reoriented NIfTI image
    """
    current_axcodes = nib.aff2axcodes(img.affine)
    if current_axcodes == target_axcodes:
        return img

    transform = ornt_transform(
        axcodes2ornt(current_axcodes),
        axcodes2ornt(target_axcodes)
    )
    reoriented_data = apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine.copy()
    new_affine = new_affine @ inv_ornt_aff(transform, img.shape)

    return nib.Nifti1Image(reoriented_data, new_affine)


def align_images(ct_img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> tuple:
    """
    Align CT and mask images to the same orientation.
    
    Args:
        ct_img: CT NIfTI image
        mask_img: Mask NIfTI image
        
    Returns:
        Tuple of (aligned_ct_data, aligned_mask_data) as numpy arrays
    """
    # Reorient both to canonical orientation (RAS)
    ct_canonical = reorient_to_canonical(ct_img)
    mask_canonical = reorient_to_canonical(mask_img)
    
    ct_data = ct_canonical.get_fdata()
    mask_data = mask_canonical.get_fdata()
    
    logger.info(f"Aligned CT shape: {ct_data.shape}")
    logger.info(f"Aligned mask shape: {mask_data.shape}")
    
    # Handle potential size mismatch by resampling mask if needed
    if ct_data.shape != mask_data.shape:
        logger.warning(f"Shape mismatch: CT {ct_data.shape} vs Mask {mask_data.shape}")
        logger.warning("Masks will be displayed in their native space - may not align perfectly")
    
    return ct_data, mask_data


def find_mask_extent(mask: np.ndarray) -> tuple:
    """
    Find the extent of the mask in the z-direction.
    
    Args:
        mask: 3D segmentation mask
        
    Returns:
        Tuple of (min_slice, max_slice, center_slice)
    """
    # Find slices that contain mask
    slices_with_mask = np.where(np.sum(mask, axis=(0, 1)) > 0)[0]
    
    if len(slices_with_mask) == 0:
        return 0, mask.shape[2] - 1, mask.shape[2] // 2
    
    min_slice = slices_with_mask[0]
    max_slice = slices_with_mask[-1]
    center_slice = (min_slice + max_slice) // 2
    
    return min_slice, max_slice, center_slice


class SegmentationViewer:
    """Interactive viewer for segmentation verification."""
    
    def __init__(
        self,
        ct_volume: np.ndarray,
        masks: Dict[str, np.ndarray],
        window_level: int = 40,
        window_width: int = 400
    ):
        """
        Initialize the viewer.
        
        Args:
            ct_volume: 3D CT volume
            masks: Dictionary of vertebra_name -> mask array
            window_level: Window level for CT display (HU)
            window_width: Window width for CT display (HU)
        """
        self.ct_volume = ct_volume
        self.masks = masks
        self.window_level = window_level
        self.window_width = window_width
        
        # Determine view range based on masks
        min_slices = []
        max_slices = []
        for mask in masks.values():
            min_s, max_s, _ = find_mask_extent(mask)
            min_slices.append(min_s)
            max_slices.append(max_s)
        
        if min_slices:
            self.slice_min = max(0, min(min_slices) - 10)
            self.slice_max = min(ct_volume.shape[2] - 1, max(max_slices) + 10)
            self.current_slice = (self.slice_min + self.slice_max) // 2
        else:
            self.slice_min = 0
            self.slice_max = ct_volume.shape[2] - 1
            self.current_slice = ct_volume.shape[2] // 2
        
        self.fig = None
        self.ax_main = None
        self.slider = None
    
    def show(self):
        """Display the interactive viewer."""
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 10))
        
        # Main image axes
        self.ax_main = plt.axes([0.1, 0.2, 0.8, 0.7])
        
        # Slider axes
        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
        
        # Create slider
        self.slider = Slider(
            ax_slider,
            'Slice',
            self.slice_min,
            self.slice_max,
            valinit=self.current_slice,
            valstep=1
        )
        self.slider.on_changed(self.update)
        
        # Initial display
        self.update(self.current_slice)
        
        # Add instructions
        self.fig.text(
            0.5, 0.95,
            'Use the slider to navigate through slices. Colored overlays show segmented structures.',
            ha='center',
            fontsize=12,
            weight='bold'
        )
        
        # Add legend
        legend_elements = []
        for name, color in VERTEBRAE_COLORS.items():
            if name in self.masks:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, alpha=0.5, label=name.replace('vertebrae_', '')))
        
        if legend_elements:
            self.ax_main.legend(
                handles=legend_elements,
                loc='upper right',
                framealpha=0.8,
                fontsize=10
            )
        
        plt.show()
    
    def update(self, slice_idx):
        """Update the display for a given slice."""
        slice_idx = int(slice_idx)
        self.current_slice = slice_idx
        
        # Clear axes
        self.ax_main.clear()
        
        # Get CT slice
        ct_slice = self.ct_volume[:, :, slice_idx]
        
        # Apply windowing
        ct_min = self.window_level - self.window_width / 2
        ct_max = self.window_level + self.window_width / 2
        ct_slice_display = np.clip(ct_slice, ct_min, ct_max)
        ct_slice_display = (ct_slice_display - ct_min) / (ct_max - ct_min)
        
        # Display CT slice
        self.ax_main.imshow(ct_slice_display, cmap='gray', origin='lower', interpolation='bilinear')
        
        # Overlay masks
        for vertebra_name, mask in self.masks.items():
            # Handle potential shape mismatch
            if slice_idx < mask.shape[2]:
                mask_slice = mask[:, :, slice_idx]
                
                if np.any(mask_slice > 0):
                    color = VERTEBRAE_COLORS.get(vertebra_name, '#00FFFF')
                    
                    # Create colored overlay
                    overlay = np.zeros((*mask_slice.shape, 4))
                    
                    # Convert hex color to RGB
                    r = int(color[1:3], 16) / 255.0
                    g = int(color[3:5], 16) / 255.0
                    b = int(color[5:7], 16) / 255.0
                    
                    overlay[mask_slice > 0] = [r, g, b, 0.4]  # Semi-transparent
                    
                    self.ax_main.imshow(overlay, origin='lower', interpolation='nearest')
                    
                    # Add contour
                    self.ax_main.contour(
                        mask_slice,
                        levels=[0.5],
                        colors=[color],
                        linewidths=2,
                        alpha=0.8
                    )
        
        # Set title with slice info
        title = f'Slice {slice_idx} / {self.ct_volume.shape[2] - 1}'
        
        # Add vertebra info if mask is present
        present_vertebrae = []
        for vertebra_name, mask in self.masks.items():
            if slice_idx < mask.shape[2] and np.any(mask[:, :, slice_idx] > 0):
                present_vertebrae.append(vertebra_name.replace('vertebrae_', ''))
        
        if present_vertebrae:
            title += f' - Present: {", ".join(present_vertebrae)}'
        
        self.ax_main.set_title(title, fontsize=14, weight='bold')
        self.ax_main.axis('off')
        
        self.fig.canvas.draw_idle()


def create_static_verification(
    ct_volume: np.ndarray,
    masks: Dict[str, np.ndarray],
    output_path: Path,
    window_level: int = 40,
    window_width: int = 400
):
    """
    Create a static verification image with multiple slices.
    
    Args:
        ct_volume: 3D CT volume
        masks: Dictionary of vertebra_name -> mask array
        output_path: Path to save the image
        window_level: Window level for CT display
        window_width: Window width for CT display
    """
    # Find representative slices for each vertebra
    all_slices = set()
    vertebra_slices = {}
    
    for vertebra_name, mask in masks.items():
        min_s, max_s, center_s = find_mask_extent(mask)
        vertebra_slices[vertebra_name] = [min_s, center_s, max_s]
        all_slices.update([min_s, center_s, max_s])
    
    # Sort slices
    slice_indices = sorted(list(all_slices))
    
    # Limit to 6 slices maximum
    if len(slice_indices) > 6:
        # Sample evenly
        step = len(slice_indices) // 6
        slice_indices = [slice_indices[i * step] for i in range(6)]
    
    # Create figure
    num_slices = len(slice_indices)
    cols = 3
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        
        # Get CT slice
        ct_slice = ct_volume[:, :, slice_idx]
        
        # Apply windowing
        ct_min = window_level - window_width / 2
        ct_max = window_level + window_width / 2
        ct_slice_display = np.clip(ct_slice, ct_min, ct_max)
        ct_slice_display = (ct_slice_display - ct_min) / (ct_max - ct_min)
        
        # Ensure we have valid data to display
        if np.any(np.isfinite(ct_slice_display)):
            # Display CT slice
            ax.imshow(ct_slice_display, cmap='gray', origin='lower', vmin=0, vmax=1)
        else:
            logger.warning(f"Slice {slice_idx} has no valid CT data to display")
        
        # Overlay masks
        for vertebra_name, mask in masks.items():
            if slice_idx < mask.shape[2]:
                mask_slice = mask[:, :, slice_idx]
                
                if np.any(mask_slice > 0):
                    color = VERTEBRAE_COLORS.get(vertebra_name, '#00FFFF')
                    
                    # Create colored overlay
                    overlay = np.zeros((*mask_slice.shape, 4))
                    r = int(color[1:3], 16) / 255.0
                    g = int(color[3:5], 16) / 255.0
                    b = int(color[5:7], 16) / 255.0
                    overlay[mask_slice > 0] = [r, g, b, 0.3]
                    
                    ax.imshow(overlay, origin='lower')
                    ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=2)
        
        # Title
        present = [v.replace('vertebrae_', '') for v, m in masks.items() 
                  if slice_idx < m.shape[2] and np.any(m[:, :, slice_idx] > 0)]
        ax.set_title(f'Slice {slice_idx}' + (f' - {", ".join(present)}' if present else ''))
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, alpha=0.5, label=name.replace('vertebrae_', ''))
        for name, color in VERTEBRAE_COLORS.items()
        if name in masks
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved verification image to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Verify segmentation by overlaying masks on CT scan'
    )
    parser.add_argument(
        '--vertebra',
        type=str,
        default='L1',
        help='Vertebra to verify (e.g., L1, L2, L3, L4, L5) or "all" for all lumbar vertebrae'
    )
    parser.add_argument(
        '--ct-dir',
        type=Path,
        default=Path('Plain5mmSTD'),
        help='Directory containing DICOM files (used when --ct-source dicom)'
    )
    parser.add_argument(
        '--ct-source',
        type=str,
        choices=['totalseg', 'dicom'],
        default='totalseg',
        help='Which CT to use for background: '
             '"totalseg" = CT in TotalSegmentator space (recommended), '
             '"dicom" = original DICOM series'
    )
    parser.add_argument(
        '--seg-dir',
        type=Path,
        default=Path('segmentations'),
        help='Directory containing segmentation masks'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'static', 'both'],
        default='interactive',
        help='Visualization mode: interactive (slider), static (image), or both'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('verification_output'),
        help='Output directory for static images'
    )
    parser.add_argument(
        '--window-level',
        type=int,
        default=40,
        help='Window level for CT display (HU)'
    )
    parser.add_argument(
        '--window-width',
        type=int,
        default=400,
        help='Window width for CT display (HU)'
    )
    
    args = parser.parse_args()
    
    # Load CT volume
    if args.ct_source == 'totalseg':
        logger.info("Loading CT volume from TotalSegmentator space...")
        ct_img = load_totalseg_ct(args.seg_dir)
        ct_volume = ct_img.get_fdata()
    else:
        logger.info("Loading CT volume from DICOM series...")
        ct_img = load_dicom_series(args.ct_dir)
        ct_volume = ct_img.get_fdata()
    
    # Determine which vertebrae to load
    if args.vertebra.lower() == 'all':
        vertebrae_to_load = LUMBAR_BODIES
    else:
        vertebra_name = f"vertebrae_{args.vertebra}"
        # Handle if user already passed "vertebrae_L1" or just "L1"
        if not args.vertebra.startswith('vertebrae_'):
             vertebra_name = f"vertebrae_{args.vertebra}"
        else:
             vertebra_name = args.vertebra

        if vertebra_name not in LUMBAR_VERTEBRAE and vertebra_name not in LUMBAR_BODIES:
            logger.warning(f"{vertebra_name} is not a known lumbar vertebra/body. Loading anyway...")
        vertebrae_to_load = [vertebra_name]
    
    # Load masks
    logger.info(f"Loading segmentation masks for: {', '.join(vertebrae_to_load)}")
    mask_images = {}
    for vertebra in vertebrae_to_load:
        mask_path = args.seg_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            mask_images[vertebra] = load_segmentation_mask(mask_path)
        else:
            logger.warning(f"Mask not found: {mask_path}")
    
    if not mask_images:
        logger.error("No masks loaded. Exiting.")
        return
    
    # Resample masks into DICOM CT grid if requested
    if args.ct_source == 'dicom':
        logger.info("Aligning masks with DICOM CT...")
        # Keep CT in its original DICOM space
        ct_volume = ct_img.get_fdata()
        ct_orient = nib.aff2axcodes(ct_img.affine)
        
        resampled_masks: Dict[str, np.ndarray] = {}
        for vertebra, mask_img in mask_images.items():
            mask_orient = nib.aff2axcodes(mask_img.affine)
            
            # Get mask data
            mask_data = mask_img.get_fdata()
            
            # Rotate 90 degrees counterclockwise in the axial plane (around z-axis)
            # k=1 rotates 90° counterclockwise (anticlockwise)
            mask_data = np.rot90(mask_data, k=1, axes=(0, 1))
            logger.info(f"{vertebra} rotated 90° counterclockwise")
            
            # If shapes match, use directly (after rotation)
            if mask_data.shape == ct_volume.shape:
                resampled_masks[vertebra] = mask_data
                logger.info(f"{vertebra} aligned: sum={mask_data.sum():.0f}, max={mask_data.max()}")
            else:
                # Shapes don't match - need resampling
                mask_nifti = nib.Nifti1Image(mask_data, mask_img.affine)
                resampled = resample_from_to(mask_nifti, ct_img, order=0)
                mask_data = resampled.get_fdata()
                resampled_masks[vertebra] = mask_data
                logger.info(f"{vertebra} resampled: sum={mask_data.sum():.0f}, max={mask_data.max()}")
        
        masks = resampled_masks
        logger.info(f"CT volume: shape={ct_volume.shape}, min={ct_volume.min():.1f}, max={ct_volume.max():.1f}")
    else:
        # TotalSegmentator space: masks already match CT grid
        masks = {vertebra: mask_img.get_fdata() for vertebra, mask_img in mask_images.items()}
    
    logger.info(f"Final CT volume shape: {ct_volume.shape}")
    for vertebra, mask in masks.items():
        logger.info(f"Final {vertebra} mask shape: {mask.shape}")
    
    # Create visualization(s)
    if args.mode in ['interactive', 'both']:
        logger.info("Launching interactive viewer...")
        viewer = SegmentationViewer(
            ct_volume,
            masks,
            window_level=args.window_level,
            window_width=args.window_width
        )
        viewer.show()
    
    if args.mode in ['static', 'both']:
        output_path = args.output / f"verification_{args.vertebra}.png"
        logger.info("Creating static verification image...")
        create_static_verification(
            ct_volume,
            masks,
            output_path,
            window_level=args.window_level,
            window_width=args.window_width
        )


if __name__ == '__main__':
    main()

