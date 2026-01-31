"""
Batch Verification Pipeline

1. Runs segmentation for ALL cases first (showing progress in Tkinter).
2. Then runs interactive verification for ALL cases.

for best performance, set NUMEXPR_MAX_THREADS to limit CPU usage, e.g. in PowerShell:
$env:NUMEXPR_MAX_THREADS="20"

Usage:
    python scripts/batch_verify_pipeline.py data/processed out/test_batch --output-csv out/result.csv
"""

print("Initializing Batch Verification Pipeline...", flush=True)

import argparse
import logging
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import time

# Set up logging early to capture imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Imports starting...")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# CRITICAL: Start with Agg backend to allow independent Tkinter window in Phase 1
import matplotlib
logger.info("Setting initial Matplotlib backend to 'Agg' for Phase 1...")
matplotlib.use('Agg', force=True)

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Import pyplot (will use Agg)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from typing import Dict, List, Optional, Tuple

from src.patient_manager import get_patient_metadata
from src.pipeline import find_patient_folders, process_single_patient
from src.visualizer import find_representative_slices
from verify_segmentation import load_dicom_series, load_segmentation_mask

logger.info("All imports complete. defining classes...")

# Color map for different vertebrae (matching verify_segmentation.py)
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
    'vertebrae_L1_body_trabecular_core': '#00FF7F',  # Spring Green
}

LUMBAR_VERTEBRAE = ['vertebrae_T11', 'vertebrae_T12', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5']
LUMBAR_BODIES = [f"{v}_body" for v in LUMBAR_VERTEBRAE]


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
        self.lbl_status.config(text=f"Part 1/2: Processing {current}/{self.total} - {message}")
        self.log(f"[{current}/{self.total}] {message}")
        self.root.update()

    def log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def close(self):
        self.root.destroy()


class VerificationViewer:
    """
    Interactive viewer for case-by-case verification with sagittal view.

    WARNING: DO NOT CHANGE IMAGE ORIENTATION/DISPLAY LOGIC.
    The transforms (fliplr, rot90, etc.) in this class are verified and fixed.
    Modifying them will break the visualization alignment.
    """

    def __init__(
        self,
        ct_volume: np.ndarray,
        masks: Dict[str, np.ndarray],
        dicom_folder: Path,
        patient_id: str,
        exam_date: Optional[str] = None,
        window_level: int = 40,
        window_width: int = 400
    ):
        """
        Initialize the verification viewer.

        Args:
            ct_volume: 3D CT volume (HU values)
            masks: Dictionary of vertebra_name -> mask array
            dicom_folder: Path to DICOM folder for metadata
            patient_id: Patient identifier
            exam_date: Exam date from DICOM
            window_level: Window level for CT display (HU)
            window_width: Window width for CT display (HU)
        """
        self.ct_volume = ct_volume
        self.masks = masks
        self.dicom_folder = dicom_folder
        self.patient_id = patient_id
        self.exam_date = exam_date
        self.window_level = window_level
        self.window_width = window_width

        # Volume dimensions: (height, width, depth) for axial view
        # Axial: (H, W) slices along depth axis
        # Sagittal: (H, D) slices along width axis
        self.axial_shape = ct_volume.shape  # (H, W, D)

        # Current slice indices
        self.axial_slice = ct_volume.shape[2] // 2  # Depth axis
        self.sagittal_slice = ct_volume.shape[1] // 2  # Width axis (for sagittal view)

        # Selected slices for HU calculation
        # Auto-select representative slices by default
        try:
            self.selected_slices = find_representative_slices(ct_volume, masks, num_slices=3)
            logger.info(f"Auto-selected slices: {self.selected_slices}")
        except Exception as e:
            logger.warning(f"Failed to auto-select slices: {e}")
            self.selected_slices: List[int] = []

        # Case status
        self.is_successful: Optional[bool] = None

        # UI elements
        self.fig = None
        self.ax_axial = None
        self.ax_sagittal = None
        self.slider_axial = None
        self.slider_sagittal = None
        self.btn_success = None
        self.btn_fail = None
        self.btn_select_slice = None
        self.btn_auto_select = None
        self.btn_done = None

        # Display state
        self.show_selected = True

        # Label mapping: Target Label -> Source Mask Name
        # Default: L1 -> vertebrae_L1_body, etc.
        self.label_mapping = {v.replace('vertebrae_', '').replace('_body', ''): v for v in LUMBAR_BODIES}
        # Ensure we have mappings for all bodies in LUMBAR_BODIES
        for v in LUMBAR_BODIES:
            short_name = v.replace('vertebrae_', '').replace('_body', '')
            if short_name not in self.label_mapping:
                self.label_mapping[short_name] = v

        # UI elements for label correction
        self.btn_shift_up = None
        self.btn_shift_down = None
        self.btn_reset = None
        self.txt_mapping = None

    def calculate_sagittal_view(self, slice_idx: int) -> np.ndarray:
        """
        Extract sagittal slice from CT volume.

        Sagittal view: side view (left-right slices, showing anterior-posterior vs superior-inferior)
        Coronal view: front view (anterior-posterior slices, showing left-right vs superior-inferior)

        Volume shape: (height, width, depth) = (H, W, D)
        - Axial: ct_volume[:, :, slice_idx] = (H, W) - top-down view
        - Sagittal: ct_volume[:, slice_idx, :] = (H, D) - side view (anterior-posterior vs superior-inferior)
        - Coronal: ct_volume[slice_idx, :, :] = (W, D) - front view (left-right vs superior-inferior)

        Args:
            slice_idx: Index along the width dimension (left-right) for sagittal view

        Returns:
            2D sagittal slice (height x depth)
        """
        # For true sagittal view (side view), extract along width axis
        # This gives (H, D) - height (anterior-posterior) vs depth (superior-inferior)
        if slice_idx < 0 or slice_idx >= self.axial_shape[1]:
            slice_idx = max(0, min(slice_idx, self.axial_shape[1] - 1))

        # Extract along width axis for sagittal view (side view)
        sagittal_slice = self.ct_volume[:, slice_idx, :]
        return sagittal_slice

    def calculate_volumetric_hu(self, vertebra_name: str) -> float:
        """
        Calculate average HU value for the entire volume of a specific vertebra.

        Args:
            vertebra_name: Target label (e.g., 'L1') OR direct mask name

        Returns:
            Average HU value within the entire mask
        """
        # Get the source mask name from mapping, or use name directly if not mapped
        mask_name = self.label_mapping.get(vertebra_name, vertebra_name)
        if not mask_name or mask_name not in self.masks:
            return 0.0

        mask = self.masks[mask_name]

        # Ensure mask matches CT shape
        if mask.shape != self.ct_volume.shape:
            logger.warning(f"Shape mismatch for {mask_name}: {mask.shape} vs {self.ct_volume.shape}")
            return 0.0

        total_sum = 0.0
        total_count = 0

        # Iterate through all slices to apply 2D transformations
        # This ensures consistency with the visual display and slice-based calculation
        for slice_idx in range(self.axial_shape[2]):
            # Get CT slice and apply same transformation as in display
            ct_slice = self.ct_volume[:, :, slice_idx]
            ct_slice = np.fliplr(ct_slice)  # Flip horizontally

            # Get mask slice
            mask_slice = mask[:, :, slice_idx]

            # Apply same transformations as in display
            mask_slice = np.rot90(mask_slice, k=-1)  # Rotate 90° clockwise
            mask_slice = np.flipud(mask_slice)  # Flip vertically

            # Only include HU values within masked regions
            if np.any(mask_slice > 0):
                # Ensure shapes match
                if ct_slice.shape == mask_slice.shape:
                    masked_hu_values = ct_slice[mask_slice > 0]
                    total_sum += np.sum(masked_hu_values)
                    total_count += masked_hu_values.size

        if total_count == 0:
            return 0.0

        return float(total_sum / total_count)

    def calculate_average_hu(self, slice_indices: List[int], vertebra_name: Optional[str] = None) -> float:
        """
        Calculate average HU value for selected slices.

        Args:
            slice_indices: List of slice indices (axial slices)
            vertebra_name: If provided, only calculate for this specific vertebra mask.
                          If None, calculates for all combined masks (legacy behavior).

        Returns:
            Average HU value within mask regions
        """
        if not slice_indices:
            return 0.0

        all_masked_values = []
        for slice_idx in slice_indices:
            if 0 <= slice_idx < self.axial_shape[2]:
                # Get CT slice and apply same transformation as in display
                ct_slice = self.ct_volume[:, :, slice_idx]
                ct_slice = np.fliplr(ct_slice)  # Flip horizontally (same as display)

                # Determine which mask(s) to use
                if vertebra_name:
                    # Specific vertebra
                    mask_name = self.label_mapping.get(vertebra_name)
                    if not mask_name or mask_name not in self.masks:
                        continue
                    masks_to_use = [self.masks[mask_name]]
                else:
                    # All masks (legacy)
                    masks_to_use = self.masks.values()

                # Combine masks
                combined_mask = np.zeros_like(ct_slice, dtype=bool)
                for mask in masks_to_use:
                    if len(mask.shape) == 3 and slice_idx < mask.shape[2]:
                        mask_slice = mask[:, :, slice_idx]
                        # Apply same transformations as in display
                        mask_slice = np.rot90(mask_slice, k=-1)  # Rotate 90° clockwise
                        mask_slice = np.flipud(mask_slice)  # Flip vertically
                        combined_mask = combined_mask | (mask_slice > 0)

                # Only include HU values within masked regions
                if np.any(combined_mask):
                    masked_hu_values = ct_slice[combined_mask]
                    all_masked_values.extend(masked_hu_values.tolist())

        if not all_masked_values:
            return 0.0

        return float(np.mean(all_masked_values))

    def window_ct(self, ct_slice: np.ndarray) -> np.ndarray:
        """Apply window/level to CT slice for display."""
        ct_min = self.window_level - self.window_width / 2
        ct_max = self.window_level + self.window_width / 2

        # Clip values to window range
        ct_display = np.clip(ct_slice, ct_min, ct_max)

        # Normalize to [0, 1] for display
        if ct_max > ct_min:
            ct_display = (ct_display - ct_min) / (ct_max - ct_min)
        else:
            # Fallback if window is invalid
            ct_display = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-10)

        # Ensure valid range
        ct_display = np.clip(ct_display, 0.0, 1.0)

        return ct_display

    def update_axial(self, slice_idx):
        """Update axial view with segmentation overlays."""
        slice_idx = int(slice_idx)
        self.axial_slice = slice_idx

        self.ax_axial.clear()

        # Get CT slice
        ct_slice = self.ct_volume[:, :, slice_idx]
        ct_display = self.window_ct(ct_slice)

        # Apply transformations for axial view:
        # - Flip CT horizontally
        ct_display = np.fliplr(ct_display)

        # Check if CT has valid data
        if np.all(np.isnan(ct_display)) or np.all(ct_display == 0):
            logger.warning(f"CT slice {slice_idx} appears to be empty or invalid")

        # Display CT slice with explicit vmin/vmax to ensure visibility
        self.ax_axial.imshow(ct_display, cmap='gray', origin='lower', interpolation='bilinear',
                            vmin=0.0, vmax=1.0)

        # Overlay segmentation masks
        present_vertebrae = []
        for vertebra_name, mask in self.masks.items():
            # Handle potential shape mismatch
            if len(mask.shape) == 3 and slice_idx < mask.shape[2]:
                mask_slice = mask[:, :, slice_idx]
            elif len(mask.shape) == 2:
                # 2D mask - use directly if it matches
                mask_slice = mask if mask.shape == ct_slice.shape else np.zeros_like(ct_slice)
            else:
                continue

            # Apply transformations for axial view:
            # - Rotate mask 90 degrees clockwise
            mask_slice = np.rot90(mask_slice, k=-1)  # k=-1 rotates 90° clockwise
            # - Flip mask vertically (not horizontally)
            mask_slice = np.flipud(mask_slice)

            if np.any(mask_slice > 0):
                # Special handling for combined vertebrae_body mask
                if vertebra_name == 'vertebrae_body':
                    color = '#FF00FF'  # Magenta for combined mask
                    present_vertebrae.append('All Vertebrae')
                else:
                    color = VERTEBRAE_COLORS.get(vertebra_name, '#00FFFF')
                    present_vertebrae.append(vertebra_name.replace('vertebrae_', ''))

                # Create colored overlay
                overlay = np.zeros((*mask_slice.shape, 4))

                # Convert hex color to RGB
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0

                overlay[mask_slice > 0] = [r, g, b, 0.4]  # Semi-transparent

                self.ax_axial.imshow(overlay, origin='lower', interpolation='nearest')

                # Add contour
                self.ax_axial.contour(
                    mask_slice,
                    levels=[0.5],
                    colors=[color],
                    linewidths=2,
                    alpha=0.8
                )

        # Highlight if selected
        if self.show_selected and slice_idx in self.selected_slices:
            # Add border
            rect = Rectangle((0, 0), ct_display.shape[1]-1, ct_display.shape[0]-1,
                           linewidth=3, edgecolor='yellow', facecolor='none')
            self.ax_axial.add_patch(rect)

        title = f'Axial Slice {slice_idx} / {self.axial_shape[2] - 1}'
        if present_vertebrae:
            title += f' - {", ".join(present_vertebrae)}'
        if slice_idx in self.selected_slices:
            title += ' [SELECTED]'
        title_color = 'darkorange' if slice_idx in self.selected_slices else 'black'
        self.ax_axial.set_title(title, fontsize=14, weight='bold', color=title_color,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax_axial.axis('off')

        # Update sagittal view to show new reference line for current axial slice
        self.update_sagittal(self.sagittal_slice)

        self.fig.canvas.draw_idle()

    def update_sagittal(self, slice_idx):
        """Update sagittal view with segmentation overlays."""
        slice_idx = int(slice_idx)
        self.sagittal_slice = slice_idx

        self.ax_sagittal.clear()

        # Get sagittal slice from CT
        sagittal_slice = self.calculate_sagittal_view(slice_idx)
        sagittal_display = self.window_ct(sagittal_slice)

        # Apply transformation for sagittal view:
        # - Rotate CT 90 degrees clockwise
        sagittal_display = np.rot90(sagittal_display, k=-1)  # k=-1 rotates 90° clockwise

        # Check if sagittal CT has valid data
        if np.all(np.isnan(sagittal_display)) or np.all(sagittal_display == 0):
            logger.warning(f"Sagittal slice {slice_idx} appears to be empty or invalid")

        # Display sagittal slice with explicit vmin/vmax to ensure visibility
        self.ax_sagittal.imshow(sagittal_display, cmap='gray', origin='lower', interpolation='bilinear',
                               aspect='auto', vmin=0.0, vmax=1.0)

        # Overlay segmentation masks on sagittal view
        for vertebra_name, mask in self.masks.items():
            # Extract sagittal slice from mask
            # CT extraction: sagittal_slice = self.ct_volume[:, slice_idx, :] (along axis 1 - width)
            # Try extracting mask along axis 2 (depth) instead: mask[:, :, slice_idx]

            if len(mask.shape) != 3:
                continue

            # Ensure mask shape matches CT shape (should be true after resampling)
            if mask.shape != self.ct_volume.shape:
                logger.warning(f"Mask shape {mask.shape} doesn't match CT shape {self.ct_volume.shape} for {vertebra_name} - skipping")
                continue

            # Extract sagittal slice along axis 0 (height) - user confirmed this is correct
            if slice_idx < 0 or slice_idx >= mask.shape[0]:
                continue

            mask_sagittal = mask[slice_idx, :, :]  # Extract along height axis (axis 0), gives (W, D)

            # Try different transformations:
            # Option 1: Flip vertically instead of horizontally
            mask_sagittal = np.flipud(mask_sagittal)  # Flip vertically

            # Verify shape matches before transformation
            # If shape doesn't match, try to reshape if sizes match
            if mask_sagittal.shape != sagittal_slice.shape:
                logger.debug(f"Mask sagittal slice shape {mask_sagittal.shape} != CT sagittal slice shape {sagittal_slice.shape} for {vertebra_name}")
                # Try to reshape if total size matches
                if mask_sagittal.size == sagittal_slice.size:
                    mask_sagittal = mask_sagittal.reshape(sagittal_slice.shape)
                    logger.debug(f"Reshaped mask sagittal slice to {mask_sagittal.shape}")
                else:
                    logger.warning(f"Cannot reshape mask sagittal slice - sizes don't match: {mask_sagittal.size} != {sagittal_slice.size}")
                    continue

            # Apply the EXACT SAME transformation as CT (in same order):
            # CT: sagittal_display = np.rot90(sagittal_display, k=-1)
            # So mask should be: mask_sagittal = np.rot90(mask_sagittal, k=-1)
            mask_sagittal = np.rot90(mask_sagittal, k=-1)  # k=-1 rotates 90° clockwise

            if np.any(mask_sagittal > 0):
                # Special handling for combined vertebrae_body mask
                if vertebra_name == 'vertebrae_body':
                    color = '#FF00FF'  # Magenta for combined mask
                else:
                    color = VERTEBRAE_COLORS.get(vertebra_name, '#00FFFF')

                # Create colored overlay
                overlay = np.zeros((*mask_sagittal.shape, 4))

                # Convert hex color to RGB
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0

                overlay[mask_sagittal > 0] = [r, g, b, 0.4]  # Semi-transparent

                self.ax_sagittal.imshow(overlay, origin='lower', interpolation='nearest', aspect='auto')

                # Add contour
                self.ax_sagittal.contour(
                    mask_sagittal,
                    levels=[0.5],
                    colors=[color],
                    linewidths=2,
                    alpha=0.8
                )

        # Show current axial slice as a reference line (cyan/blue)
        # In sagittal view, axial slice index maps to y-axis (height axis)
        # After rotation, the sagittal view shape is (D, H), so axial slice maps to y position
        if 0 <= self.axial_slice < self.axial_shape[2]:
            # Draw horizontal reference line for current axial slice
            # The axial slice index is along depth axis (axis 2), which maps to y-axis in rotated sagittal view
            self.ax_sagittal.axhline(y=self.axial_slice, color='cyan', linewidth=3, alpha=0.9, linestyle='--', label='Current Axial Slice')

        # Show selected axial slices as vertical lines (yellow)
        # In sagittal view, selected axial slices appear as vertical lines
        if self.show_selected and self.selected_slices:
            for sel_slice in self.selected_slices:
                if 0 <= sel_slice < self.axial_shape[2]:
                    # sel_slice is depth index, which maps to x-axis in sagittal view
                    self.ax_sagittal.axvline(x=sel_slice, color='yellow', linewidth=2, alpha=0.7)

        title = f'Sagittal Slice {slice_idx} / {self.axial_shape[1] - 1}'
        self.ax_sagittal.set_title(title, fontsize=14, weight='bold', color='black',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax_sagittal.axis('off')

        self.fig.canvas.draw_idle()

    def toggle_slice_selection(self, event):
        """Toggle selection of current axial slice."""
        if self.axial_slice in self.selected_slices:
            self.selected_slices.remove(self.axial_slice)
        else:
            self.selected_slices.append(self.axial_slice)
            self.selected_slices.sort()

        self.update_axial(self.axial_slice)
        self.update_sagittal(self.sagittal_slice)
        self.update_info_text()

    def auto_select_slices(self, event):
        """Auto-select representative slices."""
        try:
            self.selected_slices = find_representative_slices(self.ct_volume, self.masks, num_slices=3)
            logger.info(f"Auto-selected slices: {self.selected_slices}")
            self.update_axial(self.axial_slice)
            self.update_sagittal(self.sagittal_slice)
            self.update_info_text()
        except Exception as e:
            logger.error(f"Error auto-selecting slices: {e}")

    def mark_success(self, event):
        """Mark case as successful."""
        self.is_successful = True
        self.update_info_text()

    def mark_fail(self, event):
        """Mark case as failed."""
        self.is_successful = False
        self.update_info_text()

    def shift_labels_up(self, event):
        """
        Shift labels UP (cranial).
        Target L1 takes Source L2.
        Target L2 takes Source L3.
        ...
        """
        ordered_labels = ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']

        new_mapping = {}
        for i, target in enumerate(ordered_labels):
            if i + 1 < len(ordered_labels):
                source_label = ordered_labels[i+1]
                next_default_mask = f"vertebrae_{ordered_labels[i+1]}_body"
                new_mapping[target] = next_default_mask
            else:
                new_mapping[target] = None

        self.label_mapping.update(new_mapping)
        self.update_info_text()
        self.update_mapping_text()

    def shift_labels_down(self, event):
        """
        Shift labels DOWN (caudal).
        Target L2 takes Source L1.
        Target L3 takes Source L2.
        ...
        """
        ordered_labels = ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']

        new_mapping = {}
        for i, target in enumerate(ordered_labels):
            if i - 1 >= 0:
                prev_default_mask = f"vertebrae_{ordered_labels[i-1]}_body"
                new_mapping[target] = prev_default_mask
            else:
                new_mapping[target] = None

        self.label_mapping.update(new_mapping)
        self.update_info_text()
        self.update_mapping_text()

    def reset_labels(self, event):
        """Reset labels to default."""
        self.label_mapping = {v.replace('vertebrae_', '').replace('_body', ''): v for v in LUMBAR_BODIES}
        self.update_info_text()
        self.update_mapping_text()

    def update_mapping_text(self):
        """Update the display of current label mapping."""
        if hasattr(self, 'txt_mapping'):
            text = "Label Mapping:\n"
            ordered_labels = ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
            for label in ordered_labels:
                source = self.label_mapping.get(label)
                source_name = source.replace('vertebrae_', '').replace('_body', '') if source else "NONE"
                text += f"{label} <- {source_name}\n"
            self.txt_mapping.set_text(text)
            self.fig.canvas.draw_idle()

    def update_info_text(self):
        """Update information text display."""
        if hasattr(self, 'info_text'):
            status = "SUCCESS" if self.is_successful else ("FAILED" if self.is_successful is False else "NOT MARKED")
            status_color = 'green' if self.is_successful else ('red' if self.is_successful is False else 'gray')

            selected_str = ', '.join(map(str, self.selected_slices)) if self.selected_slices else 'None'
            avg_hu = self.calculate_average_hu(self.selected_slices) if self.selected_slices else 0.0

            # Additional logic for L1 Core vs Body check
            l1_body_hu = self.calculate_volumetric_hu('vertebrae_L1_body')
            l1_core_hu = self.calculate_volumetric_hu('vertebrae_L1_body_trabecular_core')

            info = f"Patient ID: {self.patient_id}\n"
            info += f"Exam Date: {self.exam_date or 'N/A'}\n"
            info += f"Status: {status}\n"
            info += f"Selected Slices: {selected_str}\n"
            info += f"Average HU: {avg_hu:.1f}\n"

            # Display L1 Core/Body stats
            if l1_body_hu != 0 or l1_core_hu != 0:
                info += f"L1 Body HU: {l1_body_hu:.1f}\n"
                info += f"L1 Core HU: {l1_core_hu:.1f}"

                # Warning check
                if l1_core_hu > l1_body_hu and l1_body_hu != 0:
                     info += f"\nWARNING: Core > Body HU!"

            self.info_text.set_text(info)
            # Use darker colors for better visibility
            if status_color == 'gray':
                status_color = 'black'
            elif status_color == 'green':
                status_color = 'darkgreen'
            elif status_color == 'red':
                status_color = 'darkred'

            # If there is a warning, make the text red if it wasn't already green/red
            if "WARNING" in info and status_color == 'black':
                 status_color = 'red'

            self.info_text.set_color(status_color)
            self.fig.canvas.draw_idle()

    def show(self) -> Dict:
        """
        Display the interactive viewer and return results.

        Returns:
            Dictionary with verification results
        """
        # Check backend and verify tkinter is available
        current_backend = plt.get_backend()
        logger.info(f"Current matplotlib backend: {current_backend}")

        if current_backend.lower() == 'agg':
            # Try to check if tkinter is available
            try:
                import tkinter
                logger.info("tkinter is available, attempting to switch to TkAgg backend...")
                matplotlib.use('TkAgg', force=True)
                # Need to close any existing figures and recreate
                plt.close('all')
                current_backend = plt.get_backend()
                logger.info(f"Backend after switch: {current_backend}")
            except ImportError:
                logger.error("tkinter is not available. Cannot use TkAgg backend.")
                raise RuntimeError("No interactive matplotlib backend available. Install tkinter or PyQt5.")

        if current_backend.lower() == 'agg':
            raise RuntimeError("Cannot use interactive matplotlib backend. Backend is locked to 'Agg'.")

        logger.info(f"Using backend: {current_backend} for interactive display")

        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))

        # Main title
        self.fig.suptitle(f'Verification: {self.patient_id}', fontsize=18, weight='bold', color='black')

        # Axial view (left)
        self.ax_axial = plt.subplot(2, 2, 1)

        # Sagittal view (right)
        self.ax_sagittal = plt.subplot(2, 2, 2)

        # Info panel (bottom left)
        ax_info = plt.subplot(2, 2, 3)
        ax_info.axis('off')
        self.info_text = ax_info.text(0.1, 0.5, '', fontsize=13, verticalalignment='center',
                                      family='monospace', weight='bold', color='black')

        # Controls panel (bottom right)
        ax_controls = plt.subplot(2, 2, 4)
        ax_controls.axis('off')

        # Sliders - moved lower to avoid overlapping with text
        ax_slider_axial = plt.axes([0.1, 0.08, 0.35, 0.03])
        ax_slider_sagittal = plt.axes([0.55, 0.08, 0.35, 0.03])

        self.slider_axial = Slider(
            ax_slider_axial,
            'Axial Slice',
            0,
            self.axial_shape[2] - 1,
            valinit=self.axial_slice,
            valstep=1
        )
        self.slider_axial.label.set_fontsize(12)
        self.slider_axial.label.set_weight('bold')
        self.slider_axial.on_changed(self.update_axial)

        self.slider_sagittal = Slider(
            ax_slider_sagittal,
            'Sagittal Slice',
            0,
            self.axial_shape[1] - 1,
            valinit=self.sagittal_slice,
            valstep=1
        )
        self.slider_sagittal.label.set_fontsize(12)
        self.slider_sagittal.label.set_weight('bold')
        self.slider_sagittal.on_changed(self.update_sagittal)

        # Buttons - moved lower to avoid overlapping with sliders
        btn_y = 0.02
        btn_height = 0.04
        btn_width = 0.12

        self.btn_success = Button(plt.axes([0.1, btn_y, btn_width, btn_height]), 'Mark Success')
        self.btn_success.label.set_fontsize(11)
        self.btn_success.label.set_weight('bold')
        self.btn_success.on_clicked(self.mark_success)

        self.btn_fail = Button(plt.axes([0.23, btn_y, btn_width, btn_height]), 'Mark Fail')
        self.btn_fail.label.set_fontsize(11)
        self.btn_fail.label.set_weight('bold')
        self.btn_fail.on_clicked(self.mark_fail)

        self.btn_select_slice = Button(plt.axes([0.36, btn_y, btn_width, btn_height]), 'Toggle Slice')
        self.btn_select_slice.label.set_fontsize(11)
        self.btn_select_slice.label.set_weight('bold')
        self.btn_select_slice.on_clicked(self.toggle_slice_selection)

        self.btn_auto_select = Button(plt.axes([0.49, btn_y, btn_width, btn_height]), 'Auto Select')
        self.btn_auto_select.label.set_fontsize(11)
        self.btn_auto_select.label.set_weight('bold')
        self.btn_auto_select.on_clicked(self.auto_select_slices)

        self.btn_done = Button(plt.axes([0.62, btn_y, btn_width * 1.2, btn_height]), 'Done (Save & Next)')
        self.btn_done.label.set_fontsize(11)
        self.btn_done.label.set_weight('bold')
        self.btn_done.on_clicked(lambda x: plt.close(self.fig))

        # Label Correction Controls
        ax_controls_labels = plt.axes([0.75, 0.25, 0.15, 0.2])
        ax_controls_labels.axis('off')

        self.txt_mapping = ax_controls_labels.text(0, 1, "", fontsize=10, verticalalignment='top', family='monospace')
        self.update_mapping_text()

        self.btn_shift_up = Button(plt.axes([0.75, 0.20, 0.15, 0.04]), 'Shift Up (L1<-L2)')
        self.btn_shift_up.on_clicked(self.shift_labels_up)

        self.btn_shift_down = Button(plt.axes([0.75, 0.15, 0.15, 0.04]), 'Shift Down (L2<-L1)')
        self.btn_shift_down.on_clicked(self.shift_labels_down)

        self.btn_reset = Button(plt.axes([0.75, 0.10, 0.15, 0.04]), 'Reset Labels')
        self.btn_reset.on_clicked(self.reset_labels)

        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Navigate slices using sliders\n"
            "2. Click 'Toggle Slice' to select/deselect current axial slice\n"
            "3. Mark case as Success or Fail\n"
            "4. Click 'Done' to save and proceed to next case"
        )
        ax_controls.text(0.05, 0.7, instructions, fontsize=12, verticalalignment='top',
                         family='monospace', transform=ax_controls.transAxes,
                         weight='bold', color='black')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = []
        for name, mask in self.masks.items():
            if name == 'vertebrae_body':
                legend_elements.append(Patch(facecolor='#FF00FF', alpha=0.5, label='All Vertebrae (combined)'))
            elif name in VERTEBRAE_COLORS:
                color = VERTEBRAE_COLORS[name]
                legend_elements.append(Patch(facecolor=color, alpha=0.5, label=name.replace('vertebrae_', '')))

        if legend_elements:
            self.ax_axial.legend(
                handles=legend_elements,
                loc='upper right',
                framealpha=0.9,
                fontsize=11,
                edgecolor='black',
                facecolor='white'
            )

        # Initial display
        self.update_axial(self.axial_slice)
        self.update_sagittal(self.sagittal_slice)
        self.update_info_text()

        # Show plot (blocking) - wait for user to close window
        logger.info("Displaying interactive window...")
        plt.show(block=True)

        # Ensure figure is closed
        if self.fig:
            plt.close(self.fig)

        # Return results
        # Return results
        return {
            'patient_id': self.patient_id,
            'exam_date': self.exam_date,
            'is_successful': self.is_successful,
            'selected_slices': self.selected_slices.copy(),
            'average_hu': self.calculate_average_hu(self.selected_slices), # Overall average of selected slices
            'label_mapping': self.label_mapping.copy(),
            'vertebra_hu': {
                label: self.calculate_volumetric_hu(label) # Volumetric average (all pixels)
                for label in ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
            },
            'l1_trabecular_core_hu': self.calculate_volumetric_hu('vertebrae_L1_body_trabecular_core')
        }


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


def process_case_for_verification(
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
        from src.patient_manager import create_patient_output_dir
        # Use provided forced_study_id or fallback to folder name
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


def load_masks_for_verification(segmentation_dir: Path, ct_img: nib.Nifti1Image) -> Dict[str, np.ndarray]:
    """
    Load segmentation masks and resample them to match CT volume space.

    The vertebrae_body task creates a single vertebrae_body.nii.gz file.
    We'll load it as a single mask overlay.

    Args:
        segmentation_dir: Directory containing segmentation masks
        ct_img: CT NIfTI image (used as reference for resampling)

    Returns:
        Dictionary of vertebra_name -> mask array (resampled to CT space)
    """
    masks = {}
    ct_shape = ct_img.shape[:3]  # Get spatial dimensions only

    # First, try to load individual vertebrae masks (L1-L5) and bodies
    # User requested to see only bodies, so we prioritize those and skip full vertebrae if bodies exist
    # Actually, let's just load bodies as requested
    for vertebra in LUMBAR_BODIES:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            try:
                mask_img = load_segmentation_mask(mask_path)
                mask_shape = mask_img.shape[:3]

                # Resample mask to match CT space if shapes differ
                if mask_shape != ct_shape:
                    logger.info(f"Resampling {vertebra} mask from {mask_shape} to {ct_shape} to match CT space")
                    resampled_mask_img = resample_from_to(mask_img, ct_img, order=0)  # order=0 for nearest neighbor (preserves binary mask)
                    mask_data = resampled_mask_img.get_fdata()
                    logger.info(f"Resampled {vertebra} mask: sum={mask_data.sum():.0f}, max={mask_data.max()}")
                else:
                    mask_data = mask_img.get_fdata()
                    logger.info(f"{vertebra} mask already matches CT shape: {mask_shape}")

                masks[vertebra] = mask_data
            except Exception as e:
                logger.warning(f"Failed to load mask {mask_path}: {e}")

    # Check for L1 trabecular core
    core_name = 'vertebrae_L1_body_trabecular_core'
    core_mask_path = segmentation_dir / f"{core_name}.nii.gz"
    if core_mask_path.exists():
        try:
            logger.info(f"Loading {core_name} mask")
            mask_img = load_segmentation_mask(core_mask_path)
            mask_shape = mask_img.shape[:3]

            if mask_shape != ct_shape:
                logger.info(f"Resampling {core_name} mask from {mask_shape} to {ct_shape}")
                resampled_mask_img = resample_from_to(mask_img, ct_img, order=0)
                mask_data = resampled_mask_img.get_fdata()
            else:
                mask_data = mask_img.get_fdata()

            masks[core_name] = mask_data
        except Exception as e:
            logger.warning(f"Failed to load core mask {core_mask_path}: {e}")

    # If no individual masks found, try loading vertebrae_body (single combined mask)
    if not masks:
        vertebrae_body_path = segmentation_dir / "vertebrae_body.nii.gz"
        if vertebrae_body_path.exists():
            try:
                logger.info("Loading vertebrae_body mask (combined mask from vertebrae_body task)")
                mask_img = load_segmentation_mask(vertebrae_body_path)
                mask_shape = mask_img.shape[:3]

                # Resample mask to match CT space if shapes differ
                if mask_shape != ct_shape:
                    logger.info(f"Resampling vertebrae_body mask from {mask_shape} to {ct_shape} to match CT space")
                    resampled_mask_img = resample_from_to(mask_img, ct_img, order=0)  # order=0 for nearest neighbor
                    mask_data = resampled_mask_img.get_fdata()
                    logger.info(f"Resampled vertebrae_body mask: sum={mask_data.sum():.0f}, max={mask_data.max()}")
                else:
                    mask_data = mask_img.get_fdata()
                    logger.info(f"vertebrae_body mask already matches CT shape: {mask_shape}")

                masks['vertebrae_body'] = mask_data
            except Exception as e:
                logger.warning(f"Failed to load vertebrae_body mask {vertebrae_body_path}: {e}")

    return masks


def load_ct_for_verification(dicom_folder: Path, segmentation_dir: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load CT volume for verification.
    Always loads from DICOM to ensure we have the actual CT image (not the mask).

    Returns:
        Tuple of (3D CT volume array (HU values), CT NIfTI image object)
    """
    # Always load CT from DICOM - vertebrae_body.nii.gz is the mask, not the CT
    logger.info("Loading CT from DICOM series...")
    ct_img = load_dicom_series(dicom_folder)
    ct_volume = ct_img.get_fdata()
    logger.info(f"Loaded CT volume with shape: {ct_volume.shape}")
    return ct_volume, ct_img


def run_batch_verification(
    input_path: Path,
    output_csv: Path,
    output_base_dir: Optional[Path] = None,
    fast_segmentation: bool = False,
    device: str = 'gpu',
    window_level: int = 40,
    window_width: int = 400
):
    """
    Run the batch verification pipeline.

    1. Process ALL cases to generate segmentations (showing progress window).
    2. Then verify ALL cases interactively.
    """
    import tempfile

    patient_folders = find_patient_folders(input_path)
    if not patient_folders:
        raise ValueError(f"No patient folders found in {input_path}")

    logger.info(f"Found {len(patient_folders)} patient folder(s) to verify")

    # -------------------------------------------------------------------------
    # PART 1: SEGMENTATION OF ALL CASES
    # -------------------------------------------------------------------------

    if output_base_dir is None:
        output_base_dir = Path.cwd() / 'verification_output'
        logger.info(f"No output directory specified. Using default: {output_base_dir}")
    else:
        output_base_dir = Path(output_base_dir)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix='verification_temp_'))

    # List to store cases that were successfully processed
    # Each item: {'dicom_folder': Path, 'segmentation_dir': Path, 'patient_id': str, 'exam_date': str}
    processed_cases = []

    # Track study ID uniqueness
    study_counts: Dict[str, int] = {}

    # Launch Progress Window
    progress_window = ProgressWindow(len(patient_folders), title="Batch Segmentation Progress")

    logger.info("\n=== STARTING BATCH SEGMENTATION ===")

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

            logger.info(f"Calling process_case_for_verification for {patient_folder.name}...")
            # Process case
            segmentation_dir, patient_id, exam_date, was_skipped = process_case_for_verification(
                dicom_folder=patient_folder,
                output_base_dir=output_base_dir,
                temp_dir=temp_dir,
                fast_segmentation=fast_segmentation,
                device=device,
                skip_if_exists=True,
                forced_study_id=forced_study_id
            )
            logger.info(f"Returned from process_case_for_verification for {patient_id}. Skipped: {was_skipped}")

            success_msg = f"Case {patient_id} {'(Existing)' if was_skipped else '(New Segment)'}"
            progress_window.log(success_msg)

            processed_cases.append({
                'dicom_folder': patient_folder,
                'segmentation_dir': segmentation_dir,
                'patient_id': patient_id,
                'exam_date': exam_date
            })

        except BaseException as e:
            # Catch BaseException to trap SystemExit and KeyboardInterrupt
            err_msg = f"CRITICAL ERROR processing {patient_folder.name}: {e} (Type: {type(e).__name__})"
            logger.error(err_msg, exc_info=True)
            progress_window.log("ERROR: " + err_msg)

            # Re-raise KeyboardInterrupt to allow user to abort
            if isinstance(e, KeyboardInterrupt):
                raise

    progress_window.log("Segmentation Phase Complete. Closing progress window...")
    # Give a moment to read final log
    progress_window.root.update()
    time.sleep(2)
    progress_window.close()

    # CRITICAL: Switch backend to interactive for Phase 2
    logger.info("Phase 1 complete. Switching Matplotlib backend to interactive (TkAgg)...")
    try:
        # Close any lingering plots
        plt.close('all')
        plt.switch_backend('TkAgg')
        logger.info(f"Backend switched to: {plt.get_backend()}")
    except Exception as e:
        logger.error(f"Failed to switch backend: {e}")
        # Try fallbacks
        for backend in ['Qt5Agg', 'Qt4Agg']:
            try:
                plt.switch_backend(backend)
                logger.info(f"Fallback backend switched to: {plt.get_backend()}")
                break
            except:
                continue

    logger.info("\n=== SEGMENTATION COMPLETE ===")
    logger.info(f"Successfully processed {len(processed_cases)}/{len(patient_folders)} cases")

    if not processed_cases:
        logger.error("No cases were successfully processed. Exiting.")
        return

    # -------------------------------------------------------------------------
    # PART 2: INTERACTIVE VERIFICATION
    # -------------------------------------------------------------------------

    logger.info("\n=== STARTING INTERACTIVE VERIFICATION ===")
    all_results = []

    for i, case_data in enumerate(processed_cases, 1):
        dicom_folder = case_data['dicom_folder']
        segmentation_dir = case_data['segmentation_dir']
        patient_id = case_data['patient_id']
        exam_date = case_data['exam_date']

        logger.info(f"\nVerifying case {i}/{len(processed_cases)}: {patient_id}")

        try:
            # Load CT volume
            ct_volume, ct_img = load_ct_for_verification(dicom_folder, segmentation_dir)

            # Load segmentation masks
            masks = load_masks_for_verification(segmentation_dir, ct_img)

            if not masks:
                logger.warning(f"No segmentation masks found for {patient_id}")

            # Show interactive viewer
            logger.info("Opening viewer...")

            viewer = VerificationViewer(
                ct_volume=ct_volume,
                masks=masks,
                dicom_folder=dicom_folder,
                patient_id=patient_id,
                exam_date=exam_date,
                window_level=window_level,
                window_width=window_width
            )

            result = viewer.show()
            all_results.append(result)

            logger.info("Case verified successfully")

        except Exception as e:
            logger.error(f"Error verifying {patient_id}: {e}", exc_info=True)
            all_results.append({
                'patient_id': patient_id,
                'exam_date': exam_date,
                'is_successful': None,
                'selected_slices': [],
                'average_hu': None,
                'error': str(e)
            })

    # Save results
    logger.info("\n=== SAVING RESULTS ===")
    save_verification_results(all_results, output_csv)
    logger.info(f"Done. Results in {output_csv}")


def save_verification_results(results: List[Dict], output_csv: Path):
    """
    Save verification results to CSV.

    Args:
        results: List of result dictionaries
        output_csv: Path to output CSV file
    """
    # Prepare data for CSV
    csv_data = []

    for result in results:
        patient_id = result.get('patient_id', 'UNKNOWN')
        exam_date = result.get('exam_date', '')
        is_successful = result.get('is_successful')
        selected_slices = result.get('selected_slices', [])
        average_hu = result.get('average_hu')
        l1_core_hu = result.get('l1_trabecular_core_hu')

        # Format selected slices as comma-separated string
        slice_numbers = ','.join(map(str, selected_slices)) if selected_slices else ''

        csv_data.append({
            'Exam Date': exam_date,
            'Patient ID': patient_id,
            'Status': 'Success' if is_successful else ('Failed' if is_successful is False else 'Not Marked'),
            'Selected Slice Numbers': slice_numbers,
            'Average HU (All)': f"{average_hu:.2f}" if average_hu is not None else '',
            'L1 Trabecular Core HU': f"{l1_core_hu:.2f}" if l1_core_hu is not None and l1_core_hu != 0 else '',
            # Add per-vertebra columns
            **{f"{v} HU": f"{result.get('vertebra_hu', {}).get(v, 0.0):.2f}" for v in ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']},
            **{f"{v} Source": result.get('label_mapping', {}).get(v, '') for v in ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']}
        })

    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info(f"Saved {len(csv_data)} verification results to {output_csv}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch verification pipeline: Segment all, then Verify all',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input_path', type=Path, help='Input directory containing patient DICOM folders')
    parser.add_argument('output_dir', type=Path, help='Output directory where results and segmentations will be saved')

    parser.add_argument('--output-csv', type=Path, default=None, help='Output CSV file path')
    parser.add_argument('--fast', action='store_true', help='Use fast segmentation mode')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu', help='Device (default: gpu)')
    parser.add_argument('--window-level', type=int, default=40, help='Window level')
    parser.add_argument('--window-width', type=int, default=400, help='Window width')

    args = parser.parse_args()

    if not args.input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_path}")
        sys.exit(1)

    if args.output_csv is None:
        args.output_csv = args.output_dir / 'batch_results.csv'

    try:
        run_batch_verification(
            input_path=args.input_path,
            output_csv=args.output_csv,
            output_base_dir=args.output_dir,
            fast_segmentation=args.fast,
            device=args.device,
            window_level=args.window_level,
            window_width=args.window_width
        )
    except KeyboardInterrupt:
        logger.info("\nBatch pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
