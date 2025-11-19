"""
Interactive Verification Pipeline

Shows cases one by one with sagittal view, allows user to mark success/failure,
select slices for HU calculation, and saves validation results to CSV.

Usage:
    python verify_pipeline.py --input-dir /path/to/patients --output-csv validation_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib

# Force interactive backend - must be set before importing pyplot
# Try multiple backends in order of preference
backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg']
backend_set = False

for backend_name in backends_to_try:
    try:
        matplotlib.use(backend_name, force=True)
        backend_set = True
        logger.info(f"Using matplotlib backend: {backend_name}")
        break
    except (ImportError, ValueError):
        continue

if not backend_set:
    # Last resort: try to use any available GUI backend
    import matplotlib.pyplot as plt_backend_check
    current_backend = plt_backend_check.get_backend()
    if current_backend.lower() == 'agg':
        logger.warning("No interactive backend available. Install tkinter or PyQt5 for interactive plots.")
        logger.warning("Attempting to use default backend...")
    else:
        logger.info(f"Using default matplotlib backend: {current_backend}")

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from src.patient_manager import get_patient_metadata
from src.pipeline import find_patient_folders, process_single_patient
from verify_segmentation import load_dicom_series, load_segmentation_mask

# Color map for different vertebrae (matching verify_segmentation.py)
VERTEBRAE_COLORS = {
    'vertebrae_L1': '#FF0000',  # Red
    'vertebrae_L2': '#FF8C00',  # Dark Orange
    'vertebrae_L3': '#FFD700',  # Gold
    'vertebrae_L4': '#00FF00',  # Lime
    'vertebrae_L5': '#0000FF',  # Blue
}

LUMBAR_VERTEBRAE = ['vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5']


class VerificationViewer:
    """Interactive viewer for case-by-case verification with sagittal view."""
    
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
        self.btn_done = None
        
        # Display state
        self.show_selected = True
        
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
    
    def calculate_average_hu(self, slice_indices: List[int]) -> float:
        """
        Calculate average HU value for selected slices within masked regions only.
        
        Args:
            slice_indices: List of slice indices (axial slices)
            
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
                
                # Combine all masks to get total masked region
                combined_mask = np.zeros_like(ct_slice, dtype=bool)
                for mask in self.masks.values():
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
    
    def mark_success(self, event):
        """Mark case as successful."""
        self.is_successful = True
        self.update_info_text()
    
    def mark_fail(self, event):
        """Mark case as failed."""
        self.is_successful = False
        self.update_info_text()
    
    def update_info_text(self):
        """Update information text display."""
        if hasattr(self, 'info_text'):
            status = "SUCCESS" if self.is_successful else ("FAILED" if self.is_successful is False else "NOT MARKED")
            status_color = 'green' if self.is_successful else ('red' if self.is_successful is False else 'gray')
            
            selected_str = ', '.join(map(str, self.selected_slices)) if self.selected_slices else 'None'
            avg_hu = self.calculate_average_hu(self.selected_slices) if self.selected_slices else 0.0
            
            info = f"Patient ID: {self.patient_id}\n"
            info += f"Exam Date: {self.exam_date or 'N/A'}\n"
            info += f"Status: {status}\n"
            info += f"Selected Slices: {selected_str}\n"
            info += f"Average HU: {avg_hu:.1f}"
            
            self.info_text.set_text(info)
            # Use darker colors for better visibility
            if status_color == 'gray':
                status_color = 'black'
            elif status_color == 'green':
                status_color = 'darkgreen'
            elif status_color == 'red':
                status_color = 'darkred'
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
                logger.error("Please install tkinter or PyQt5:")
                logger.error("  - tkinter: Usually comes with Python, but may need: sudo apt-get install python3-tk (Linux)")
                logger.error("  - PyQt5: pip install PyQt5")
                raise RuntimeError("No interactive matplotlib backend available. Install tkinter or PyQt5.")
        
        if current_backend.lower() == 'agg':
            logger.error("Matplotlib backend is still 'Agg' after attempting to switch.")
            logger.error("This may be because another module has locked the backend.")
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
        
        self.btn_done = Button(plt.axes([0.49, btn_y, btn_width * 1.5, btn_height]), 'Done (Save & Next)')
        self.btn_done.label.set_fontsize(11)
        self.btn_done.label.set_weight('bold')
        self.btn_done.on_clicked(lambda x: plt.close(self.fig))
        
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
        return {
            'patient_id': self.patient_id,
            'exam_date': self.exam_date,
            'is_successful': self.is_successful,
            'selected_slices': self.selected_slices.copy(),
            'average_hu': self.calculate_average_hu(self.selected_slices) if self.selected_slices else None
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
    
    # Check for individual vertebrae masks (L1-L5)
    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            return True
    
    return False


def process_case_for_verification(
    dicom_folder: Path,
    output_base_dir: Path,
    temp_dir: Optional[Path] = None,
    fast_segmentation: bool = False,
    device: str = 'gpu',
    skip_if_exists: bool = True
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
        patient_output_dir = create_patient_output_dir(output_base_dir, patient_id)
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
            keep_temp_files=True  # Keep temp files for verification
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
    
    # First, try to load individual vertebrae masks (L1-L5)
    for vertebra in LUMBAR_VERTEBRAE:
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


def run_verification_pipeline(
    input_path: Path,
    output_csv: Path,
    output_base_dir: Optional[Path] = None,
    fast_segmentation: bool = False,
    device: str = 'gpu',
    window_level: int = 40,
    window_width: int = 400
):
    """
    Run the verification pipeline for all cases.
    
    Args:
        input_path: Path to input directory containing patient folders
        output_csv: Path to output CSV file for validation results
        output_base_dir: Base directory for processing outputs (if None, uses temp)
        fast_segmentation: Use fast segmentation mode
        device: Device for segmentation ('gpu' or 'cpu')
        window_level: Window level for CT display
        window_width: Window width for CT display
    """
    import tempfile
    
    # Find patient folders
    patient_folders = find_patient_folders(input_path)
    
    if not patient_folders:
        raise ValueError(f"No patient folders found in {input_path}")
    
    logger.info(f"Found {len(patient_folders)} patient folder(s) to verify")
    
    # Create output directory if not provided
    if output_base_dir is None:
        # Use a default directory in the current working directory
        output_base_dir = Path.cwd() / 'verification_output'
        logger.info(f"No output directory specified. Using default: {output_base_dir}")
    else:
        output_base_dir = Path(output_base_dir)
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")
    logger.info("Existing segmentations will be reused if found.")
    
    # Create temporary directory for processing (only for intermediate files)
    temp_dir = Path(tempfile.mkdtemp(prefix='verification_temp_'))
    
    # Results storage
    all_results = []
    
    # Process each case
    for i, patient_folder in enumerate(patient_folders, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing case {i}/{len(patient_folders)}: {patient_folder.name}")
        logger.info(f"{'='*60}")
        
        try:
            # Process case through pipeline (or use existing segmentations)
            segmentation_dir, patient_id, exam_date, was_skipped = process_case_for_verification(
                dicom_folder=patient_folder,
                output_base_dir=output_base_dir,
                temp_dir=temp_dir,
                fast_segmentation=fast_segmentation,
                device=device,
                skip_if_exists=True
            )
            
            if was_skipped:
                logger.info(f"Using existing segmentations for {patient_id}")
            else:
                logger.info(f"Completed segmentation for {patient_id}")
            
            # Load CT volume
            ct_volume, ct_img = load_ct_for_verification(patient_folder, segmentation_dir)
            
            # Load segmentation masks and resample to match CT space
            masks = load_masks_for_verification(segmentation_dir, ct_img)
            
            if not masks:
                logger.warning(f"No segmentation masks found for {patient_id}")
            
            # Show interactive viewer (blocks until user closes window)
            logger.info(f"\nOpening interactive viewer for case {i}/{len(patient_folders)}...")
            logger.info("Please validate the segmentation in the interactive window.")
            logger.info("Click 'Done' button when finished to proceed to next case.\n")
            
            viewer = VerificationViewer(
                ct_volume=ct_volume,
                masks=masks,
                dicom_folder=patient_folder,
                patient_id=patient_id,
                exam_date=exam_date,
                window_level=window_level,
                window_width=window_width
            )
            
            # This will block until user closes the window
            result = viewer.show()
            all_results.append(result)
            
            logger.info(f"\nCase {i} completed: Success={result['is_successful']}, "
                       f"Slices={result['selected_slices']}, Avg HU={result['average_hu']}")
            logger.info("Moving to next case...\n")
            
        except Exception as e:
            logger.error(f"Error processing case {patient_folder.name}: {e}", exc_info=True)
            # Add failed case to results
            metadata = get_patient_metadata(patient_folder)
            all_results.append({
                'patient_id': metadata.get('patient_id') or patient_folder.name,
                'exam_date': metadata.get('study_date'),
                'is_successful': None,
                'selected_slices': [],
                'average_hu': None,
                'error': str(e)
            })
    
    # Save results to CSV only after all cases are validated
    logger.info(f"\n{'='*60}")
    logger.info("All cases validated. Saving results to CSV...")
    save_verification_results(all_results, output_csv)
    
    logger.info(f"\n{'='*60}")
    logger.info("Verification pipeline complete!")
    logger.info(f"Processed {len(patient_folders)} cases")
    logger.info(f"Results saved to {output_csv}")
    logger.info(f"{'='*60}")


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
        
        # Format selected slices as comma-separated string
        slice_numbers = ','.join(map(str, selected_slices)) if selected_slices else ''
        
        csv_data.append({
            'Exam Date': exam_date,
            'Patient ID': patient_id,
            'Status': 'Success' if is_successful else ('Failed' if is_successful is False else 'Not Marked'),
            'Selected Slice Numbers': slice_numbers,
            'Average HU': f"{average_hu:.2f}" if average_hu is not None else ''
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"Saved {len(csv_data)} verification results to {output_csv}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interactive verification pipeline for CT segmentation cases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all cases in a directory
  python verify_pipeline.py --input-dir /path/to/patients --output-csv results.csv
  
  # Use CPU instead of GPU
  python verify_pipeline.py --input-dir /path/to/patients --output-csv results.csv --device cpu
  
  # Adjust CT window settings
  python verify_pipeline.py --input-dir /path/to/patients --output-csv results.csv --window-level 400 --window-width 1800
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing patient DICOM folders'
    )
    
    parser.add_argument(
        '--output-csv',
        type=Path,
        required=True,
        help='Output CSV file path for validation results'
    )
    
    parser.add_argument(
        '--output-base-dir',
        type=Path,
        default=None,
        help='Base directory for processing outputs (default: ./verification_output). '
             'If segmentations already exist in this directory, they will be reused to skip reprocessing.'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast segmentation mode'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['gpu', 'cpu'],
        default='gpu',
        help='Device for segmentation (default: gpu)'
    )
    
    parser.add_argument(
        '--window-level',
        type=int,
        default=40,
        help='Window level for CT display (HU, default: 40)'
    )
    
    parser.add_argument(
        '--window-width',
        type=int,
        default=400,
        help='Window width for CT display (HU, default: 400)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    try:
        run_verification_pipeline(
            input_path=args.input_dir,
            output_csv=args.output_csv,
            output_base_dir=args.output_base_dir,
            fast_segmentation=args.fast,
            device=args.device,
            window_level=args.window_level,
            window_width=args.window_width
        )
    except KeyboardInterrupt:
        logger.info("\nVerification pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

