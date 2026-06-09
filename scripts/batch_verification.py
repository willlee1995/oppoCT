"""
Batch Verification Script

Runs the INTERACTIVE VERIFICATION phase for already-segmented patient cases.
This script is designed to be run after `batch_segmentation.py` has completed.

Usage:
    python scripts/batch_verification.py data/processed out/test_batch --output-csv out/results.csv
"""

print("Initializing Batch Verification Pipeline...", flush=True)

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up logging early to capture imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Imports starting...")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
# Use TkAgg for interactive display
matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider
from nibabel.processing import resample_from_to

from src.dicom_processor import dicom_to_nifti
from src.patient_manager import get_patient_metadata, create_patient_output_dir
from src.pipeline import find_patient_folders
from verify_segmentation import load_segmentation_mask

logger.info("All imports complete.")

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
    'vertebrae_L1_body_trabecular_core': '#00FF7F',  # Spring Green
}

LUMBAR_VERTEBRAE = ['vertebrae_T11', 'vertebrae_T12', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5']
LUMBAR_BODIES = [f"{v}_body" for v in LUMBAR_VERTEBRAE]


def _get_fdata_float32(img: nib.Nifti1Image) -> np.ndarray:
    """Return image data as float32 without asking nibabel to cache a float64 copy."""
    try:
        return img.get_fdata(dtype=np.float32, caching="unchanged")
    except TypeError:
        return img.get_fdata(dtype=np.float32)


def _mask_data_as_bool(img: nib.Nifti1Image) -> np.ndarray:
    """Return a compact binary mask array."""
    return _get_fdata_float32(img) > 0


def _array_size_mb(array: Optional[np.ndarray]) -> float:
    if array is None:
        return 0.0
    return float(array.nbytes) / (1024.0 * 1024.0)


def _process_rss_mb() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def log_memory_checkpoint(label: str) -> None:
    rss_mb = _process_rss_mb()
    if rss_mb is not None:
        logger.info("Memory checkpoint %s: RSS %.1f MB", label, rss_mb)


def log_viewer_array_memory(ct_volume: np.ndarray, masks: Dict[str, np.ndarray]) -> None:
    mask_mb = sum(_array_size_mb(mask) for mask in masks.values())
    logger.info(
        "Viewer arrays: CT %.1f MB (%s), masks %.1f MB across %d mask(s)",
        _array_size_mb(ct_volume),
        ct_volume.dtype,
        mask_mb,
        len(masks),
    )


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

        # Volume dimensions are the NIfTI voxel axes, typically (left-right, anterior-posterior, inferior-superior).
        self.axial_shape = ct_volume.shape

        # Current slice indices
        self.axial_slice = ct_volume.shape[2] // 2  # Depth axis
        self.sagittal_slice = ct_volume.shape[0] // 2  # Left-right axis

        # Case status
        self.is_successful: Optional[bool] = None

        # UI elements
        self.fig = None
        self.ax_axial = None
        self.ax_sagittal = None
        self.slider_axial = None
        self.slider_sagittal = None
        self.info_text = None
        self.txt_mapping = None
        self._sagittal_crosshair = None
        self._widgets = []
        self._widget_cids = []
        self._hu_cache: Dict[str, float] = {}
        self._rgba_cache: Dict[str, Tuple[float, float, float, float]] = {}

        # Label mapping
        self.label_mapping = {v.replace('vertebrae_', '').replace('_body', ''): v for v in LUMBAR_BODIES}
        for v in LUMBAR_BODIES:
            short_name = v.replace('vertebrae_', '').replace('_body', '')
            if short_name not in self.label_mapping:
                self.label_mapping[short_name] = v

    @staticmethod
    def _format_axial_slice(image_slice: np.ndarray) -> np.ndarray:
        """Display axial slices with left-right on the horizontal axis."""
        return image_slice.T

    @staticmethod
    def _format_sagittal_slice(image_slice: np.ndarray) -> np.ndarray:
        """Display sagittal slices with superior-inferior on the vertical axis."""
        return np.rot90(image_slice, k=-1)

    @staticmethod
    def _mask_pixels(mask_slice: np.ndarray) -> np.ndarray:
        """Return a boolean view when masks are already compact booleans."""
        return mask_slice if mask_slice.dtype == np.bool_ else mask_slice > 0

    def calculate_sagittal_view(self, slice_idx: int) -> np.ndarray:
        """Extract a true sagittal slice from CT volume."""
        if slice_idx < 0 or slice_idx >= self.axial_shape[0]:
            slice_idx = max(0, min(slice_idx, self.axial_shape[0] - 1))
        sagittal_slice = self.ct_volume[slice_idx, :, :]
        return sagittal_slice

    def calculate_volumetric_hu(self, vertebra_name: str) -> float:
        """Calculate average HU value for the entire volume of a specific vertebra."""
        mask_name = self.label_mapping.get(vertebra_name, vertebra_name)
        if not mask_name or mask_name not in self.masks:
            return 0.0

        if mask_name in self._hu_cache:
            return self._hu_cache[mask_name]

        mask = self.masks[mask_name]
        if mask.shape != self.ct_volume.shape:
            logger.warning(f"Shape mismatch for {mask_name}: {mask.shape} vs {self.ct_volume.shape}")
            return 0.0

        mask_bool = mask if mask.dtype == np.bool_ else mask > 0
        total_count = int(np.count_nonzero(mask_bool))

        if total_count == 0:
            self._hu_cache[mask_name] = 0.0
            return 0.0

        total_sum = np.sum(self.ct_volume, where=mask_bool, dtype=np.float64)
        average_hu = float(total_sum / total_count)
        self._hu_cache[mask_name] = average_hu
        return average_hu

    def calculate_average_hu(self, slice_indices: List[int], vertebra_name: Optional[str] = None) -> float:
        """Calculate average HU value for selected slices."""
        if not slice_indices:
            return 0.0

        total_sum = 0.0
        total_count = 0
        for slice_idx in slice_indices:
            if 0 <= slice_idx < self.axial_shape[2]:
                ct_slice = self.ct_volume[:, :, slice_idx]

                if vertebra_name:
                    mask_name = self.label_mapping.get(vertebra_name)
                    if not mask_name or mask_name not in self.masks:
                        continue
                    masks_to_use = [self.masks[mask_name]]
                else:
                    masks_to_use = self.masks.values()

                combined_mask = np.zeros_like(ct_slice, dtype=bool)
                for mask in masks_to_use:
                    if len(mask.shape) == 3 and slice_idx < mask.shape[2]:
                        mask_slice = mask[:, :, slice_idx]
                        if mask_slice.shape == ct_slice.shape:
                            combined_mask |= self._mask_pixels(mask_slice)

                if np.any(combined_mask):
                    total_sum += float(np.sum(ct_slice, where=combined_mask, dtype=np.float64))
                    total_count += int(np.count_nonzero(combined_mask))

        if total_count == 0:
            return 0.0
        return float(total_sum / total_count)

    def window_ct(self, ct_slice: np.ndarray) -> np.ndarray:
        """Apply window/level to CT slice for display."""
        ct_min = self.window_level - self.window_width / 2
        ct_max = self.window_level + self.window_width / 2
        ct_display = np.clip(ct_slice, ct_min, ct_max).astype(np.float32, copy=False)
        if ct_max > ct_min:
            ct_display = (ct_display - ct_min) / np.float32(ct_max - ct_min)
        else:
            ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-10)
        ct_display = np.clip(ct_display, 0.0, 1.0).astype(np.float32, copy=False)
        return ct_display

    def _rgba_for_color(self, color: str, alpha: float = 0.4) -> Tuple[float, float, float, float]:
        cache_key = f"{color}:{alpha}"
        if cache_key not in self._rgba_cache:
            self._rgba_cache[cache_key] = (
                int(color[1:3], 16) / 255.0,
                int(color[3:5], 16) / 255.0,
                int(color[5:7], 16) / 255.0,
                alpha,
            )
        return self._rgba_cache[cache_key]

    def update_axial(self, slice_idx):
        """Update axial view with segmentation overlays."""
        slice_idx = int(slice_idx)
        self.axial_slice = slice_idx
        self.ax_axial.clear()

        ct_slice = self.ct_volume[:, :, slice_idx]
        ct_display = self._format_axial_slice(self.window_ct(ct_slice))

        self.ax_axial.imshow(ct_display, cmap='gray', origin='upper', interpolation='bilinear',
                            vmin=0.0, vmax=1.0)

        present_vertebrae = []
        overlay = np.zeros((*ct_display.shape, 4), dtype=np.float32)
        overlay_has_pixels = False
        contour_specs = []
        for vertebra_name, mask in self.masks.items():
            if len(mask.shape) == 3 and slice_idx < mask.shape[2]:
                mask_slice = mask[:, :, slice_idx]
            elif len(mask.shape) == 2:
                mask_slice = mask if mask.shape == ct_slice.shape else np.zeros_like(ct_slice)
            else:
                continue

            mask_slice = self._format_axial_slice(mask_slice)

            mask_pixels = self._mask_pixels(mask_slice)

            if np.any(mask_pixels):
                if vertebra_name == 'vertebrae_body':
                    color = '#FF00FF'
                    present_vertebrae.append('All Vertebrae')
                else:
                    color = VERTEBRAE_COLORS.get(vertebra_name, '#00FFFF')
                    present_vertebrae.append(vertebra_name.replace('vertebrae_', ''))

                overlay[mask_pixels] = self._rgba_for_color(color)
                overlay_has_pixels = True
                contour_specs.append((mask_slice, color))

        if overlay_has_pixels:
            self.ax_axial.imshow(overlay, origin='upper', interpolation='nearest')
            for mask_slice, color in contour_specs:
                self.ax_axial.contour(mask_slice, levels=[0.5], colors=[color], linewidths=2, alpha=0.8)

        title = f'Axial Slice {slice_idx} / {self.axial_shape[2] - 1}'
        if present_vertebrae:
            title += f' - {", ".join(present_vertebrae)}'
        self.ax_axial.set_title(title, fontsize=14, weight='bold', color='black',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax_axial.axis('off')

        if self._sagittal_crosshair is not None:
            self._sagittal_crosshair.set_ydata([self.axial_slice, self.axial_slice])
        self.fig.canvas.draw_idle()

    def update_sagittal(self, slice_idx):
        """Update sagittal view with segmentation overlays."""
        slice_idx = int(slice_idx)
        self.sagittal_slice = slice_idx
        self.ax_sagittal.clear()
        self._sagittal_crosshair = None

        sagittal_slice = self.calculate_sagittal_view(slice_idx)
        sagittal_display = self.window_ct(sagittal_slice)
        sagittal_display = self._format_sagittal_slice(sagittal_display)

        self.ax_sagittal.imshow(sagittal_display, cmap='gray', origin='lower', interpolation='bilinear',
                               aspect='auto', vmin=0.0, vmax=1.0)

        overlay = np.zeros((*sagittal_display.shape, 4), dtype=np.float32)
        overlay_has_pixels = False
        contour_specs = []
        for vertebra_name, mask in self.masks.items():
            if len(mask.shape) != 3:
                continue
            if mask.shape != self.ct_volume.shape:
                continue
            if slice_idx < 0 or slice_idx >= mask.shape[0]:
                continue

            mask_sagittal = mask[slice_idx, :, :]
            if mask_sagittal.shape != sagittal_slice.shape:
                if mask_sagittal.size == sagittal_slice.size:
                    mask_sagittal = mask_sagittal.reshape(sagittal_slice.shape)
                else:
                    continue

            mask_sagittal = self._format_sagittal_slice(mask_sagittal)

            mask_pixels = self._mask_pixels(mask_sagittal)

            if np.any(mask_pixels):
                if vertebra_name == 'vertebrae_body':
                    color = '#FF00FF'
                else:
                    color = VERTEBRAE_COLORS.get(vertebra_name, '#00FFFF')

                overlay[mask_pixels] = self._rgba_for_color(color)
                overlay_has_pixels = True
                contour_specs.append((mask_sagittal, color))

        if overlay_has_pixels:
            self.ax_sagittal.imshow(overlay, origin='lower', interpolation='nearest', aspect='auto')
            for mask_sagittal, color in contour_specs:
                self.ax_sagittal.contour(mask_sagittal, levels=[0.5], colors=[color], linewidths=2, alpha=0.8)

        if 0 <= self.axial_slice < self.axial_shape[2]:
            self._sagittal_crosshair = self.ax_sagittal.axhline(
                y=self.axial_slice, color='cyan', linewidth=3, alpha=0.9, linestyle='--'
            )

        title = f'Sagittal Slice {slice_idx} / {self.axial_shape[0] - 1}'
        self.ax_sagittal.set_title(title, fontsize=14, weight='bold', color='black',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax_sagittal.axis('off')
        self.fig.canvas.draw_idle()

    def mark_success(self, event):
        """Mark case as successful."""
        self.is_successful = True
        self.update_info_text()

    def mark_fail(self, event):
        """Mark case as failed."""
        self.is_successful = False
        self.update_info_text()

    def shift_labels_up(self, event):
        """Shift labels UP (cranial)."""
        ordered_labels = ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
        new_mapping = {}
        for i, target in enumerate(ordered_labels):
            if i + 1 < len(ordered_labels):
                next_default_mask = f"vertebrae_{ordered_labels[i+1]}_body"
                new_mapping[target] = next_default_mask
            else:
                new_mapping[target] = None
        self.label_mapping.update(new_mapping)
        self.update_info_text()
        self.update_mapping_text()

    def shift_labels_down(self, event):
        """Shift labels DOWN (caudal)."""
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
        if self.txt_mapping is not None:
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
        if self.info_text is not None:
            status = "SUCCESS" if self.is_successful else ("FAILED" if self.is_successful is False else "NOT MARKED")

            l1_body_hu = self.calculate_volumetric_hu('vertebrae_L1_body')
            l1_core_hu = self.calculate_volumetric_hu('vertebrae_L1_body_trabecular_core')

            info = f"Patient ID: {self.patient_id}\n"
            info += f"Exam Date: {self.exam_date or 'N/A'}\n"
            info += f"Status: {status}\n"

            if l1_body_hu != 0 or l1_core_hu != 0:
                info += f"L1 Body HU: {l1_body_hu:.1f}\n"
                info += f"L1 Core HU: {l1_core_hu:.1f}"
                if l1_core_hu > l1_body_hu and l1_body_hu != 0:
                     info += f"\nWARNING: Core > Body HU!"

            self.info_text.set_text(info)
            status_color = 'black'
            if self.is_successful:
                status_color = 'darkgreen'
            elif self.is_successful is False:
                status_color = 'darkred'
            if "WARNING" in info and status_color == 'black':
                 status_color = 'red'
            self.info_text.set_color(status_color)
            self.fig.canvas.draw_idle()

    def close(self) -> None:
        """Release Matplotlib handles and large arrays retained by the viewer."""
        for widget, callback_id in self._widget_cids:
            try:
                widget.disconnect(callback_id)
            except Exception:
                pass
        self._widget_cids.clear()
        self._widgets.clear()

        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            try:
                self.fig.clf()
            except Exception:
                pass

        self.fig = None
        self.ax_axial = None
        self.ax_sagittal = None
        self.slider_axial = None
        self.slider_sagittal = None
        self.info_text = None
        self.txt_mapping = None
        self._sagittal_crosshair = None
        self._rgba_cache.clear()
        self._hu_cache.clear()
        self.ct_volume = None
        self.masks = {}

    def show(self, tk_root=None) -> Dict:
        """Display the interactive viewer and return results.

        When ``tk_root`` is the running Tkinter root (e.g. batch GUI), use a
        non-blocking show plus a cooperative event loop so the figure window
        actually appears while the main app is already in ``mainloop()``.
        Calling ``plt.show(block=True)`` from a Tk button handler often yields
        no visible window or an immediate return on some setups.
        """
        try:
            return self._show_impl(tk_root=tk_root)
        except Exception:
            self.close()
            raise

    def _show_impl(self, tk_root=None) -> Dict:
        current_backend = plt.get_backend()
        logger.info(f"Current matplotlib backend: {current_backend}")

        if current_backend.lower() == "agg":
            try:
                import tkinter  # noqa: F401

                logger.info("tkinter is available, attempting to switch to TkAgg backend...")
                matplotlib.use("TkAgg", force=True)
                plt.close("all")
                current_backend = plt.get_backend()
                logger.info(f"Backend after switch: {current_backend}")
            except ImportError:
                logger.error("tkinter is not available. Cannot use TkAgg backend.")
                raise RuntimeError(
                    "No interactive matplotlib backend available. Install tkinter or PyQt5."
                ) from None

        if current_backend.lower() == "agg":
            logger.error("Matplotlib backend is still 'Agg' after attempting to switch.")
            raise RuntimeError(
                "Cannot use interactive matplotlib backend. Backend is locked to 'Agg'."
            )

        logger.info(f"Using backend: {current_backend} for interactive display")

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Verification: {self.patient_id}', fontsize=18, weight='bold', color='black')

        self.ax_axial = plt.subplot(2, 2, 1)
        self.ax_sagittal = plt.subplot(2, 2, 2)

        ax_info = plt.subplot(2, 2, 3)
        ax_info.axis('off')
        self.info_text = ax_info.text(0.1, 0.5, '', fontsize=13, verticalalignment='center',
                                      family='monospace', weight='bold', color='black')

        ax_controls = plt.subplot(2, 2, 4)
        ax_controls.axis('off')

        ax_slider_axial = plt.axes([0.1, 0.08, 0.35, 0.03])
        ax_slider_sagittal = plt.axes([0.55, 0.08, 0.35, 0.03])

        self.slider_axial = Slider(ax_slider_axial, 'Axial Slice', 0, self.axial_shape[2] - 1,
                                   valinit=self.axial_slice, valstep=1)
        self.slider_axial.label.set_fontsize(12)
        self.slider_axial.label.set_weight('bold')
        self._widget_cids.append((self.slider_axial, self.slider_axial.on_changed(self.update_axial)))

        self.slider_sagittal = Slider(ax_slider_sagittal, 'Sagittal Slice', 0, self.axial_shape[0] - 1,
                                      valinit=self.sagittal_slice, valstep=1)
        self.slider_sagittal.label.set_fontsize(12)
        self.slider_sagittal.label.set_weight('bold')
        self._widget_cids.append((self.slider_sagittal, self.slider_sagittal.on_changed(self.update_sagittal)))

        btn_y = 0.02
        btn_height = 0.04
        btn_width = 0.12

        btn_success = Button(plt.axes([0.1, btn_y, btn_width, btn_height]), 'Mark Success')
        btn_success.label.set_fontsize(11)
        btn_success.label.set_weight('bold')
        self._widget_cids.append((btn_success, btn_success.on_clicked(self.mark_success)))

        btn_fail = Button(plt.axes([0.23, btn_y, btn_width, btn_height]), 'Mark Fail')
        btn_fail.label.set_fontsize(11)
        btn_fail.label.set_weight('bold')
        self._widget_cids.append((btn_fail, btn_fail.on_clicked(self.mark_fail)))

        btn_done = Button(plt.axes([0.36, btn_y, btn_width * 1.2, btn_height]), 'Done (Save & Next)')
        btn_done.label.set_fontsize(11)
        btn_done.label.set_weight('bold')
        self._widget_cids.append((btn_done, btn_done.on_clicked(lambda x: plt.close(self.fig))))

        ax_controls_labels = plt.axes([0.75, 0.25, 0.15, 0.2])
        ax_controls_labels.axis('off')
        self.txt_mapping = ax_controls_labels.text(0, 1, "", fontsize=10, verticalalignment='top', family='monospace')
        self.update_mapping_text()

        btn_shift_up = Button(plt.axes([0.75, 0.20, 0.15, 0.04]), 'Shift Up (L1<-L2)')
        self._widget_cids.append((btn_shift_up, btn_shift_up.on_clicked(self.shift_labels_up)))

        btn_shift_down = Button(plt.axes([0.75, 0.15, 0.15, 0.04]), 'Shift Down (L2<-L1)')
        self._widget_cids.append((btn_shift_down, btn_shift_down.on_clicked(self.shift_labels_down)))

        btn_reset = Button(plt.axes([0.75, 0.10, 0.15, 0.04]), 'Reset Labels')
        self._widget_cids.append((btn_reset, btn_reset.on_clicked(self.reset_labels)))
        self._widgets = [
            self.slider_axial,
            self.slider_sagittal,
            btn_success,
            btn_fail,
            btn_done,
            btn_shift_up,
            btn_shift_down,
            btn_reset,
        ]

        instructions = (
            "Instructions:\n"
            "1. Navigate slices using sliders\n"
            "2. Mark case as Success or Fail\n"
            "3. Click 'Done' to save and proceed to next case"
        )
        ax_controls.text(0.05, 0.7, instructions, fontsize=12, verticalalignment='top',
                         family='monospace', transform=ax_controls.transAxes,
                         weight='bold', color='black')

        legend_elements = []
        for name, mask in self.masks.items():
            if name == 'vertebrae_body':
                legend_elements.append(Patch(facecolor='#FF00FF', alpha=0.5, label='All Vertebrae (combined)'))
            elif name in VERTEBRAE_COLORS:
                color = VERTEBRAE_COLORS[name]
                legend_elements.append(Patch(facecolor=color, alpha=0.5, label=name.replace('vertebrae_', '')))

        if legend_elements:
            self.ax_axial.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=11,
                                edgecolor='black', facecolor='white')

        self.update_axial(self.axial_slice)
        self.update_sagittal(self.sagittal_slice)
        self.update_info_text()

        logger.info("Displaying interactive window...")
        if tk_root is not None:
            plt.show(block=False)
            try:
                mgr_win = self.fig.canvas.manager.window
                mgr_win.lift()
                try:
                    mgr_win.attributes("-topmost", True)
                    mgr_win.attributes("-topmost", False)
                except Exception:
                    pass
                mgr_win.focus_force()
            except Exception:
                logger.debug("Could not lift matplotlib viewer window", exc_info=True)

            try:
                while plt.fignum_exists(self.fig.number):
                    tk_root.update()
                    tk_root.update_idletasks()
                    try:
                        self.fig.canvas.flush_events()
                    except Exception:
                        break
                    plt.pause(0.02)
            except Exception:
                logger.debug("Viewer event loop ended with exception", exc_info=True)
        else:
            plt.show(block=True)

        if self.fig:
            plt.close(self.fig)

        result = {
            'patient_id': self.patient_id,
            'exam_date': self.exam_date,
            'is_successful': self.is_successful,
            'selected_slices': [],
            'average_hu': None,
            'label_mapping': self.label_mapping.copy(),
            'vertebra_hu': {
                label: self.calculate_volumetric_hu(label)
                for label in ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
            },
            'l1_trabecular_core_hu': self.calculate_volumetric_hu('vertebrae_L1_body_trabecular_core')
        }
        self.close()
        return result


def check_segmentations_exist(segmentation_dir: Path) -> bool:
    """Check if segmentation files already exist in the directory."""
    if not segmentation_dir.exists():
        return False

    vertebrae_body_path = segmentation_dir / "vertebrae_body.nii.gz"
    if vertebrae_body_path.exists():
        return True

    for vertebra in LUMBAR_VERTEBRAE:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            return True
        body_path = segmentation_dir / f"{vertebra}_body.nii.gz"
        if body_path.exists():
            return True

    return False


TRABECULAR_CORE_FILENAME = "vertebrae_L1_body_trabecular_core.nii.gz"


def assess_trabecular_core(segmentation_dir: Path) -> Tuple[bool, str]:
    """
    Return (ok, detail) for L1 trabecular core segmentation output.

    Checks that the mask file exists and contains at least one foreground voxel.
    """
    if not segmentation_dir.exists():
        return False, "segmentation directory not found"

    core_path = segmentation_dir / TRABECULAR_CORE_FILENAME
    if not core_path.exists():
        return False, f"missing {TRABECULAR_CORE_FILENAME}"

    try:
        mask_img = load_segmentation_mask(core_path)
        data = _mask_data_as_bool(mask_img)
        if np.count_nonzero(data) == 0:
            return False, "trabecular core mask is empty"
    except Exception as exc:
        return False, f"failed to read trabecular core mask: {exc}"

    return True, "ok"


def segmentation_outputs_complete(segmentation_dir: Path) -> bool:
    """True when segmentation outputs include a non-empty L1 trabecular core mask."""
    ok, _ = assess_trabecular_core(segmentation_dir)
    return ok


def load_masks_for_verification(segmentation_dir: Path, ct_img: nib.Nifti1Image) -> Dict[str, np.ndarray]:
    """Load segmentation masks and resample them to match CT volume space."""
    masks = {}
    ct_shape = ct_img.shape[:3]

    def mask_matches_ct(mask_img: nib.Nifti1Image) -> bool:
        shape_match = mask_img.shape[:3] == ct_shape
        affine_match = np.allclose(mask_img.affine, ct_img.affine, atol=1e-3)
        return shape_match and affine_match

    def mask_data_in_ct_space(mask_img: nib.Nifti1Image, name: str) -> np.ndarray:
        if mask_matches_ct(mask_img):
            return _mask_data_as_bool(mask_img)
        logger.info(
            "Resampling %s mask from shape %s to CT shape %s",
            name,
            mask_img.shape[:3],
            ct_shape,
        )
        resampled_img = resample_from_to(mask_img, ct_img, order=0)
        return _mask_data_as_bool(resampled_img)

    for vertebra in LUMBAR_BODIES:
        mask_path = segmentation_dir / f"{vertebra}.nii.gz"
        if mask_path.exists():
            try:
                mask_img = load_segmentation_mask(mask_path)
                masks[vertebra] = mask_data_in_ct_space(mask_img, vertebra)
            except Exception as e:
                logger.warning(f"Failed to load mask {mask_path}: {e}")

    core_name = 'vertebrae_L1_body_trabecular_core'
    core_mask_path = segmentation_dir / f"{core_name}.nii.gz"
    if core_mask_path.exists():
        try:
            mask_img = load_segmentation_mask(core_mask_path)
            masks[core_name] = mask_data_in_ct_space(mask_img, core_name)
        except Exception as e:
            logger.warning(f"Failed to load core mask {core_mask_path}: {e}")

    if not masks:
        vertebrae_body_path = segmentation_dir / "vertebrae_body.nii.gz"
        if vertebrae_body_path.exists():
            try:
                mask_img = load_segmentation_mask(vertebrae_body_path)
                masks['vertebrae_body'] = mask_data_in_ct_space(mask_img, 'vertebrae_body')
            except Exception as e:
                logger.warning(f"Failed to load vertebrae_body mask: {e}")

    return masks


def load_ct_for_verification(
    dicom_folder: Path,
    segmentation_dir: Path,
    series_instance_uid: Optional[str] = None,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load CT volume for verification."""
    suid = (series_instance_uid or "").strip()
    if suid:
        logger.info("Loading CT from DICOM (series %s)...", suid[:20] + ("…" if len(suid) > 20 else ""))
    else:
        logger.info("Loading CT from DICOM series...")
    ct_img, _ = dicom_to_nifti(
        dicom_folder,
        output_path=None,
        series_instance_uid=suid or None,
    )
    ct_volume = _get_fdata_float32(ct_img)
    logger.info(f"Loaded CT volume with shape: {ct_volume.shape}, dtype: {ct_volume.dtype}")
    return ct_volume, ct_img


def save_verification_results(results: List[Dict], output_csv: Path):
    """Save verification results to CSV."""
    csv_data = []

    for result in results:
        patient_id = result.get('patient_id', 'UNKNOWN')
        exam_date = result.get('exam_date', '')
        is_successful = result.get('is_successful')
        selected_slices = result.get('selected_slices', [])
        average_hu = result.get('average_hu')
        l1_core_hu = result.get('l1_trabecular_core_hu')

        slice_numbers = ','.join(map(str, selected_slices)) if selected_slices else ''

        csv_data.append({
            'Exam Date': exam_date,
            'Patient ID': patient_id,
            'Status': 'Success' if is_successful else ('Failed' if is_successful is False else 'Not Marked'),
            'Selected Slice Numbers': slice_numbers,
            'Average HU (All)': f"{average_hu:.2f}" if average_hu is not None else '',
            'L1 Trabecular Core HU': f"{l1_core_hu:.2f}" if l1_core_hu is not None and l1_core_hu != 0 else '',
            **{f"{v} HU": f"{result.get('vertebra_hu', {}).get(v, 0.0):.2f}" for v in ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']},
            **{f"{v} Source": result.get('label_mapping', {}).get(v, '') for v in ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']}
        })

    df = pd.DataFrame(csv_data)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(csv_data)} verification results to {output_csv}")


def run_batch_verification(
    input_path: Path,
    output_base_dir: Path,
    output_csv: Path,
    window_level: int = 40,
    window_width: int = 400
):
    """
    Run the batch verification pipeline (Phase 2 only).

    Iterates over patient folders on disk and verifies segmentations interactively.
    """
    patient_folders = find_patient_folders(input_path)
    if not patient_folders:
        raise ValueError(f"No patient folders found in {input_path}")

    logger.info(f"Found {len(patient_folders)} patient folder(s)")
    output_base_dir = Path(output_base_dir)

    # Find cases that have segmentations
    cases_to_verify = []
    for patient_folder in patient_folders:
        try:
            metadata = get_patient_metadata(patient_folder)
            patient_id = metadata['patient_id'] or patient_folder.name
            exam_date = metadata['study_date']
            study_folder_name = patient_folder.name

            patient_output_dir = create_patient_output_dir(output_base_dir, patient_id, study_id=study_folder_name)
            segmentation_dir = patient_output_dir / 'segmentations'

            if check_segmentations_exist(segmentation_dir):
                cases_to_verify.append({
                    'dicom_folder': patient_folder,
                    'segmentation_dir': segmentation_dir,
                    'patient_id': patient_id,
                    'exam_date': exam_date
                })
            else:
                logger.warning(f"No segmentations found for {patient_id} at {segmentation_dir}, skipping.")
        except Exception as e:
            logger.error(f"Error processing {patient_folder}: {e}")

    logger.info(f"Found {len(cases_to_verify)} cases with segmentations to verify")

    if not cases_to_verify:
        logger.error("No cases found with segmentations. Run batch_segmentation.py first.")
        return

    logger.info("\n=== STARTING INTERACTIVE VERIFICATION ===")
    all_results = []

    for i, case_data in enumerate(cases_to_verify, 1):
        dicom_folder = case_data['dicom_folder']
        segmentation_dir = case_data['segmentation_dir']
        patient_id = case_data['patient_id']
        exam_date = case_data['exam_date']

        logger.info(f"\nVerifying case {i}/{len(cases_to_verify)}: {patient_id}")

        viewer = None
        ct_volume = None
        ct_img = None
        masks = {}
        try:
            log_memory_checkpoint(f"before loading viewer case {i}")
            ct_volume, ct_img = load_ct_for_verification(dicom_folder, segmentation_dir)
            masks = load_masks_for_verification(segmentation_dir, ct_img)
            log_viewer_array_memory(ct_volume, masks)
            log_memory_checkpoint(f"after loading viewer case {i}")

            if not masks:
                logger.warning(f"No segmentation masks found for {patient_id}")

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
        finally:
            if viewer is not None:
                viewer.close()
            del viewer, ct_volume, ct_img, masks
            gc.collect()
            log_memory_checkpoint(f"after closing viewer case {i}")

    logger.info("\n=== SAVING RESULTS ===")
    save_verification_results(all_results, output_csv)
    logger.info(f"Done. Results in {output_csv}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch verification pipeline: Verify already-segmented patients interactively',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input_path', type=Path, help='Input directory containing patient DICOM folders')
    parser.add_argument('output_dir', type=Path, help='Output directory where segmentations are saved')

    parser.add_argument('--output-csv', type=Path, default=None, help='Output CSV file path')
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
            output_base_dir=args.output_dir,
            output_csv=args.output_csv,
            window_level=args.window_level,
            window_width=args.window_width
        )
    except KeyboardInterrupt:
        logger.info("\nBatch verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
