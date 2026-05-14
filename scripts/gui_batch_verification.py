"""
Tkinter batch workflow for CSV-driven segmentation and verification.

Steps: (1) create batch CSV from DICOM folders, (2) load CSV and assign a DICOM series
per study row, (3) run batch segmentation, (4) verify and update QC columns in the CSV.

This module intentionally orchestrates existing pipeline and viewer code only:
- segmentation is delegated to src.pipeline.process_single_patient()
- verification uses batch_verification.VerificationViewer and its loaders unchanged
"""

from __future__ import annotations

# Matplotlib must select TkAgg before pyplot (or tkinter) initializes the GUI toolkit,
# otherwise the viewer can run on Agg and plt.show() returns without a window.
import matplotlib

matplotlib.use("TkAgg", force=True)

import csv
import logging
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_verification import (  # noqa: E402
    VerificationViewer,
    check_segmentations_exist,
    load_ct_for_verification,
    load_masks_for_verification,
)
from src.dicom_processor import (  # noqa: E402
    enumerate_dicom_series,
    single_ct_series_fields_if_unique,
)
from src.patient_manager import get_patient_metadata, normalize_patient_id  # noqa: E402
from src.pipeline import find_patient_folders, process_single_patient  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _ThreadSafeGuiTextStream:
    """Buffer text from worker threads; flush to Tk on the main thread via after().

    tqdm and nnU-Net log to stderr very frequently. Scheduling ``after(0, ...)`` and
    updating a Tk ``Text`` widget on every tiny write floods the event loop and can
    make Windows show "(Not Responding)" even though work runs on a worker thread.
    We coalesce writes and only reschedule drains with a small delay so the UI can
    breathe between batches.
    """

    def __init__(self, root: tk.Tk, append_callback, throttle_ms: int = 50):
        self._root = root
        self._append = append_callback
        self._throttle_ms = max(16, int(throttle_ms))
        self._buffer = ""
        self._lock = threading.Lock()
        self._flush_scheduled = False

    def write(self, s: str) -> int:
        if s is None:
            return 0
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return 0
        with self._lock:
            self._buffer += s
            if not self._flush_scheduled:
                self._flush_scheduled = True
                self._root.after(0, self._drain)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            if not self._flush_scheduled:
                self._flush_scheduled = True
                self._root.after(0, self._drain)

    def _drain(self) -> None:
        with self._lock:
            chunk = self._buffer
            self._buffer = ""
        if chunk:
            self._append(chunk)
        with self._lock:
            more = bool(self._buffer)
        if more:
            self._root.after(self._throttle_ms, self._drain)
        else:
            with self._lock:
                self._flush_scheduled = False

    def isatty(self) -> bool:
        return False


class _TeeTextStream:
    """Write to the original console stream and to the GUI mirror."""

    def __init__(self, original, gui_stream: _ThreadSafeGuiTextStream):
        self._original = original
        self._gui = gui_stream

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        if self._original is not None:
            try:
                self._original.write(s)
            except Exception:
                pass
        return self._gui.write(s)

    def flush(self) -> None:
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass
        self._gui.flush()

    def isatty(self) -> bool:
        if self._original is not None and hasattr(self._original, "isatty"):
            try:
                return bool(self._original.isatty())
            except Exception:
                pass
        return False


class _GuiLogHandler(logging.Handler):
    """Mirror log records through the same throttled stream as stdout/stderr."""

    def __init__(self, gui_stream: _ThreadSafeGuiTextStream):
        super().__init__(level=logging.INFO)
        self._gui_stream = gui_stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            self._gui_stream.write(msg)
        except Exception:
            self.handleError(record)


CSV_COLUMNS = [
    "case_id",
    "patient_id",
    "patient_name",
    "exam_date",
    "dicom_folder",
    "output_base_dir",
    "study_id",
    "series_instance_uid",
    "series_description",
    "patient_output_dir",
    "segmentation_dir",
    "status",
    "error",
    "was_skipped",
    "segmentation_duration_seconds",
    "updated_at",
    "selected_slices",
    "average_hu_all",
    "l1_trabecular_core_hu",
    "T11 HU",
    "T12 HU",
    "L1 HU",
    "L2 HU",
    "L3 HU",
    "L4 HU",
    "L5 HU",
    "T11 Source",
    "T12 Source",
    "L1 Source",
    "L2 Source",
    "L3 Source",
    "L4 Source",
    "L5 Source",
]

TERMINAL_SEGMENTATION_STATUSES = {"not checked", "success", "failed", "not applicable"}
QC_STATUSES = ["All", "Not Checked", "Success", "Failed", "Not Applicable", "Failed Pipeline"]
VERTEBRA_LABELS = ["T11", "T12", "L1", "L2", "L3", "L4", "L5"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pytorch_cuda_summary() -> str:
    """Human-readable PyTorch / CUDA status for the batch GUI (may import torch)."""
    try:
        import torch
    except Exception as exc:
        return f"Could not import torch:\n{exc}"
    lines: List[str] = [f"PyTorch {torch.__version__}"]
    compiled = getattr(torch.version, "cuda", None)
    if compiled not in (None, "0.0"):
        lines.append(f"Built with CUDA {compiled}")
    else:
        lines.append("This PyTorch wheel is CPU-only (no CUDA libraries bundled).")
    try:
        avail = bool(torch.cuda.is_available())
    except Exception as exc:
        lines.append(f"torch.cuda.is_available() raised: {exc}")
        return "\n".join(lines)
    lines.append(f"torch.cuda.is_available(): {avail}")
    if avail:
        try:
            n = int(torch.cuda.device_count())
            lines.append(f"device_count: {n}")
            if n > 0:
                lines.append(f"device 0: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            lines.append(f"CUDA device query failed: {exc}")
    else:
        lines.append(
            "No GPU for this process: install an NVIDIA driver and a CUDA-enabled "
            "PyTorch build, or choose Device: cpu."
        )
    return "\n".join(lines)


def _safe_study_id(study_id: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(study_id))


def expected_patient_output_dir(output_base_dir: Path, patient_id: str, study_id: str) -> Path:
    return Path(output_base_dir) / normalize_patient_id(patient_id) / _safe_study_id(study_id)


def autofill_series_if_single_ct(row: Dict[str, str]) -> None:
    """When exactly one CT series exists under the case folder, set CSV series columns."""
    folder = Path(row["dicom_folder"])
    try:
        pair = single_ct_series_fields_if_unique(folder)
    except Exception:
        return
    if not pair:
        return
    uid, desc = pair
    row["series_instance_uid"] = uid
    row["series_description"] = desc


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CSV_COLUMNS)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_case_rows(input_path: Path, output_base_dir: Path) -> List[Dict[str, str]]:
    patient_folders = find_patient_folders(input_path)
    if not patient_folders:
        raise ValueError(f"No patient folders found in {input_path}")

    rows: List[Dict[str, str]] = []
    study_counts: Dict[str, int] = {}

    for patient_folder in patient_folders:
        metadata = get_patient_metadata(patient_folder)
        patient_id = metadata.get("patient_id") or patient_folder.name
        patient_name = metadata.get("patient_name") or ""
        exam_date = metadata.get("study_date") or ""
        study_name = patient_folder.name

        key = f"{patient_id}_{study_name}"
        count = study_counts.get(key, 0)
        study_counts[key] = count + 1
        study_id = study_name if count == 0 else f"{study_name}_{count}"

        patient_output_dir = expected_patient_output_dir(output_base_dir, patient_id, study_id)
        segmentation_dir = patient_output_dir / "segmentations"

        rows.append(
            {
                "case_id": uuid.uuid4().hex[:12],
                "patient_id": patient_id,
                "patient_name": patient_name,
                "exam_date": exam_date,
                "dicom_folder": str(patient_folder),
                "output_base_dir": str(output_base_dir),
                "study_id": study_id,
                "series_instance_uid": "",
                "series_description": "",
                "patient_output_dir": str(patient_output_dir),
                "segmentation_dir": str(segmentation_dir),
                "status": "pending",
                "error": "",
                "was_skipped": "",
                "segmentation_duration_seconds": "",
                "updated_at": _now(),
            }
        )

    return rows


def process_segmentation_row(
    row: Dict[str, str],
    output_base_dir_override: Optional[Path],
    temp_dir: Path,
    fast_segmentation: bool,
    device: str,
) -> Dict[str, str]:
    dicom_folder = Path(row["dicom_folder"])
    output_base_dir = output_base_dir_override or Path(row.get("output_base_dir") or ".")
    study_id = row.get("study_id") or dicom_folder.name
    patient_id = row.get("patient_id") or dicom_folder.name
    patient_output_dir = expected_patient_output_dir(output_base_dir, patient_id, study_id)
    segmentation_dir = patient_output_dir / "segmentations"

    row["output_base_dir"] = str(output_base_dir)
    row["patient_output_dir"] = str(patient_output_dir)
    row["segmentation_dir"] = str(segmentation_dir)

    start_time = time.perf_counter()
    if check_segmentations_exist(segmentation_dir):
        row["status"] = "not checked"
        row["error"] = ""
        row["was_skipped"] = "true"
        row["segmentation_duration_seconds"] = f"{time.perf_counter() - start_time:.2f}"
        row["updated_at"] = _now()
        return row

    series_uid = (row.get("series_instance_uid") or "").strip() or None

    result = process_single_patient(
        dicom_folder=dicom_folder,
        output_base_dir=output_base_dir,
        temp_dir=temp_dir,
        fast_segmentation=fast_segmentation,
        device=device,
        keep_temp_files=True,
        forced_study_id=study_id,
        series_instance_uid=series_uid,
    )

    if result.get("status") != "success":
        raise RuntimeError(result.get("error") or "Pipeline processing failed")

    row["status"] = "not checked"
    row["error"] = ""
    row["was_skipped"] = "false"
    row["segmentation_duration_seconds"] = f"{time.perf_counter() - start_time:.2f}"
    row["updated_at"] = _now()
    return row


def apply_verification_result(row: Dict[str, str], result: Dict) -> Dict[str, str]:
    is_successful = result.get("is_successful")
    if is_successful is True:
        row["status"] = "success"
    elif is_successful is False:
        row["status"] = "failed"
    else:
        row["status"] = "not checked"

    selected_slices = result.get("selected_slices") or []
    row["selected_slices"] = ",".join(str(s) for s in selected_slices)

    average_hu = result.get("average_hu")
    row["average_hu_all"] = f"{average_hu:.2f}" if average_hu is not None else ""

    l1_core_hu = result.get("l1_trabecular_core_hu")
    row["l1_trabecular_core_hu"] = (
        f"{l1_core_hu:.2f}" if l1_core_hu is not None and l1_core_hu != 0 else ""
    )

    vertebra_hu = result.get("vertebra_hu", {})
    label_mapping = result.get("label_mapping", {})
    for label in VERTEBRA_LABELS:
        value = vertebra_hu.get(label, 0.0)
        row[f"{label} HU"] = f"{value:.2f}"
        row[f"{label} Source"] = label_mapping.get(label, "")

    row["error"] = ""
    row["updated_at"] = _now()
    return row


class BatchVerificationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("oppoCT Batch Verification")
        self.root.geometry("1040x780")

        self.frames: Dict[str, ttk.Frame] = {}
        self.w1_rows: List[Dict[str, str]] = []
        self.sp_csv_path: Optional[Path] = None
        self.sp_rows: List[Dict[str, str]] = []
        self.sp_series_items: List[Dict[str, object]] = []
        self._w2_cuda_check_running = False
        self.w3_csv_path: Optional[Path] = None
        self.w3_rows: List[Dict[str, str]] = []
        self.w3_display_indices: List[int] = []
        self._w1_scanning = False

        self._init_styles()
        self._create_main_menu()
        self._create_list_frame()
        self._create_select_series_frame()
        self._create_segmentation_frame()
        self._create_qc_frame()
        self.show_frame("main")

    def _init_styles(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"))
        style.configure("Menu.TButton", font=("Helvetica", 14), padding=10)

    def show_frame(self, name: str) -> None:
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[name].pack(fill="both", expand=True)

    def _create_main_menu(self) -> None:
        frame = ttk.Frame(self.root, padding=20)
        self.frames["main"] = frame

        ttk.Label(frame, text="oppoCT Batch Verification", style="Title.TLabel").pack(pady=40)
        ttk.Button(
            frame,
            text="1. Create CSV List",
            style="Menu.TButton",
            command=lambda: self.show_frame("list"),
        ).pack(fill="x", padx=120, pady=10)
        ttk.Button(
            frame,
            text="2. Select series (load CSV)",
            style="Menu.TButton",
            command=lambda: self.show_frame("select_series"),
        ).pack(fill="x", padx=120, pady=10)
        ttk.Button(
            frame,
            text="3. Segment In Batch",
            style="Menu.TButton",
            command=lambda: self.show_frame("segment"),
        ).pack(fill="x", padx=120, pady=10)
        ttk.Button(
            frame,
            text="4. Verify And Mark CSV",
            style="Menu.TButton",
            command=lambda: self.show_frame("qc"),
        ).pack(fill="x", padx=120, pady=10)
        ttk.Button(frame, text="Exit", style="Menu.TButton", command=self.root.quit).pack(
            fill="x", padx=120, pady=40
        )

        note = (
            "Batch segmentation relies on TotalSegmentator "
            "(Wasserthal et al., Radiology: Artificial Intelligence 2023; doi:10.1148/ryai.230024), "
            "whose models follow the U-Net family "
            "(Ronneberger et al., MICCAI 2015). "
        )
        ttk.Label(frame, text=note, wraplength=760, justify=tk.CENTER).pack(side=tk.BOTTOM, pady=20)

    def _create_list_frame(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        self.frames["list"] = frame

        ttk.Label(frame, text="Workflow 1: Create CSV List", font=("Helvetica", 14, "bold")).pack(pady=5)
        ttk.Button(frame, text="Back to Main Menu", command=lambda: self.show_frame("main")).pack(
            anchor="nw", pady=5
        )

        inputs = ttk.LabelFrame(frame, text="Directories", padding=10)
        inputs.pack(fill="x", pady=5)

        self.w1_input_var = tk.StringVar()
        self.w1_output_var = tk.StringVar()

        ttk.Label(inputs, text="DICOM root:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(inputs, textvariable=self.w1_input_var, width=70).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(inputs, text="Browse", command=self.w1_browse_input).grid(row=0, column=2, padx=5)

        ttk.Label(inputs, text="Output root:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(inputs, textvariable=self.w1_output_var, width=70).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(inputs, text="Browse", command=self.w1_browse_output).grid(row=1, column=2, padx=5)

        self.w1_scan_button = ttk.Button(inputs, text="Scan Cases", command=self.w1_scan)
        self.w1_scan_button.grid(row=2, column=1, sticky="w", pady=8)

        self.w1_status_var = tk.StringVar(value="Status: waiting for scan")
        ttk.Label(frame, textvariable=self.w1_status_var).pack(anchor="w", pady=5)

        list_frame = ttk.LabelFrame(frame, text="Cases", padding=5)
        list_frame.pack(fill="both", expand=True)
        self.w1_cases = tk.Listbox(list_frame, exportselection=False)
        self.w1_cases.pack(fill="both", expand=True, side=tk.LEFT)
        scroll = ttk.Scrollbar(list_frame, command=self.w1_cases.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.w1_cases.config(yscrollcommand=scroll.set)

        actions = ttk.Frame(frame, padding=5)
        actions.pack(fill="x")
        self.w1_save_button = ttk.Button(actions, text="Save Batch CSV", command=self.w1_save_csv, state=tk.DISABLED)
        self.w1_save_button.pack(side=tk.RIGHT)

    def w1_browse_input(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.w1_input_var.set(folder)

    def w1_browse_output(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.w1_output_var.set(folder)

    def w1_scan(self) -> None:
        input_path = Path(self.w1_input_var.get())
        output_base_dir = Path(self.w1_output_var.get())
        if not input_path.exists():
            messagebox.showerror("Invalid Input", "Choose a valid DICOM root folder.")
            return
        if not str(output_base_dir):
            messagebox.showerror("Invalid Output", "Choose an output root folder.")
            return
        if self._w1_scanning:
            return

        self._w1_scanning = True
        self.w1_scan_button.config(state=tk.DISABLED)
        self.w1_status_var.set("Status: scanning cases (background thread; UI should stay responsive)...")

        def worker() -> None:
            try:
                rows = build_case_rows(input_path, output_base_dir)
                for row in rows:
                    autofill_series_if_single_ct(row)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._w1_scan_finish_error(e))
                return
            self.root.after(0, lambda r=rows: self._w1_scan_finish_ok(r))

        threading.Thread(target=worker, daemon=True).start()

    def _w1_scan_finish_ok(self, rows: List[Dict[str, str]]) -> None:
        self._w1_scanning = False
        self.w1_scan_button.config(state=tk.NORMAL)
        self.w1_rows = rows
        self.w1_cases.delete(0, tk.END)
        for row in rows:
            label = (
                f"{row['patient_id']} | {row['exam_date']} | "
                f"{Path(row['dicom_folder']).name} | {row['status']}"
            )
            self.w1_cases.insert(tk.END, label)
        self.w1_status_var.set(f"Status: found {len(rows)} case(s)")
        self.w1_save_button.config(state=tk.NORMAL)

    def _w1_scan_finish_error(self, exc: Exception) -> None:
        self._w1_scanning = False
        self.w1_scan_button.config(state=tk.NORMAL)
        logger.exception("Failed to scan cases")
        messagebox.showerror("Scan Failed", str(exc))
        self.w1_status_var.set("Status: scan failed")

    def w1_save_csv(self) -> None:
        if not self.w1_rows:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="oppoct_batch_jobs.csv",
        )
        if not save_path:
            return

        try:
            write_csv_rows(Path(save_path), self.w1_rows)
            messagebox.showinfo("Saved", f"Saved batch CSV:\n{save_path}")
        except Exception as exc:
            logger.exception("Failed to save CSV")
            messagebox.showerror("Save Failed", str(exc))

    def _create_select_series_frame(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        self.frames["select_series"] = frame

        ttk.Label(frame, text="Workflow 2: Select series & update CSV", font=("Helvetica", 14, "bold")).pack(
            pady=5
        )
        ttk.Button(frame, text="Back to Main Menu", command=lambda: self.show_frame("main")).pack(
            anchor="nw", pady=5
        )

        cfg = ttk.LabelFrame(frame, text="Batch CSV", padding=10)
        cfg.pack(fill="x", pady=5)
        self.sp_csv_var = tk.StringVar()
        ttk.Label(cfg, text="CSV file:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(cfg, textvariable=self.sp_csv_var, width=62).grid(row=0, column=1, padx=5, pady=2, sticky="we")
        ttk.Button(cfg, text="Browse", command=self.sp_browse_csv).grid(row=0, column=2, padx=5)
        ttk.Button(cfg, text="Load CSV", command=self.sp_load_csv).grid(row=0, column=3, padx=5)
        cfg.columnconfigure(1, weight=1)

        self.sp_status_var = tk.StringVar(value="Load a batch CSV, then choose a study and a series.")
        ttk.Label(frame, textvariable=self.sp_status_var, wraplength=920, justify=tk.LEFT).pack(
            anchor="w", pady=(4, 2)
        )

        pane = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        pane.pack(fill="both", expand=True, pady=8)

        left = ttk.LabelFrame(pane, text="Studies (CSV rows)", padding=6)
        self.sp_study_list = tk.Listbox(left, exportselection=False, height=22)
        self.sp_study_list.pack(side=tk.LEFT, fill="both", expand=True)
        sb_l = ttk.Scrollbar(left, command=self.sp_study_list.yview)
        sb_l.pack(side=tk.RIGHT, fill=tk.Y)
        self.sp_study_list.config(yscrollcommand=sb_l.set)
        self.sp_study_list.bind("<<ListboxSelect>>", self.sp_on_study_select)
        pane.add(left, weight=1)

        right = ttk.LabelFrame(pane, text="Series (DICOM under selected study)", padding=6)
        self.sp_series_list = tk.Listbox(right, exportselection=False, height=22)
        self.sp_series_list.pack(side=tk.LEFT, fill="both", expand=True)
        sb_r = ttk.Scrollbar(right, command=self.sp_series_list.yview)
        sb_r.pack(side=tk.RIGHT, fill=tk.Y)
        self.sp_series_list.config(yscrollcommand=sb_r.set)
        pane.add(right, weight=1)

        actions = ttk.Frame(frame, padding=(0, 8, 0, 0))
        actions.pack(fill="x")
        ttk.Button(actions, text="Select — mark on CSV", command=self.sp_mark_selection_on_csv).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(actions, text="Clear series for selected study", command=self.sp_clear_series_for_selected).pack(
            side=tk.LEFT
        )

    def sp_browse_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.sp_csv_var.set(path)

    def sp_load_csv(self) -> None:
        path = Path(self.sp_csv_var.get().strip())
        if not path.is_file():
            messagebox.showerror("Invalid CSV", "Choose an existing batch CSV file.")
            return
        try:
            self.sp_rows = read_csv_rows(path)
        except Exception as exc:
            logger.exception("Failed to read CSV")
            messagebox.showerror("Read failed", str(exc))
            return
        self.sp_csv_path = path
        self.sp_series_list.delete(0, tk.END)
        self.sp_series_items = []
        self.sp_refresh_study_listbox()
        self.sp_status_var.set(f"Loaded {len(self.sp_rows)} row(s) from {path.name}")

    def sp_refresh_study_listbox(self) -> None:
        self.sp_study_list.delete(0, tk.END)
        for row in self.sp_rows:
            suid = (row.get("series_instance_uid") or "").strip()
            flag = "series OK" if suid else "no series"
            folder = Path(row.get("dicom_folder", "") or ".")
            label = (
                f"{row.get('patient_id', '')} | {row.get('study_id', '')} | {row.get('exam_date', '')} | "
                f"{folder.name} | {flag}"
            )
            self.sp_study_list.insert(tk.END, label)

    def sp_selected_study_index(self) -> Optional[int]:
        sel = self.sp_study_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def sp_selected_series_index(self) -> Optional[int]:
        sel = self.sp_series_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def sp_on_study_select(self, _event=None) -> None:
        self.sp_series_list.delete(0, tk.END)
        self.sp_series_items = []
        idx = self.sp_selected_study_index()
        if idx is None or idx >= len(self.sp_rows):
            return
        row = self.sp_rows[idx]
        folder = Path(row.get("dicom_folder", ""))
        if not folder.is_dir():
            self.sp_status_var.set(f"DICOM folder not found:\n{folder}")
            return
        try:
            items = enumerate_dicom_series(folder)
        except Exception as exc:
            logger.exception("Series enumeration failed")
            self.sp_status_var.set(f"Series scan failed: {exc}")
            messagebox.showerror("Series scan failed", str(exc))
            return
        self.sp_series_items = items
        if not items:
            self.sp_status_var.set(f"No DICOM files under {folder.name}")
            return
        for it in items:
            uid = str(it.get("series_instance_uid", "") or "")
            desc = str(it.get("series_description", "") or "")
            mod = str(it.get("modality", "") or "")
            n = int(it.get("num_instances", 0) or 0)
            uid_disp = uid if uid else "(missing SeriesInstanceUID)"
            line = f"{mod or '?'} | n={n} | {desc[:48]}{'…' if len(desc) > 48 else ''} | {uid_disp}"
            self.sp_series_list.insert(tk.END, line)
        self.sp_status_var.set(f"Study {row.get('patient_id', '')}: {len(items)} series in {folder.name}")

    def sp_mark_selection_on_csv(self) -> None:
        if self.sp_csv_path is None or not self.sp_rows:
            messagebox.showerror("No CSV", "Load a batch CSV first.")
            return
        si = self.sp_selected_study_index()
        if si is None:
            messagebox.showinfo("Select a study", "Select a study in the left list.")
            return
        ri = self.sp_selected_series_index()
        if ri is None:
            messagebox.showinfo("Select a series", "Select a series in the right list.")
            return
        if ri < 0 or ri >= len(self.sp_series_items):
            messagebox.showerror("Invalid selection", "Reload the study or pick a series again.")
            return
        pick = self.sp_series_items[ri]
        self.sp_rows[si]["series_instance_uid"] = str(pick.get("series_instance_uid", "") or "")
        self.sp_rows[si]["series_description"] = str(pick.get("series_description", "") or "")
        self.sp_rows[si]["updated_at"] = _now()
        try:
            write_csv_rows(self.sp_csv_path, self.sp_rows)
        except Exception as exc:
            logger.exception("Failed to write CSV")
            messagebox.showerror("Save failed", str(exc))
            return
        self.sp_refresh_study_listbox()
        self.sp_study_list.selection_set(si)
        self.sp_study_list.see(si)
        self.sp_on_study_select()
        messagebox.showinfo("Saved", "Series selection written to the CSV.")

    def sp_clear_series_for_selected(self) -> None:
        if self.sp_csv_path is None or not self.sp_rows:
            messagebox.showerror("No CSV", "Load a batch CSV first.")
            return
        si = self.sp_selected_study_index()
        if si is None:
            messagebox.showinfo("Select a study", "Select a study in the left list.")
            return
        self.sp_rows[si]["series_instance_uid"] = ""
        self.sp_rows[si]["series_description"] = ""
        self.sp_rows[si]["updated_at"] = _now()
        try:
            write_csv_rows(self.sp_csv_path, self.sp_rows)
        except Exception as exc:
            logger.exception("Failed to write CSV")
            messagebox.showerror("Save failed", str(exc))
            return
        self.sp_refresh_study_listbox()
        self.sp_study_list.selection_set(si)
        self.sp_study_list.see(si)
        self.sp_on_study_select()
        messagebox.showinfo("Saved", "Cleared series fields for this study in the CSV.")

    def _create_segmentation_frame(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        self.frames["segment"] = frame

        ttk.Label(frame, text="Workflow 3: Segment In Batch", font=("Helvetica", 14, "bold")).pack(pady=5)
        ttk.Button(frame, text="Back to Main Menu", command=lambda: self.show_frame("main")).pack(
            anchor="nw", pady=5
        )

        config = ttk.LabelFrame(frame, text="Configuration", padding=10)
        config.pack(fill="x", pady=5)

        self.w2_csv_var = tk.StringVar()
        self.w2_output_override_var = tk.StringVar()
        self.w2_device_var = tk.StringVar(value="gpu")
        self.w2_fast_var = tk.BooleanVar(value=False)
        self.w2_retry_failed_var = tk.BooleanVar(value=False)

        ttk.Label(config, text="Batch CSV:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(config, textvariable=self.w2_csv_var, width=70).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(config, text="Browse", command=self.w2_browse_csv).grid(row=0, column=2, padx=5)

        ttk.Label(config, text="Output override:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(config, textvariable=self.w2_output_override_var, width=70).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(config, text="Browse", command=self.w2_browse_output).grid(row=1, column=2, padx=5)

        ttk.Label(config, text="Device:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ttk.Combobox(config, textvariable=self.w2_device_var, values=["gpu", "cpu"], width=8, state="readonly").grid(
            row=2, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Checkbutton(config, text="Fast mode", variable=self.w2_fast_var).grid(
            row=2, column=1, sticky="w", padx=90, pady=2
        )
        ttk.Checkbutton(config, text="Retry failed_pipeline rows", variable=self.w2_retry_failed_var).grid(
            row=2, column=1, sticky="w", padx=190, pady=2
        )

        cuda_frame = ttk.LabelFrame(frame, text="PyTorch / CUDA (workflow 3)", padding=10)
        cuda_frame.pack(fill="x", pady=5)
        self.w2_cuda_status_var = tk.StringVar(
            value='Click "Check CUDA / PyTorch" to see whether GPU inference is available.'
        )
        ttk.Button(cuda_frame, text="Check CUDA / PyTorch", command=self.w2_check_cuda).pack(anchor="w")
        ttk.Label(
            cuda_frame,
            textvariable=self.w2_cuda_status_var,
            wraplength=900,
            justify=tk.LEFT,
        ).pack(fill="x", pady=(6, 0))

        self.w2_start_button = ttk.Button(frame, text="Start Batch Segmentation", command=self.w2_start)
        self.w2_start_button.pack(pady=10)

        progress = ttk.LabelFrame(frame, text="Progress", padding=10)
        progress.pack(fill="both", expand=True)

        self.w2_status_var = tk.StringVar(value="Study: 0 / 0")
        ttk.Label(progress, textvariable=self.w2_status_var).pack(anchor="w")

        self.w2_progress_var = tk.DoubleVar()
        ttk.Progressbar(progress, variable=self.w2_progress_var, maximum=100).pack(fill="x", pady=5)

        self.w2_log_text = tk.Text(progress, height=18, state=tk.DISABLED)
        self.w2_log_text.pack(fill="both", expand=True)

    def w2_browse_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.w2_csv_var.set(path)

    def w2_browse_output(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.w2_output_override_var.set(folder)

    def _w2_append_raw(self, text: str) -> None:
        if not text:
            return
        self.w2_log_text.config(state=tk.NORMAL)
        self.w2_log_text.insert(tk.END, text)
        self.w2_log_text.see(tk.END)
        # Very long logs slow Text widget layout; trim from the top occasionally.
        try:
            last_line = int(str(self.w2_log_text.index("end-1c")).split(".", maxsplit=1)[0])
            if last_line > 2800:
                self.w2_log_text.delete("1.0", "800.0")
        except (tk.TclError, ValueError, IndexError):
            pass
        self.w2_log_text.config(state=tk.DISABLED)

    def w2_log(self, message: str) -> None:
        self._w2_append_raw(message + "\n")

    def w2_check_cuda(self) -> None:
        """Run torch import / CUDA probes off the Tk main thread (can take several seconds)."""

        if self._w2_cuda_check_running:
            return
        self._w2_cuda_check_running = True
        self.w2_cuda_status_var.set("Checking… (first torch import can take a few seconds)")

        def worker() -> None:
            try:
                summary = _pytorch_cuda_summary()
            finally:
                self._w2_cuda_check_running = False
            self.root.after(0, self.w2_cuda_status_var.set, summary)

        threading.Thread(target=worker, daemon=True).start()

    def w2_start(self) -> None:
        csv_path = Path(self.w2_csv_var.get())
        if not csv_path.exists():
            messagebox.showerror("Invalid CSV", "Choose a valid batch CSV.")
            return

        if self.w2_device_var.get().strip().lower() == "gpu":
            try:
                import torch
            except Exception as exc:
                messagebox.showerror("PyTorch", f"Cannot import torch:\n{exc}")
                return
            if not torch.cuda.is_available():
                proceed = messagebox.askyesno(
                    "GPU not available",
                    "PyTorch reports CUDA is not available, so segmentation will run on the CPU.\n\n"
                    "That is much slower; Windows may show (Not Responding) in the title bar while the CPU is busy.\n\n"
                    'Use "Check CUDA / PyTorch" above to see details.\n\n'
                    "Continue with CPU inference anyway?",
                )
                if not proceed:
                    return

        output_override = self.w2_output_override_var.get().strip()
        override_path = Path(output_override) if output_override else None
        self.w2_start_button.config(state=tk.DISABLED)
        self.w2_progress_var.set(0)
        self.w2_log_text.config(state=tk.NORMAL)
        self.w2_log_text.delete("1.0", tk.END)
        self.w2_log_text.config(state=tk.DISABLED)

        threading.Thread(
            target=self._run_segmentation_thread,
            args=(csv_path, override_path, self.w2_fast_var.get(), self.w2_device_var.get(), self.w2_retry_failed_var.get()),
            daemon=True,
        ).start()

    def _run_segmentation_thread(
        self,
        csv_path: Path,
        output_override: Optional[Path],
        fast_segmentation: bool,
        device: str,
        retry_failed: bool,
    ) -> None:
        gui_stream = _ThreadSafeGuiTextStream(self.root, self._w2_append_raw, throttle_ms=50)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        log_handler: Optional[_GuiLogHandler] = None
        show_completion_dialog = False
        try:
            sys.stdout = _TeeTextStream(old_stdout, gui_stream)
            sys.stderr = _TeeTextStream(old_stderr, gui_stream)
            log_handler = _GuiLogHandler(gui_stream)
            log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.root.addHandler(log_handler)

            try:
                rows = read_csv_rows(csv_path)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Read Failed", str(exc)))
                self.root.after(0, self.w2_start_button.config, {"state": tk.NORMAL})
                return

            pending: List[Tuple[int, Dict[str, str]]] = []
            for idx, row in enumerate(rows):
                status = row.get("status", "pending").lower()
                if status == "pending" or (retry_failed and status == "failed_pipeline"):
                    pending.append((idx, row))

            total = len(pending)
            self.root.after(0, self.w2_log, f"Found {total} pending case(s).")

            temp_dir = Path(tempfile.mkdtemp(prefix="oppoct_batch_gui_"))
            for count, (idx, row) in enumerate(pending, 1):
                label = row.get("patient_id") or row.get("dicom_folder", f"row {idx}")
                self.root.after(0, self.w2_status_var.set, f"Study: {count} / {total}")
                self.root.after(0, self.w2_progress_var.set, ((count - 1) / max(total, 1)) * 100)
                self.root.after(0, self.w2_log, f"Processing {label}")

                try:
                    rows[idx] = process_segmentation_row(
                        row, output_override, temp_dir, fast_segmentation, device
                    )
                    write_csv_rows(csv_path, rows)
                    skipped = rows[idx].get("was_skipped") == "true"
                    suffix = "existing segmentation" if skipped else "new segmentation"
                    self.root.after(0, self.w2_log, f"Finished {label} ({suffix})")
                except Exception as exc:
                    logger.exception("Segmentation failed for %s", label)
                    rows[idx]["status"] = "failed_pipeline"
                    rows[idx]["error"] = str(exc)
                    rows[idx]["updated_at"] = _now()
                    write_csv_rows(csv_path, rows)
                    self.root.after(0, self.w2_log, f"ERROR {label}: {exc}")

            self.root.after(0, self.w2_progress_var.set, 100)
            self.root.after(0, self.w2_status_var.set, f"Study: {total} / {total}")
            self.root.after(0, self.w2_log, "Batch segmentation complete.")
            show_completion_dialog = True
        finally:
            if log_handler is not None:
                try:
                    logging.root.removeHandler(log_handler)
                except ValueError:
                    pass
            sys.stdout, sys.stderr = old_stdout, old_stderr
            try:
                old_stdout.flush()
            except Exception:
                pass
            try:
                old_stderr.flush()
            except Exception:
                pass
            gui_stream.flush()

        self.root.after(0, self.w2_start_button.config, {"state": tk.NORMAL})
        if show_completion_dialog:
            self.root.after(0, lambda: messagebox.showinfo("Done", "Batch segmentation complete."))

    def _create_qc_frame(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        self.frames["qc"] = frame

        ttk.Label(frame, text="Workflow 4: Verify And Mark CSV", font=("Helvetica", 14, "bold")).pack(pady=5)
        ttk.Button(frame, text="Back to Main Menu", command=lambda: self.show_frame("main")).pack(
            anchor="nw", pady=5
        )

        config = ttk.LabelFrame(frame, text="Load Batch CSV", padding=10)
        config.pack(fill="x", pady=5)

        self.w3_csv_var = tk.StringVar()
        ttk.Label(config, text="Batch CSV:").grid(row=0, column=0, sticky="e", padx=5)
        ttk.Entry(config, textvariable=self.w3_csv_var, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(config, text="Browse", command=self.w3_browse_csv).grid(row=0, column=2, padx=5)
        ttk.Button(config, text="Load Data", command=self.w3_load_data).grid(row=0, column=3, padx=5)

        pane = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        pane.pack(fill="both", expand=True, pady=5)

        left = ttk.Frame(pane)
        pane.add(left, weight=1)

        self.w3_filter_var = tk.StringVar(value="All")
        ttk.Combobox(left, textvariable=self.w3_filter_var, values=QC_STATUSES, state="readonly").pack(fill="x")
        self.w3_filter_var.trace_add("write", lambda *_: self.w3_refresh_list())

        self.w3_cases = tk.Listbox(left, exportselection=False)
        self.w3_cases.pack(fill="both", expand=True, side=tk.LEFT)
        scroll = ttk.Scrollbar(left, command=self.w3_cases.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.w3_cases.config(yscrollcommand=scroll.set)
        self.w3_cases.bind("<<ListboxSelect>>", self.w3_on_case_select)

        right = ttk.Frame(pane, padding=5)
        pane.add(right, weight=2)

        self.w3_details = tk.Text(right, height=16, state=tk.DISABLED)
        self.w3_details.pack(fill="both", expand=True)

        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=10)
        self.w3_pass_button = ttk.Button(
            actions, text="Mark Pass", command=self.w3_mark_pass, state=tk.DISABLED
        )
        self.w3_pass_button.pack(side=tk.LEFT, padx=5)
        self.w3_fail_button = ttk.Button(
            actions, text="Mark Fail", command=self.w3_mark_fail, state=tk.DISABLED
        )
        self.w3_fail_button.pack(side=tk.LEFT, padx=5)
        self.w3_open_button = ttk.Button(
            actions, text="Open Existing Viewer", command=self.w3_open_viewer, state=tk.DISABLED
        )
        self.w3_open_button.pack(side=tk.LEFT, padx=5)
        self.w3_na_button = ttk.Button(
            actions, text="Mark Not Applicable", command=self.w3_mark_not_applicable, state=tk.DISABLED
        )
        self.w3_na_button.pack(side=tk.LEFT, padx=5)

        hint = ttk.Label(
            right,
            text="Mark Pass/Fail updates the CSV immediately. Open Viewer to review slices and capture HU metrics.",
            wraplength=560,
        )
        hint.pack(anchor="w", pady=(0, 4))

    def w3_browse_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.w3_csv_var.set(path)

    def w3_load_data(self) -> None:
        csv_path = Path(self.w3_csv_var.get())
        if not csv_path.exists():
            messagebox.showerror("Invalid CSV", "Choose a valid batch CSV.")
            return

        try:
            self.w3_csv_path = csv_path
            self.w3_rows = read_csv_rows(csv_path)
            self.w3_refresh_list()
        except Exception as exc:
            logger.exception("Failed to load QC CSV")
            messagebox.showerror("Load Failed", str(exc))

    def w3_refresh_list(self) -> None:
        self.w3_cases.delete(0, tk.END)
        self.w3_display_indices.clear()
        selected_filter = self.w3_filter_var.get().lower()

        def sort_key(item: Tuple[int, Dict[str, str]]) -> Tuple[int, str]:
            _, row = item
            status = row.get("status", "pending").lower()
            priority = 0 if status == "not checked" else 1
            return priority, row.get("patient_id", "")

        for original_idx, row in sorted(enumerate(self.w3_rows), key=sort_key):
            status = row.get("status", "pending").lower()
            if selected_filter != "all" and selected_filter != status:
                continue
            label = f"[{status.upper()}] {row.get('patient_id', '')} | {row.get('exam_date', '')}"
            self.w3_cases.insert(tk.END, label)
            self.w3_display_indices.append(original_idx)

        self.w3_pass_button.config(state=tk.DISABLED)
        self.w3_fail_button.config(state=tk.DISABLED)
        self.w3_open_button.config(state=tk.DISABLED)
        self.w3_na_button.config(state=tk.DISABLED)

    def w3_selected_original_index(self) -> Optional[int]:
        selection = self.w3_cases.curselection()
        if not selection:
            return None
        return self.w3_display_indices[selection[0]]

    def w3_on_case_select(self, _event=None) -> None:
        idx = self.w3_selected_original_index()
        if idx is None:
            return
        row = self.w3_rows[idx]

        details = [
            f"Case ID: {row.get('case_id', '')}",
            f"Patient ID: {row.get('patient_id', '')}",
            f"Patient Name: {row.get('patient_name', '')}",
            f"Exam Date: {row.get('exam_date', '')}",
            f"Status: {row.get('status', '')}",
            f"SeriesInstanceUID (segmentation): {row.get('series_instance_uid', '')}",
            f"Series description: {row.get('series_description', '')}",
            f"DICOM Folder: {row.get('dicom_folder', '')}",
            f"Segmentation Dir: {row.get('segmentation_dir', '')}",
            f"Selected Slices: {row.get('selected_slices', '')}",
            f"Average HU: {row.get('average_hu_all', '')}",
            f"L1 Trabecular Core HU: {row.get('l1_trabecular_core_hu', '')}",
            f"Error: {row.get('error', '')}",
        ]

        self.w3_details.config(state=tk.NORMAL)
        self.w3_details.delete("1.0", tk.END)
        self.w3_details.insert(tk.END, "\n".join(details))
        self.w3_details.config(state=tk.DISABLED)
        self.w3_pass_button.config(state=tk.NORMAL)
        self.w3_fail_button.config(state=tk.NORMAL)
        self.w3_open_button.config(state=tk.NORMAL)
        self.w3_na_button.config(state=tk.NORMAL)

    def w3_open_viewer(self) -> None:
        if self.w3_csv_path is None:
            return
        idx = self.w3_selected_original_index()
        if idx is None:
            return

        row = self.w3_rows[idx]
        dicom_folder = Path(row.get("dicom_folder", ""))
        segmentation_dir = Path(row.get("segmentation_dir", ""))

        if not dicom_folder.exists():
            messagebox.showerror("Missing DICOM", f"DICOM folder not found:\n{dicom_folder}")
            return
        if not check_segmentations_exist(segmentation_dir):
            messagebox.showerror("Missing Segmentation", f"No segmentation masks found:\n{segmentation_dir}")
            return

        try:
            suid = (row.get("series_instance_uid") or "").strip() or None
            ct_volume, ct_img = load_ct_for_verification(
                dicom_folder, segmentation_dir, series_instance_uid=suid
            )
            masks = load_masks_for_verification(segmentation_dir, ct_img)
            viewer = VerificationViewer(
                ct_volume=ct_volume,
                masks=masks,
                dicom_folder=dicom_folder,
                patient_id=row.get("patient_id") or dicom_folder.name,
                exam_date=row.get("exam_date") or None,
            )
            result = viewer.show(tk_root=self.root)
            self.w3_rows[idx] = apply_verification_result(row, result)
            write_csv_rows(self.w3_csv_path, self.w3_rows)
            self.w3_refresh_list()
            messagebox.showinfo("Saved", "Verification result saved to CSV.")
        except Exception as exc:
            logger.exception("Verification failed")
            row["error"] = str(exc)
            row["updated_at"] = _now()
            write_csv_rows(self.w3_csv_path, self.w3_rows)
            messagebox.showerror("Verification Failed", str(exc))

    def w3_mark_not_applicable(self) -> None:
        self._w3_write_qc_status("not applicable", "Case marked not applicable.")

    def w3_mark_pass(self) -> None:
        self._w3_write_qc_status("success", "Case marked pass (success).")

    def w3_mark_fail(self) -> None:
        self._w3_write_qc_status("failed", "Case marked fail.")

    def _w3_write_qc_status(self, status: str, saved_message: str) -> None:
        if self.w3_csv_path is None:
            return
        idx = self.w3_selected_original_index()
        if idx is None:
            return

        self.w3_rows[idx]["status"] = status
        self.w3_rows[idx]["updated_at"] = _now()
        self.w3_rows[idx]["error"] = ""
        write_csv_rows(self.w3_csv_path, self.w3_rows)
        self.w3_refresh_list()
        messagebox.showinfo("Saved", saved_message)


def launch_gui() -> None:
    root = tk.Tk()
    BatchVerificationApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
