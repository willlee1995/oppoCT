"""
Standalone single-case DICOM workflow.

This script is additive and intentionally imports the existing segmentation
pipeline and verification viewer without modifying them.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("TkAgg", force=True)

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
from src.dicom_network import (  # noqa: E402
    DicomScpConfig,
    DicomScuConfig,
    DicomStoreSCP,
    ReceivedInstance,
    send_secondary_capture,
)
from src.dicom_processor import enumerate_dicom_series  # noqa: E402
from src.dicom_secondary_capture import (  # noqa: E402
    create_secondary_capture_report,
    find_reference_dicom,
)
from src.patient_manager import get_patient_metadata  # noqa: E402
from src.pipeline import find_patient_folders, process_single_patient  # noqa: E402


logger = logging.getLogger(__name__)

MAX_LOG_LINES = 2000
MAX_LOG_LINE_CHARS = 4000
APP_BG = "#f3f6f8"
CARD_BG = "#ffffff"
ACCENT = "#1f6f8b"
ACCENT_DARK = "#154f63"
TEXT_DARK = "#1f2933"
TEXT_MUTED = "#52606d"


@dataclass
class CaseState:
    dicom_folder: Path
    patient_id: str
    patient_name: str
    exam_date: str
    study_id: str
    series_items: List[Dict[str, object]] = field(default_factory=list)
    series_instance_uid: str = ""
    series_description: str = ""
    status: str = "pending"
    output_dir: Optional[Path] = None
    segmentation_dir: Optional[Path] = None
    qc_result: Optional[Dict] = None
    secondary_capture_path: Optional[Path] = None


class CollapsibleSection(ttk.Frame):
    """Small disclosure panel used to keep setup controls out of the main workflow."""

    def __init__(self, parent, title: str, *, expanded: bool = False):
        super().__init__(parent, style="Card.TFrame")
        self.title = title
        self.expanded = expanded

        self.header_button = ttk.Button(
            self,
            text=self._button_text(),
            command=self.toggle,
            style="Link.TButton",
        )
        self.header_button.grid(row=0, column=0, sticky="ew")
        self.body = ttk.Frame(self, padding=(12, 8, 12, 12), style="Card.TFrame")
        self.columnconfigure(0, weight=1)
        if self.expanded:
            self.body.grid(row=1, column=0, sticky="ew")

    def _button_text(self) -> str:
        marker = "[-]" if self.expanded else "[+]"
        return f"{marker} {self.title}"

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.header_button.configure(text=self._button_text())
        if self.expanded:
            self.body.grid(row=1, column=0, sticky="ew")
        else:
            self.body.grid_remove()


def append_bounded_log(text_widget: tk.Text, message: str, *, max_lines: int = MAX_LOG_LINES) -> None:
    line = str(message)
    if len(line) > MAX_LOG_LINE_CHARS:
        line = line[:MAX_LOG_LINE_CHARS] + "... [truncated]"

    text_widget.config(state=tk.NORMAL)
    text_widget.insert(tk.END, line + "\n")
    try:
        line_count = int(float(text_widget.index("end-1c").split(".")[0]))
    except (ValueError, tk.TclError):
        line_count = 0
    if line_count > max_lines:
        text_widget.delete("1.0", f"{line_count - max_lines + 1}.0")
    text_widget.see(tk.END)
    text_widget.config(state=tk.DISABLED)


class SingleCaseWorkflowApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("oppoCT Single-Case DICOM Workflow")
        self.root.geometry("1220x860")
        self.root.configure(bg=APP_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.style = ttk.Style(self.root)
        self._configure_style()

        self.scp: Optional[DicomStoreSCP] = None
        self.cases: List[CaseState] = []
        self.selected_case_index: Optional[int] = None
        self.selected_series_index: Optional[int] = None
        self._receive_timer: Optional[threading.Timer] = None
        self._worker_running = False

        self.incoming_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "incoming"))
        self.output_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "output_single_case"))
        self.temp_dir_var = tk.StringVar(value=str(Path(tempfile.gettempdir()) / "oppoct_single_case"))
        self.device_var = tk.StringVar(value="gpu")
        self.fast_var = tk.BooleanVar(value=False)

        self.scp_ae_var = tk.StringVar(value="OPPOCT_SCP")
        self.scp_host_var = tk.StringVar(value="0.0.0.0")
        self.scp_port_var = tk.StringVar(value="11112")

        self.scu_ae_var = tk.StringVar(value="OPPOCT_SCU")
        self.dest_ae_var = tk.StringVar(value="ANY-SCP")
        self.dest_host_var = tk.StringVar(value="127.0.0.1")
        self.dest_port_var = tk.StringVar(value="11112")
        self.status_var = tk.StringVar(value="Ready. Start SCP or scan an incoming folder.")

        self._build_ui()
        self._refresh_action_state()

    def _configure_style(self) -> None:
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure(".", font=("Segoe UI", 9), background=APP_BG, foreground=TEXT_DARK)
        self.style.configure("TFrame", background=APP_BG)
        self.style.configure("Card.TFrame", background=CARD_BG)
        self.style.configure("TLabelframe", background=APP_BG, bordercolor="#d9e2ec")
        self.style.configure(
            "TLabelframe.Label",
            background=APP_BG,
            foreground=TEXT_DARK,
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure("TLabel", background=APP_BG, foreground=TEXT_DARK)
        self.style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT_DARK)
        self.style.configure("Muted.TLabel", background=APP_BG, foreground=TEXT_MUTED)
        self.style.configure("Title.TLabel", background=APP_BG, foreground=ACCENT_DARK, font=("Segoe UI", 22, "bold"))
        self.style.configure(
            "CoverTitle.TLabel",
            background=CARD_BG,
            foreground=ACCENT_DARK,
            font=("Segoe UI", 22, "bold"),
        )
        self.style.configure(
            "CoverSubtitle.TLabel",
            background=CARD_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 11),
        )
        self.style.configure("Subtitle.TLabel", background=APP_BG, foreground=TEXT_MUTED, font=("Segoe UI", 11))
        self.style.configure("TButton", padding=(10, 5))
        self.style.configure("Accent.TButton", padding=(12, 6), foreground="#ffffff", background=ACCENT)
        self.style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#9fb3c8")])
        self.style.configure("Link.TButton", anchor="w", padding=(10, 7), background=CARD_BG, foreground=ACCENT_DARK)
        self.style.map("Link.TButton", background=[("active", "#e6f6fb")])
        self.style.configure("TNotebook", background=APP_BG, borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 9, "bold"))

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        cover_frame = ttk.Frame(self.notebook, padding=20)
        workflow_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(cover_frame, text="Cover")
        self.notebook.add(workflow_frame, text="Workflow")

        self._build_cover_page(cover_frame)

        root_frame = workflow_frame
        root_frame.columnconfigure(0, weight=1)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(1, weight=1)
        root_frame.rowconfigure(3, weight=1)

        settings_row = ttk.Frame(root_frame)
        settings_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        settings_row.columnconfigure(0, weight=1)
        settings_row.columnconfigure(1, weight=1)

        self._build_paths_frame(settings_row)
        self._build_network_frame(settings_row)
        self._build_case_frame(root_frame)
        self._build_actions_frame(root_frame)
        self._build_log_frame(root_frame)

    def _build_cover_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        card = ttk.Frame(parent, padding=28, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text="oppoCT Single-Case DICOM Workflow", style="CoverTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(
            card,
            text="Receive one CT study, manually scan it, segment lumbar vertebrae, QC with the existing viewer, then pass-and-send or fail.",
            style="CoverSubtitle.TLabel",
            wraplength=980,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="w", pady=(0, 18))

        body = (
            "Workflow\n"
            "1. Start the DICOM SCP and receive a study into the incoming folder.\n"
            "2. Click Scan Incoming Folder and choose the CT series when more than one is present.\n"
            "3. Run segmentation, then open the unchanged QC viewer for review and vertebral label shifting.\n"
            "4. Mark fail locally, or mark pass and send a DICOM Secondary Capture report.\n\n"
            "References\n"
            "- Wasserthal J, et al. TotalSegmentator: robust segmentation of 104 anatomical structures in CT images. arXiv:2208.05868.\n"
            "- pydicom project documentation for DICOM parsing and writing.\n"
            "- pynetdicom project documentation for DICOM networking, C-STORE SCP, and C-STORE SCU.\n"
            "- Existing oppoCT batch verification workflow and VerificationViewer are reused unchanged.\n\n"
            "Disclaimer\n"
            "This software is intended for research and workflow assistance only. It is not a standalone diagnostic device, "
            "does not replace physician interpretation, and generated segmentations or reports must be reviewed by qualified clinical staff. "
            "Confirm DICOM routing, patient identity, and institutional compliance before sending results to PACS or another clinical system."
        )
        text = tk.Text(
            card,
            height=18,
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            bg=CARD_BG,
            fg=TEXT_DARK,
            font=("Segoe UI", 10),
        )
        text.grid(row=2, column=0, sticky="nsew", pady=(0, 18))
        text.insert("1.0", body)
        text.config(state=tk.DISABLED)

        ttk.Button(
            card,
            text="Open Workflow",
            style="Accent.TButton",
            command=lambda: self.notebook.select(1),
        ).grid(row=3, column=0, sticky="w")

    def _build_paths_frame(self, parent: ttk.Frame) -> None:
        section = CollapsibleSection(parent, "Folder and segmentation settings", expanded=False)
        section.grid(row=0, column=0, sticky="new", padx=(0, 5))
        frame = section.body
        frame.columnconfigure(1, weight=1)

        self._path_row(frame, 0, "Incoming folder", self.incoming_dir_var)
        self._path_row(frame, 1, "Output folder", self.output_dir_var)
        self._path_row(frame, 2, "Temp folder", self.temp_dir_var)

        ttk.Label(frame, text="Device", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.device_var,
            values=["gpu", "cpu"],
            state="readonly",
            width=8,
        ).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Checkbutton(frame, text="Fast segmentation", variable=self.fast_var).grid(
            row=3, column=2, sticky="w", padx=6
        )

    def _build_network_frame(self, parent: ttk.Frame) -> None:
        section = CollapsibleSection(parent, "DICOM network settings", expanded=False)
        section.grid(row=0, column=1, sticky="new", padx=(5, 0))
        frame = section.body
        for col in (1, 3):
            frame.columnconfigure(col, weight=1)

        ttk.Label(frame, text="SCP AE", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.scp_ae_var, width=14).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(frame, text="Host", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Entry(frame, textvariable=self.scp_host_var, width=14).grid(row=0, column=3, sticky="ew", pady=3)
        ttk.Label(frame, text="Port", style="Card.TLabel").grid(row=0, column=4, sticky="w", padx=(8, 0), pady=3)
        ttk.Entry(frame, textvariable=self.scp_port_var, width=7).grid(row=0, column=5, sticky="w", pady=3)

        ttk.Label(frame, text="SCU AE", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.scu_ae_var, width=14).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(frame, text="Dest AE", style="Card.TLabel").grid(row=1, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Entry(frame, textvariable=self.dest_ae_var, width=14).grid(row=1, column=3, sticky="ew", pady=3)
        ttk.Label(frame, text="Dest", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.dest_host_var, width=14).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(frame, text="Port", style="Card.TLabel").grid(row=2, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Entry(frame, textvariable=self.dest_port_var, width=7).grid(row=2, column=3, sticky="w", pady=3)

        self.start_scp_button = ttk.Button(frame, text="Start SCP", command=self.start_scp)
        self.start_scp_button.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.stop_scp_button = ttk.Button(frame, text="Stop SCP", command=self.stop_scp)
        self.stop_scp_button.grid(row=3, column=1, sticky="w", pady=(8, 0))

    def _build_case_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Manual Scan Results", padding=10)
        frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        left = ttk.Frame(frame)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="Studies").grid(row=0, column=0, sticky="w")
        self.case_list = tk.Listbox(left, height=8, exportselection=False)
        self.case_list.configure(
            bg="#ffffff",
            fg=TEXT_DARK,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#bcccdc",
            relief=tk.FLAT,
        )
        self.case_list.grid(row=1, column=0, sticky="nsew")
        self.case_list.bind("<<ListboxSelect>>", self.on_case_select)
        case_scroll = ttk.Scrollbar(left, orient="vertical", command=self.case_list.yview)
        case_scroll.grid(row=1, column=1, sticky="ns")
        self.case_list.configure(yscrollcommand=case_scroll.set)

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Series").grid(row=0, column=0, sticky="w")
        self.series_list = tk.Listbox(right, height=8, exportselection=False)
        self.series_list.configure(
            bg="#ffffff",
            fg=TEXT_DARK,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#bcccdc",
            relief=tk.FLAT,
        )
        self.series_list.grid(row=1, column=0, sticky="nsew")
        self.series_list.bind("<<ListboxSelect>>", self.on_series_select)
        series_scroll = ttk.Scrollbar(right, orient="vertical", command=self.series_list.yview)
        series_scroll.grid(row=1, column=1, sticky="ns")
        self.series_list.configure(yscrollcommand=series_scroll.set)

        self.case_detail_var = tk.StringVar(value="No study selected.")
        ttk.Label(frame, textvariable=self.case_detail_var, justify=tk.LEFT, wraplength=1080).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

    def _build_actions_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Workflow Actions", padding=10)
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        self.scan_button = ttk.Button(
            frame,
            text="Scan Incoming Folder",
            style="Accent.TButton",
            command=self.scan_incoming,
        )
        self.scan_button.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.segment_button = ttk.Button(frame, text="Run Segmentation", command=self.run_segmentation)
        self.segment_button.grid(row=0, column=1, sticky="w", padx=(0, 8))
        self.qc_button = ttk.Button(frame, text="Open QC Viewer", command=self.open_qc_viewer)
        self.qc_button.grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.pass_send_button = ttk.Button(frame, text="Pass and Send", command=self.pass_and_send)
        self.pass_send_button.grid(row=0, column=3, sticky="w", padx=(0, 8))
        self.fail_button = ttk.Button(frame, text="Mark Fail", command=self.mark_fail)
        self.fail_button.grid(row=0, column=4, sticky="w", padx=(0, 8))

        ttk.Label(frame, textvariable=self.status_var, wraplength=1080, justify=tk.LEFT).grid(
            row=1, column=0, columnspan=5, sticky="ew", pady=(8, 0)
        )

    def _build_log_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Log", padding=10)
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        parent.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(frame, height=12, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.configure(
            bg="#fbfdff",
            fg=TEXT_DARK,
            insertbackground=TEXT_DARK,
            highlightthickness=1,
            highlightbackground="#bcccdc",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _path_row(self, frame: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(frame, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse", command=lambda v=variable: self.browse_folder(v)).grid(
            row=row, column=2, sticky="w", padx=(6, 0), pady=4
        )

    def browse_folder(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=variable.get() or str(PROJECT_ROOT))
        if path:
            variable.set(path)

    def log(self, message: str) -> None:
        append_bounded_log(self.log_text, message)

    def log_from_thread(self, message: str) -> None:
        self.root.after(0, self.log, message)

    def clear_log(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_scp(self) -> None:
        if self.scp is not None and self.scp.is_running:
            return
        try:
            config = DicomScpConfig(
                ae_title=self.scp_ae_var.get().strip() or "OPPOCT_SCP",
                host=self.scp_host_var.get().strip() or "0.0.0.0",
                port=int(self.scp_port_var.get().strip()),
                incoming_root=Path(self.incoming_dir_var.get()),
            )
            self.scp = DicomStoreSCP(
                config,
                on_instance_received=self._on_instance_received,
                on_association_released=self._on_association_released,
            )
            self.scp.start()
        except Exception as exc:
            logger.exception("Failed to start DICOM SCP")
            messagebox.showerror("SCP Start Failed", str(exc))
            return

        self.status_var.set(f"DICOM SCP listening as {config.ae_title} on {config.host}:{config.port}.")
        self.log(f"SCP started. Incoming folder: {config.incoming_root}")
        self._refresh_action_state()

    def stop_scp(self) -> None:
        if self.scp is not None:
            self.scp.stop()
            self.scp = None
        if self._receive_timer is not None:
            self._receive_timer.cancel()
            self._receive_timer = None
        self.status_var.set("DICOM SCP stopped.")
        self.log("SCP stopped.")
        self._refresh_action_state()

    def _on_instance_received(self, item: ReceivedInstance) -> None:
        self.root.after(0, self._record_received_instance, item)
        if self._receive_timer is not None:
            self._receive_timer.cancel()
        self._receive_timer = threading.Timer(
            2.0,
            lambda: self.root.after(0, self._notify_study_ready, item.study_instance_uid),
        )
        self._receive_timer.daemon = True
        self._receive_timer.start()

    def _on_association_released(self) -> None:
        self.root.after(0, self.status_var.set, "DICOM association released. Study may be ready to scan.")

    def _record_received_instance(self, item: ReceivedInstance) -> None:
        self.status_var.set(
            f"Receiving study {item.study_instance_uid}: {item.modality or '?'} instance {item.sop_instance_uid}"
        )
        self.log(
            "Received "
            f"PatientID={item.patient_id or '?'} Study={item.study_instance_uid} "
            f"Series={item.series_instance_uid} File={item.file_path.name}"
        )

    def _notify_study_ready(self, study_uid: str) -> None:
        self.status_var.set(f"Study {study_uid} received. Click Scan Incoming Folder when ready.")
        self.log(f"Study {study_uid} appears ready for manual scan.")

    def scan_incoming(self) -> None:
        incoming = Path(self.incoming_dir_var.get())
        output_base = Path(self.output_dir_var.get())
        self.clear_log()
        self.log(f"Scanning incoming folder: {incoming}")

        try:
            folders = find_patient_folders(incoming)
            cases: List[CaseState] = []
            for folder in folders:
                metadata = get_patient_metadata(folder)
                patient_id = metadata.get("patient_id") or folder.name
                patient_name = metadata.get("patient_name") or ""
                exam_date = metadata.get("study_date") or ""
                series_items = enumerate_dicom_series(folder)
                case = CaseState(
                    dicom_folder=folder,
                    patient_id=patient_id,
                    patient_name=patient_name,
                    exam_date=exam_date,
                    study_id=folder.name,
                    series_items=series_items,
                )
                self._autoselect_single_ct_series(case)
                if output_base.exists():
                    self._attach_existing_output(case, output_base)
                cases.append(case)
        except Exception as exc:
            logger.exception("Incoming scan failed")
            messagebox.showerror("Scan Failed", str(exc))
            self.status_var.set("Scan failed.")
            return

        self.cases = cases
        self.selected_case_index = None
        self.selected_series_index = None
        self._refresh_case_list()
        self.status_var.set(f"Scan complete: {len(cases)} study/studies found.")
        self.log(f"Scan complete: {len(cases)} study/studies found.")
        self._refresh_action_state()

    def _autoselect_single_ct_series(self, case: CaseState) -> None:
        ct_items = [item for item in case.series_items if str(item.get("modality", "")).upper() == "CT"]
        if len(ct_items) == 1:
            item = ct_items[0]
            case.series_instance_uid = str(item.get("series_instance_uid", "") or "")
            case.series_description = str(item.get("series_description", "") or "")

    def _attach_existing_output(self, case: CaseState, output_base: Path) -> None:
        normalized_patient = "".join(ch for ch in case.patient_id.upper() if ch.isalnum() or ch == "_") or "UNKNOWN"
        output_dir = output_base / normalized_patient / case.study_id
        segmentation_dir = output_dir / "segmentations"
        if check_segmentations_exist(segmentation_dir):
            case.output_dir = output_dir
            case.segmentation_dir = segmentation_dir
            case.status = "not checked"

    def _refresh_case_list(self) -> None:
        self.case_list.delete(0, tk.END)
        for case in self.cases:
            label = (
                f"[{case.status.upper()}] {case.patient_id} | {case.exam_date or '?'} | "
                f"{case.dicom_folder.name} | {len(case.series_items)} series"
            )
            self.case_list.insert(tk.END, label)

    def on_case_select(self, _event=None) -> None:
        selection = self.case_list.curselection()
        if not selection:
            self.selected_case_index = None
            self.selected_series_index = None
            self._refresh_series_list()
            self._refresh_action_state()
            return
        self.selected_case_index = int(selection[0])
        self.selected_series_index = None
        self._refresh_series_list()
        self._refresh_case_details()
        self._refresh_action_state()

    def _refresh_series_list(self) -> None:
        self.series_list.delete(0, tk.END)
        case = self.current_case()
        if case is None:
            return
        selected_index = None
        for index, item in enumerate(case.series_items):
            uid = str(item.get("series_instance_uid", "") or "")
            desc = str(item.get("series_description", "") or "")
            modality = str(item.get("modality", "") or "")
            count = item.get("num_instances", "")
            uid_short = uid[-12:] if uid else "NO_UID"
            marker = "*" if uid and uid == case.series_instance_uid else " "
            self.series_list.insert(tk.END, f"{marker} {modality or '?'} | n={count} | {desc[:48]} | ...{uid_short}")
            if uid and uid == case.series_instance_uid:
                selected_index = index
        if selected_index is not None:
            self.series_list.selection_set(selected_index)
            self.selected_series_index = selected_index

    def on_series_select(self, _event=None) -> None:
        case = self.current_case()
        selection = self.series_list.curselection()
        if case is None or not selection:
            self.selected_series_index = None
            self._refresh_action_state()
            return
        index = int(selection[0])
        self.selected_series_index = index
        item = case.series_items[index]
        case.series_instance_uid = str(item.get("series_instance_uid", "") or "")
        case.series_description = str(item.get("series_description", "") or "")
        self._refresh_case_details()
        self._refresh_action_state()

    def _refresh_case_details(self) -> None:
        case = self.current_case()
        if case is None:
            self.case_detail_var.set("No study selected.")
            return
        self.case_detail_var.set(
            "\n".join(
                [
                    f"Patient ID: {case.patient_id}",
                    f"Patient Name: {case.patient_name}",
                    f"Exam Date: {case.exam_date}",
                    f"DICOM Folder: {case.dicom_folder}",
                    f"Selected SeriesInstanceUID: {case.series_instance_uid}",
                    f"Series Description: {case.series_description}",
                    f"Status: {case.status}",
                    f"Segmentation Dir: {case.segmentation_dir or ''}",
                ]
            )
        )

    def current_case(self) -> Optional[CaseState]:
        if self.selected_case_index is None:
            return None
        if self.selected_case_index < 0 or self.selected_case_index >= len(self.cases):
            return None
        return self.cases[self.selected_case_index]

    def run_segmentation(self) -> None:
        case = self.current_case()
        if case is None:
            return
        if not case.series_instance_uid:
            messagebox.showerror("Select CT Series", "Choose a CT series before segmentation.")
            return
        selected_item = case.series_items[self.selected_series_index] if self.selected_series_index is not None else {}
        if str(selected_item.get("modality", "")).upper() != "CT":
            messagebox.showerror("Select CT Series", "The selected series is not a CT series.")
            return
        if self._worker_running:
            return

        output_base = Path(self.output_dir_var.get())
        temp_dir = Path(self.temp_dir_var.get())
        output_base.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        case.status = "segmenting"
        case.qc_result = None
        self._worker_running = True
        self._refresh_case_list()
        self._refresh_case_details()
        self._refresh_action_state()
        self.log(f"Starting segmentation for {case.patient_id} ({case.series_description or case.series_instance_uid})")

        def worker() -> None:
            try:
                result = process_single_patient(
                    dicom_folder=case.dicom_folder,
                    output_base_dir=output_base,
                    temp_dir=temp_dir,
                    fast_segmentation=self.fast_var.get(),
                    device=self.device_var.get(),
                    keep_temp_files=True,
                    forced_study_id=case.study_id,
                    series_instance_uid=case.series_instance_uid,
                )
            except Exception as exc:
                logger.exception("Segmentation worker failed")
                self.root.after(0, self._segmentation_finished, case, None, exc)
                return
            self.root.after(0, self._segmentation_finished, case, result, None)

        threading.Thread(target=worker, daemon=True).start()

    def _segmentation_finished(self, case: CaseState, result: Optional[Dict], exc: Optional[Exception]) -> None:
        self._worker_running = False
        if exc is not None:
            case.status = "failed_pipeline"
            self.status_var.set("Segmentation failed.")
            self.log(f"Segmentation failed: {exc}")
            messagebox.showerror("Segmentation Failed", str(exc))
        elif result is None or result.get("status") != "success":
            case.status = "failed_pipeline"
            error = "" if result is None else str(result.get("error") or "Pipeline failed")
            self.status_var.set("Segmentation failed.")
            self.log(f"Segmentation failed: {error}")
            messagebox.showerror("Segmentation Failed", error)
        else:
            case.status = "not checked"
            case.output_dir = Path(str(result.get("output_dir")))
            case.segmentation_dir = case.output_dir / "segmentations"
            self.status_var.set("Segmentation complete. Open QC Viewer.")
            self.log(f"Segmentation complete: {case.segmentation_dir}")

        self._refresh_case_list()
        self._refresh_case_details()
        self._refresh_action_state()

    def open_qc_viewer(self) -> None:
        case = self.current_case()
        if case is None or case.segmentation_dir is None:
            return
        if not check_segmentations_exist(case.segmentation_dir):
            messagebox.showerror("Missing Segmentation", f"No segmentation masks found:\n{case.segmentation_dir}")
            return

        try:
            self.status_var.set("Loading QC viewer...")
            self.root.update_idletasks()
            ct_volume, ct_img = load_ct_for_verification(
                case.dicom_folder,
                case.segmentation_dir,
                series_instance_uid=(case.series_instance_uid or None),
            )
            masks = load_masks_for_verification(case.segmentation_dir, ct_img)
            viewer = VerificationViewer(
                ct_volume=ct_volume,
                masks=masks,
                dicom_folder=case.dicom_folder,
                patient_id=case.patient_id,
                exam_date=case.exam_date or None,
            )
            result = viewer.show(tk_root=self.root)
        except Exception as exc:
            logger.exception("QC viewer failed")
            self.status_var.set("QC viewer failed.")
            self.log(f"QC viewer failed: {exc}")
            messagebox.showerror("QC Failed", str(exc))
            return

        case.qc_result = result
        if result.get("is_successful") is True:
            case.status = "success"
            self.status_var.set("QC marked pass. Click Pass and Send to transmit the Secondary Capture.")
            self.log("QC marked pass.")
        elif result.get("is_successful") is False:
            case.status = "failed"
            self.status_var.set("QC marked fail. No DICOM send will be performed.")
            self.log("QC marked fail.")
        else:
            case.status = "not checked"
            self.status_var.set("QC closed without pass/fail. Reopen viewer or mark fail.")
            self.log("QC closed without pass/fail.")

        self._refresh_case_list()
        self._refresh_case_details()
        self._refresh_action_state()

    def mark_fail(self) -> None:
        case = self.current_case()
        if case is None:
            return
        case.status = "failed"
        self.status_var.set("Case marked fail. No DICOM send will be performed.")
        self.log(f"Case {case.patient_id} marked fail.")
        self._refresh_case_list()
        self._refresh_case_details()
        self._refresh_action_state()

    def pass_and_send(self) -> None:
        case = self.current_case()
        if case is None or case.qc_result is None or case.qc_result.get("is_successful") is not True:
            return
        if case.output_dir is None:
            messagebox.showerror("Missing Output", "Segmentation output directory is not available.")
            return
        if self._worker_running:
            return

        try:
            config = DicomScuConfig(
                ae_title=self.scu_ae_var.get().strip() or "OPPOCT_SCU",
                destination_ae_title=self.dest_ae_var.get().strip() or "ANY-SCP",
                destination_host=self.dest_host_var.get().strip() or "127.0.0.1",
                destination_port=int(self.dest_port_var.get().strip()),
            )
        except ValueError as exc:
            messagebox.showerror("Invalid DICOM Destination", str(exc))
            return

        self._worker_running = True
        self._refresh_action_state()
        self.status_var.set("Creating and sending Secondary Capture...")
        self.log("Creating Secondary Capture report and sending via DICOM SCU.")

        def worker() -> None:
            try:
                source = find_reference_dicom(case.dicom_folder, case.series_instance_uid or None)
                send_dir = case.output_dir / "dicom_send"
                output_path = send_dir / "oppoct_qc_secondary_capture.dcm"
                created = create_secondary_capture_report(source, output_path, case.qc_result or {})
                result = send_secondary_capture(created, config)
            except Exception as exc:
                logger.exception("Pass/send failed")
                self.root.after(0, self._send_finished, case, None, exc)
                return
            self.root.after(0, self._send_finished, case, (created, result), None)

        threading.Thread(target=worker, daemon=True).start()

    def _send_finished(self, case: CaseState, payload, exc: Optional[Exception]) -> None:
        self._worker_running = False
        if exc is not None:
            self.status_var.set("DICOM send failed.")
            self.log(f"DICOM send failed: {exc}")
            messagebox.showerror("DICOM Send Failed", str(exc))
        else:
            created, result = payload
            case.secondary_capture_path = Path(created)
            if result.ok:
                self.status_var.set("Pass confirmed and Secondary Capture sent successfully.")
                self.log(f"Sent Secondary Capture: {created}")
                messagebox.showinfo("DICOM Send Complete", "Secondary Capture sent successfully.")
            else:
                self.status_var.set("Secondary Capture was created, but DICOM send failed.")
                self.log(f"DICOM send failed: {result.error}")
                messagebox.showerror("DICOM Send Failed", result.error)
        self._refresh_case_details()
        self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        case = self.current_case()
        scp_running = self.scp is not None and self.scp.is_running
        self.start_scp_button.config(state=tk.DISABLED if scp_running else tk.NORMAL)
        self.stop_scp_button.config(state=tk.NORMAL if scp_running else tk.DISABLED)

        can_segment = (
            case is not None
            and bool(case.series_instance_uid)
            and not self._worker_running
            and case.status not in {"segmenting"}
        )
        can_qc = (
            case is not None
            and case.segmentation_dir is not None
            and not self._worker_running
            and check_segmentations_exist(case.segmentation_dir)
        )
        can_send = (
            case is not None
            and case.qc_result is not None
            and case.qc_result.get("is_successful") is True
            and not self._worker_running
        )
        can_fail = case is not None and not self._worker_running

        self.scan_button.config(state=tk.NORMAL if not self._worker_running else tk.DISABLED)
        self.segment_button.config(state=tk.NORMAL if can_segment else tk.DISABLED)
        self.qc_button.config(state=tk.NORMAL if can_qc else tk.DISABLED)
        self.pass_send_button.config(state=tk.NORMAL if can_send else tk.DISABLED)
        self.fail_button.config(state=tk.NORMAL if can_fail else tk.DISABLED)

    def on_close(self) -> None:
        if self._receive_timer is not None:
            self._receive_timer.cancel()
            self._receive_timer = None
        if self.scp is not None:
            try:
                self.scp.stop()
            except Exception:
                logger.exception("Failed to stop SCP on close")
        self.root.destroy()


def launch_gui() -> None:
    root = tk.Tk()
    SingleCaseWorkflowApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
