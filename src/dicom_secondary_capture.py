"""
Create report-style DICOM Secondary Capture objects for single-case QC results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pydicom
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from .dicom_processor import iter_dicom_file_paths_streaming


COPY_STUDY_PATIENT_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "AccessionNumber",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "StudyID",
    "StudyDescription",
    "ReferringPhysicianName",
    "FrameOfReferenceUID",
]


def find_reference_dicom(dicom_folder: Path, series_instance_uid: Optional[str] = None) -> Path:
    """Return a readable source DICOM, preferring the selected CT series when provided."""

    fallback: Optional[Path] = None
    desired_uid = (series_instance_uid or "").strip()
    for path in iter_dicom_file_paths_streaming(dicom_folder):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if fallback is None:
            fallback = path
        if desired_uid and str(getattr(ds, "SeriesInstanceUID", "") or "").strip() == desired_uid:
            return path
        if not desired_uid:
            return path

    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"No readable DICOM files found under {dicom_folder}")


def _format_hu(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _report_lines(qc_result: Dict, app_name: str) -> Iterable[str]:
    patient_id = qc_result.get("patient_id") or ""
    exam_date = qc_result.get("exam_date") or ""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    status = "PASS" if qc_result.get("is_successful") is True else "FAIL"

    yield f"{app_name} Single-Case QC Report"
    yield ""
    yield f"Patient ID: {patient_id}"
    yield f"Exam Date: {exam_date}"
    yield f"QC Status: {status}"
    yield f"Created: {timestamp}"
    yield ""
    yield "Vertebral HU"

    vertebra_hu = qc_result.get("vertebra_hu") or {}
    label_mapping = qc_result.get("label_mapping") or {}
    for label in ["T11", "T12", "L1", "L2", "L3", "L4", "L5"]:
        value = _format_hu(vertebra_hu.get(label))
        source = label_mapping.get(label, "")
        yield f"  {label}: {value or 'n/a'} HU    source: {source or 'n/a'}"

    l1_core_hu = _format_hu(qc_result.get("l1_trabecular_core_hu"))
    yield ""
    yield f"L1 trabecular core HU: {l1_core_hu or 'n/a'}"
    yield ""
    yield "Note: This image is a Secondary Capture summary generated after human QC."


def _render_report_image(lines: Iterable[str], width: int = 1400, height: int = 1000) -> np.ndarray:
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.95
    for index, line in enumerate(lines):
        if index == 0:
            ax.text(0.05, y, line, fontsize=28, fontweight="bold", va="top", color="black")
            y -= 0.07
        elif line == "":
            y -= 0.035
        elif line == "Vertebral HU":
            ax.text(0.05, y, line, fontsize=20, fontweight="bold", va="top", color="black")
            y -= 0.045
        else:
            ax.text(0.05, y, line, fontsize=17, family="monospace", va="top", color="black")
            y -= 0.04
        if y < 0.05:
            break

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def create_secondary_capture_report(
    source_dicom_path: Path,
    output_path: Path,
    qc_result: Dict,
    *,
    app_name: str = "oppoCT",
    series_description: str = "oppoCT QC Secondary Capture",
) -> Path:
    """Create a DICOM Secondary Capture report image from a viewer QC result."""

    source = pydicom.dcmread(str(source_dicom_path), stop_before_pixels=True)
    now = datetime.now()
    content_date = now.strftime("%Y%m%d")
    content_time = now.strftime("%H%M%S")
    image = _render_report_image(_report_lines(qc_result, app_name))

    sop_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    for keyword in COPY_STUDY_PATIENT_TAGS:
        if hasattr(source, keyword):
            setattr(ds, keyword, getattr(source, keyword))

    if not getattr(ds, "StudyInstanceUID", None):
        ds.StudyInstanceUID = generate_uid()
    if not getattr(ds, "PatientID", None):
        ds.PatientID = str(qc_result.get("patient_id") or "UNKNOWN")

    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.Modality = "OT"
    ds.SeriesDescription = series_description
    ds.SeriesNumber = 9901
    ds.InstanceNumber = 1
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.ConversionType = "WSD"
    ds.Manufacturer = app_name
    ds.SoftwareVersions = "single-case-workflow"
    ds.ContentDate = content_date
    ds.ContentTime = content_time
    ds.InstanceCreationDate = content_date
    ds.InstanceCreationTime = content_time
    ds.BurnedInAnnotation = "YES"
    ds.LossyImageCompression = "00"

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows = int(image.shape[0])
    ds.Columns = int(image.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = image.tobytes()

    ds.save_as(str(output_path), write_like_original=False)
    return output_path
