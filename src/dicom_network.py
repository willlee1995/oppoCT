"""
DICOM networking helpers for the standalone single-case workflow.

This module is intentionally independent from the existing batch scripts. It writes
received C-STORE instances to a normal folder tree so the current file-based
pipeline can scan and process them without changes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

logger = logging.getLogger(__name__)


ReceivedInstanceCallback = Callable[["ReceivedInstance"], None]
AssociationReleasedCallback = Callable[[], None]


@dataclass(frozen=True)
class DicomScpConfig:
    ae_title: str = "OPPOCT_SCP"
    host: str = "0.0.0.0"
    port: int = 11112
    incoming_root: Path = Path("incoming")


@dataclass(frozen=True)
class DicomScuConfig:
    ae_title: str = "OPPOCT_SCU"
    destination_ae_title: str = "ANY-SCP"
    destination_host: str = "127.0.0.1"
    destination_port: int = 11112


@dataclass(frozen=True)
class ReceivedInstance:
    file_path: Path
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    patient_id: str
    patient_name: str
    modality: str


@dataclass(frozen=True)
class DicomStoreStatus:
    file_path: Path
    status: int
    status_text: str


@dataclass(frozen=True)
class DicomSendResult:
    ok: bool
    statuses: List[DicomStoreStatus] = field(default_factory=list)
    error: str = ""


def _require_pynetdicom():
    try:
        import pynetdicom  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pynetdicom is required for DICOM SCP/SCU. Install with "
            "`python -m pip install -r requirements-single-case.txt`."
        ) from exc


def _clean_component(value: object, default: str) -> str:
    text = str(value or "").strip() or default
    safe = []
    for char in text:
        if char.isalnum() or char in "._-":
            safe.append(char)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or default


def _dataset_file_meta(ds, fallback_transfer_syntax=None) -> FileMetaDataset:
    file_meta = FileMetaDataset()
    source_meta = getattr(ds, "file_meta", None)
    if source_meta is not None:
        for elem in source_meta:
            file_meta.add(elem)

    sop_class = getattr(ds, "SOPClassUID", None) or getattr(file_meta, "MediaStorageSOPClassUID", None)
    sop_instance = getattr(ds, "SOPInstanceUID", None) or getattr(
        file_meta, "MediaStorageSOPInstanceUID", None
    )

    file_meta.MediaStorageSOPClassUID = sop_class or generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_instance or generate_uid()
    file_meta.TransferSyntaxUID = (
        getattr(file_meta, "TransferSyntaxUID", None)
        or fallback_transfer_syntax
        or ExplicitVRLittleEndian
    )
    file_meta.ImplementationClassUID = getattr(file_meta, "ImplementationClassUID", None) or generate_uid()
    return file_meta


class DicomStoreSCP:
    """Background C-STORE SCP that writes instances under the configured incoming root."""

    def __init__(
        self,
        config: DicomScpConfig,
        on_instance_received: Optional[ReceivedInstanceCallback] = None,
        on_association_released: Optional[AssociationReleasedCallback] = None,
    ):
        self.config = config
        self.on_instance_received = on_instance_received
        self.on_association_released = on_association_released
        self._server = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        _require_pynetdicom()
        from pynetdicom import AE, evt
        from pynetdicom.presentation import AllStoragePresentationContexts

        with self._lock:
            if self._server is not None:
                return

            self.config.incoming_root.mkdir(parents=True, exist_ok=True)
            ae = AE(ae_title=self.config.ae_title)
            for context in AllStoragePresentationContexts:
                ae.add_supported_context(context.abstract_syntax)

            handlers = [(evt.EVT_C_STORE, self._handle_c_store)]
            if self.on_association_released is not None:
                handlers.append((evt.EVT_RELEASED, self._handle_association_released))

            self._server = ae.start_server(
                (self.config.host, int(self.config.port)),
                block=False,
                evt_handlers=handlers,
            )
            logger.info(
                "DICOM SCP listening as %s on %s:%s",
                self.config.ae_title,
                self.config.host,
                self.config.port,
            )

    def stop(self) -> None:
        with self._lock:
            if self._server is None:
                return
            self._server.shutdown()
            self._server = None
            logger.info("DICOM SCP stopped")

    def _handle_association_released(self, _event) -> None:
        if self.on_association_released is not None:
            try:
                self.on_association_released()
            except Exception:
                logger.exception("DICOM association release callback failed")

    def _handle_c_store(self, event):
        ds = event.dataset
        ds.file_meta = _dataset_file_meta(
            ds,
            fallback_transfer_syntax=getattr(getattr(event, "context", None), "transfer_syntax", None),
        )

        study_uid = _clean_component(getattr(ds, "StudyInstanceUID", None), "NO_STUDY_UID")
        series_uid = _clean_component(getattr(ds, "SeriesInstanceUID", None), "NO_SERIES_UID")
        sop_uid = _clean_component(getattr(ds, "SOPInstanceUID", None), generate_uid())
        modality = str(getattr(ds, "Modality", "") or "").strip()
        patient_id = str(getattr(ds, "PatientID", "") or "").strip()
        patient_name = str(getattr(ds, "PatientName", "") or "").strip()

        study_dir = self.config.incoming_root / study_uid
        series_dir = study_dir / series_uid
        series_dir.mkdir(parents=True, exist_ok=True)
        file_path = series_dir / f"{sop_uid}.dcm"

        ds.save_as(str(file_path), write_like_original=False)
        received = ReceivedInstance(
            file_path=file_path,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_uid,
            patient_id=patient_id,
            patient_name=patient_name,
            modality=modality,
        )

        if self.on_instance_received is not None:
            try:
                self.on_instance_received(received)
            except Exception:
                logger.exception("DICOM instance callback failed")

        return 0x0000


def send_dicom_files(paths: Iterable[Path], config: DicomScuConfig) -> DicomSendResult:
    """Send one or more DICOM files to a remote C-STORE SCP."""

    _require_pynetdicom()
    from pynetdicom import AE

    file_paths = [Path(p) for p in paths]
    if not file_paths:
        return DicomSendResult(ok=False, error="No DICOM files were provided for sending.")

    datasets: List[FileDataset] = []
    for path in file_paths:
        try:
            datasets.append(pydicom.dcmread(str(path)))
        except Exception as exc:
            return DicomSendResult(ok=False, error=f"Could not read DICOM file {path}: {exc}")

    ae = AE(ae_title=config.ae_title)
    requested = set()
    for ds in datasets:
        sop_class = getattr(ds, "SOPClassUID", None) or getattr(
            getattr(ds, "file_meta", None), "MediaStorageSOPClassUID", None
        )
        if not sop_class:
            return DicomSendResult(ok=False, error="DICOM file is missing SOPClassUID.")
        transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ExplicitVRLittleEndian)
        key = (str(sop_class), str(transfer_syntax))
        if key not in requested:
            ae.add_requested_context(sop_class, [transfer_syntax, ExplicitVRLittleEndian])
            requested.add(key)

    assoc = ae.associate(
        config.destination_host,
        int(config.destination_port),
        ae_title=config.destination_ae_title,
    )
    if not assoc.is_established:
        return DicomSendResult(
            ok=False,
            error=(
                "Could not establish DICOM association to "
                f"{config.destination_ae_title}@{config.destination_host}:{config.destination_port}."
            ),
        )

    statuses: List[DicomStoreStatus] = []
    try:
        for path, ds in zip(file_paths, datasets):
            status = assoc.send_c_store(ds)
            code = int(getattr(status, "Status", 0xC000)) if status is not None else 0xC000
            status_text = "Success" if code == 0x0000 else f"DICOM status 0x{code:04X}"
            statuses.append(DicomStoreStatus(file_path=path, status=code, status_text=status_text))
    finally:
        assoc.release()

    ok = all(item.status == 0x0000 for item in statuses)
    error = "" if ok else "; ".join(f"{s.file_path.name}: {s.status_text}" for s in statuses)
    return DicomSendResult(ok=ok, statuses=statuses, error=error)


def send_secondary_capture(path: Path, config: DicomScuConfig) -> DicomSendResult:
    return send_dicom_files([path], config)
