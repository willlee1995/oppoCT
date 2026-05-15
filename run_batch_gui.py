#!/usr/bin/env python
"""Launch the CSV-driven Tkinter batch verification workflow."""

import os
import sys

from src.totalseg_runtime import apply_totalseg_runtime_paths


def _prepare_frozen_runtime() -> None:
    """Configure paths/streams before importing TotalSegmentator-dependent modules."""
    apply_totalseg_runtime_paths()

    if not getattr(sys, "frozen", False):
        return

    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", os.path.join(base_dir, "totalsegmentator_weights"))

    class _DummyStream:
        def write(self, *args, **kwargs):
            pass

        def flush(self):
            pass

    if sys.stdout is None:
        sys.stdout = _DummyStream()
    if sys.stderr is None:
        sys.stderr = _DummyStream()


_prepare_frozen_runtime()


def _abort_frozen_import(message: str) -> None:
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "oppoCT-Batch-QC", 0x10)
        except Exception:
            pass
    raise SystemExit(1)


if getattr(sys, "frozen", False):
    # Fail fast with a clear message if the exe (embedded PYZ) and _internal
    # native wheels drift (common after pip upgrades + exe-only builds).
    try:
        import numpy as _np  # noqa: F401
    except ImportError as exc:
        _abort_frozen_import(
            "This install is inconsistent: the app executable and the "
            "_internal folder do not match (numpy/matplotlib native code).\n\n"
            "Typical causes: pip-upgraded packages after an exe-only build, or "
            "mixing an old _internal tree with a new exe.\n\n"
            "Fix from the oppoCT repo (same venv you use to run PyInstaller):\n"
            "  python build_batch_gui.py\n"
            "If it still fails:\n"
            "  python build_batch_gui.py --clean\n"
            "and remove dist\\oppoCT-Batch-QC-Package before rebuilding.\n\n"
            f"ImportError: {exc}"
        )

try:
    from scripts.gui_batch_verification import launch_gui
except OSError as exc:
    if sys.platform == "win32" and (
        getattr(exc, "winerror", None) == 1455
        or "1455" in str(exc)
        or "paging file" in str(exc).lower()
    ):
        _abort_frozen_import(
            "PyTorch could not load because Windows ran out of virtual memory "
            "(paging file too small).\n\n"
            "Close other applications, increase the system paging file size, "
            "or reboot and try again.\n\n"
            f"{exc}"
        )
    raise


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    launch_gui()
