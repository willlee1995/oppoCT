#!/usr/bin/env python
"""Launch the CSV-driven Tkinter batch verification workflow."""

import os
import sys


def _prepare_frozen_runtime() -> None:
    """Configure paths/streams before importing TotalSegmentator-dependent modules."""
    if not getattr(sys, "frozen", False):
        return

    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    os.environ["TOTALSEG_WEIGHTS_PATH"] = os.path.join(base_dir, "totalsegmentator_weights")

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

from scripts.gui_batch_verification import launch_gui


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    launch_gui()
