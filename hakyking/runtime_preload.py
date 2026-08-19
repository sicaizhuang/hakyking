from __future__ import annotations

import os


_NEURAL_PITCH_PRELOAD_DONE = False
_NEURAL_PITCH_PRELOAD_ERROR: str | None = None


def preload_neural_pitch_runtime() -> str | None:
    """Load ONNX Runtime before Qt to avoid Windows DLL initialization issues."""

    global _NEURAL_PITCH_PRELOAD_DONE, _NEURAL_PITCH_PRELOAD_ERROR
    if _NEURAL_PITCH_PRELOAD_DONE:
        return _NEURAL_PITCH_PRELOAD_ERROR
    _NEURAL_PITCH_PRELOAD_DONE = True

    # ONNX Runtime is optional and some Windows installations fail while
    # loading its native DLLs. Keep the stable WORLD/pYIN path as the default;
    # developers can explicitly opt in after the RMVPE runtime passes checks.
    if os.environ.get("HAKYKING_PRELOAD_NEURAL_PITCH", "0") != "1":
        return None

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

    try:
        import onnxruntime  # noqa: F401

        # Importing rmvpe_onnx here keeps its ONNX provider bindings warm for
        # later worker threads, but does not instantiate or download the model.
        import rmvpe_onnx  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional neural tracker fallback
        _NEURAL_PITCH_PRELOAD_ERROR = str(exc)
    return _NEURAL_PITCH_PRELOAD_ERROR
