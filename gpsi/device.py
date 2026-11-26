"""Device selection: use CUDA when available, otherwise fall back to CPU."""

import torch


def get_device(requested=None):
    """Return the torch device string to use.

    Args:
        requested: ``None``/``"auto"`` to pick CUDA if available (else CPU),
            or an explicit device string such as ``"cpu"`` or ``"cuda:0"``.
            Requesting CUDA on a machine without it falls back to CPU with a
            warning instead of failing.
    """
    if requested in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print(f"[gpsi] WARNING: device '{requested}' requested but CUDA is not available; using CPU.")
        return "cpu"
    return str(requested)
