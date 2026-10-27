"""PyTorch compute device selection."""

import torch


def resolve_device(preference: str = "auto") -> str:
    """Resolve compute device based on preference and hardware availability.

    Args:
        preference: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        The device string ('cuda', 'mps', or 'cpu').
    """
    pref = preference.lower().strip()
    if pref != "auto":
        return pref

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
