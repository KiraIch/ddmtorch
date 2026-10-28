"""Device and dtype helpers for CPU / Apple Silicon MPS execution."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``"auto"`` to MPS when available, otherwise CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine")
    return resolved


def resolve_dtype(device: str | torch.device, dtype: torch.dtype | None = None) -> torch.dtype:
    """Pick a safe default dtype.

    MPS does not implement float64, so MPS runs in float32 by default.
    CPU runs in float64 for numerical stability.
    """
    dev = resolve_device(device)
    if dtype is not None:
        if dev.type == "mps" and dtype == torch.float64:
            raise ValueError("MPS does not support float64; use float32 on MPS")
        return dtype
    return torch.float32 if dev.type == "mps" else torch.float64
