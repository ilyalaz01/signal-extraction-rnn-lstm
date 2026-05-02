"""Device resolution — maps a config string to a ``torch.device``.

Called once at SDK initialisation; all downstream services receive the
resolved ``torch.device`` object. See PLAN.md § 11.5 and ADR-006 (deferred).

Public surface:
    resolve_device(device_str) → torch.device
"""

from __future__ import annotations

import torch


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    Args:
        device_str: ``'cpu'``, ``'cuda'``, or ``'auto'``.
            ``'auto'`` selects ``cuda`` when available, else ``cpu``.

    Raises:
        ValueError: if ``device_str`` is not one of the accepted values.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str in ("cpu", "cuda"):
        return torch.device(device_str)
    raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device_str!r}")
