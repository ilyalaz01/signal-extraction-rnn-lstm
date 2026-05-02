"""Device resolution — maps config string to torch.device.

Called once at SDK initialisation; all downstream services receive the
resolved torch.device object. See PLAN.md § 11.5 and ADR-006 (deferred).

Public surface:
    resolve_device(device_str) → object  (torch.device in M2)
"""


def resolve_device(device_str: str) -> object:
    """Resolve a device string to a torch.device.

    Args:
        device_str: one of 'cpu', 'cuda', or 'auto'.
            'auto' selects 'cuda' if torch.cuda.is_available(), else 'cpu'.

    Returns:
        torch.device instance.

    Raises:
        ValueError: if device_str is not one of the accepted values.
    """
    raise NotImplementedError("M2")
