"""Model registry — build(kind, config) is the only public surface.

Concrete model classes are internal to this package; the SDK calls build()
and receives an opaque object. See PLAN.md § 14 item 2 and PRD_models.md.
"""

__all__ = ["build"]


def build(kind: str, config: object) -> object:
    """Instantiate and return a model for the given kind.

    Args:
        kind: one of 'fc', 'rnn', 'lstm'.
        config: full config dict (model sub-section will be extracted).

    Returns:
        Instantiated model (a SignalExtractor subclass).

    Raises:
        ValueError: if kind is not in the registry.
    """
    raise NotImplementedError("M3")
