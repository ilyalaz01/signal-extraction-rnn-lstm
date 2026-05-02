"""RNG seeding — seeds Python, NumPy, and PyTorch for reproducibility.

Called once at SDK initialisation. See PLAN.md § 11.1 and NFR-1 / NFR-2
in PRD.md.

Public surface:
    seed_everything(seed) → None
"""


def seed_everything(seed: int) -> None:
    """Seed all random sources for full reproducibility.

    Seeds: random, numpy.random, torch (CPU + all CUDA devices).
    Also calls torch.use_deterministic_algorithms(True) where supported.

    Args:
        seed: integer seed value (read from config.runtime.seed).
    """
    raise NotImplementedError("M2")
