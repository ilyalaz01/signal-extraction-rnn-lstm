"""Signal generation service.

Generates clean and noisy sinusoidal components and composite signals
per the model in HOMEWORK_BRIEF.md § 3 and PRD_signal_generation.md.

Public surface:
    generate_corpus(config) → Corpus
"""


def generate_corpus(config: object) -> object:
    """Generate the 10-vector sinusoid corpus.

    Args:
        config: validated config dict (signal section will be extracted).

    Returns:
        Corpus dataclass with fields:
            clean[i]  — numpy array, shape (N_SAMPLES,), i ∈ {0..3}
            noisy[i]  — numpy array, shape (N_SAMPLES,), i ∈ {0..3}
            clean_sum — numpy array, shape (N_SAMPLES,)
            noisy_sum — numpy array, shape (N_SAMPLES,)

    See PRD_signal_generation.md § 5 for full tensor contracts.
    """
    raise NotImplementedError("M2")
