"""Dataset construction service.

Window sampling, train/val/test splitting, and torch Dataset / DataLoader
wrappers. See PRD_dataset_construction.md for tensor contracts and
ADR-016 / ADR-017 for split strategy and shuffling decisions.

Public surface:
    build_dataset(corpus, config) → SplitDatasets
"""


def build_dataset(corpus: object, config: object) -> object:
    """Sample windows from corpus and return labelled DataLoader splits.

    Args:
        corpus: Corpus produced by signal_gen.generate_corpus().
        config: validated config dict (dataset + runtime sections used).

    Returns:
        SplitDatasets(train, val, test) — three torch DataLoader objects.
        Each batch yields (X, y) where:
            X: (B, 14)  for FC  or  (B, 10, 5) for RNN/LSTM
            y: (B, 10)  — target clean window

    See PRD_dataset_construction.md § 5 for the full sampling procedure.
    """
    raise NotImplementedError("M2")
