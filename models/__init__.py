"""Modelos neurais usados pelas políticas de streaming."""


def __getattr__(name: str):
    if name == "StreamingRNN":
        from models.rnn import StreamingRNN

        return StreamingRNN

    if name == "LoadedRnnModel":
        from models.checkpoint import LoadedRnnModel

        return LoadedRnnModel

    if name in {"RnnSample", "FeatureNormalizer", "StreamingRnnDataset"}:
        from models import dataset

        return getattr(dataset, name)

    if name in {"TrainConfig", "EpochMetrics"}:
        from models import train

        return getattr(train, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "StreamingRNN",
    "LoadedRnnModel",
    "RnnSample",
    "FeatureNormalizer",
    "StreamingRnnDataset",
    "TrainConfig",
    "EpochMetrics",
]
