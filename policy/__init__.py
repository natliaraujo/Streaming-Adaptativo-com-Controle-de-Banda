"""Políticas de seleção de servidor e qualidade de streaming."""

from policy.rate_based import RateBasedFixedServerPolicy
from policy.probe_buffer import ProbeBufferAwarePolicy
from policy.streaming_policy import StreamingPolicy
from policy.training_collection_policy import TrainingDataCollectionPolicy


def __getattr__(name: str):
    if name == "RnnStreamingPolicy":
        from policy.rnn import RnnStreamingPolicy

        return RnnStreamingPolicy

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RateBasedFixedServerPolicy",
    "ProbeBufferAwarePolicy",
    "RnnStreamingPolicy",
    "StreamingPolicy",
    "TrainingDataCollectionPolicy"
]
