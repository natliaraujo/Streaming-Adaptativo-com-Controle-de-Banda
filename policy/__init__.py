"""Políticas de seleção de servidor e qualidade de streaming."""

from typing import TYPE_CHECKING

from policy.rate_based import RateBasedFixedServerPolicy
from policy.probe_buffer import ProbeBufferAwarePolicy
from policy.streaming_policy import StreamingPolicy
from policy.training_collection_policy import TrainingDataCollectionPolicy

if TYPE_CHECKING:
    from policy.rnn import RnnStreamingPolicy


def __getattr__(name: str) -> object:
    if name == "RnnStreamingPolicy":
        from policy.rnn import RnnStreamingPolicy

        return RnnStreamingPolicy

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RateBasedFixedServerPolicy",
    "ProbeBufferAwarePolicy",
    "RnnStreamingPolicy",
    "StreamingPolicy",
    "TrainingDataCollectionPolicy",
]
