"""Monitoramento de servidores e montagem de features para políticas."""

from monitoring.observation_store import ObservationStore, ServerObservation
from monitoring.server_monitor import MonitorConfig, ServerMonitor
from monitoring.fault_injecting_store import (
    FaultInjectingObservationStore,
    FaultInjectionConfig,
    FaultWindow,
    build_random_fault_schedule,
)


def __getattr__(name: str):
    if name in {"FeatureConfig", "FeatureHistory", "PlayerState"}:
        from monitoring import feature_builder

        return getattr(feature_builder, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FaultInjectingObservationStore",
    "FaultInjectionConfig",
    "FaultWindow",
    "build_random_fault_schedule",
    "FeatureConfig",
    "FeatureHistory",
    "PlayerState",
    "ObservationStore",
    "ServerObservation",
    "MonitorConfig",
    "ServerMonitor",
]
