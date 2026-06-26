"""
Constrói os vetores e sequências de entrada usados pela RNN.

Este módulo transforma observações dos servidores e estado do player em vetores
numéricos. Também mantém a janela temporal recente necessária para alimentar a
GRU usada pela política 3.
"""

from collections import deque
from dataclasses import dataclass

import torch

from monitoring.observation_store import ServerObservation


@dataclass(frozen=True)
class PlayerState:
    """Estado recente do player usado como entrada da RNN."""

    buffer_level_s: float
    last_bitrate_kbps: float
    last_download_time_s: float
    last_rebuffer_event: int
    last_server_index: int
    startup_phase: int


@dataclass(frozen=True)
class FeatureConfig:
    """Configuração usada na construção das features."""

    sequence_length: int
    server_a_id: str = "A"
    server_b_id: str = "B"
    startup_segments: int = 10


class FeatureHistory:
    """Mantém uma janela temporal de vetores de features."""

    def __init__(self, sequence_length: int, feature_size: int) -> None:
        """
        Inicializa o histórico.

        Args:
            sequence_length: Número de timesteps na sequência.
            feature_size: Número de features por timestep.
        """
        self.sequence_length: int = sequence_length
        self.feature_size: int = feature_size
        self._items: deque[list[float]] = deque(maxlen=sequence_length)

    def append(self, feature_vector: list[float]) -> None:
        """
        Adiciona um vetor de features ao histórico.

        Args:
            feature_vector: Vetor numérico de features.

        Raises:
            ValueError: Se o vetor tiver tamanho incorreto.
        """
        if len(feature_vector) != self.feature_size:
            raise ValueError(
                f"Feature vector com tamanho {len(feature_vector)}, "
                f"esperado {self.feature_size}."
            )

        self._items.append(feature_vector)

    def is_ready(self) -> bool:
        """
        Verifica se o histórico já possui uma sequência completa.

        Returns:
            `True` se houver `sequence_length` vetores armazenados.
        """
        return len(self._items) == self.sequence_length

    def to_sequence(self) -> list[list[float]]:
        """
        Retorna a sequência temporal como lista de listas.

        Returns:
            Sequência no formato `(sequence_length, feature_size)`.

        Raises:
            ValueError: Se o histórico ainda não estiver completo.
        """
        if not self.is_ready():
            raise ValueError("Histórico insuficiente para montar sequência da RNN.")

        return [list(item) for item in self._items]

    def to_tensor(self) -> torch.Tensor:
        """
        Converte o histórico diretamente para tensor sem normalização.

        Returns:
            Tensor com shape `(1, sequence_length, feature_size)`.
        """
        return torch.tensor(
            [self.to_sequence()],
            dtype=torch.float32,
        )


def _value_or_zero(value: float | None) -> float:
    """Converte `None` para zero."""
    return 0.0 if value is None else float(value)


def _success_as_float(observation: ServerObservation | None) -> float:
    """Converte o status de sucesso em `0.0` ou `1.0`."""
    if observation is None:
        return 0.0

    return 1.0 if observation.success else 0.0


def build_feature_vector(
    observations: dict[str, ServerObservation],
    player_state: PlayerState,
    config: FeatureConfig,
) -> list[float]:
    """
    Constrói o vetor de features de um timestep.

    Args:
        observations: Observações recentes dos servidores.
        player_state: Estado atual do player.
        config: Configuração dos identificadores dos servidores.

    Returns:
        Vetor de features com tamanho 16.
    """
    obs_a: ServerObservation | None = observations.get(config.server_a_id)
    obs_b: ServerObservation | None = observations.get(config.server_b_id)

    throughput_a: float = _value_or_zero(
        None if obs_a is None else obs_a.throughput_kbps
    )
    latency_a: float = _value_or_zero(
        None if obs_a is None else obs_a.latency_ms
    )
    jitter_a: float = _value_or_zero(
        None if obs_a is None else obs_a.jitter_ms
    )

    throughput_b: float = _value_or_zero(
        None if obs_b is None else obs_b.throughput_kbps
    )
    latency_b: float = _value_or_zero(
        None if obs_b is None else obs_b.latency_ms
    )
    jitter_b: float = _value_or_zero(
        None if obs_b is None else obs_b.jitter_ms
    )

    return [
        throughput_a,
        latency_a,
        jitter_a,
        _success_as_float(obs_a),
        throughput_b,
        latency_b,
        jitter_b,
        _success_as_float(obs_b),
        throughput_a - throughput_b,
        latency_a - latency_b,
        player_state.buffer_level_s,
        player_state.last_bitrate_kbps,
        player_state.last_download_time_s,
        float(player_state.last_rebuffer_event),
        float(player_state.last_server_index),
        float(player_state.startup_phase),
    ]
