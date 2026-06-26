"""
Armazena observações recentes dos servidores de forma thread-safe.

As threads de monitoramento atualizam este repositório com medições periódicas
dos servidores. A thread principal do experimento lê snapshots consistentes
dessas observações para montar as entradas da RNN.
"""

import threading
from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class ServerObservation:
    """Representa uma observação recente de um servidor."""

    server_id: str
    throughput_kbps: float | None
    latency_ms: float | None
    jitter_ms: float | None
    success: bool
    timestamp_s: float
    error: str | None = None


class ObservationStore:
    """Armazena as observações dos servidores com proteção por lock."""

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._data: dict[str, ServerObservation] = {}

    def update(self, observation: ServerObservation) -> None:
        """
        Atualiza a observação de um servidor.

        Args:
            observation: Nova observação do servidor.
        """
        with self._lock:
            self._data[observation.server_id] = observation

    def snapshot(self) -> dict[str, ServerObservation]:
        """
        Retorna uma cópia das observações atuais.

        Returns:
            Dicionário indexado pelo identificador do servidor.
        """
        with self._lock:
            return dict(self._data)

    def get(self, server_id: str) -> ServerObservation | None:
        """
        Retorna a observação mais recente de um servidor.

        Args:
            server_id: Identificador do servidor.

        Returns:
            Observação do servidor ou `None`.
        """
        with self._lock:
            return self._data.get(server_id)


def make_failed_observation(server_id: str, error: str) -> ServerObservation:
    """
    Cria uma observação de falha.

    Args:
        server_id: Identificador do servidor.
        error: Mensagem de erro.

    Returns:
        Observação marcada como malsucedida.
    """
    return ServerObservation(
        server_id=server_id,
        throughput_kbps=None,
        latency_ms=None,
        jitter_ms=None,
        success=False,
        timestamp_s=time(),
        error=error,
    )
