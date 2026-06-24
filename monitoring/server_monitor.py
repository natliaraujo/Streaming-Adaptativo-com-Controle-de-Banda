"""
Define a thread responsável por monitorar continuamente um servidor.

Cada monitor executa probes periódicos em um servidor específico, mede latência,
vazão aproximada e jitter, e registra a observação no `ObservationStore`.

Esse monitoramento é usado pela política 3 baseada em RNN.
"""

import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from domain.manifest import ServerInfo
from monitoring.observation_store import ObservationStore, ServerObservation


@dataclass(frozen=True)
class MonitorConfig:
    """Configuração de uma thread de monitoramento."""

    interval_s: float = 1.0
    timeout_s: float = 1.0
    probe_path: str = "/segment/240p?seg=1"
    max_probe_bytes: int = 8192
    chunk_size: int = 1024


class ServerMonitor(threading.Thread):
    """Thread que monitora periodicamente um servidor."""

    def __init__(
        self,
        server: ServerInfo,
        store: ObservationStore,
        config: MonitorConfig,
    ) -> None:
        """
        Inicializa o monitor.

        Args:
            server: Servidor que será monitorado.
            store: Repositório compartilhado de observações.
            config: Configuração do monitor.
        """
        super().__init__(daemon=True)
        self.server: ServerInfo = server
        self.store: ObservationStore = store
        self.config: MonitorConfig = config
        self._stop_event: threading.Event = threading.Event()

    def stop(self) -> None:
        """Solicita a parada da thread."""
        self._stop_event.set()

    def run(self) -> None:
        """Executa o loop de monitoramento."""
        while not self._stop_event.is_set():
            observation: ServerObservation = self._probe_server()
            self.store.update(observation)
            time.sleep(self.config.interval_s)

    def _probe_server(self) -> ServerObservation:
        """
        Mede o servidor por meio de uma requisição HTTP leve.

        Returns:
            Observação contendo latência, vazão, jitter e status.
        """
        url: str = f"{self.server.url}{self.config.probe_path}"
        start_s: float = time.time()
        last_chunk_time_s: float = start_s
        chunk_intervals_s: list[float] = []
        bytes_received: int = 0

        try:
            with urllib.request.urlopen(url, timeout=self.config.timeout_s) as response:
                success: bool = 200 <= response.status < 300

                while bytes_received < self.config.max_probe_bytes:
                    remaining_bytes: int = self.config.max_probe_bytes - bytes_received
                    read_size: int = min(self.config.chunk_size, remaining_bytes)

                    chunk: bytes = response.read(read_size)

                    if not chunk:
                        break

                    now_s: float = time.time()
                    chunk_intervals_s.append(now_s - last_chunk_time_s)
                    last_chunk_time_s = now_s

                    bytes_received += len(chunk)

            elapsed_s: float = max(time.time() - start_s, 1e-9)
            latency_ms: float = elapsed_s * 1000.0
            throughput_kbps: float = (bytes_received * 8) / (elapsed_s * 1000.0)

            jitter_ms: float = (
                statistics.stdev(chunk_intervals_s) * 1000.0
                if len(chunk_intervals_s) > 1
                else 0.0
            )

            return ServerObservation(
                server_id=self.server.id,
                throughput_kbps=throughput_kbps,
                latency_ms=latency_ms,
                jitter_ms=jitter_ms,
                success=success,
                timestamp_s=time.time(),
                error=None if success else f"HTTP status {response.status}",
            )

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed_s = max(time.time() - start_s, 1e-9)

            return ServerObservation(
                server_id=self.server.id,
                throughput_kbps=None,
                latency_ms=elapsed_s * 1000.0,
                jitter_ms=None,
                success=False,
                timestamp_s=time.time(),
                error=str(exc),
            )
