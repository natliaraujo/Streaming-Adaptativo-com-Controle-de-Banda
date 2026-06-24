"""Injeta janelas temporárias de falha nas observações dos servidores."""

import random
import time
from dataclasses import dataclass

from monitoring.observation_store import ObservationStore, ServerObservation


@dataclass(frozen=True)
class FaultWindow:
    """Janela de indisponibilidade relativa ao início da coleta."""

    server_id: str
    start_after_s: float
    duration_s: float

    @property
    def end_after_s(self) -> float:
        """Calcula o instante relativo de recuperação.

        Returns:
            Soma do início relativo e da duração da falha, em segundos.
        """
        return self.start_after_s + self.duration_s


@dataclass(frozen=True)
class FaultInjectionConfig:
    """Configuração das falhas simuladas."""

    windows: tuple[FaultWindow, ...]
    failed_latency_ms: float = 10_000.0


def build_random_fault_schedule(
    server_ids: list[str],
    faults_per_server: int,
    min_initial_delay_s: float,
    min_healthy_gap_s: float,
    max_healthy_gap_s: float,
    min_failure_duration_s: float,
    max_failure_duration_s: float,
    seed: int,
) -> tuple[FaultWindow, ...]:
    """Cria um cronograma reproduzível de falhas temporárias.

    Cada servidor recebe a mesma quantidade de falhas. A ordem é embaralhada com
    uma semente local e as janelas são separadas por períodos saudáveis, evitando
    indisponibilidade simulada simultânea dos dois servidores.

    Args:
        server_ids: Identificadores que participarão da simulação.
        faults_per_server: Quantidade de janelas atribuída a cada servidor.
        min_initial_delay_s: Período saudável mínimo antes da primeira janela.
        min_healthy_gap_s: Menor intervalo saudável entre falhas.
        max_healthy_gap_s: Maior intervalo saudável entre falhas.
        min_failure_duration_s: Menor duração possível de uma falha.
        max_failure_duration_s: Maior duração possível de uma falha.
        seed: Semente que torna o cronograma reproduzível.

    Returns:
        Janelas ordenadas cronologicamente e sem sobreposição.

    Raises:
        ValueError: Se não houver servidores ou se quantidades e intervalos forem
            inválidos.
    """
    if not server_ids:
        raise ValueError("É necessário informar ao menos um servidor.")

    if faults_per_server < 1:
        raise ValueError("faults_per_server deve ser maior que zero.")

    if not 0.0 <= min_healthy_gap_s <= max_healthy_gap_s:
        raise ValueError("Intervalo de tempo saudável inválido.")

    if not 0.0 < min_failure_duration_s <= max_failure_duration_s:
        raise ValueError("Intervalo de duração de falha inválido.")

    rng = random.Random(seed)
    failure_order: list[str] = server_ids * faults_per_server
    rng.shuffle(failure_order)

    windows: list[FaultWindow] = []
    cursor_s: float = min_initial_delay_s

    for server_id in failure_order:
        cursor_s += rng.uniform(min_healthy_gap_s, max_healthy_gap_s)
        duration_s: float = rng.uniform(
            min_failure_duration_s,
            max_failure_duration_s,
        )
        windows.append(
            FaultWindow(
                server_id=server_id,
                start_after_s=cursor_s,
                duration_s=duration_s,
            )
        )
        cursor_s += duration_s

    return tuple(windows)


class FaultInjectingObservationStore(ObservationStore):
    """
    Wrapper de ObservationStore que injeta falha simulada em um servidor.

    A classe delega as atualizações para um `ObservationStore` real, mas altera
    os snapshots retornados quando a falha simulada está ativa.
    """

    def __init__(
        self,
        base_store: ObservationStore,
        config: FaultInjectionConfig,
    ) -> None:
        """
        Inicializa o wrapper de injeção de falha.

        Args:
            base_store: Store real atualizado pelos monitores.
            config: Configuração da falha simulada.
        """
        super().__init__()
        self.base_store: ObservationStore = base_store
        self.config: FaultInjectionConfig = config
        self.start_time_s: float = time.monotonic()

    def update(self, observation: ServerObservation) -> None:
        """
        Atualiza a observação no store base.

        Args:
            observation: Observação coletada por um monitor.
        """
        self.base_store.update(observation)

    def snapshot(self) -> dict[str, ServerObservation]:
        """
        Retorna um snapshot com possível falha simulada.

        Returns:
            Observações atuais, com o servidor configurado como falho se a janela
            de falha estiver ativa.
        """
        observations: dict[str, ServerObservation] = self.base_store.snapshot()

        elapsed_s: float = time.monotonic() - self.start_time_s

        for window in self.config.windows:
            if window.start_after_s <= elapsed_s < window.end_after_s:
                observations[window.server_id] = self._failed_observation(
                    window.server_id
                )

        return observations

    def get(self, server_id: str) -> ServerObservation | None:
        """
        Retorna a observação de um servidor.

        Args:
            server_id: Identificador do servidor.

        Returns:
            Observação real ou simulada.
        """
        return self.snapshot().get(server_id)

    def _failed_observation(self, server_id: str) -> ServerObservation:
        """Cria uma amostra sintética de indisponibilidade.

        Args:
            server_id: Servidor afetado pela janela ativa.

        Returns:
            Observação malsucedida, com vazão zero e latência elevada.
        """
        return ServerObservation(
            server_id=server_id,
            throughput_kbps=0.0,
            latency_ms=self.config.failed_latency_ms,
            jitter_ms=0.0,
            success=False,
            timestamp_s=time.time(),
            error="simulated temporary server failure",
        )
