"""Implementa a política 2 com servidor primário e seleção buffer-aware."""

from config import BUFFER_MIN_S, BUFFER_TARGET_S
from domain.action import StreamingAction
from domain.manifest import Manifest, ServerInfo
from monitoring.observation_store import ServerObservation
from player.buffer import BufferManager
from policy.quality_selector import ThresholdBufferAwareQualitySelector
from policy.streaming_policy import StreamingPolicy


class ProbeBufferAwarePolicy(StreamingPolicy):
    """Política 2: servidor primário fixo e qualidade buffer-aware."""

    def __init__(
        self,
        history_size: int,
        safety_factor: float,
    ) -> None:
        """Configura a janela de fallback e os limiares da Política 2.

        Args:
            history_size: Amostras recentes usadas se a EWMA ainda não existir.
            safety_factor: Fração da vazão usada abaixo do buffer confortável.
        """
        self.history_size: int = history_size
        self.quality_selector = ThresholdBufferAwareQualitySelector(
            safety_factor=safety_factor,
            min_buffer_s=BUFFER_MIN_S,
            target_buffer_s=BUFFER_TARGET_S,
        )

    def select_action(
        self,
        manifest: Manifest,
        buffer: BufferManager,
        observations: dict[str, ServerObservation],
        throughput_history_kbps: list[float],
        throughput_ewma_kbps: float | None = None,
    ) -> StreamingAction:
        """Escolhe servidor primário e qualidade por EWMA mais buffer.

        O servidor secundário nunca é escolhido diretamente por esta política;
        somente o runner pode ativá-lo durante failover. A qualidade é mínima sob
        buffer baixo e progressivamente mais ousada quando há reserva.

        Args:
            manifest: Manifesto com servidor primário e qualidades disponíveis.
            buffer: Estado usado para determinar a zona de segurança.
            observations: Probes recebidos por compatibilidade, mas não usados.
            throughput_history_kbps: Histórico usado antes de existir uma EWMA.
            throughput_ewma_kbps: Estimativa preferencial de vazão.

        Returns:
            Ação no servidor primário com representação buffer-aware.
        """
        server: ServerInfo = min(
            manifest.servers,
            key=lambda item: item.priority,
        )

        estimated_throughput: float = self._estimate_throughput(
            throughput_ewma_kbps=throughput_ewma_kbps,
            throughput_history_kbps=throughput_history_kbps,
            fallback_bitrate_kbps=manifest.representations[0].bitrate_kbps,
        )

        representation = self.quality_selector.select(
            representations=manifest.representations,
            estimated_throughput_kbps=estimated_throughput,
            buffer_level_s=buffer.level_s,
        )

        return StreamingAction(
            server=server,
            representation=representation,
        )

    def _estimate_throughput(
        self,
        throughput_ewma_kbps: float | None,
        throughput_history_kbps: list[float],
        fallback_bitrate_kbps: int,
    ) -> float:
        """Determina a vazão disponível para a decisão da Política 2.

        Args:
            throughput_ewma_kbps: EWMA atual, quando já inicializada.
            throughput_history_kbps: Medições anteriores de download.
            fallback_bitrate_kbps: Menor bitrate do manifesto para inicialização.

        Returns:
            EWMA quando disponível; caso contrário, média recente ou bitrate de
            fallback.
        """
        if throughput_ewma_kbps is not None:
            return throughput_ewma_kbps

        if throughput_history_kbps:
            recent: list[float] = throughput_history_kbps[-self.history_size:]
            return sum(recent) / len(recent)

        return float(fallback_bitrate_kbps)
