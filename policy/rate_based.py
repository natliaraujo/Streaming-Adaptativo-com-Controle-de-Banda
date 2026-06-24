"""Implementa a política 1 com servidor fixo e seleção por vazão."""

from domain import StreamingAction, Manifest, ServerInfo
from monitoring.observation_store import ServerObservation
from player import BufferManager
from policy.quality_selector import RateBasedQualitySelector
from policy.streaming_policy import StreamingPolicy


class RateBasedFixedServerPolicy(StreamingPolicy):
    """Política 1: servidor fixo e qualidade baseada em vazão média."""

    def __init__(
        self,
        history_size: int,
        safety_factor: float,
    ) -> None:
        """Configura a política baseline baseada em EWMA.

        Args:
            history_size: Mantido por compatibilidade com a configuração pública.
            safety_factor: Fração conservadora da EWMA disponível para vídeo.
        """
        self.quality_selector = RateBasedQualitySelector(
            history_size=history_size,
            safety_factor=safety_factor,
        )

    def select_action(
        self,
        manifest: Manifest,
        buffer: BufferManager,
        observations: dict[str, ServerObservation],
        throughput_history_kbps: list[float],
        throughput_ewma_kbps: float | None = None,
    ) -> StreamingAction:
        """Escolhe o servidor primário e qualidade pela EWMA de vazão.

        A Política 1 ignora buffer e probes para servir como baseline. A EWMA
        suaviza oscilações recentes antes da aplicação do fator de segurança.

        Args:
            manifest: Manifesto com servidores e representações ordenadas.
            buffer: Estado do player, deliberadamente ignorado pelo baseline.
            observations: Probes dos servidores, deliberadamente ignorados.
            throughput_history_kbps: Histórico mantido pela interface, mas não
                utilizado diretamente.
            throughput_ewma_kbps: EWMA usada para selecionar a representação.

        Returns:
            Ação com servidor de maior prioridade e representação baseada na EWMA.
        """
        server: ServerInfo = sorted(
            manifest.servers,
            key=lambda item: item.priority,
        )[0]

        representation = self.quality_selector.select(
            representations=manifest.representations,
            estimated_throughput_kbps=throughput_ewma_kbps,
        )

        return StreamingAction(
            server=server,
            representation=representation,
        )
