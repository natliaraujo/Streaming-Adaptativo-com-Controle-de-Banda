"""
Define uma política exploratória para coleta de dados de treinamento.

Esta política não usa RNN. Ela é usada antes do treinamento para gerar um CSV
com exemplos variados de estado da rede, buffer, servidores e decisões tomadas.

A política alterna entre os servidores saudáveis para evitar um dataset enviesado
para apenas um servidor. Quando uma falha simulada é injetada no servidor A, a
política passa a escolher apenas servidores ainda saudáveis.
"""

from domain.action import StreamingAction
from domain.manifest import Manifest, Representation, ServerInfo
from monitoring.observation_store import ServerObservation
from player.buffer import BufferManager
from policy.quality_selector import BufferAwareQualitySelector
from policy.streaming_policy import StreamingPolicy


class TrainingDataCollectionPolicy(StreamingPolicy):
    """
    Política exploratória usada para gerar dataset da RNN.

    A política escolhe servidores de forma alternada entre os saudáveis e escolhe
    a qualidade usando uma estratégia buffer-aware simples.
    """

    def __init__(
        self,
        safety_factor: float,
        history_size: int,
    ) -> None:
        """
        Inicializa a política de coleta.

        Args:
            safety_factor: Fator de segurança para seleção de qualidade.
            history_size: Número de amostras recentes usadas para estimar vazão
                quando não houver observação direta do servidor.
        """
        self.quality_selector = BufferAwareQualitySelector(
            safety_factor=safety_factor,
        )
        self.history_size: int = history_size
        self.step: int = 0

    def select_action(
        self,
        manifest: Manifest,
        buffer: BufferManager,
        observations: dict[str, ServerObservation],
        throughput_history_kbps: list[float],
        throughput_ewma_kbps: float | None = None,
    ) -> StreamingAction:
        """
        Escolhe servidor e qualidade para coleta de dados.

        Args:
            manifest: Manifesto do experimento.
            buffer: Estado atual do buffer.
            observations: Observações recentes dos servidores.
            throughput_history_kbps: Histórico de vazões reais dos downloads.
            throughput_ewma_kbps: EWMA recebida pela interface comum. A coleta
                prefere probes por servidor e usa o histórico como fallback.

        Returns:
            Ação contendo servidor e representação.
        """
        server: ServerInfo = self._choose_server(
            manifest=manifest,
            observations=observations,
        )

        estimated_throughput_kbps: float = self._estimate_server_throughput(
            server=server,
            observations=observations,
            throughput_history_kbps=throughput_history_kbps,
            fallback_bitrate_kbps=manifest.representations[0].bitrate_kbps,
        )

        representation: Representation = self.quality_selector.select(
            representations=manifest.representations,
            predicted_throughput_kbps=estimated_throughput_kbps,
            buffer_level_s=buffer.level_s,
            segment_duration_s=manifest.segment_duration_s,
        )

        self.step += 1

        return StreamingAction(
            server=server,
            representation=representation,
        )

    def _choose_server(
        self,
        manifest: Manifest,
        observations: dict[str, ServerObservation],
    ) -> ServerInfo:
        """
        Escolhe um servidor saudável alternando entre as opções disponíveis.

        Args:
            manifest: Manifesto do experimento.
            observations: Observações recentes dos servidores.

        Returns:
            Servidor escolhido.
        """
        healthy_servers: list[ServerInfo] = []

        for server in manifest.servers:
            observation: ServerObservation | None = observations.get(server.id)

            if observation is None:
                healthy_servers.append(server)
            elif observation.success:
                healthy_servers.append(server)

        if not healthy_servers:
            return sorted(manifest.servers, key=lambda item: item.priority)[0]

        selected_index: int = self.step % len(healthy_servers)

        return healthy_servers[selected_index]

    def _estimate_server_throughput(
        self,
        server: ServerInfo,
        observations: dict[str, ServerObservation],
        throughput_history_kbps: list[float],
        fallback_bitrate_kbps: int,
    ) -> float:
        """
        Estima a vazão disponível para o servidor escolhido.

        Args:
            server: Servidor escolhido.
            observations: Observações recentes dos servidores.
            throughput_history_kbps: Histórico de vazões reais.
            fallback_bitrate_kbps: Bitrate usado como fallback conservador.

        Returns:
            Vazão estimada em kbps.
        """
        observation: ServerObservation | None = observations.get(server.id)

        if (
            observation is not None
            and observation.success
            and observation.throughput_kbps is not None
            and observation.throughput_kbps > 0
        ):
            return observation.throughput_kbps

        if throughput_history_kbps:
            recent: list[float] = throughput_history_kbps[-self.history_size:]
            return sum(recent) / len(recent)

        if server.bandwidth_kbps is not None:
            return server.bandwidth_kbps

        return float(fallback_bitrate_kbps)
