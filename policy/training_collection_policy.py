"""
Define uma política exploratória para coleta de dados de treinamento.

Esta política não usa RNN nem tenta escolher a melhor qualidade. Ela gera um CSV
com exemplos variados de estado da rede, buffer, servidores e representações.

A política alterna entre os servidores saudáveis para evitar um dataset enviesado
para apenas um servidor e percorre as qualidades em ciclos embaralhados. Quando
o buffer está crítico, usa a menor representação; quando está apenas baixo,
continua explorando com probabilidade controlada.
"""

from random import Random

from domain.action import StreamingAction
from domain.manifest import Manifest, Representation, ServerInfo
from monitoring.observation_store import ServerObservation
from player.buffer import BufferManager
from policy.streaming_policy import StreamingPolicy


class TrainingDataCollectionPolicy(StreamingPolicy):
    """
    Política exploratória usada para gerar dataset da RNN.

    A política escolhe pares servidor-representação com shuffle bag: embaralha
    as combinações saudáveis disponíveis, consome todos os pares, e só então
    embaralha de novo. Assim, há aleatoriedade sem perder cobertura.
    """

    def __init__(
        self,
        seed: int | None = None,
        low_buffer_explore_probability: float = 0.30,
    ) -> None:
        """
        Inicializa a política de coleta.

        Args:
            seed: Semente opcional para tornar a exploração reproduzível.
            low_buffer_explore_probability: Probabilidade de continuar a
                exploração quando o buffer está abaixo do mínimo, mas ainda
                acima do nível crítico.
        """
        if not 0.0 <= low_buffer_explore_probability <= 1.0:
            raise ValueError(
                "low_buffer_explore_probability deve estar entre 0 e 1."
            )
        self.random = Random(seed)
        self.low_buffer_explore_probability = low_buffer_explore_probability
        self._server_cycle: list[str] = []
        self._action_cycle: list[tuple[str, int]] = []

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
            throughput_history_kbps: Recebido pela interface comum, mas não usado
                para escolher qualidade durante a coleta exploratória.
            throughput_ewma_kbps: Recebido pela interface comum, mas não usado.

        Returns:
            Ação contendo servidor e representação.
        """
        server, representation = self._choose_action(
            manifest=manifest,
            buffer=buffer,
            observations=observations,
        )

        return StreamingAction(
            server=server,
            representation=representation,
        )

    def _choose_action(
        self,
        manifest: Manifest,
        buffer: BufferManager,
        observations: dict[str, ServerObservation],
    ) -> tuple[ServerInfo, Representation]:
        """Escolhe servidor e representação por exploração balanceada."""
        if not manifest.representations:
            raise ValueError("Manifest sem representações para coleta de treino.")

        healthy_servers: list[ServerInfo] = self._healthy_servers(
            manifest=manifest,
            observations=observations,
        )
        if not healthy_servers:
            server = sorted(manifest.servers, key=lambda item: item.priority)[0]
            return server, manifest.representations[0]

        if buffer.level_s <= buffer.critical_level_s:
            server = self._choose_server_from_cycle(healthy_servers)
            return server, manifest.representations[0]

        recovery_threshold_s: float = max(
            buffer.min_level_s,
            manifest.segment_duration_s,
        )
        if (
            buffer.level_s < recovery_threshold_s
            and self.random.random() >= self.low_buffer_explore_probability
        ):
            server = self._choose_server_from_cycle(healthy_servers)
            return server, manifest.representations[0]

        healthy_by_id = {server.id: server for server in healthy_servers}
        valid_pairs: set[tuple[str, int]] = {
            (server.id, index)
            for server in healthy_servers
            for index in range(len(manifest.representations))
        }
        self._action_cycle = [
            pair
            for pair in self._action_cycle
            if pair in valid_pairs
        ]

        if not self._action_cycle:
            self._action_cycle = sorted(valid_pairs)
            self.random.shuffle(self._action_cycle)

        server_id, representation_index = self._action_cycle.pop(0)
        return healthy_by_id[server_id], manifest.representations[representation_index]

    def _healthy_servers(
        self,
        manifest: Manifest,
        observations: dict[str, ServerObservation],
    ) -> list[ServerInfo]:
        """
        Lista servidores saudáveis para a coleta.

        Args:
            manifest: Manifesto do experimento.
            observations: Observações recentes dos servidores.

        Returns:
            Servidores saudáveis no estado atual.
        """
        healthy_servers: list[ServerInfo] = []

        for server in manifest.servers:
            observation: ServerObservation | None = observations.get(server.id)

            if observation is None:
                healthy_servers.append(server)
            elif observation.success:
                healthy_servers.append(server)

        return healthy_servers

    def _choose_server_from_cycle(
        self,
        healthy_servers: list[ServerInfo],
    ) -> ServerInfo:
        """Escolhe servidor com shuffle bag durante recuperação de buffer."""
        healthy_by_id = {server.id: server for server in healthy_servers}
        self._server_cycle = [
            server_id
            for server_id in self._server_cycle
            if server_id in healthy_by_id
        ]

        if not self._server_cycle:
            self._server_cycle = list(healthy_by_id)
            self.random.shuffle(self._server_cycle)

        selected_id = self._server_cycle.pop(0)
        return healthy_by_id[selected_id]

