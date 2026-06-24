"""
Define a interface geral para políticas de streaming adaptativo.

Diferentemente de uma política ABR clássica, uma política de streaming escolhe
uma ação completa: o servidor de origem e a representação de qualidade do
próximo segmento.

Essa interface permite comparar políticas com diferentes níveis de complexidade,
como baseline com servidor fixo, heurística com probes e política preditiva com
RNN.
"""

from abc import ABC, abstractmethod

from domain.action import StreamingAction
from domain.manifest import Manifest
from monitoring.observation_store import ServerObservation
from player.buffer import BufferManager


class StreamingPolicy(ABC):
    """Interface comum para todas as políticas de streaming adaptativo.

    Toda política deve escolher uma ação completa, isto é, um servidor e uma
    representação de qualidade. Políticas simples podem ignorar observações de
    monitoramento; políticas avançadas, como a RNN, podem usá-las.
    """

    @abstractmethod
    def select_action(
        self,
        manifest: Manifest,
        buffer: BufferManager,
        observations: dict[str, ServerObservation],
        throughput_history_kbps: list[float],
        throughput_ewma_kbps: float | None = None,
    ) -> StreamingAction:
        """Escolhe servidor e representação para o próximo segmento.

        Args:
            manifest: Servidores e representações disponíveis.
            buffer: Estado atual do player.
            observations: Últimos probes conhecidos por servidor.
            throughput_history_kbps: Vazões dos downloads anteriores.
            throughput_ewma_kbps: EWMA atual, quando inicializada.

        Returns:
            Ação completa de servidor e qualidade.
        """
        pass

    def update_last_download_state(
        self,
        bitrate_kbps: float,
        download_time_s: float,
        rebuffer_event: int,
        server_index: int,
    ) -> None:
        """Atualiza o estado interno após um download.

        Políticas que não usam feedback do último download podem ignorar este
        método. A política RNN usa essas informações como parte da próxima
        entrada da rede.

        Args:
            bitrate_kbps: Bitrate nominal da representação baixada.
            download_time_s: Duração do download bem-sucedido.
            rebuffer_event: Indicador de stall durante a operação.
            server_index: Posição do servidor efetivo no manifesto.
        """
        return None
