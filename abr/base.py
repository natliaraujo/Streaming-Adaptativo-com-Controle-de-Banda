"""Contrato comum para políticas de Adaptive Bitrate (ABR)."""

from abc import ABC, abstractmethod

from domain.manifest import Representation


class AbrPolicy(ABC):
    """Interface para estratégias que escolhem uma representação de vídeo."""

    @abstractmethod
    def select_representation(
        self,
        representations: list[Representation],
        throughput_history_kbps: list[float],
        buffer_level_s: float,
        segment_duration_s: float,
    ) -> Representation:
        """
        Escolhe a representação para o próximo segmento.

        Args:
            representations: Representações disponíveis no manifesto,
                normalmente ordenadas por bitrate crescente.
            throughput_history_kbps: Histórico de vazão medida em kbps.
            buffer_level_s: Nível atual do buffer em segundos.
            segment_duration_s: Duração nominal de cada segmento em segundos.

        Returns:
            A representação que a política deseja baixar em seguida.
        """

        pass
