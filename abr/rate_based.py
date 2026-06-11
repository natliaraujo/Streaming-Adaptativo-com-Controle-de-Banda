"""Política ABR baseada na média recente de vazão medida."""

from abr.base import AbrPolicy
from domain.manifest import Representation


class RateBasedAbrPolicy(AbrPolicy):
    """Seleciona a maior qualidade que cabe em uma estimativa segura de vazão."""

    def __init__(self, history_size: int, safety_factor: float) -> None:
        """
        Inicializa a política baseada em taxa.

        Args:
            history_size: Quantidade máxima de amostras recentes usadas na média.
            safety_factor: Fator multiplicativo aplicado à média para evitar
                escolher bitrates muito próximos da vazão estimada.
        """

        self.history_size = history_size
        self.safety_factor = safety_factor

    def select_representation(
        self,
        representations: list[Representation],
        throughput_history_kbps: list[float],
        buffer_level_s: float,
        segment_duration_s: float,
    ) -> Representation:
        """
        Retorna a maior representação com bitrate abaixo da vazão segura.

        A política usa a média das últimas `history_size` vazões. Quando ainda
        não há histórico, assume o bitrate da menor representação como estimativa
        inicial. Os argumentos de buffer fazem parte da interface ABR, mas esta
        implementação baseline não os utiliza.
        """

        if throughput_history_kbps:
            recent = throughput_history_kbps[-self.history_size:]
            avg_throughput = sum(recent) / len(recent)
        else:
            avg_throughput = representations[0].bitrate_kbps

        safe_bandwidth = avg_throughput * self.safety_factor

        chosen = representations[0]

        for rep in representations:
            if rep.bitrate_kbps <= safe_bandwidth:
                chosen = rep
            else:
                break

        return chosen
