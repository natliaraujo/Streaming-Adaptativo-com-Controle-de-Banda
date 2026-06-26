"""
Define a ação escolhida por uma política de streaming.

Uma ação representa a decisão completa tomada antes do download de um segmento:
qual servidor será usado e qual representação de qualidade será solicitada.

Essa abstração permite que políticas mais simples escolham apenas a qualidade
com servidor fixo, enquanto políticas mais avançadas escolhem simultaneamente
servidor e qualidade.
"""

from dataclasses import dataclass

from domain import Representation, ServerInfo


@dataclass(frozen=True)
class StreamingAction:
    """
    Representa uma decisão completa de streaming.

    A ação indica qual servidor deve ser usado e qual representação de qualidade
    deve ser baixada para o próximo segmento. Políticas preditivas podem anexar
    as previsões numéricas usadas na decisão.
    """

    server: ServerInfo
    representation: Representation
    predicted_server_a_throughput_kbps: float | None = None
    predicted_server_b_throughput_kbps: float | None = None
    predicted_selected_throughput_kbps: float | None = None
