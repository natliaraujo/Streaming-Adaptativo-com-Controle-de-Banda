"""Modelo de métricas coletadas para cada segmento baixado."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentMetrics:
    """Conjunto de medições gravadas no CSV para um segmento."""

    segment: int
    """Número sequencial do segmento no experimento."""

    timestamp: str
    """Instante de coleta em ISO 8601."""

    server_id: str
    """Identificador do servidor usado no download."""

    quality: str
    """Qualidade escolhida pela política ABR."""

    bitrate_kbps: int
    """Bitrate nominal da representação escolhida."""

    throughput_kbps: float
    """Vazão efetiva medida durante o download."""

    download_time_s: float
    """Duração total do download em segundos."""

    jitter_network_ms: float
    """Jitter calculado a partir dos intervalos entre chunks HTTP."""

    jitter_ewma_ms: float
    """Média móvel exponencial do jitter de rede."""

    buffer_level_s: float
    """Nível do buffer após adicionar o segmento baixado."""

    buffer_can_play: int
    """Indicador inteiro de buffer suficiente para reproduzir um segmento."""

    rebuffer_event: int
    """Indicador inteiro de ocorrência de rebuffering no segmento."""

    stall_duration_s: float
    """Tempo acumulado de stall observado durante o download."""

    failover_total: int
    """Quantidade acumulada de trocas para servidores de fallback."""
