"""
Define as métricas coletadas durante os experimentos.

As métricas descrevem o resultado observado para cada segmento baixado,
incluindo qualidade, servidor, vazão, jitter, buffer, rebuffering e, quando
disponível, observações dos servidores e previsões usadas pela política RNN.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentMetrics:
    """Métricas registradas para um segmento."""

    segment: int
    timestamp: str
    startup_phase: int
    server_id: str
    quality: str
    bitrate_kbps: int
    throughput_kbps: float
    throughput_ewma_kbps: float | None
    download_time_s: float
    jitter_network_ms: float
    jitter_ewma_ms: float
    buffer_level_s: float
    buffer_can_play: int
    rebuffer_event: int
    stall_duration_s: float
    playback_wait_s: float
    failover_event: int
    failover_duration_s: float
    failover_total: int

    rnn_predicted_a_throughput_kbps: float | None = None
    rnn_predicted_b_throughput_kbps: float | None = None
    rnn_predicted_selected_throughput_kbps: float | None = None

    probe_a_ok: int | None = None
    probe_a_latency_ms: float | None = None
    probe_a_throughput_kbps: float | None = None
    probe_a_jitter_ms: float | None = None

    probe_b_ok: int | None = None
    probe_b_latency_ms: float | None = None
    probe_b_throughput_kbps: float | None = None
    probe_b_jitter_ms: float | None = None
