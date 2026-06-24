"""
Escreve as métricas dos experimentos em arquivos CSV.

Este módulo recebe estruturas `SegmentMetrics` e as converte para linhas
tabulares usadas na análise e no treinamento da RNN.
"""

import csv
from typing import TextIO

from domain.metrics import SegmentMetrics


class CsvMetricsWriter:
    """Escreve métricas de segmentos em CSV."""

    HEADER: list[str] = [
        "segment",
        "timestamp",
        "server_id",
        "quality",
        "bitrate_kbps",
        "throughput_kbps",
        "throughput_ewma_kbps",
        "download_time_s",
        "jitter_network_ms",
        "jitter_ewma_ms",
        "buffer_level_s",
        "buffer_can_play",
        "rebuffer_event",
        "stall_duration_s",
        "playback_wait_s",
        "failover_event",
        "failover_duration_s",
        "failover_total",
        "probe_a_ok",
        "probe_a_latency_ms",
        "probe_a_throughput_kbps",
        "probe_a_jitter_ms",
        "probe_b_ok",
        "probe_b_latency_ms",
        "probe_b_throughput_kbps",
        "probe_b_jitter_ms",
    ]

    def __init__(self, path: str) -> None:
        """
        Abre o arquivo CSV e escreve o cabeçalho.

        Args:
            path: Caminho do arquivo CSV.
        """
        self.file: TextIO = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)

    def write(self, metrics: SegmentMetrics) -> None:
        """
        Escreve uma linha de métricas.

        Args:
            metrics: Métricas do segmento.
        """
        self.writer.writerow(
            [
                metrics.segment,
                metrics.timestamp,
                metrics.server_id,
                metrics.quality,
                metrics.bitrate_kbps,
                round(metrics.throughput_kbps, 2),
                self._format_optional_float(metrics.throughput_ewma_kbps),
                round(metrics.download_time_s, 3),
                round(metrics.jitter_network_ms, 2),
                round(metrics.jitter_ewma_ms, 2),
                round(metrics.buffer_level_s, 2),
                metrics.buffer_can_play,
                metrics.rebuffer_event,
                round(metrics.stall_duration_s, 3),
                round(metrics.playback_wait_s, 3),
                metrics.failover_event,
                round(metrics.failover_duration_s, 3),
                metrics.failover_total,
                self._format_optional_int(metrics.probe_a_ok),
                self._format_optional_float(metrics.probe_a_latency_ms),
                self._format_optional_float(metrics.probe_a_throughput_kbps),
                self._format_optional_float(metrics.probe_a_jitter_ms),
                self._format_optional_int(metrics.probe_b_ok),
                self._format_optional_float(metrics.probe_b_latency_ms),
                self._format_optional_float(metrics.probe_b_throughput_kbps),
                self._format_optional_float(metrics.probe_b_jitter_ms),
            ]
        )
        self.file.flush()

    def close(self) -> None:
        """Fecha o arquivo CSV e libera seu descritor.

        O runner chama este método em ``finally`` para preservar as amostras já
        gravadas mesmo quando o experimento é interrompido por erro.
        """
        self.file.close()

    def _format_optional_float(self, value: float | None) -> str:
        """Converte uma medição opcional para a representação do CSV.

        Args:
            value: Medição numérica ou ``None`` quando não houve probe.

        Returns:
            Número arredondado como texto, ou célula vazia para ``None``.
        """
        if value is None:
            return ""

        return str(round(value, 3))

    def _format_optional_int(self, value: int | None) -> str:
        """Converte um indicador opcional para a representação do CSV.

        Args:
            value: Indicador inteiro ou ``None`` quando não houve probe.

        Returns:
            Inteiro como texto, ou célula vazia para ``None``.
        """
        if value is None:
            return ""

        return str(value)
