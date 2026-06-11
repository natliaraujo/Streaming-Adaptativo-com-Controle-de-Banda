"""Escrita das métricas de segmentos no formato CSV do experimento."""

import csv

from domain.metrics import SegmentMetrics


class CsvMetricsWriter:
    """Grava uma linha de métricas por segmento baixado."""

    # Cabeçalho fixo usado pelos CSVs de saída do projeto.
    HEADER = [
        "segment",
        "timestamp",
        "server_id",
        "quality",
        "bitrate_kbps",
        "vazao_kbps",
        "download_time_s",
        "jitter_network_ms",
        "jitter_ewma_ms",
        "buffer_level_s",
        "buffer_can_play",
        "rebuffer_event",
        "stall_duration_s",
        "failover_total",
    ]

    def __init__(self, path: str) -> None:
        """Abre o arquivo de saída e escreve o cabeçalho do CSV."""

        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)

    def write(self, metrics: SegmentMetrics) -> None:
        """Escreve uma linha com os valores arredondados das métricas."""

        self.writer.writerow([
            metrics.segment,
            metrics.timestamp,
            metrics.server_id,
            metrics.quality,
            metrics.bitrate_kbps,
            round(metrics.throughput_kbps, 2),
            round(metrics.download_time_s, 3),
            round(metrics.jitter_network_ms, 2),
            round(metrics.jitter_ewma_ms, 2),
            round(metrics.buffer_level_s, 2),
            metrics.buffer_can_play,
            metrics.rebuffer_event,
            round(metrics.stall_duration_s, 3),
            metrics.failover_total,
        ])

    def close(self) -> None:
        """Fecha o arquivo CSV associado ao escritor."""

        self.file.close()
