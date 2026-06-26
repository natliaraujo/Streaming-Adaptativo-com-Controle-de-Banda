"""
Escreve as métricas dos experimentos em arquivos CSV.

Este módulo recebe estruturas `SegmentMetrics` e as converte para linhas
tabulares usadas na análise e no treinamento da RNN.
"""

import csv
from pathlib import Path
from typing import TextIO

from domain.metrics import SegmentMetrics


class CsvMetricsWriter:
    """Escreve métricas de segmentos em CSV."""

    HEADER: list[str] = [
        "segment",
        "timestamp",
        "startup_phase",
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
        "rnn_predicted_a_throughput_kbps",
        "rnn_predicted_b_throughput_kbps",
        "rnn_predicted_selected_throughput_kbps",
        "probe_a_ok",
        "probe_a_latency_ms",
        "probe_a_throughput_kbps",
        "probe_a_jitter_ms",
        "probe_b_ok",
        "probe_b_latency_ms",
        "probe_b_throughput_kbps",
        "probe_b_jitter_ms",
    ]

    def __init__(self, path: str, append: bool = False) -> None:
        """
        Abre o arquivo CSV e escreve ou reaproveita o cabeçalho.

        Args:
            path: Caminho do arquivo CSV.
            append: Quando ``True``, adiciona linhas ao fim de um CSV existente.
                O cabeçalho existente precisa ser igual ao esquema atual.
        """
        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.header: list[str] = list(self.HEADER)
        should_write_header = True
        mode = "w"
        if append and csv_path.exists() and csv_path.stat().st_size > 0:
            existing_header = self._read_existing_header(csv_path)
            unknown_columns = [
                column
                for column in existing_header
                if column not in self.HEADER
            ]
            if unknown_columns:
                raise ValueError(
                    f"CSV existente com cabeçalho incompatível: {csv_path}. "
                    "Colunas desconhecidas: "
                    + ", ".join(unknown_columns)
                )
            self.header = existing_header
            should_write_header = False
            mode = "a"

        self.file: TextIO = open(csv_path, mode, newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if should_write_header:
            self.writer.writerow(self.header)

    def write(self, metrics: SegmentMetrics) -> None:
        """
        Escreve uma linha de métricas.

        Args:
            metrics: Métricas do segmento.
        """
        row = self._row_from_metrics(metrics)
        self.writer.writerow([row.get(column, "") for column in self.header])
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

    def _read_existing_header(self, path: Path) -> list[str]:
        """Lê o cabeçalho de um CSV existente."""
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            return next(reader, [])

    def _row_from_metrics(self, metrics: SegmentMetrics) -> dict[str, object]:
        """Converte métricas para um dicionário indexado pelo cabeçalho atual."""
        return {
            "segment": metrics.segment,
            "timestamp": metrics.timestamp,
            "startup_phase": metrics.startup_phase,
            "server_id": metrics.server_id,
            "quality": metrics.quality,
            "bitrate_kbps": metrics.bitrate_kbps,
            "throughput_kbps": round(metrics.throughput_kbps, 2),
            "throughput_ewma_kbps": self._format_optional_float(
                metrics.throughput_ewma_kbps
            ),
            "download_time_s": round(metrics.download_time_s, 3),
            "jitter_network_ms": round(metrics.jitter_network_ms, 2),
            "jitter_ewma_ms": round(metrics.jitter_ewma_ms, 2),
            "buffer_level_s": round(metrics.buffer_level_s, 2),
            "buffer_can_play": metrics.buffer_can_play,
            "rebuffer_event": metrics.rebuffer_event,
            "stall_duration_s": round(metrics.stall_duration_s, 3),
            "playback_wait_s": round(metrics.playback_wait_s, 3),
            "failover_event": metrics.failover_event,
            "failover_duration_s": round(metrics.failover_duration_s, 3),
            "failover_total": metrics.failover_total,
            "rnn_predicted_a_throughput_kbps": self._format_optional_float(
                metrics.rnn_predicted_a_throughput_kbps
            ),
            "rnn_predicted_b_throughput_kbps": self._format_optional_float(
                metrics.rnn_predicted_b_throughput_kbps
            ),
            "rnn_predicted_selected_throughput_kbps": self._format_optional_float(
                metrics.rnn_predicted_selected_throughput_kbps
            ),
            "probe_a_ok": self._format_optional_int(metrics.probe_a_ok),
            "probe_a_latency_ms": self._format_optional_float(
                metrics.probe_a_latency_ms
            ),
            "probe_a_throughput_kbps": self._format_optional_float(
                metrics.probe_a_throughput_kbps
            ),
            "probe_a_jitter_ms": self._format_optional_float(
                metrics.probe_a_jitter_ms
            ),
            "probe_b_ok": self._format_optional_int(metrics.probe_b_ok),
            "probe_b_latency_ms": self._format_optional_float(
                metrics.probe_b_latency_ms
            ),
            "probe_b_throughput_kbps": self._format_optional_float(
                metrics.probe_b_throughput_kbps
            ),
            "probe_b_jitter_ms": self._format_optional_float(
                metrics.probe_b_jitter_ms
            ),
        }
