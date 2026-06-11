"""Orquestração do experimento de streaming adaptativo."""

from datetime import datetime, timezone, timedelta

from abr.base import AbrPolicy
from domain.manifest import Manifest
from domain.metrics import SegmentMetrics
from experiment.csv_writer import CsvMetricsWriter
from failover.server_selector import PriorityServerSelector
from network.segment_downloader import download_segment
from player.buffer import BufferManager


class ExperimentRunner:
    """Executa downloads sequenciais e registra métricas por segmento."""

    def __init__(
        self,
        manifest: Manifest,
        abr_policy: AbrPolicy,
        csv_writer: CsvMetricsWriter,
        num_segments: int,
        alpha_ewma: float,
    ) -> None:
        """
        Prepara os componentes necessários para rodar o experimento.

        Args:
            manifest: Manifesto com servidores, representações e duração.
            abr_policy: Política usada para escolher a qualidade de cada segmento.
            csv_writer: Escritor responsável por persistir as métricas.
            num_segments: Quantidade de segmentos que serão baixados.
            alpha_ewma: Peso da amostra atual no cálculo de jitter EWMA.
        """

        self.manifest = manifest
        self.abr_policy = abr_policy
        self.csv_writer = csv_writer
        self.num_segments = num_segments
        self.alpha_ewma = alpha_ewma

        self.buffer = BufferManager()
        self.server_selector = PriorityServerSelector(manifest.servers)
        self.throughput_history_kbps: list[float] = []
        self.jitter_ewma_ms: float = 0.0

    def run(self) -> None:
        """
        Executa o ciclo de seleção ABR, download, failover e coleta de métricas.

        Para cada segmento, a política escolhe uma representação, o downloader
        mede a rede, o buffer é atualizado e uma linha é escrita no CSV. Em caso
        de falha no servidor atual, o runner tenta o próximo servidor por
        prioridade.
        """

        print(f"Iniciando download de {self.num_segments} segmentos...")

        for seg_num in range(1, self.num_segments + 1):
            self.buffer.drain()

            chosen_rep = self.abr_policy.select_representation(
                representations=self.manifest.representations,
                throughput_history_kbps=self.throughput_history_kbps,
                buffer_level_s=self.buffer.level_s,
                segment_duration_s=self.manifest.segment_duration_s,
            )

            server = self.server_selector.get_current_server()

            buffer_can_play = int(
                self.buffer.level_s >= self.manifest.segment_duration_s
            )

            timestamp = datetime.now(
                timezone(timedelta(hours=-3))
            ).isoformat()

            try:
                segment_path = f"{chosen_rep.url_path}?seg={seg_num}"

                result = download_segment(
                    buffer=self.buffer,
                    server_url=server.url,
                    path=segment_path,
                    nominal_bitrate_kbps=chosen_rep.bitrate_kbps,
                )

            except Exception as e:
                print(f"Erro no segmento {seg_num} usando servidor {server.id}: {e}")

                server = self.server_selector.mark_failed_and_switch()

                segment_path = f"{chosen_rep.url_path}?seg={seg_num}"

                result = download_segment(
                    buffer=self.buffer,
                    server_url=server.url,
                    path=segment_path,
                    nominal_bitrate_kbps=chosen_rep.bitrate_kbps,
                )

            self.throughput_history_kbps.append(result.throughput_kbps)

            self.jitter_ewma_ms = (
                self.alpha_ewma * result.jitter_network_ms
                + (1 - self.alpha_ewma) * self.jitter_ewma_ms
            )

            self.buffer.add_segment(self.manifest.segment_duration_s)

            stall_duration_s = self.buffer.get_stall_and_reset()
            rebuffer_event = int(stall_duration_s > 0)

            metrics = SegmentMetrics(
                segment=seg_num,
                timestamp=timestamp,
                server_id=server.id,
                quality=chosen_rep.quality,
                bitrate_kbps=chosen_rep.bitrate_kbps,
                throughput_kbps=result.throughput_kbps,
                download_time_s=result.download_time_s,
                jitter_network_ms=result.jitter_network_ms,
                jitter_ewma_ms=self.jitter_ewma_ms,
                buffer_level_s=self.buffer.level_s,
                buffer_can_play=buffer_can_play,
                rebuffer_event=rebuffer_event,
                stall_duration_s=stall_duration_s,
                failover_total=self.server_selector.failover_total,
            )

            self.csv_writer.write(metrics)

            print(
                f"Seg {seg_num:2d}: {chosen_rep.quality:5s} "
                f"Servidor={server.id:5s} "
                f"Vazão={result.throughput_kbps:6.1f} kbps "
                f"Buffer={self.buffer.level_s:.2f}s "
                f"Rebuffer={rebuffer_event}"
            )

        self.csv_writer.close()
