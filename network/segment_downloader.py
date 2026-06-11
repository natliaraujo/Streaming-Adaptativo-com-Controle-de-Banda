"""Download HTTP de segmentos e cálculo de métricas de rede."""

import http.client
import statistics
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from config import HTTP_TIMEOUT_S
from player.buffer import BufferManager


@dataclass(frozen=True)
class DownloadResult:
    """Resultado de um download de segmento."""

    bytes_received: int
    """Total de bytes recebidos do servidor."""

    download_time_s: float
    """Tempo total de download em segundos."""

    throughput_kbps: float
    """Vazão efetiva calculada a partir dos bytes recebidos."""

    jitter_network_ms: float
    """Desvio padrão dos intervalos entre chunks, em milissegundos."""


def download_segment(
    buffer: BufferManager,
    server_url: str,
    path: str,
    nominal_bitrate_kbps: int,
) -> DownloadResult:
    """
    Baixa um segmento via HTTP e mede vazão, tempo e jitter.

    Durante a leitura dos chunks, o buffer é drenado para aproximar o consumo de
    reprodução enquanto o cliente aguarda a chegada do segmento.

    Args:
        buffer: Gerenciador de buffer que será atualizado durante o download.
        server_url: URL base do servidor escolhido.
        path: Caminho HTTP do segmento, incluindo query string quando houver.
        nominal_bitrate_kbps: Bitrate usado como fallback se o tempo medido for
            zero.

    Returns:
        Métricas de rede coletadas para o segmento.

    Raises:
        ValueError: Se a URL do servidor não possuir host.
        Exception: Se a resposta HTTP não for 200.
    """

    url_parts = urlparse(server_url)
    host = url_parts.hostname
    port = url_parts.port

    if host is None:
        raise ValueError(f"URL de servidor inválida: {server_url}")

    conn = http.client.HTTPConnection(host, port, timeout=HTTP_TIMEOUT_S)

    try:
        conn.request("GET", path)
        response = conn.getresponse()

        if response.status != 200:
            raise Exception(f"HTTP error {response.status}")

        bytes_received = 0
        chunk_intervals: list[float] = []

        buffer.start_download()
        start_download = time.time()
        last_chunk_time = start_download

        while True:
            chunk = response.read(1024)

            if not chunk:
                break

            now = time.time()

            buffer.drain()
            chunk_intervals.append(now - last_chunk_time)
            last_chunk_time = now

            bytes_received += len(chunk)

        download_time_s = time.time() - start_download

        if download_time_s > 0:
            throughput_kbps = (bytes_received * 8) / (download_time_s * 1000)
        else:
            throughput_kbps = nominal_bitrate_kbps

        if len(chunk_intervals) > 1:
            jitter_network_ms = statistics.stdev(chunk_intervals) * 1000
        else:
            jitter_network_ms = 0.0

        return DownloadResult(
            bytes_received=bytes_received,
            download_time_s=download_time_s,
            throughput_kbps=throughput_kbps,
            jitter_network_ms=jitter_network_ms,
        )

    finally:
        conn.close()
