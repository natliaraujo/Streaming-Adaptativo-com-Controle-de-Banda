"""
Cliente baseline para streaming adaptativo sobre HTTP.

O cliente implementa a primeira política do projeto: um ABR baseado em vazão
(Rate-Based ABR). A cada segmento, ele estima a banda disponível usando uma
média móvel das últimas medições, aplica um fator de segurança e escolhe a
maior representação cujo bitrate nominal cabe nessa banda conservadora.

Além da decisão de qualidade, o módulo simula o buffer de reprodução, mede
vazão e jitter no nível da aplicação, registra rebuffering e grava as métricas
em CSV para análise posterior e geração de gráficos.
"""

import http.client
import json
import csv
import time
import statistics
import os
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import urlparse

type ServerInfo = dict[str, Any]
type Representation = dict[str, Any]

# URL do manifest fornecido pela infraestrutura do projeto.
MANIFEST_URL: str = "http://137.131.178.229:8080/manifest"

# Parâmetros do experimento baseline.
NUM_SEGMENTS: int = 10
ABR_HISTORY_SIZE: int = 5
SAFETY_FACTOR: float = 0.8
ALPHA_EWMA: float = 0.3


class BufferManager:
    """
    Simula o buffer de reprodução do cliente.

    O nível do buffer é medido em segundos de vídeo. Durante o download de um
    segmento, o buffer é drenado pelo tempo real decorrido; quando o download
    termina, a duração do novo segmento é adicionada. Se o nível chega abaixo
    de zero, o player entra em rebuffering e o tempo de espera é acumulado.

    Attributes:
        level: Quantidade de vídeo disponível para reprodução, em segundos.
        stall_accumulated: Tempo de rebuffering acumulado desde o último reset.
        in_stall: Indica se o player está atualmente sem dados para reproduzir.
        last_time: Último instante usado como referência para drenagem.
    """

    def __init__(self) -> None:
        self.level: float = 0.0
        self.stall_accumulated: float = 0.0
        self.in_stall: bool = False
        # Evita um stall artificial na partida: a primeira drenagem deve contar
        # apenas o tempo desde a criação do buffer.
        self.last_time: float = time.time()

    def start_download(self) -> None:
        """
        Reinicia a referência temporal no início de um download.

        Isso separa o tempo gasto transferindo o segmento do tempo ocioso entre
        eventos, permitindo que `drain()` desconte apenas o intervalo correto.
        """
        self.last_time = time.time()

    def drain(self) -> None:
        """
        Drena o buffer pelo tempo real decorrido desde a última medição.

        A reprodução consome um segundo de buffer por segundo real. Quando o
        nível fica negativo, o valor excedente é registrado como rebuffering e
        o nível volta para zero, que representa o player parado aguardando dados.
        """
        now: float = time.time()
        elapsed: float = now - self.last_time
        self.level -= elapsed

        if self.level < 0:
            if not self.in_stall:
                self.in_stall = True
            self.stall_accumulated += abs(self.level)
            self.level = 0.0
        self.last_time = now

    def add_segment(self, duration: float) -> None:
        """
        Adiciona um segmento recém-baixado ao buffer.

        Args:
            duration: Duração do segmento em segundos.
        """
        self.level += duration
        self.in_stall = False

    def get_stall_and_reset(self) -> float:
        """
        Retorna o rebuffering acumulado desde a última consulta.

        Returns:
            Duração total de stall em segundos, antes de zerar o acumulador.
        """
        s: float = self.stall_accumulated
        self.stall_accumulated = 0.0
        return s


def download_segment_v2(
    buffer: BufferManager,
    server_url: str,
    path: str,
    nominal_bitrate: int,
) -> tuple[int, float, float, float]:
    """
    Baixa um segmento e mede as métricas de rede observadas pelo cliente.

    A função lê a resposta HTTP em chunks de 1024 bytes. A cada chunk recebido,
    o buffer é drenado pelo tempo real decorrido e o intervalo entre chunks é
    armazenado para estimar o jitter. A vazão é calculada no nível da aplicação,
    usando os bytes recebidos e o tempo total do download.

    Args:
        buffer: Gerenciador do buffer de reprodução.
        server_url: URL base do servidor escolhido.
        path: Caminho do segmento no servidor, incluindo query string.
        nominal_bitrate: Bitrate da representação escolhida, usado apenas como
            fallback se o tempo medido for zero.

    Returns:
        Tupla com bytes recebidos, tempo de download em segundos, vazão em kbps
        e jitter em milissegundos.

    Raises:
        ValueError: Se a URL do servidor não tiver host.
        Exception: Se o servidor responder com status HTTP diferente de 200.
    """
    url_parts = urlparse(server_url)
    host: str | None = url_parts.hostname
    port: int | None = url_parts.port

    if host is None:
        raise ValueError(f"URL de servidor inválida: {server_url}")

    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("GET", path)
    response = conn.getresponse()
    if response.status != 200:
        raise Exception(f"HTTP error {response.status}")

    data: bytes = b""
    chunk_intervals: list[float] = []

    buffer.start_download()
    start_download: float = time.time()
    last_chunk_time: float = start_download

    while True:
        chunk: bytes = response.read(1024)
        if not chunk:
            break
        now: float = time.time()

        buffer.drain()
        chunk_intervals.append(now - last_chunk_time)
        last_chunk_time = now
        data += chunk

    download_time: float = time.time() - start_download
    bytes_received: int = len(data)

    # Cálculo da vazão em kbps (proteção contra divisão por zero)
    if download_time > 0:
        vazao_kbps: float = (bytes_received * 8) / (download_time * 1000)
    else:
        vazao_kbps = nominal_bitrate

    if len(chunk_intervals) > 1:
        jitter_net: float = statistics.stdev(chunk_intervals) * 1000
    else:
        jitter_net = 0.0

    conn.close()
    return bytes_received, download_time, vazao_kbps, jitter_net


def load_manifest(url: str) -> tuple[float, list[ServerInfo], list[Representation]]:
    """
    Baixa e interpreta o manifest JSON do conteúdo.

    O manifest informa a duração dos segmentos, a lista de servidores
    disponíveis e as representações de qualidade. As representações são
    ordenadas por bitrate para simplificar a decisão ABR.

    Args:
        url: URL completa do endpoint `/manifest`.

    Returns:
        Duração do segmento, lista de servidores e lista de representações.
    """
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        manifest: dict[str, Any] = json.loads(resp.read().decode('utf-8'))
    segment_duration: float = manifest["segment_duration_s"]
    servers: list[ServerInfo] = manifest["servers"]
    representations: list[Representation] = manifest["representations"]

    representations.sort(key=lambda r: r["bitrate_kbps"])
    return segment_duration, servers, representations


def main() -> None:
    """
    Executa a política baseline Rate-Based ABR.

    Para cada segmento, o fluxo é:
    1. drenar o buffer pelo tempo decorrido;
    2. estimar a vazão pela média móvel das últimas medições;
    3. aplicar o fator de segurança;
    4. escolher a maior qualidade que cabe na banda segura;
    5. baixar o segmento e medir vazão/jitter;
    6. atualizar buffer, EWMA de jitter e CSV de métricas.

    O CSV gerado é a base para gráficos, comparação com políticas futuras e
    correlação com capturas no Wireshark.
    """
    print("Baixando manifesto...")
    segment_duration, servers, representations = load_manifest(MANIFEST_URL)
    server_url: str = servers[0]["url"]
    server_id: str = servers[0]["id"]

    buffer: BufferManager = BufferManager()
    throughput_history: list[float] = []
    jitter_ewma: float = 0.0

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    csv_path: str = os.path.join(script_dir, "metricas_baseline.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "segment", "timestamp", "server_id", "quality", "bitrate_kbps",
        "vazao_kbps", "download_time_s", "jitter_network_ms", "jitter_ewma_ms",
        "buffer_level_s", "buffer_can_play", "rebuffer_event", "stall_duration_s",
        "failover_total"
    ])

    print(f"Iniciando download de {NUM_SEGMENTS} segmentos...")

    for seg_num in range(1, NUM_SEGMENTS + 1):
        # Drena o tempo ocioso entre o fim do evento anterior e esta decisão.
        buffer.drain()

        if throughput_history:
            avg_vazao = sum(throughput_history[-ABR_HISTORY_SIZE:]) / len(throughput_history[-ABR_HISTORY_SIZE:])
        else:
            avg_vazao = representations[0]["bitrate_kbps"]

        safe_bandwidth = avg_vazao * SAFETY_FACTOR

        chosen_rep: Representation | None = None
        for rep in representations:
            if rep["bitrate_kbps"] <= safe_bandwidth:
                chosen_rep = rep
            else:
                break
        if chosen_rep is None:
            chosen_rep = representations[0]  # fallback para a mínima

        # Marca se o player teria conteúdo suficiente para tocar durante outro
        # segmento inteiro antes de iniciar este download.
        buffer_can_play: int = 1 if buffer.level >= segment_duration else 0

        timestamp: str = datetime.now(timezone(timedelta(hours=-3))).isoformat()

        try:
            segment_path: str = f"{chosen_rep['url_path']}?seg={seg_num}"
            bytes_rec, download_time, vazao, jitter_net = download_segment_v2(
                buffer, server_url, segment_path, chosen_rep['bitrate_kbps']
            )
        except Exception as e:
            print(f"Erro no segmento {seg_num}: {e}")
            # Em caso de erro, atribui valores de fallback para não interromper a execução
            bytes_rec = chosen_rep["segment_bytes"]
            download_time = 1.0
            vazao = 0.0             # vazão zero indica falha
            jitter_net = 0.0

        throughput_history.append(vazao)
        if len(throughput_history) > ABR_HISTORY_SIZE:
            throughput_history.pop(0)

        jitter_ewma = ALPHA_EWMA * jitter_net + (1 - ALPHA_EWMA) * jitter_ewma

        buffer.add_segment(segment_duration)
        stall_duration: float = buffer.get_stall_and_reset()
        rebuffer_event: int = 1 if stall_duration > 0 else 0

        writer.writerow([
            seg_num,
            timestamp,
            server_id,
            chosen_rep["quality"],
            chosen_rep["bitrate_kbps"],
            round(vazao, 2),
            round(download_time, 3),
            round(jitter_net, 2),
            round(jitter_ewma, 2),
            round(buffer.level, 2),
            buffer_can_play,
            rebuffer_event,
            round(stall_duration, 3),
            0
        ])

        print(f"Seg {seg_num:2d}: {chosen_rep['quality']:5s} "
              f"Vazão={vazao:6.1f} kbps  Buffer={buffer.level:.2f}s  "
              f"Rebuffer={rebuffer_event}")

    csv_file.close()
    print(f"CSV gerado: {csv_path}")

if __name__ == "__main__":
    main()
