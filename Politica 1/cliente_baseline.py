import http.client
import json
import csv
import time
import statistics
import datetime
from urllib.parse import urlparse

# ------------------------------------------------------------
# 1. Configurações
# ------------------------------------------------------------
MANIFEST_URL = "http://137.131.178.229:8080/manifest"
NUM_SEGMENTS = 20            # Número de segmentos para teste
ABR_HISTORY_SIZE = 5         # Quantas vazões passadas considerar na média
SAFETY_FACTOR = 0.8          # Fator de segurança (usa 80% da vazão média)
ALPHA_EWMA = 0.3             # Fator de suavização para jitter EWMA

# ------------------------------------------------------------
# 2. Classe BufferManager
# ------------------------------------------------------------
class BufferManager:
    def __init__(self):
        self.level = 0.0                # segundos de vídeo disponíveis
        self.stall_accumulated = 0.0    # tempo total em rebuffering desde o último reset
        self.in_stall = False           # indica se esta atualmente em stall
        self.last_time = time.time()    # instante da última drenagem

    def start_download(self):
        """Chamado no início do download de um segmento."""
        self.last_time = time.time()

    def drain(self):
        """Drena o buffer pelo tempo real decorrido desde a última chamada."""
        now = time.time()
        elapsed = now - self.last_time
        self.level -= elapsed
        if self.level < 0:
            if not self.in_stall:
                self.in_stall = True
            self.stall_accumulated += abs(self.level)
            self.level = 0.0
        self.last_time = now

    def add_segment(self, duration):
        """Adiciona um segmento recém-baixado ao buffer."""
        self.level += duration
        self.in_stall = False

    def get_stall_and_reset(self):
        """Retorna o tempo total de stall deste segmento e zera o acumulador."""
        s = self.stall_accumulated
        self.stall_accumulated = 0.0
        return s

# ------------------------------------------------------------
# 3. Download de segmento com medição de vazão e jitter
# ------------------------------------------------------------
def download_segment_v2(buffer, server_url, path, nominal_bitrate):
    """
    Baixa um segmento via HTTP, drena o buffer durante a transferência
    e retorna: bytes_recebidos, download_time, vazao_kbps, jitter_network_ms
    """
    url_parts = urlparse(server_url)
    host = url_parts.hostname
    port = url_parts.port

    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("GET", path)
    response = conn.getresponse()
    if response.status != 200:
        raise Exception(f"HTTP error {response.status}")

    data = b""
    chunk_intervals = []       # intervalos entre chegadas de chunks (para jitter)

    buffer.start_download()    # marca o last_time
    start_download = time.time()
    last_chunk_time = start_download

    while True:
        chunk = response.read(1024)
        if not chunk:
            break
        now = time.time()
        # Drena o buffer com o tempo real decorrido
        buffer.drain()
        # Calcula intervalo desde o último chunk (para jitter)
        chunk_intervals.append(now - last_chunk_time)
        last_chunk_time = now
        data += chunk

    download_time = time.time() - start_download
    bytes_received = len(data)

    # Proteção contra divisão por zero
    if download_time > 0:
        vazao_kbps = (bytes_received * 8) / (download_time * 1000)
    else:
        vazao_kbps = nominal_bitrate  # fallback para bitrate nominal (improvável)

    # Jitter de rede: desvio padrão dos intervalos entre chunks (em ms)
    if len(chunk_intervals) > 1:
        jitter_net = statistics.stdev(chunk_intervals) * 1000
    else:
        jitter_net = 0.0

    conn.close()
    return bytes_received, download_time, vazao_kbps, jitter_net

# ------------------------------------------------------------
# 4. Baixar e interpretar o manifesto
# ------------------------------------------------------------
def load_manifest(url):
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        manifest = json.loads(resp.read().decode('utf-8'))
    segment_duration = manifest["segment_duration_s"]
    servers = manifest["servers"]
    representations = manifest["representations"]
    representations.sort(key=lambda r: r["bitrate_kbps"])
    return segment_duration, servers, representations

# ------------------------------------------------------------
# 5. Loop principal (Baseline Rate‑Based)
# ------------------------------------------------------------
def main():
    print("Baixando manifesto...")
    segment_duration, servers, representations = load_manifest(MANIFEST_URL)
    server_url = servers[0]["url"]
    server_id = servers[0]["id"]

    # Inicializa buffer e métricas
    buffer = BufferManager()
    throughput_history = []
    jitter_ewma = 0.0

    # Abre arquivo CSV
    csv_file = open("metricas_baseline.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "segment", "timestamp", "server_id", "quality", "bitrate_kbps",
        "vazao_kbps", "download_time_s", "jitter_network_ms", "jitter_ewma_ms",
        "buffer_level_s", "buffer_can_play", "rebuffer_event", "stall_duration_s",
        "failover_total"
    ])

    print(f"Iniciando download de {NUM_SEGMENTS} segmentos...")



if __name__ == "__main__":
    main()