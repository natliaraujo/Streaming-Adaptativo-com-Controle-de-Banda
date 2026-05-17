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
NUM_SEGMENTS = 10              # Número de segmentos para teste (alterar conforme necessário)
ABR_HISTORY_SIZE = 5           # Quantas medições de vazão passadas são usadas na média móvel
SAFETY_FACTOR = 0.8            # Fator de segurança: usa apenas 80% da vazão média estimada
ALPHA_EWMA = 0.3               # Fator de suavização para o jitter EWMA (0 < alpha <= 1)

# ------------------------------------------------------------
# 2. Classe BufferManager
#    Simula o buffer de reprodução do cliente (playout buffer),
#    como descrito no Capítulo 2 do Kurose (aplicações multimídia).
#    O buffer é medido em segundos de vídeo e drenado em tempo real,
#    detectando rebuffering quando o nível fica negativo.
# ------------------------------------------------------------
class BufferManager:
    def __init__(self):
        self.level = 0.0               # ocupação do buffer (segundos de vídeo)
        self.stall_accumulated = 0.0   # tempo acumulado em rebuffering desde o último reset
        self.in_stall = False          # indica se o player está atualmente congelado
        # Inicializa com o instante atual para que o primeiro drain() não desconte
        # um intervalo gigante (evita stall artificial na partida).
        self.last_time = time.time()

    def start_download(self):
        """
        Chamado no início do download de um segmento.
        Reinicia a referência temporal para que a drenagem durante a transferência
        meça apenas o tempo gasto no download, separadamente do tempo ocioso.
        """
        self.last_time = time.time()

    def drain(self):
        """
        Drena o buffer com base no tempo real decorrido desde a última chamada.
        Representa a exibição contínua do vídeo: cada segundo real consome 1 segundo de buffer.
        Se o nível se torna negativo, o player entra em rebuffering (stall).
        """
        now = time.time()
        elapsed = now - self.last_time
        self.level -= elapsed
        # Detecta início ou continuação de stall
        if self.level < 0:
            if not self.in_stall:
                self.in_stall = True
            # Acumula o tempo negativo (tempo que o usuário esperou)
            self.stall_accumulated += abs(self.level)
            self.level = 0.0   # o buffer na prática não fica negativo, estaciona em zero
        self.last_time = now

    def add_segment(self, duration):
        """
        Adiciona um segmento recém-baixado ao buffer.
        `duration` é a duração do segmento em segundos (ex.: 2.0 s).
        Também encerra o estado de stall, pois novos dados chegaram.
        """
        self.level += duration
        self.in_stall = False

    def get_stall_and_reset(self):
        """
        Retorna o tempo total de stall acumulado desde o último reset
        e zera o acumulador para o próximo segmento.
        """
        s = self.stall_accumulated
        self.stall_accumulated = 0.0
        return s

# ------------------------------------------------------------
# 3. Download de segmento com medição de vazão e jitter
#    Implementa a métrica de vazão da camada de aplicação (throughput)
#    e o jitter como o desvio padrão dos intervalos entre chegadas de chunks,
#    conforme discutido no Kurose (Capítulo 2, Seção 2.6).
# ------------------------------------------------------------
def download_segment_v2(buffer, server_url, path, nominal_bitrate):
    """
    Baixa um segmento via HTTP, drena o buffer durante a transferência
    e retorna: bytes_recebidos, download_time (s), vazao_kbps, jitter_network_ms
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
    chunk_intervals = []       # intervalos entre chegadas de chunks consecutivos (para jitter)

    buffer.start_download()    # sincroniza o relógio do buffer para esta transferência
    start_download = time.time()
    last_chunk_time = start_download

    while True:
        chunk = response.read(1024)   # lê até 1024 bytes por vez
        if not chunk:
            break
        now = time.time()
        # Drena o buffer com o tempo real gasto para este chunk
        buffer.drain()
        # Calcula o intervalo desde o último chunk (para medição de jitter)
        chunk_intervals.append(now - last_chunk_time)
        last_chunk_time = now
        data += chunk

    download_time = time.time() - start_download
    bytes_received = len(data)

    # Cálculo da vazão em kbps (proteção contra divisão por zero)
    if download_time > 0:
        vazao_kbps = (bytes_received * 8) / (download_time * 1000)
    else:
        vazao_kbps = nominal_bitrate  # fallback (caso extremamente raro)

    # Jitter de rede: desvio padrão dos intervalos entre chunks (em ms)
    if len(chunk_intervals) > 1:
        jitter_net = statistics.stdev(chunk_intervals) * 1000
    else:
        jitter_net = 0.0

    conn.close()
    return bytes_received, download_time, vazao_kbps, jitter_net

# ------------------------------------------------------------
# 4. Baixar e interpretar o manifesto
#    O manifesto é o arquivo de descrição do conteúdo (como um MPD do DASH).
# ------------------------------------------------------------
def load_manifest(url):
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        manifest = json.loads(resp.read().decode('utf-8'))
    segment_duration = manifest["segment_duration_s"]
    servers = manifest["servers"]
    representations = manifest["representations"]
    # Ordena as qualidades por bitrate (menor para maior) para facilitar a escolha ABR
    representations.sort(key=lambda r: r["bitrate_kbps"])
    return segment_duration, servers, representations

# ------------------------------------------------------------
# 5. Loop principal (Baseline Rate‑Based)
#    Implementa o algoritmo ABR mais simples: escolhe a maior qualidade
#    cujo bitrate seja menor ou igual a uma estimativa conservadora da banda.
#    Esse algoritmo é suscetível a oscilações e rebufferings, conforme será
#    analisado nas próximas tarefas.
# ------------------------------------------------------------
def main():
    print("Baixando manifesto...")
    segment_duration, servers, representations = load_manifest(MANIFEST_URL)
    server_url = servers[0]["url"]   # servidor principal (A)
    server_id = servers[0]["id"]

    # Inicializa buffer e métricas
    buffer = BufferManager()
    throughput_history = []   # histórico das últimas vazões para a média móvel
    jitter_ewma = 0.0         # EWMA do jitter de rede (alfa = ALPHA_EWMA)

    # Abre arquivo CSV para registro das métricas
    csv_file = open("metricas_baseline.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "segment", "timestamp", "server_id", "quality", "bitrate_kbps",
        "vazao_kbps", "download_time_s", "jitter_network_ms", "jitter_ewma_ms",
        "buffer_level_s", "buffer_can_play", "rebuffer_event", "stall_duration_s",
        "failover_total"
    ])

    print(f"Iniciando download de {NUM_SEGMENTS} segmentos...")

    for seg_num in range(1, NUM_SEGMENTS + 1):
        # ---- 1. Drenagem do tempo ocioso ----
        # O buffer é drenado pelo tempo real passado desde o último evento
        # (fim do download anterior ou início do programa).
        buffer.drain()

        # ---- 2. Decisão ABR (Rate‑Based) ----
        # Estimativa de vazão: média das últimas N medições (ou menor qualidade se não houver histórico)
        if throughput_history:
            avg_vazao = sum(throughput_history[-ABR_HISTORY_SIZE:]) / len(throughput_history[-ABR_HISTORY_SIZE:])
        else:
            avg_vazao = representations[0]["bitrate_kbps"]   # começa com a menor qualidade

        # Banda segura = estimativa * fator de segurança (para ser conservador)
        safe_bandwidth = avg_vazao * SAFETY_FACTOR

        # Seleciona a maior qualidade cujo bitrate nominal seja <= banda segura
        chosen_rep = None
        for rep in representations:
            if rep["bitrate_kbps"] <= safe_bandwidth:
                chosen_rep = rep
            else:
                break
        if chosen_rep is None:
            chosen_rep = representations[0]  # fallback para a mínima

        # ---- 3. buffer_can_play ----
        # Indica se o buffer atual é suficiente para reprodução contínua
        # enquanto o próximo segmento é baixado (pelo menos 2s de buffer).
        buffer_can_play = 1 if buffer.level >= segment_duration else 0

        # Timestamp do início do download (UTC, seguindo a recomendação de usar timezone-aware)
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        # ---- 4. Download do segmento ----
        try:
            segment_path = f"{chosen_rep['url_path']}?seg={seg_num}"
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
            # Nota: o buffer não foi drenado durante a tentativa; o próximo buffer.drain()
            # no início do loop tratará o tempo ocioso.

        # ---- 5. Atualizar histórico de vazão ----
        throughput_history.append(vazao)
        if len(throughput_history) > ABR_HISTORY_SIZE:
            throughput_history.pop(0)

        # ---- 6. Jitter EWMA (suavização exponencial) ----
        jitter_ewma = ALPHA_EWMA * jitter_net + (1 - ALPHA_EWMA) * jitter_ewma

        # ---- 7. Finalizar buffer: adicionar segmento e capturar stall ----
        buffer.add_segment(segment_duration)
        stall_duration = buffer.get_stall_and_reset()
        rebuffer_event = 1 if stall_duration > 0 else 0

        # ---- 8. Escrever no CSV ----
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
            0    # failover_total (ainda sem failover nesta entrega)
        ])

        # Exibe progresso no terminal
        print(f"Seg {seg_num:2d}: {chosen_rep['quality']:5s} "
              f"Vazão={vazao:6.1f} kbps  Buffer={buffer.level:.2f}s  "
              f"Rebuffer={rebuffer_event}")

    csv_file.close()
    print(f"CSV gerado: metricas_baseline.csv")

if __name__ == "__main__":
    main()