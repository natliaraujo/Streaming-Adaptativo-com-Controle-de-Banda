"""
Define os parâmetros globais de configuração do cliente de streaming adaptativo.

Este módulo centraliza constantes usadas pelos experimentos, como URL do
manifest, número de segmentos baixados, parâmetros das políticas ABR, fatores de
segurança, tamanho de histórico e timeouts de rede.

Manter esses valores em um único módulo evita duplicação de constantes entre os
scripts de execução, políticas, downloader e módulos de análise.
"""

# URL do manifest fornecido pela infraestrutura do projeto.
# Usa só o servidor principal
MANIFEST_URL: str = "http://137.131.178.229:8080/manifest"

# Parâmetros do experimento baseline.
NUM_SEGMENTS: int = 100
BUFFER_MAX_S: float = 30.0
BUFFER_TARGET_S: float = 15.0
BUFFER_MIN_S: float = 4.0
BUFFER_CRITICAL_S: float = 1.0
TRAINING_NUM_SEGMENTS: int = 200
TRAINING_FAULTS_PER_SERVER: int = 4
TRAINING_FAULT_SEED: int = 2026
TRAINING_FAULT_INITIAL_DELAY_S: float = 4.0
TRAINING_FAULT_MIN_GAP_S: float = 2.0
TRAINING_FAULT_MAX_GAP_S: float = 6.0
TRAINING_FAULT_MIN_DURATION_S: float = 3.0
TRAINING_FAULT_MAX_DURATION_S: float = 7.0
RNN_SEQUENCE_LENGTH: int = 10
RNN_FEATURE_SIZE: int = 15
RNN_HIDDEN_SIZE: int = 64
ABR_HISTORY_SIZE: int = 5
SAFETY_FACTOR: float = 0.92
ALPHA_EWMA: float = 0.3
HTTP_TIMEOUT_S: int = 15
HEALTH_CHECK_TIMEOUT_S: float = 2.0
NETWORK_RETRY_DELAY_S: float = 2.0
NETWORK_RECOVERY_MAX_WAIT_S: float = 120.0
