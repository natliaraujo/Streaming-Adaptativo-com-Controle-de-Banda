"""Constantes de configuração usadas pelo experimento baseline."""

# URL do manifest fornecido pela infraestrutura do projeto.
# Usa só o servidor principal
MANIFEST_URL: str = "http://137.131.178.229:8080/manifest"

# Parâmetros do experimento baseline.
NUM_SEGMENTS: int = 10
ABR_HISTORY_SIZE: int = 5
SAFETY_FACTOR: float = 0.8
ALPHA_EWMA: float = 0.3
HTTP_TIMEOUT_S: int = 10
