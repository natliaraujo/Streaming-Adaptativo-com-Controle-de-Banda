"""Controle simples do buffer de reprodução durante os downloads."""

import time


class BufferManager:
    """Simula consumo de buffer e acumula tempo de stall."""

    def __init__(self) -> None:
        """Inicializa o buffer vazio e prepara os contadores temporais."""

        self.level_s: float = 0.0
        self.stall_accumulated_s: float = 0.0
        self.in_stall: bool = False
        self.last_time: float = time.time()

    def start_download(self) -> None:
        """Marca o instante inicial usado para drenar o buffer no download."""

        self.last_time = time.time()

    def drain(self) -> None:
        """
        Consome buffer de acordo com o tempo real decorrido.

        Se o tempo decorrido ultrapassar o nível disponível, o buffer é zerado e
        a diferença é acumulada como stall para ser reportada nas métricas.
        """

        now: float = time.time()
        elapsed: float = now - self.last_time

        self.level_s -= elapsed

        if self.level_s < 0:
            if not self.in_stall:
                self.in_stall = True

            self.stall_accumulated_s += abs(self.level_s)
            self.level_s = 0.0

        self.last_time = now

    def add_segment(self, duration_s: float) -> None:
        """Adiciona ao buffer a duração de um segmento baixado com sucesso."""

        self.level_s += duration_s
        self.in_stall = False

    def get_stall_and_reset(self) -> float:
        """Retorna o stall acumulado desde a última chamada e zera o contador."""

        stall_s = self.stall_accumulated_s
        self.stall_accumulated_s = 0.0
        return stall_s
