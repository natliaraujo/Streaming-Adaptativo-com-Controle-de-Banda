"""Simula um buffer finito de reprodução medido em segundos de vídeo."""

import time


class BufferManager:
    """Simula consumo de buffer e acumula tempo de stall."""

    def __init__(
        self,
        max_level_s: float,
        target_level_s: float,
        min_level_s: float,
        critical_level_s: float,
    ) -> None:
        """Inicializa um buffer vazio com seus limiares operacionais.

        O nível máximo limita a memória simulada. Os demais limiares não alteram
        o nível diretamente: eles ficam disponíveis para as políticas decidirem
        quando agir de forma conservadora ou agressiva.

        Args:
            max_level_s: Capacidade máxima do buffer, em segundos de vídeo.
            target_level_s: Nível considerado confortável pelas políticas.
            min_level_s: Nível abaixo do qual há risco elevado de rebuffering.
            critical_level_s: Nível crítico, menor ou igual ao nível mínimo.

        Raises:
            ValueError: Se os limiares forem negativos, estiverem fora de ordem
                ou se o alvo exceder a capacidade máxima.
        """
        if not 0.0 <= critical_level_s <= min_level_s <= target_level_s:
            raise ValueError("Limiares de buffer devem estar em ordem crescente.")
        if target_level_s > max_level_s:
            raise ValueError("O alvo do buffer não pode exceder o nível máximo.")

        self.max_level_s: float = max_level_s
        self.target_level_s: float = target_level_s
        self.min_level_s: float = min_level_s
        self.critical_level_s: float = critical_level_s
        self.level_s: float = 0.0
        self.stall_accumulated_s: float = 0.0

    def consume(self, elapsed_s: float) -> None:
        """Consome buffer durante um intervalo de reprodução.

        Se houver vídeo suficiente, apenas reduz ``level_s``. Caso contrário,
        zera o buffer e acumula como stall a parcela do intervalo que não pôde
        ser reproduzida.

        Args:
            elapsed_s: Tempo real transcorrido, em segundos.

        Raises:
            ValueError: Se ``elapsed_s`` for negativo.
        """
        if elapsed_s < 0.0:
            raise ValueError("O tempo consumido não pode ser negativo.")

        if elapsed_s <= self.level_s:
            self.level_s -= elapsed_s
            return

        missing_s: float = elapsed_s - self.level_s
        self.level_s = 0.0
        self.stall_accumulated_s += missing_s

    def add_segment(self, segment_duration_s: float) -> None:
        """Adiciona um segmento baixado respeitando a capacidade máxima.

        Args:
            segment_duration_s: Duração de reprodução do segmento, em segundos.

        Raises:
            ValueError: Se a duração informada for negativa.
        """
        if segment_duration_s < 0.0:
            raise ValueError("A duração do segmento não pode ser negativa.")

        self.level_s = min(
            self.level_s + segment_duration_s,
            self.max_level_s,
        )

    def get_stall_and_reset(self) -> float:
        """Obtém e reinicia o stall acumulado do segmento atual.

        Returns:
            Duração total de stall acumulada, em segundos.
        """

        stall_s = self.stall_accumulated_s
        self.stall_accumulated_s = 0.0
        return stall_s

    def wait_if_full(self, resume_margin_s: float) -> float:
        """Pausa o download somente quando o buffer está cheio.

        A estratégia mantém o player próximo da capacidade máxima sem executar
        downloads que seriam descartados. Ao atingir ``max_level_s``, espera a
        margem solicitada e consome esse mesmo intervalo do buffer.

        Args:
            resume_margin_s: Espaço temporal que deve ser liberado antes de
                retomar os downloads. Normalmente é a duração de um segmento.

        Returns:
            Tempo efetivamente aguardado. Retorna zero se o buffer não estiver
            cheio.

        Raises:
            ValueError: Se a margem solicitada for negativa.
        """
        if resume_margin_s < 0.0:
            raise ValueError("A margem de retomada não pode ser negativa.")

        if self.level_s < self.max_level_s:
            return 0.0

        wait_s: float = min(resume_margin_s, self.level_s)
        time.sleep(wait_s)
        self.consume(wait_s)
        return wait_s
