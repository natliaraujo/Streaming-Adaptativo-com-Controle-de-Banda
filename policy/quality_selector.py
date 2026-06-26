"""Seleciona representações de vídeo a partir de vazão estimada e buffer."""

import math

from domain import Representation


class RateBasedQualitySelector:
    """Seleciona qualidade usando uma estimativa de vazão e fator de segurança."""

    def __init__(
        self,
        history_size: int,
        safety_factor: float,
    ) -> None:
        """Configura a margem de segurança da política rate-based.

        Args:
            history_size: Mantido por compatibilidade com a configuração pública.
            safety_factor: Fração da vazão estimada considerada utilizável.
        """
        self.history_size: int = history_size
        self.safety_factor: float = safety_factor

    def select(
        self,
        representations: list[Representation],
        estimated_throughput_kbps: float | None,
    ) -> Representation:
        """Escolhe a maior representação compatível com a vazão segura.

        Sem estimativa, começa pela menor qualidade. Nos demais casos, aplica o
        fator de segurança à EWMA fornecida pela política e percorre os bitrates
        já ordenados do manifesto.

        Args:
            representations: Qualidades disponíveis, ordenadas por bitrate.
            estimated_throughput_kbps: EWMA atual de vazão, quando inicializada.

        Returns:
            Maior representação cujo bitrate cabe na vazão segura.
        """
        if estimated_throughput_kbps is None:
            return representations[0]

        safe_throughput: float = estimated_throughput_kbps * self.safety_factor

        chosen: Representation = representations[0]

        for rep in representations:
            if rep.bitrate_kbps <= safe_throughput:
                chosen = rep
            else:
                break

        return chosen

class SmoothBufferAwareQualitySelector:
    """Seleciona qualidade com ajuste contínuo pelo desvio do buffer."""

    def __init__(
        self,
        safety_factor: float,
        target_buffer_s: float,
        max_buffer_s: float,
        response_power: float,
        max_boost_factor: float,
    ) -> None:
        """Configura a curva de agressividade da Política 2.

        No alvo de buffer, a política usa a EWMA integral. Abaixo do alvo, aplica
        uma redução suave até ``safety_factor``. Acima do alvo, aplica um boost
        exponencial suave até ``max_boost_factor`` quando o buffer se aproxima do
        máximo. ``response_power`` controla a curvatura da resposta.

        Args:
            safety_factor: Menor fração da EWMA usada com buffer muito baixo.
            target_buffer_s: Nível de buffer em que a política fica neutra.
            max_buffer_s: Nível usado para normalizar o boost positivo.
            response_power: Potência aplicada ao desvio normalizado do buffer.
            max_boost_factor: Maior multiplicador permitido para a EWMA.
        """
        self.safety_factor: float = safety_factor
        self.target_buffer_s: float = target_buffer_s
        self.max_buffer_s: float = max_buffer_s
        self.response_power: float = response_power
        self.max_boost_factor: float = max_boost_factor

    def select(
        self,
        representations: list[Representation],
        estimated_throughput_kbps: float,
        buffer_level_s: float,
    ) -> Representation:
        """Escolhe qualidade por EWMA ajustada continuamente pelo buffer.

        A função transforma a distância entre ``buffer_level_s`` e
        ``target_buffer_s`` em um fator multiplicativo. Como o fator é contínuo,
        pequenas variações no buffer produzem pequenas variações na agressividade
        da seleção, sem mudanças abruptas em limiares fixos.

        Args:
            representations: Qualidades disponíveis, ordenadas por bitrate.
            estimated_throughput_kbps: EWMA de vazão usada pela Política 2.
            buffer_level_s: Segundos de vídeo disponíveis antes do download.

        Returns:
            Maior representação cujo bitrate cabe na EWMA ajustada pelo buffer.
        """
        if estimated_throughput_kbps <= 0.0:
            return representations[0]

        buffer_factor: float = self._buffer_factor(buffer_level_s)
        available_kbps: float = estimated_throughput_kbps * buffer_factor

        chosen: Representation = representations[0]
        for representation in representations:
            if representation.bitrate_kbps <= available_kbps:
                chosen = representation
            else:
                break

        return chosen

    def _buffer_factor(self, buffer_level_s: float) -> float:
        """Calcula o multiplicador suave aplicado à EWMA."""
        if buffer_level_s >= self.target_buffer_s:
            denominator: float = max(
                self.max_buffer_s - self.target_buffer_s,
                1e-9,
            )
            normalized_gap: float = min(
                (buffer_level_s - self.target_buffer_s) / denominator,
                1.0,
            )
            shaped_gap: float = normalized_gap ** self.response_power
            return self.max_boost_factor ** shaped_gap

        denominator = max(self.target_buffer_s, 1e-9)
        normalized_gap = min(
            (self.target_buffer_s - buffer_level_s) / denominator,
            1.0,
        )
        shaped_gap = normalized_gap ** self.response_power
        return math.pow(self.safety_factor, shaped_gap)


class BufferAwareQualitySelector:
    """Seleciona qualidade considerando vazão prevista e nível do buffer."""

    def __init__(
        self,
        safety_factor: float,
    ) -> None:
        """Configura o seletor usado pela Política 3 e pela coleta da RNN.

        Args:
            safety_factor: Fração da vazão prevista usada na zona intermediária.
        """
        self.safety_factor: float = safety_factor

    def select(
        self,
        representations: list[Representation],
        predicted_throughput_kbps: float,
        buffer_level_s: float,
        segment_duration_s: float,
    ) -> Representation:
        """Escolhe qualidade usando previsão de vazão e sobrevivência do buffer.

        Este seletor preserva a estratégia original da Política 3. Ele varia a
        margem conforme o buffer e aceita uma representação quando sua taxa cabe
        na previsão ou quando o tempo previsto de download ainda deixa reserva
        suficiente no player.

        Args:
            representations: Qualidades disponíveis, ordenadas por bitrate.
            predicted_throughput_kbps: Vazão futura prevista pela RNN ou monitor.
            buffer_level_s: Nível atual do buffer em segundos.
            segment_duration_s: Duração nominal de um segmento.

        Returns:
            Maior representação aceita pelos critérios de taxa ou reserva.
        """
        low_buffer_s: float = segment_duration_s
        high_buffer_s: float = 3.0 * segment_duration_s
        min_reserve_s: float = 0.5 * segment_duration_s

        if buffer_level_s < low_buffer_s:
            effective_safety_factor: float = 0.65
        elif buffer_level_s < high_buffer_s:
            effective_safety_factor = self.safety_factor
        else:
            effective_safety_factor = 1.05

        safe_throughput_kbps: float = (
            predicted_throughput_kbps * effective_safety_factor
        )

        chosen: Representation = representations[0]

        for rep in representations:
            if predicted_throughput_kbps <= 0:
                predicted_download_time_s: float = float("inf")
            else:
                predicted_download_time_s = (
                    rep.segment_bytes * 8
                ) / (predicted_throughput_kbps * 1000)

            rate_fits: bool = rep.bitrate_kbps <= safe_throughput_kbps

            buffer_survives: bool = (
                buffer_level_s - predicted_download_time_s
            ) >= min_reserve_s

            if rate_fits or buffer_survives:
                chosen = rep
            else:
                break

        return chosen
