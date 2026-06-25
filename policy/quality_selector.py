"""Seleciona representações de vídeo a partir de vazão estimada e buffer."""

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


class ThresholdBufferAwareQualitySelector:
    """Seleciona qualidade por EWMA e pelos limiares globais do buffer."""

    def __init__(
        self,
        safety_factor: float,
        min_buffer_s: float,
        target_buffer_s: float,
    ) -> None:
        """Configura os limiares usados pela Política 2.

        Args:
            safety_factor: Fração da EWMA utilizável abaixo do nível confortável.
            min_buffer_s: Nível que força a menor representação.
            target_buffer_s: Nível a partir do qual a política pode experimentar
                um degrau acima da taxa sustentada.
        """
        self.safety_factor: float = safety_factor
        self.min_buffer_s: float = min_buffer_s
        self.target_buffer_s: float = target_buffer_s

    def select(
        self,
        representations: list[Representation],
        estimated_throughput_kbps: float,
        buffer_level_s: float,
    ) -> Representation:
        """Escolhe qualidade combinando EWMA e reserva de buffer.

        A estratégia possui três comportamentos: abaixo do mínimo força a menor
        qualidade; entre mínimo e alvo aplica o fator de segurança; no alvo ou
        acima dele usa a vazão integral e experimenta no máximo um degrau acima.
        A promoção controlada transforma buffer acumulado em margem para testar
        uma qualidade melhor sem saltos arbitrários.

        Args:
            representations: Qualidades disponíveis, ordenadas por bitrate.
            estimated_throughput_kbps: EWMA de vazão usada pela Política 2.
            buffer_level_s: Segundos de vídeo disponíveis antes do download.

        Returns:
            Representação escolhida segundo a zona atual do buffer.
        """
        if buffer_level_s < self.min_buffer_s:
            return representations[0]

        effective_safety_factor: float = (
            1.0 if buffer_level_s >= self.target_buffer_s
            else self.safety_factor
        )
        available_kbps: float = estimated_throughput_kbps * effective_safety_factor

        chosen_index: int = 0
        for index, representation in enumerate(representations):
            if representation.bitrate_kbps <= available_kbps:
                chosen_index = index
            else:
                break

        # Com buffer confortável, usa a reserva para experimentar um degrau acima.
        if (
            buffer_level_s >= self.target_buffer_s
            and buffer_level_s <= self.target_buffer_s + 10
            and estimated_throughput_kbps > 0.0
            and chosen_index < len(representations) - 1
        ):
            chosen_index += 1
            
        if (
            buffer_level_s >= self.target_buffer_s + 10
            and estimated_throughput_kbps > 0.0
            and chosen_index < len(representations) - 1
        ):
            chosen_index = len(representations) - 1

        return representations[chosen_index]


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
