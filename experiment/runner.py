"""Coordena políticas, downloads, buffer, failover e métricas do experimento."""

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from config import (
    BUFFER_CRITICAL_S,
    BUFFER_MAX_S,
    BUFFER_MIN_S,
    BUFFER_TARGET_S,
    HEALTH_CHECK_TIMEOUT_S,
    NETWORK_RECOVERY_MAX_WAIT_S,
    NETWORK_RETRY_DELAY_S,
    RNN_STARTUP_SEGMENTS,
)
from domain.action import StreamingAction
from domain.manifest import Manifest, Representation, ServerInfo, normalize_server_id
from domain.metrics import SegmentMetrics
from experiment.csv_writer import CsvMetricsWriter
from monitoring.observation_store import ObservationStore, ServerObservation
from network.segment_downloader import DownloadResult, download_segment
from network.server_probe import probe_server_health
from player.buffer import BufferManager
from policy.streaming_policy import StreamingPolicy


ServerFailurePredicate = Callable[[int, ServerInfo], bool]


def display_server_id(server_id: str) -> str:
    """Padroniza identificadores de servidor usados em logs e CSVs."""
    return normalize_server_id(server_id)


def estimate_download_time_s(
    segment_bytes: int,
    throughput_kbps: float | None,
) -> float:
    """Estima o tempo necessário para baixar um segmento.

    Args:
        segment_bytes: Tamanho nominal do segmento em bytes.
        throughput_kbps: Vazão estimada em kilobits por segundo.

    Returns:
        Tempo estimado em segundos, ou infinito quando a vazão é ausente ou não
        positiva.
    """
    if throughput_kbps is None or throughput_kbps <= 0.0:
        return float("inf")

    return (segment_bytes * 8) / (throughput_kbps * 1000.0)


class ExperimentRunner:
    """Executa downloads sequenciais e registra métricas por segmento."""

    def __init__(
        self,
        manifest: Manifest,
        policy: StreamingPolicy,
        csv_writer: CsvMetricsWriter,
        num_segments: int,
        alpha_jitter_ewma: float,
        alpha_throughput_ewma: float,
        observation_store: ObservationStore | None = None,
        simulated_server_failure: ServerFailurePredicate | None = None,
    ) -> None:
        """Prepara o estado compartilhado de uma execução.

        O runner cria um buffer vazio, inicia as EWMAs sem histórico e mantém a
        lista de servidores falhos entre segmentos para evitar novas tentativas
        enquanto o endpoint de saúde não indicar recuperação.

        Args:
            manifest: Servidores, representações e duração dos segmentos.
            policy: Estratégia que escolhe servidor e representação.
            csv_writer: Destino das métricas produzidas.
            num_segments: Quantidade de segmentos do experimento.
            alpha_jitter_ewma: Peso da amostra mais recente no jitter_ewma.
            alpha_throughput_ewma: Peso da amostra mais recente no throughput_ewma.
            observation_store: Snapshot opcional dos monitores de servidor.
            simulated_server_failure: Predicado opcional que marca um servidor
                como indisponível em um segmento. É usado somente por
                experimentos controlados de failover.
        """
        self.manifest: Manifest = manifest
        self.policy: StreamingPolicy = policy
        self.csv_writer: CsvMetricsWriter = csv_writer
        self.num_segments: int = num_segments
        self.alpha_jitter_ewma: float = alpha_jitter_ewma
        self.alpha_throughput_ewma: float = alpha_throughput_ewma
        self.observation_store: ObservationStore | None = observation_store
        self.simulated_server_failure = simulated_server_failure
        self.current_segment: int = 0

        self.buffer = BufferManager(
            max_level_s=BUFFER_MAX_S,
            target_level_s=BUFFER_TARGET_S,
            min_level_s=BUFFER_MIN_S,
            critical_level_s=BUFFER_CRITICAL_S,
        )
        self.throughput_history_kbps: list[float] = []
        self.throughput_ewma_kbps: float | None = None
        self.jitter_ewma_ms: float = 0.0
        self.failover_total: int = 0
        self.failed_server_ids: set[str] = set()

    def run(self) -> None:
        """Executa todos os segmentos e fecha o CSV mesmo em caso de erro.

        A lista de representações é impressa antes do loop para tornar explícitos
        os bitrates realmente recebidos do manifesto.
        """
        print(f"Iniciando download de {self.num_segments} segmentos...")
        print("Representações disponíveis no manifest:")
        for representation in self.manifest.representations:
            print(f"  {representation.quality}: {representation.bitrate_kbps} kbps")

        try:
            for seg_num in range(1, self.num_segments + 1):
                self._run_segment(seg_num)
        finally:
            self.csv_writer.close()

    def _run_segment(self, seg_num: int) -> None:
        """Executa o ciclo completo de uma amostra do experimento.

        A ordem estratégica é: coletar observações, consultar a política, baixar
        com recuperação de rede, atualizar EWMAs, consumir o tempo real da
        operação, adicionar o segmento, controlar o buffer cheio e persistir as
        métricas.

        Args:
            seg_num: Número sequencial do segmento, iniciado em um.
        """
        self.current_segment = seg_num
        observations: dict[str, ServerObservation] = self._get_observations()
        self._refresh_failed_servers()

        action: StreamingAction = self.policy.select_action(
            manifest=self.manifest,
            buffer=self.buffer,
            observations=observations,
            throughput_history_kbps=self.throughput_history_kbps,
            throughput_ewma_kbps=self.throughput_ewma_kbps,
        )

        server: ServerInfo = self._redirect_if_failed(action.server)
        chosen_rep: Representation = action.representation
        segment_path: str = f"{chosen_rep.url_path}?seg={seg_num}"
        timestamp: str = datetime.now(
            timezone(timedelta(hours=-3))
        ).isoformat()

        (
            result,
            server,
            failover_event,
            failover_duration_s,
            playback_elapsed_s,
        ) = (
            self._download_with_failover(
                server=server,
                path=segment_path,
                nominal_bitrate_kbps=chosen_rep.bitrate_kbps,
                seg_num=seg_num,
            )
        )

        self.throughput_history_kbps.append(result.throughput_kbps)
        self._update_ewmas(result)

        # Simulação explícita do player durante e após o download.
        self.buffer.consume(playback_elapsed_s)
        self.buffer.add_segment(self.manifest.segment_duration_s)

        stall_duration_s: float = self.buffer.get_stall_and_reset()
        rebuffer_event: int = int(stall_duration_s > 0.0)

        estimated_next_download_time_s: float = estimate_download_time_s(
            segment_bytes=chosen_rep.segment_bytes,
            throughput_kbps=self.throughput_ewma_kbps or result.throughput_kbps,
        )
        buffer_can_play: int = int(
            self.buffer.level_s > estimated_next_download_time_s
        )

        playback_wait_s: float = self.buffer.wait_if_full(
            resume_margin_s=self.manifest.segment_duration_s
        )

        self.policy.update_last_download_state(
            bitrate_kbps=chosen_rep.bitrate_kbps,
            download_time_s=result.download_time_s,
            rebuffer_event=rebuffer_event,
            server_index=self._server_index(server.id),
        )

        metrics = self._build_metrics(
            seg_num=seg_num,
            timestamp=timestamp,
            action=action,
            server=server,
            representation=chosen_rep,
            result=result,
            buffer_can_play=buffer_can_play,
            rebuffer_event=rebuffer_event,
            stall_duration_s=stall_duration_s,
            playback_wait_s=playback_wait_s,
            failover_event=failover_event,
            failover_duration_s=failover_duration_s,
            observations=observations,
        )
        self.csv_writer.write(metrics)

        print(
            f"Seg {seg_num:3d}: {chosen_rep.quality:5s} "
            f"Servidor={display_server_id(server.id):5s} "
            f"Vazão={result.throughput_kbps:7.1f} kbps "
            f"EWMA={self.throughput_ewma_kbps:7.1f} kbps "
            f"Buffer={self.buffer.level_s:5.2f}s "
            f"Rebuffer={rebuffer_event} Failover={failover_event}"
        )

    def _download_with_failover(
        self,
        server: ServerInfo,
        path: str,
        nominal_bitrate_kbps: int,
        seg_num: int,
    ) -> tuple[DownloadResult, ServerInfo, int, float, float]:
        """Baixa um segmento, alternando servidores durante indisponibilidade.

        Uma falha marca o servidor atual, inicia a medição do failover e procura
        um alternativo aprovado por ``/health``. Se todos estiverem fora do ar,
        aguarda a recuperação sem avançar o número do segmento. O tempo total da
        operação é retornado para que a queda seja consumida pelo buffer.

        Args:
            server: Servidor escolhido originalmente pela política.
            path: Caminho HTTP do segmento, incluindo a query string.
            nominal_bitrate_kbps: Bitrate usado pelo downloader como fallback.
            seg_num: Número do segmento, usado nas mensagens de diagnóstico.

        Returns:
            Tupla com resultado do download, servidor efetivamente usado,
            indicador de failover, duração do failover e tempo total percebido
            pelo player.

        Raises:
            RuntimeError: Se nenhum servidor se recuperar dentro do limite
                configurado.
        """
        operation_started_s: float = time.monotonic()
        recovery_deadline_s: float = (
            operation_started_s + NETWORK_RECOVERY_MAX_WAIT_S
        )
        failover_event: int = 0
        failover_started_s: float | None = None
        failover_duration_s: float = 0.0

        while True:
            try:
                if self._is_simulated_failure(server):
                    raise ConnectionError(
                        "falha simulada do servidor "
                        f"{server.id} no segmento {seg_num}"
                    )
                result = download_segment(
                    server_url=server.url,
                    path=path,
                    nominal_bitrate_kbps=nominal_bitrate_kbps,
                )
                playback_elapsed_s: float = (
                    time.monotonic() - operation_started_s
                )
                return (
                    result,
                    server,
                    failover_event,
                    failover_duration_s,
                    playback_elapsed_s,
                )
            except Exception as exc:
                print(
                    f"Erro no segmento {seg_num} usando servidor "
                    f"{display_server_id(server.id)}: {exc}"
                )
                self.failed_server_ids.add(server.id)

                if failover_started_s is None:
                    failover_started_s = time.monotonic()

                fallback_server = self._wait_for_healthy_fallback_server(
                    failed_server_id=server.id,
                    deadline_s=recovery_deadline_s,
                )
                if fallback_server is None:
                    raise RuntimeError(
                        "A rede não se recuperou dentro de "
                        f"{NETWORK_RECOVERY_MAX_WAIT_S:.0f}s."
                    ) from exc

                server = fallback_server
                failover_event = 1
                self.failover_total += 1
                failover_duration_s = time.monotonic() - failover_started_s

    def _update_ewmas(self, result: DownloadResult) -> None:
        """Atualiza as EWMAs após um download bem-sucedido.

        A primeira amostra inicializa a EWMA de vazão diretamente. As amostras
        seguintes combinam medição atual e histórico pelo ``alpha_ewma``.

        Args:
            result: Métricas de rede do segmento concluído.
        """
        if self.throughput_ewma_kbps is None:
            self.throughput_ewma_kbps = result.throughput_kbps
        else:
            self.throughput_ewma_kbps = (
                self.alpha_throughput_ewma * result.throughput_kbps
                + (1.0 - self.alpha_throughput_ewma) * self.throughput_ewma_kbps
            )

        self.jitter_ewma_ms = (
            self.alpha_jitter_ewma * result.jitter_network_ms
            + (1.0 - self.alpha_jitter_ewma) * self.jitter_ewma_ms
        )

    def _refresh_failed_servers(self) -> None:
        """Revalida servidores falhos antes da próxima decisão.

        Um servidor só volta a ser elegível quando seu endpoint ``/health``
        responde positivamente.
        """
        for server in self.manifest.servers:
            if server.id not in self.failed_server_ids:
                continue
            if self._is_simulated_failure(server):
                continue
            if probe_server_health(
                server,
                timeout_s=HEALTH_CHECK_TIMEOUT_S,
            ).ok:
                self.failed_server_ids.remove(server.id)

    def _redirect_if_failed(self, server: ServerInfo) -> ServerInfo:
        """Redireciona uma escolha que aponta para servidor ainda falho.

        Args:
            server: Servidor selecionado pela política.

        Returns:
            Alternativo saudável, quando disponível. Caso nenhum health check
            confirme um alternativo, retorna o servidor original para permitir
            uma tentativa real de recuperação.
        """
        if server.id not in self.failed_server_ids:
            return server

        fallback = self._choose_healthy_fallback_server(server.id)
        if fallback is None:
            # A tentativa real ainda pode funcionar mesmo se o health check oscilar.
            return server
        return fallback

    def _wait_for_healthy_fallback_server(
        self,
        failed_server_id: str,
        deadline_s: float,
    ) -> ServerInfo | None:
        """Aguarda um servidor alternativo ficar saudável.

        Args:
            failed_server_id: Servidor que falhou na tentativa mais recente.
            deadline_s: Instante monotônico máximo para encerrar a espera.

        Returns:
            Primeiro alternativo saudável por prioridade, ou ``None`` ao atingir
            o prazo.
        """
        while True:
            fallback = self._choose_healthy_fallback_server(failed_server_id)
            if fallback is not None:
                return fallback

            remaining_s: float = deadline_s - time.monotonic()
            if remaining_s <= 0.0:
                return None

            wait_s: float = min(NETWORK_RETRY_DELAY_S, remaining_s)
            print(
                "Todos os servidores estão indisponíveis; "
                f"nova tentativa em {wait_s:.1f}s..."
            )
            time.sleep(wait_s)

    def _choose_healthy_fallback_server(
        self,
        failed_server_id: str,
    ) -> ServerInfo | None:
        """Procura uma alternativa saudável seguindo a prioridade do manifesto.

        Args:
            failed_server_id: Servidor que não deve ser selecionado nesta busca.

        Returns:
            Primeiro servidor com health check positivo, ou ``None`` quando não
            houver candidato saudável.
        """
        candidates = sorted(self.manifest.servers, key=lambda item: item.priority)
        for candidate in candidates:
            if candidate.id == failed_server_id:
                continue
            if self._is_simulated_failure(candidate):
                continue
            if probe_server_health(
                candidate,
                timeout_s=HEALTH_CHECK_TIMEOUT_S,
            ).ok:
                self.failed_server_ids.discard(candidate.id)
                return candidate
        return None

    def _is_simulated_failure(self, server: ServerInfo) -> bool:
        """Indica se o experimento tornou um servidor indisponível.

        O predicado é consultado tanto antes do download quanto nos fluxos de
        recuperação. Assim, um servidor que caiu não reaparece apenas porque o
        endpoint real continua saudável durante a simulação.

        Args:
            server: Servidor cuja disponibilidade deve ser consultada.

        Returns:
            ``True`` somente quando há um injetor configurado e ele marca o
            servidor como falho no segmento atual.
        """
        if self.simulated_server_failure is None:
            return False
        return self.simulated_server_failure(self.current_segment, server)

    def _get_observations(self) -> dict[str, ServerObservation]:
        """Obtém um snapshot consistente das observações de monitoramento.

        Returns:
            Cópia das observações indexada por servidor, ou dicionário vazio
            quando o experimento não usa monitores.
        """
        if self.observation_store is None:
            return {}
        return self.observation_store.snapshot()

    def _server_index(self, server_id: str) -> int:
        """Converte o identificador do servidor para a posição no manifesto.

        Args:
            server_id: Identificador textual procurado.

        Returns:
            Índice do servidor, usado pelas features da Política 3.

        Raises:
            ValueError: Se o servidor não fizer parte do manifesto.
        """
        for index, server in enumerate(self.manifest.servers):
            if server.id == server_id:
                return index
        raise ValueError(f"Servidor não encontrado no manifest: {server_id}")

    def _build_metrics(
        self,
        seg_num: int,
        timestamp: str,
        action: StreamingAction,
        server: ServerInfo,
        representation: Representation,
        result: DownloadResult,
        buffer_can_play: int,
        rebuffer_event: int,
        stall_duration_s: float,
        playback_wait_s: float,
        failover_event: int,
        failover_duration_s: float,
        observations: dict[str, ServerObservation],
    ) -> SegmentMetrics:
        """Consolida estado do player, rede, failover e probes em uma amostra.

        Args:
            seg_num: Número sequencial do segmento.
            timestamp: Horário ISO 8601 do início da operação.
            action: Decisão original da política, incluindo previsões opcionais.
            server: Servidor que concluiu o download.
            representation: Qualidade escolhida para o segmento.
            result: Medições do download bem-sucedido.
            buffer_can_play: Indicador de sobrevivência ao próximo download.
            rebuffer_event: Indicador de stall no segmento.
            stall_duration_s: Duração acumulada do stall.
            playback_wait_s: Pausa feita por buffer cheio.
            failover_event: Indicador de troca de servidor.
            failover_duration_s: Tempo para confirmar o alternativo.
            observations: Snapshot dos probes no momento da decisão.

        Returns:
            Estrutura imutável pronta para persistência no CSV.
        """
        obs_a = self._observation_by_index(observations, 0)
        obs_b = self._observation_by_index(observations, 1)

        return SegmentMetrics(
            segment=seg_num,
            timestamp=timestamp,
            startup_phase=int(seg_num <= RNN_STARTUP_SEGMENTS),
            server_id=display_server_id(server.id),
            quality=representation.quality,
            bitrate_kbps=representation.bitrate_kbps,
            throughput_kbps=result.throughput_kbps,
            throughput_ewma_kbps=self.throughput_ewma_kbps,
            download_time_s=result.download_time_s,
            jitter_network_ms=result.jitter_network_ms,
            jitter_ewma_ms=self.jitter_ewma_ms,
            buffer_level_s=self.buffer.level_s,
            buffer_can_play=buffer_can_play,
            rebuffer_event=rebuffer_event,
            stall_duration_s=stall_duration_s,
            playback_wait_s=playback_wait_s,
            failover_event=failover_event,
            failover_duration_s=failover_duration_s,
            failover_total=self.failover_total,
            rnn_predicted_a_throughput_kbps=(
                action.predicted_server_a_throughput_kbps
            ),
            rnn_predicted_b_throughput_kbps=(
                action.predicted_server_b_throughput_kbps
            ),
            rnn_predicted_selected_throughput_kbps=(
                action.predicted_selected_throughput_kbps
            ),
            probe_a_ok=None if obs_a is None else int(obs_a.success),
            probe_a_latency_ms=None if obs_a is None else obs_a.latency_ms,
            probe_a_throughput_kbps=None if obs_a is None else obs_a.throughput_kbps,
            probe_a_jitter_ms=None if obs_a is None else obs_a.jitter_ms,
            probe_b_ok=None if obs_b is None else int(obs_b.success),
            probe_b_latency_ms=None if obs_b is None else obs_b.latency_ms,
            probe_b_throughput_kbps=None if obs_b is None else obs_b.throughput_kbps,
            probe_b_jitter_ms=None if obs_b is None else obs_b.jitter_ms,
        )

    def _observation_by_index(
        self,
        observations: dict[str, ServerObservation],
        index: int,
    ) -> ServerObservation | None:
        """Obtém a observação associada a uma posição do manifesto.

        Args:
            observations: Snapshot indexado por identificador.
            index: Posição esperada do servidor no manifesto.

        Returns:
            Observação correspondente, ou ``None`` se o índice ou a observação
            não estiver disponível.
        """
        if index >= len(self.manifest.servers):
            return None
        return observations.get(self.manifest.servers[index].id)
