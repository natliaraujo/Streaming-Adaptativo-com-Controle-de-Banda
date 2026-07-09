"""
Implementa a política 3 baseada em RNN.

A política usa uma sequência temporal de observações dos servidores e do estado
do player para prever os probes futuros dos servidores A e B e a vazão real
esperada do próximo download. Os probes ranqueiam o servidor; a vazão real
prevista seleciona a qualidade segura para o buffer.
"""

import torch

from domain.action import StreamingAction
from domain.manifest import Manifest, Representation, ServerInfo
from models.dataset import FeatureNormalizer
from models.rnn import StreamingRNN
from monitoring.feature_builder import (
    FeatureConfig,
    FeatureHistory,
    PlayerState,
    build_feature_vector,
)
from monitoring.observation_store import ServerObservation
from player.buffer import BufferManager
from policy.quality_selector import BufferAwareQualitySelector
from policy.streaming_policy import StreamingPolicy


class RnnStreamingPolicy(StreamingPolicy):
    """Política de streaming que usa RNN para prever probes e vazão real."""

    def __init__(
        self,
        model: StreamingRNN,
        feature_config: FeatureConfig,
        feature_history: FeatureHistory,
        normalizer: FeatureNormalizer,
        safety_factor: float = 0.8,
        server_selection_margin_kbps: float = 30.0,
        observed_probe_weight: float = 0.70,
    ) -> None:
        """
        Inicializa a política RNN.

        Args:
            model: Modelo RNN pré-treinado.
            feature_config: Configuração de features.
            feature_history: Janela temporal de features.
            normalizer: Normalizador salvo no treinamento.
            safety_factor: Fator de segurança para seleção de qualidade.
            server_selection_margin_kbps: Margem mínima para trocar de servidor
                quando os scores A/B estão quase empatados.
            observed_probe_weight: Peso do probe atual no score híbrido de
                ranking. O restante vem da previsão da RNN.
        """
        if server_selection_margin_kbps < 0.0:
            raise ValueError("server_selection_margin_kbps não pode ser negativo.")
        if not 0.0 <= observed_probe_weight <= 1.0:
            raise ValueError("observed_probe_weight deve estar entre 0 e 1.")

        self.model: StreamingRNN = model
        self.feature_config: FeatureConfig = feature_config
        self.feature_history: FeatureHistory = feature_history
        self.normalizer: FeatureNormalizer = normalizer
        self.server_selection_margin_kbps = server_selection_margin_kbps
        self.observed_probe_weight = observed_probe_weight
        self.quality_selector = BufferAwareQualitySelector(
            safety_factor=safety_factor,
        )

        self.last_bitrate_kbps: float = 0.0
        self.last_download_time_s: float = 0.0
        self.last_rebuffer_event: int = 0
        self.last_server_index: int = 0
        self.decision_count: int = 0

    def update_last_download_state(
        self,
        bitrate_kbps: float,
        download_time_s: float,
        rebuffer_event: int,
        server_index: int,
    ) -> None:
        """Atualiza as features de feedback usadas no próximo timestep.

        Args:
            bitrate_kbps: Bitrate nominal do segmento concluído.
            download_time_s: Tempo do download bem-sucedido.
            rebuffer_event: Indicador de rebuffering observado.
            server_index: Índice do servidor que concluiu o segmento.
        """
        self.last_bitrate_kbps = bitrate_kbps
        self.last_download_time_s = download_time_s
        self.last_rebuffer_event = rebuffer_event
        self.last_server_index = server_index

    def select_action(
        self,
        manifest: Manifest,
        buffer: BufferManager,
        observations: dict[str, ServerObservation],
        throughput_history_kbps: list[float],
        throughput_ewma_kbps: float | None = None,
    ) -> StreamingAction:
        """
        Escolhe servidor e representação para o próximo segmento.

        Args:
            manifest: Manifesto do experimento.
            buffer: Buffer atual do player.
            observations: Observações recentes dos servidores.
            throughput_history_kbps: Histórico de vazões reais.
            throughput_ewma_kbps: EWMA da interface comum; a RNN não a usa
                diretamente, pois produz sua própria previsão.

        Returns:
            Ação contendo servidor e representação.
        """
        player_state = PlayerState(
            buffer_level_s=buffer.level_s,
            last_bitrate_kbps=self.last_bitrate_kbps,
            last_download_time_s=self.last_download_time_s,
            last_rebuffer_event=self.last_rebuffer_event,
            last_server_index=self.last_server_index,
            startup_phase=int(
                self.decision_count < self.feature_config.startup_segments
            ),
        )

        feature_vector: list[float] = build_feature_vector(
            observations=observations,
            player_state=player_state,
            config=self.feature_config,
        )

        self.feature_history.append(feature_vector)
        self.decision_count += 1

        if not self.feature_history.is_ready():
            return self._fallback_action(
                manifest=manifest,
                throughput_history_kbps=throughput_history_kbps,
                buffer_level_s=buffer.level_s,
            )

        raw_sequence: list[list[float]] = self.feature_history.to_sequence()
        normalized_sequence: list[list[float]] = (
            self.normalizer.normalize_sequence(raw_sequence)
        )

        x: torch.Tensor = torch.tensor(
            [normalized_sequence],
            dtype=torch.float32,
        )

        self.model.eval()

        with torch.no_grad():
            prediction: torch.Tensor = self.model(x)

        predicted_values: list[float] = self.normalizer.denormalize_target(
            [
                float(prediction[0, index].item())
                for index in range(prediction.shape[1])
            ]
        )
        predicted_a_probe: float = max(0.0, predicted_values[0])
        predicted_b_probe: float = max(0.0, predicted_values[1])
        predicted_download_throughput: float = max(
            0.0,
            predicted_values[2],
        )

        server_a: ServerInfo = manifest.servers[0]
        server_b: ServerInfo = manifest.servers[1]

        observed_a_probe = self._observed_probe_throughput(
            observations,
            server_a.id,
        )
        observed_b_probe = self._observed_probe_throughput(
            observations,
            server_b.id,
        )
        score_a = self._server_ranking_score(
            predicted_probe=predicted_a_probe,
            observed_probe=observed_a_probe,
        )
        score_b = self._server_ranking_score(
            predicted_probe=predicted_b_probe,
            observed_probe=observed_b_probe,
        )
        server = self._select_server_from_scores(
            manifest=manifest,
            server_a=server_a,
            score_a=score_a,
            server_b=server_b,
            score_b=score_b,
        )
        predicted_selected_probe_throughput = (
            predicted_a_probe
            if server.id == server_a.id
            else predicted_b_probe
        )

        representation: Representation = self.quality_selector.select(
            representations=manifest.representations,
            predicted_throughput_kbps=predicted_download_throughput,
            buffer_level_s=buffer.level_s,
            segment_duration_s=manifest.segment_duration_s,
        )

        return StreamingAction(
            server=server,
            representation=representation,
            predicted_server_a_throughput_kbps=predicted_a_probe,
            predicted_server_b_throughput_kbps=predicted_b_probe,
            predicted_selected_throughput_kbps=predicted_selected_probe_throughput,
            predicted_download_throughput_kbps=predicted_download_throughput,
        )

    def _observed_probe_throughput(
        self,
        observations: dict[str, ServerObservation],
        server_id: str,
    ) -> float | None:
        """Retorna o probe atual de um servidor quando ele é confiável."""
        observation = observations.get(server_id)
        if observation is None or not observation.success:
            return None
        if observation.throughput_kbps is None or observation.throughput_kbps <= 0.0:
            return None
        return observation.throughput_kbps

    def _server_ranking_score(
        self,
        predicted_probe: float,
        observed_probe: float | None,
    ) -> float:
        """Combina previsão da RNN e probe medido para ranquear servidores."""
        if observed_probe is None:
            return predicted_probe
        return (
            self.observed_probe_weight * observed_probe
            + (1.0 - self.observed_probe_weight) * predicted_probe
        )

    def _select_server_from_scores(
        self,
        manifest: Manifest,
        server_a: ServerInfo,
        score_a: float,
        server_b: ServerInfo,
        score_b: float,
    ) -> ServerInfo:
        """Escolhe servidor usando margem para evitar trocas por ruído."""
        margin = self.server_selection_margin_kbps
        if score_a >= score_b + margin:
            return server_a
        if score_b >= score_a + margin:
            return server_b
        if 0 <= self.last_server_index < len(manifest.servers):
            return manifest.servers[self.last_server_index]
        return sorted(manifest.servers, key=lambda item: item.priority)[0]

    def _fallback_action(
        self,
        manifest: Manifest,
        throughput_history_kbps: list[float],
        buffer_level_s: float,
    ) -> StreamingAction:
        """Escolhe uma ação enquanto ainda não há sequência completa para a RNN.

        A estratégia usa o servidor prioritário e estima vazão pela média do
        histórico, pela banda nominal ou pelo menor bitrate, nessa ordem.

        Args:
            manifest: Manifesto com servidores e representações.
            throughput_history_kbps: Vazões reais já observadas.
            buffer_level_s: Reserva atual do player.

        Returns:
            Ação heurística usada durante o aquecimento da janela temporal.
        """
        server: ServerInfo = sorted(
            manifest.servers,
            key=lambda item: item.priority,
        )[0]

        if throughput_history_kbps:
            estimated_throughput: float = (
                sum(throughput_history_kbps) / len(throughput_history_kbps)
            )
        elif server.bandwidth_kbps is not None:
            estimated_throughput = server.bandwidth_kbps
        else:
            estimated_throughput = manifest.representations[0].bitrate_kbps

        representation: Representation = self.quality_selector.select(
            representations=manifest.representations,
            predicted_throughput_kbps=estimated_throughput,
            buffer_level_s=buffer_level_s,
            segment_duration_s=manifest.segment_duration_s,
        )

        return StreamingAction(
            server=server,
            representation=representation,
        )
