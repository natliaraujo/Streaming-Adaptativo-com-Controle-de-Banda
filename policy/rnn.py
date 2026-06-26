"""
Implementa a política 3 baseada em RNN.

A política usa uma sequência temporal de observações dos servidores e do estado
do player para prever a vazão futura dos servidores A e B. Depois escolhe o
servidor com maior vazão prevista e seleciona uma qualidade segura para o buffer.
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
    """Política de streaming que usa RNN para prever vazão futura."""

    def __init__(
        self,
        model: StreamingRNN,
        feature_config: FeatureConfig,
        feature_history: FeatureHistory,
        normalizer: FeatureNormalizer,
        safety_factor: float = 0.8,
    ) -> None:
        """
        Inicializa a política RNN.

        Args:
            model: Modelo RNN pré-treinado.
            feature_config: Configuração de features.
            feature_history: Janela temporal de features.
            normalizer: Normalizador salvo no treinamento.
            safety_factor: Fator de segurança para seleção de qualidade.
        """
        self.model: StreamingRNN = model
        self.feature_config: FeatureConfig = feature_config
        self.feature_history: FeatureHistory = feature_history
        self.normalizer: FeatureNormalizer = normalizer
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

        predicted_a: float = max(0.0, float(prediction[0, 0].item()))
        predicted_b: float = max(0.0, float(prediction[0, 1].item()))

        server_a: ServerInfo = manifest.servers[0]
        server_b: ServerInfo = manifest.servers[1]

        if predicted_a >= predicted_b:
            server: ServerInfo = server_a
            predicted_throughput_kbps: float = predicted_a
        else:
            server = server_b
            predicted_throughput_kbps = predicted_b

        representation: Representation = self.quality_selector.select(
            representations=manifest.representations,
            predicted_throughput_kbps=predicted_throughput_kbps,
            buffer_level_s=buffer.level_s,
            segment_duration_s=manifest.segment_duration_s,
        )

        return StreamingAction(
            server=server,
            representation=representation,
            predicted_server_a_throughput_kbps=predicted_a,
            predicted_server_b_throughput_kbps=predicted_b,
            predicted_selected_throughput_kbps=predicted_throughput_kbps,
        )

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
