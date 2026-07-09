"""
Treina a rede neural recorrente usada pela política 3.

Este módulo lê um CSV de métricas contendo observações dos servidores, constrói
sequências temporais, treina uma GRU para prever os probes futuros dos
servidores A/B e a vazão real do próximo download, e salva um checkpoint com os
pesos do modelo e os parâmetros de normalização.

Exemplo de uso:

    python -m models.train \
        --csv outputs/metricas_policy3_rnn.csv \
        --output outputs/models/rnn_policy.pt \
        --server-a A \
        --server-b B
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    RNN_FEATURE_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_SEQUENCE_LENGTH,
    RNN_TARGET_SIZE,
)
from models.dataset import (
    FeatureNormalizer,
    RnnSample,
    StreamingRnnDataset,
    compute_feature_normalizer,
    load_rnn_samples_from_csv,
    split_samples,
)
from models.rnn import StreamingRNN


@dataclass(frozen=True)
class TrainConfig:
    """Configuração de treinamento da RNN."""

    csv_path: Path
    output_path: Path
    server_a_id: str
    server_b_id: str
    sequence_length: int
    probe_target_smoothing_window: int
    feature_size: int
    output_size: int
    hidden_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    train_ratio: float
    device: str
    weight_decay: float
    dropout: float
    l1_lambda: float
    grad_clip_norm: float
    huber_beta: float
    probe_loss_weight: float
    download_loss_weight: float
    rank_loss_weight: float
    probe_diff_loss_weight: float
    rank_logit_scale_kbps: float
    rank_margin_threshold_kbps: float


@dataclass(frozen=True)
class EpochMetrics:
    """Métricas agregadas de uma época."""

    loss: float


class RnnPredictionLoss(nn.Module):
    """Loss robusta com termo explícito de ranking entre servidores A/B."""

    def __init__(
        self,
        normalizer: FeatureNormalizer,
        probe_loss_weight: float,
        download_loss_weight: float,
        rank_loss_weight: float,
        probe_diff_loss_weight: float,
        huber_beta: float,
        rank_logit_scale_kbps: float,
        rank_margin_threshold_kbps: float,
    ) -> None:
        super().__init__()
        if normalizer.target_mean is None or normalizer.target_std is None:
            raise ValueError("Loss de ranking requer normalização dos alvos.")
        if rank_logit_scale_kbps <= 0.0:
            raise ValueError("rank_logit_scale_kbps deve ser positivo.")
        if rank_margin_threshold_kbps < 0.0:
            raise ValueError("rank_margin_threshold_kbps não pode ser negativo.")
        if huber_beta <= 0.0:
            raise ValueError("huber_beta deve ser positivo.")
        if min(
            probe_loss_weight,
            download_loss_weight,
            rank_loss_weight,
            probe_diff_loss_weight,
        ) < 0.0:
            raise ValueError("Pesos da loss não podem ser negativos.")

        self.rank_loss_weight = rank_loss_weight
        self.probe_diff_loss_weight = probe_diff_loss_weight
        self.huber_beta = huber_beta
        self.rank_logit_scale_kbps = rank_logit_scale_kbps
        self.rank_margin_threshold_kbps = rank_margin_threshold_kbps
        self.register_buffer(
            "target_mean",
            torch.tensor(normalizer.target_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "target_std",
            torch.tensor(normalizer.target_std, dtype=torch.float32),
        )
        self.register_buffer(
            "target_weights",
            torch.tensor(
                [
                    probe_loss_weight,
                    probe_loss_weight,
                    download_loss_weight,
                ],
                dtype=torch.float32,
            ),
        )

    def _denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.target_std + self.target_mean

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula regressão robusta e ranking A/B."""
        regression = F.smooth_l1_loss(
            prediction,
            target,
            beta=self.huber_beta,
            reduction="none",
        )
        regression_loss = (regression * self.target_weights).mean()

        predicted_raw = self._denormalize(prediction)
        target_raw = self._denormalize(target)
        predicted_probe_diff = predicted_raw[:, 0] - predicted_raw[:, 1]
        target_probe_diff = target_raw[:, 0] - target_raw[:, 1]

        confident_mask = (
            target_probe_diff.abs() >= self.rank_margin_threshold_kbps
        )
        if confident_mask.any():
            rank_target = (target_probe_diff[confident_mask] >= 0.0).float()
            rank_logits = (
                predicted_probe_diff[confident_mask]
                / self.rank_logit_scale_kbps
            )
            rank_loss = F.binary_cross_entropy_with_logits(
                rank_logits,
                rank_target,
            )
        else:
            rank_loss = prediction.sum() * 0.0

        diff_loss = F.smooth_l1_loss(
            predicted_probe_diff / self.rank_logit_scale_kbps,
            target_probe_diff / self.rank_logit_scale_kbps,
            beta=1.0,
        )

        return (
            regression_loss
            + self.rank_loss_weight * rank_loss
            + self.probe_diff_loss_weight * diff_loss
        )


def parse_args() -> TrainConfig:
    """
    Lê os argumentos de linha de comando.

    Returns:
        Configuração de treinamento.
    """
    parser = argparse.ArgumentParser(
        description="Treina a RNN da política 3.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/rnn_training_data.csv"),
        help="Caminho do CSV de treino coletado antes da política RNN.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/models/rnn_policy.pt"),
        help="Caminho do checkpoint gerado.",
    )

    parser.add_argument(
        "--server-a",
        type=str,
        default="A",
        help="Identificador do servidor A no manifest/CSV.",
    )

    parser.add_argument(
        "--server-b",
        type=str,
        default="B",
        help="Identificador do servidor B no manifest/CSV.",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=RNN_SEQUENCE_LENGTH,
        help="Tamanho da janela temporal da RNN.",
    )

    parser.add_argument(
        "--probe-target-window",
        type=int,
        default=3,
        help="Janela de mediana dos probes usados como alvo A/B.",
    )

    parser.add_argument(
        "--feature-size",
        type=int,
        default=RNN_FEATURE_SIZE,
        help="Quantidade de features por timestep.",
    )

    parser.add_argument(
        "--output-size",
        type=int,
        default=RNN_TARGET_SIZE,
        help="Quantidade de saídas previstas pela RNN.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=RNN_HIDDEN_SIZE,
        help="Tamanho do estado oculto da GRU.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas de treinamento.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-4,
        help="Regularização L2/weight decay aplicada pelo AdamW.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
        help="Dropout aplicado ao estado oculto antes da camada final.",
    )

    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.0,
        help="Regularização L1 opcional aplicada aos pesos.",
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Norma máxima do gradiente. Use 0 para desativar.",
    )

    parser.add_argument(
        "--huber-beta",
        type=float,
        default=1.0,
        help="Beta da SmoothL1/Huber loss em escala normalizada.",
    )

    parser.add_argument(
        "--probe-loss-weight",
        type=float,
        default=1.25,
        help="Peso da regressão dos probes A/B.",
    )

    parser.add_argument(
        "--download-loss-weight",
        type=float,
        default=0.75,
        help="Peso da regressão da vazão real de download.",
    )

    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.75,
        help="Peso da loss de classificação do melhor probe A/B.",
    )

    parser.add_argument(
        "--probe-diff-loss-weight",
        type=float,
        default=0.30,
        help="Peso da loss da diferença probeA - probeB.",
    )

    parser.add_argument(
        "--rank-logit-scale-kbps",
        type=float,
        default=80.0,
        help="Escala em kbps usada no logit do ranking A/B.",
    )

    parser.add_argument(
        "--rank-margin-threshold-kbps",
        type=float,
        default=30.0,
        help="Margem mínima entre probes para cobrar a loss de ranking A/B.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fração das amostras usada para treino.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Dispositivo de treino. Em auto, usa GPU CUDA se disponível.",
    )

    args = parser.parse_args()

    return TrainConfig(
        csv_path=args.csv,
        output_path=args.output,
        server_a_id=args.server_a,
        server_b_id=args.server_b,
        sequence_length=args.sequence_length,
        probe_target_smoothing_window=args.probe_target_window,
        feature_size=args.feature_size,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
        device=args.device,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        l1_lambda=args.l1_lambda,
        grad_clip_norm=args.grad_clip,
        huber_beta=args.huber_beta,
        probe_loss_weight=args.probe_loss_weight,
        download_loss_weight=args.download_loss_weight,
        rank_loss_weight=args.rank_loss_weight,
        probe_diff_loss_weight=args.probe_diff_loss_weight,
        rank_logit_scale_kbps=args.rank_logit_scale_kbps,
        rank_margin_threshold_kbps=args.rank_margin_threshold_kbps,
    )


def select_device(requested_device: str) -> torch.device:
    """
    Seleciona o dispositivo de treinamento.

    Args:
        requested_device: `auto`, `cuda` ou `cpu`.

    Returns:
        Dispositivo PyTorch selecionado.

    Raises:
        RuntimeError: Se CUDA for solicitado explicitamente e não estiver
            disponível no PyTorch instalado.
    """
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA foi solicitada, mas não está disponível. "
                "Instale uma build do PyTorch com CUDA ou use --device cpu."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_dataloaders(
    config: TrainConfig,
    pin_memory: bool,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    FeatureNormalizer,
]:
    """
    Cria os DataLoaders de treino e validação.

    Args:
        config: Configuração de treinamento.

    Returns:
        Tupla `(train_loader, val_loader, normalizer)`.
    """
    samples: list[RnnSample] = load_rnn_samples_from_csv(
        csv_path=config.csv_path,
        sequence_length=config.sequence_length,
        server_a_id=config.server_a_id,
        server_b_id=config.server_b_id,
        probe_target_smoothing_window=config.probe_target_smoothing_window,
    )
    sample_feature_size = len(samples[0].x[0])
    if sample_feature_size != config.feature_size:
        raise ValueError(
            "Feature size configurado incompatível com o dataset: "
            f"config={config.feature_size}, dataset={sample_feature_size}."
        )
    sample_output_size = len(samples[0].y)
    if sample_output_size != config.output_size:
        raise ValueError(
            "Output size configurado incompatível com o dataset: "
            f"config={config.output_size}, dataset={sample_output_size}."
        )

    train_samples, val_samples = split_samples(
        samples=samples,
        train_ratio=config.train_ratio,
    )

    normalizer: FeatureNormalizer = compute_feature_normalizer(train_samples)

    train_dataset = StreamingRnnDataset(
        samples=train_samples,
        normalizer=normalizer,
    )

    val_dataset = StreamingRnnDataset(
        samples=val_samples,
        normalizer=normalizer,
    )

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, normalizer


def train_one_epoch(
    model: StreamingRNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l1_lambda: float,
    grad_clip_norm: float,
) -> EpochMetrics:
    """
    Executa uma época de treinamento.

    Args:
        model: Modelo RNN.
        loader: DataLoader de treino.
        loss_fn: Função de perda.
        optimizer: Otimizador.
        device: Dispositivo de execução.

    Returns:
        Métricas agregadas da época.
    """
    model.train()

    total_loss: float = 0.0
    total_batches: int = 0

    for x_batch, y_batch in loader:
        use_non_blocking = device.type == "cuda"
        x_batch = x_batch.to(device, non_blocking=use_non_blocking)
        y_batch = y_batch.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad()

        prediction: torch.Tensor = model(x_batch)
        loss: torch.Tensor = loss_fn(prediction, y_batch)

        if l1_lambda > 0.0:
            l1_penalty = torch.zeros((), device=device)
            for parameter in model.parameters():
                l1_penalty = l1_penalty + parameter.abs().sum()
            loss = loss + l1_lambda * l1_penalty

        loss.backward()
        if grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return EpochMetrics(
        loss=total_loss / max(total_batches, 1),
    )


def evaluate(
    model: StreamingRNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """
    Avalia o modelo em validação.

    Args:
        model: Modelo RNN.
        loader: DataLoader de validação.
        loss_fn: Função de perda.
        device: Dispositivo de execução.

    Returns:
        Métricas agregadas de validação.
    """
    model.eval()

    total_loss: float = 0.0
    total_batches: int = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            use_non_blocking = device.type == "cuda"
            x_batch = x_batch.to(device, non_blocking=use_non_blocking)
            y_batch = y_batch.to(device, non_blocking=use_non_blocking)

            prediction: torch.Tensor = model(x_batch)
            loss: torch.Tensor = loss_fn(prediction, y_batch)

            total_loss += float(loss.item())
            total_batches += 1

    return EpochMetrics(
        loss=total_loss / max(total_batches, 1),
    )


def save_checkpoint(
    model: StreamingRNN,
    normalizer: FeatureNormalizer,
    config: TrainConfig,
) -> None:
    """
    Salva o checkpoint do modelo treinado.

    Args:
        model: Modelo treinado.
        normalizer: Normalizador calculado a partir do conjunto de treino.
        config: Configuração de treinamento.
    """
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "target_mean": normalizer.target_mean,
        "target_std": normalizer.target_std,
        "target_clip_min": normalizer.target_clip_min,
        "target_clip_max": normalizer.target_clip_max,
        "sequence_length": config.sequence_length,
        "probe_target_smoothing_window": config.probe_target_smoothing_window,
        "feature_size": config.feature_size,
        "output_size": config.output_size,
        "hidden_size": config.hidden_size,
        "dropout": config.dropout,
        "loss_config": {
            "type": "smooth_l1_plus_rank",
            "huber_beta": config.huber_beta,
            "probe_loss_weight": config.probe_loss_weight,
            "download_loss_weight": config.download_loss_weight,
            "rank_loss_weight": config.rank_loss_weight,
            "probe_diff_loss_weight": config.probe_diff_loss_weight,
            "rank_logit_scale_kbps": config.rank_logit_scale_kbps,
            "rank_margin_threshold_kbps": config.rank_margin_threshold_kbps,
            "weight_decay": config.weight_decay,
            "l1_lambda": config.l1_lambda,
            "grad_clip_norm": config.grad_clip_norm,
        },
        "server_a_id": config.server_a_id,
        "server_b_id": config.server_b_id,
    }

    torch.save(checkpoint, config.output_path)


def main() -> None:
    """Executa o treinamento da RNN."""
    config: TrainConfig = parse_args()

    device: torch.device = select_device(config.device)

    print(f"Usando dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Lendo dataset: {config.csv_path}")

    train_loader, val_loader, normalizer = create_dataloaders(
        config,
        pin_memory=device.type == "cuda",
    )

    model = StreamingRNN(
        input_size=config.feature_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        dropout=config.dropout,
    ).to(device)

    loss_fn: nn.Module = RnnPredictionLoss(
        normalizer=normalizer,
        probe_loss_weight=config.probe_loss_weight,
        download_loss_weight=config.download_loss_weight,
        rank_loss_weight=config.rank_loss_weight,
        probe_diff_loss_weight=config.probe_diff_loss_weight,
        huber_beta=config.huber_beta,
        rank_logit_scale_kbps=config.rank_logit_scale_kbps,
        rank_margin_threshold_kbps=config.rank_margin_threshold_kbps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss: float | None = None

    for epoch in range(1, config.epochs + 1):
        train_metrics: EpochMetrics = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            l1_lambda=config.l1_lambda,
            grad_clip_norm=config.grad_clip_norm,
        )

        val_metrics: EpochMetrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        improved: bool = (
            best_val_loss is None
            or val_metrics.loss < best_val_loss
        )

        if improved:
            best_val_loss = val_metrics.loss
            save_checkpoint(
                model=model,
                normalizer=normalizer,
                config=config,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"{'*' if improved else ''}"
        )

    print(f"Melhor checkpoint salvo em: {config.output_path}")


if __name__ == "__main__":
    main()
