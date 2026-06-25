"""
Treina a rede neural recorrente usada pela política 3.

Este módulo lê um CSV de métricas contendo observações dos servidores, constrói
sequências temporais, treina uma GRU para prever a vazão futura dos servidores A
e B e salva um checkpoint com os pesos do modelo e os parâmetros de normalização.

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
from torch.utils.data import DataLoader

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
    feature_size: int
    hidden_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    train_ratio: float


@dataclass(frozen=True)
class EpochMetrics:
    """Métricas agregadas de uma época."""

    loss: float


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
        default=10,
        help="Tamanho da janela temporal da RNN.",
    )

    parser.add_argument(
        "--feature-size",
        type=int,
        default=15,
        help="Quantidade de features por timestep.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
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
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fração das amostras usada para treino.",
    )

    args = parser.parse_args()

    return TrainConfig(
        csv_path=args.csv,
        output_path=args.output,
        server_a_id=args.server_a,
        server_b_id=args.server_b,
        sequence_length=args.sequence_length,
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
    )


def create_dataloaders(
    config: TrainConfig,
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
    )

    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, normalizer


def train_one_epoch(
    model: StreamingRNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        prediction: torch.Tensor = model(x_batch)
        loss: torch.Tensor = loss_fn(prediction, y_batch)

        loss.backward()
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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

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
        "sequence_length": config.sequence_length,
        "feature_size": config.feature_size,
        "hidden_size": config.hidden_size,
        "server_a_id": config.server_a_id,
        "server_b_id": config.server_b_id,
    }

    torch.save(checkpoint, config.output_path)


def main() -> None:
    """Executa o treinamento da RNN."""
    config: TrainConfig = parse_args()

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Usando dispositivo: {device}")
    print(f"Lendo dataset: {config.csv_path}")

    train_loader, val_loader, normalizer = create_dataloaders(config)

    model = StreamingRNN(
        input_size=config.feature_size,
        hidden_size=config.hidden_size,
        output_size=2,
    ).to(device)

    loss_fn: nn.Module = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    best_val_loss: float | None = None

    for epoch in range(1, config.epochs + 1):
        train_metrics: EpochMetrics = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
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
