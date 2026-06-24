"""
Carrega checkpoints de modelos RNN treinados.

Este módulo concentra a lógica de leitura dos pesos da rede neural e dos
metadados necessários para inferência, como normalização das features,
tamanho da sequência temporal e identificadores dos servidores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.dataset import FeatureNormalizer
from models.rnn import StreamingRNN


@dataclass(frozen=True)
class LoadedRnnModel:
    """Agrupa o modelo RNN carregado e seus metadados de treinamento."""

    model: StreamingRNN
    normalizer: FeatureNormalizer
    sequence_length: int
    feature_size: int
    hidden_size: int
    server_a_id: str
    server_b_id: str


def load_rnn_checkpoint(checkpoint_path: Path) -> LoadedRnnModel:
    """
    Carrega a RNN pré-treinada a partir de um checkpoint.

    Args:
        checkpoint_path: Caminho do arquivo `.pt` salvo por `models.train`.

    Returns:
        Objeto contendo modelo carregado, normalizador e metadados.

    Raises:
        FileNotFoundError: Se o checkpoint não existir.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            "Treine o modelo antes com `python -m models.train`."
        )

    if checkpoint_path.stat().st_size == 0:
        raise ValueError(
            f"Checkpoint vazio: {checkpoint_path}. "
            "Treine o modelo novamente com `python -m models.train`."
        )

    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
    except (EOFError, RuntimeError) as exc:
        raise ValueError(
            f"Checkpoint inválido ou corrompido: {checkpoint_path}. "
            "Treine o modelo novamente com `python -m models.train`."
        ) from exc

    feature_size: int = int(checkpoint["feature_size"])
    hidden_size: int = int(checkpoint["hidden_size"])

    model = StreamingRNN(
        input_size=feature_size,
        hidden_size=hidden_size,
        output_size=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = FeatureNormalizer(
        mean=list(checkpoint["normalizer_mean"]),
        std=list(checkpoint["normalizer_std"]),
    )

    return LoadedRnnModel(
        model=model,
        normalizer=normalizer,
        sequence_length=int(checkpoint["sequence_length"]),
        feature_size=feature_size,
        hidden_size=hidden_size,
        server_a_id=str(checkpoint["server_a_id"]),
        server_b_id=str(checkpoint["server_b_id"]),
    )
