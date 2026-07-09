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

from domain.manifest import normalize_server_id
from models.dataset import FeatureNormalizer
from models.rnn import StreamingRNN


@dataclass(frozen=True)
class LoadedRnnModel:
    """Agrupa o modelo RNN carregado e seus metadados de treinamento."""

    model: StreamingRNN
    normalizer: FeatureNormalizer
    sequence_length: int
    probe_target_smoothing_window: int
    feature_size: int
    output_size: int
    hidden_size: int
    dropout: float
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
    output_size: int = int(checkpoint.get("output_size", 2))
    hidden_size: int = int(checkpoint["hidden_size"])
    dropout: float = float(checkpoint.get("dropout", 0.0))

    model = StreamingRNN(
        input_size=feature_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = FeatureNormalizer(
        mean=list(checkpoint["normalizer_mean"]),
        std=list(checkpoint["normalizer_std"]),
        target_mean=(
            None
            if checkpoint.get("target_mean") is None
            else list(checkpoint["target_mean"])
        ),
        target_std=(
            None
            if checkpoint.get("target_std") is None
            else list(checkpoint["target_std"])
        ),
        target_clip_min=(
            None
            if checkpoint.get("target_clip_min") is None
            else list(checkpoint["target_clip_min"])
        ),
        target_clip_max=(
            None
            if checkpoint.get("target_clip_max") is None
            else list(checkpoint["target_clip_max"])
        ),
    )

    return LoadedRnnModel(
        model=model,
        normalizer=normalizer,
        sequence_length=int(checkpoint["sequence_length"]),
        probe_target_smoothing_window=int(
            checkpoint.get("probe_target_smoothing_window", 1)
        ),
        feature_size=feature_size,
        output_size=output_size,
        hidden_size=hidden_size,
        dropout=dropout,
        server_a_id=normalize_server_id(str(checkpoint["server_a_id"])),
        server_b_id=normalize_server_id(str(checkpoint["server_b_id"])),
    )
