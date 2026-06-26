"""
Constrói datasets supervisionados para treinamento da RNN.

Este módulo lê arquivos CSV de métricas dos experimentos e transforma as linhas
em sequências temporais. Cada amostra contém uma janela de estados recentes como
entrada e a vazão futura observada dos servidores A e B como alvo.

A RNN é treinada para prever:

    [throughput_A_kbps, throughput_B_kbps]

a partir de uma sequência de features contendo métricas dos servidores, estado
do buffer e informações do último download.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RnnSample:
    """Representa uma amostra supervisionada para treinamento da RNN."""

    x: list[list[float]]
    y: list[float]


@dataclass(frozen=True)
class FeatureNormalizer:
    """
    Normaliza features usando média e desvio padrão.

    Attributes:
        mean: Média de cada feature.
        std: Desvio padrão de cada feature. Valores zero são substituídos por 1.
    """

    mean: list[float]
    std: list[float]

    def normalize_sequence(self, sequence: list[list[float]]) -> list[list[float]]:
        """
        Normaliza uma sequência temporal.

        Args:
            sequence: Sequência no formato `(sequence_length, feature_size)`.

        Returns:
            Sequência normalizada no mesmo formato.
        """
        normalized: list[list[float]] = []

        for timestep in sequence:
            normalized.append(
                [
                    (value - self.mean[index]) / self.std[index]
                    for index, value in enumerate(timestep)
                ]
            )

        return normalized

    def normalize_target(self, target: list[float]) -> list[float]:
        """
        Normaliza o alvo de vazão.

        Esta implementação não normaliza o alvo. O modelo aprende diretamente
        valores em kbps. Para o tamanho deste projeto, isso é suficiente.

        Args:
            target: Lista `[throughput_A, throughput_B]`.

        Returns:
            O próprio alvo sem normalização.
        """
        return target


class StreamingRnnDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset PyTorch para treinamento da RNN de streaming.

    Cada item retornado possui:

        x: Tensor com shape `(sequence_length, feature_size)`.
        y: Tensor com shape `(2,)`, representando a vazão futura dos servidores
           A e B.
    """

    def __init__(
        self,
        samples: list[RnnSample],
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        """
        Inicializa o dataset.

        Args:
            samples: Lista de amostras supervisionadas.
            normalizer: Normalizador de features. Se fornecido, normaliza `x`.
        """
        self.samples: list[RnnSample] = samples
        self.normalizer: FeatureNormalizer | None = normalizer

    def __len__(self) -> int:
        """
        Retorna a quantidade de amostras do dataset.

        Returns:
            Número de amostras.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna uma amostra do dataset.

        Args:
            index: Índice da amostra.

        Returns:
            Tupla `(x, y)`, em que `x` é uma sequência temporal e `y` é o alvo.
        """
        sample: RnnSample = self.samples[index]

        x_values: list[list[float]]

        if self.normalizer is None:
            x_values = sample.x
        else:
            x_values = self.normalizer.normalize_sequence(sample.x)

        y_values: list[float]

        if self.normalizer is None:
            y_values = sample.y
        else:
            y_values = self.normalizer.normalize_target(sample.y)

        x_tensor: torch.Tensor = torch.tensor(x_values, dtype=torch.float32)
        y_tensor: torch.Tensor = torch.tensor(y_values, dtype=torch.float32)

        return x_tensor, y_tensor


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """
    Lê um CSV de métricas.

    Args:
        csv_path: Caminho do arquivo CSV.

    Returns:
        Lista de linhas do CSV.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo estiver vazio.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: list[dict[str, str]] = list(reader)

    if not rows:
        raise ValueError(f"CSV sem dados: {csv_path}")

    return rows


def get_float(
    row: dict[str, str],
    column: str,
    default: float = 0.0,
) -> float:
    """
    Obtém uma coluna como float.

    Args:
        row: Linha do CSV.
        column: Nome da coluna.
        default: Valor usado se a coluna estiver ausente ou vazia.

    Returns:
        Valor convertido para float.
    """
    value: str | None = row.get(column)

    if value is None or value == "":
        return default

    return float(value)


def get_int(
    row: dict[str, str],
    column: str,
    default: int = 0,
) -> int:
    """
    Obtém uma coluna como int.

    Args:
        row: Linha do CSV.
        column: Nome da coluna.
        default: Valor usado se a coluna estiver ausente ou vazia.

    Returns:
        Valor convertido para int.
    """
    value: str | None = row.get(column)

    if value is None or value == "":
        return default

    return int(float(value))


def validate_required_columns(
    rows: list[dict[str, str]],
    required_columns: set[str],
) -> None:
    """
    Valida se o CSV possui as colunas necessárias.

    Args:
        rows: Linhas do CSV.
        required_columns: Conjunto de colunas obrigatórias.

    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente.
    """
    if not rows:
        raise ValueError("Não há linhas para validar.")

    available_columns: set[str] = set(rows[0].keys())
    missing_columns: set[str] = required_columns - available_columns

    if missing_columns:
        missing: str = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")


def server_id_to_index(
    server_id: str,
    server_a_id: str,
    server_b_id: str,
) -> int:
    """
    Converte o identificador do servidor em índice numérico.

    Args:
        server_id: Identificador lido do CSV.
        server_a_id: Identificador do servidor A.
        server_b_id: Identificador do servidor B.

    Returns:
        0 para servidor A, 1 para servidor B e -1 para desconhecido.
    """
    if server_id == server_a_id:
        return 0

    if server_id == server_b_id:
        return 1

    return -1


def row_to_feature_vector(
    row: dict[str, str],
    server_a_id: str,
    server_b_id: str,
    startup_segments: int,
) -> list[float]:
    """
    Converte uma linha do CSV em vetor de features.

    A ordem das features deve ser a mesma usada em `FeatureHistory` durante a
    execução da política RNN.

    Args:
        row: Linha do CSV.
        server_a_id: Identificador do servidor A.
        server_b_id: Identificador do servidor B.

    Returns:
        Vetor de features com tamanho 16.
    """
    throughput_a: float = get_float(row, "probe_a_throughput_kbps")
    latency_a: float = get_float(row, "probe_a_latency_ms")
    jitter_a: float = get_float(row, "probe_a_jitter_ms")
    success_a: float = float(get_int(row, "probe_a_ok"))

    throughput_b: float = get_float(row, "probe_b_throughput_kbps")
    latency_b: float = get_float(row, "probe_b_latency_ms")
    jitter_b: float = get_float(row, "probe_b_jitter_ms")
    success_b: float = float(get_int(row, "probe_b_ok"))

    diff_throughput: float = throughput_a - throughput_b
    diff_latency: float = latency_a - latency_b

    buffer_level_s: float = get_float(row, "buffer_level_s")
    last_bitrate_kbps: float = get_float(row, "bitrate_kbps")
    last_download_time_s: float = get_float(row, "download_time_s")
    last_rebuffer_event: float = float(get_int(row, "rebuffer_event"))

    last_server_index: float = float(
        server_id_to_index(
            server_id=row.get("server_id", ""),
            server_a_id=server_a_id,
            server_b_id=server_b_id,
        )
    )
    startup_phase: float = get_float(
        row,
        "startup_phase",
        default=float(get_int(row, "segment") <= startup_segments),
    )

    return [
        throughput_a,
        latency_a,
        jitter_a,
        success_a,
        throughput_b,
        latency_b,
        jitter_b,
        success_b,
        diff_throughput,
        diff_latency,
        buffer_level_s,
        last_bitrate_kbps,
        last_download_time_s,
        last_rebuffer_event,
        last_server_index,
        startup_phase,
    ]


def row_to_target(row: dict[str, str]) -> list[float]:
    """
    Converte uma linha do CSV no alvo supervisionado.

    O alvo é a vazão observada dos dois servidores no instante futuro.

    Args:
        row: Linha do CSV.

    Returns:
        Lista `[throughput_A, throughput_B]`.
    """
    return [
        get_float(row, "probe_a_throughput_kbps"),
        get_float(row, "probe_b_throughput_kbps"),
    ]


def split_rows_by_segment_runs(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """
    Divide linhas anexadas em execuções independentes pelo reinício do segmento.

    Quando `scripts/run_collect_training_data.py` adiciona novas coletas ao mesmo
    CSV, o campo `segment` volta a começar em 1. Essa quebra impede que uma
    janela temporal da RNN misture o fim de uma coleta com o início da próxima.

    Args:
        rows: Linhas do CSV na ordem original.

    Returns:
        Lista de execuções contíguas.
    """
    if not rows or "segment" not in rows[0]:
        return [rows]

    runs: list[list[dict[str, str]]] = []
    current_run: list[dict[str, str]] = []
    previous_segment: int | None = None

    for row in rows:
        segment = get_int(row, "segment")
        if (
            previous_segment is not None
            and current_run
            and segment <= previous_segment
        ):
            runs.append(current_run)
            current_run = []

        current_run.append(row)
        previous_segment = segment

    if current_run:
        runs.append(current_run)

    return runs


def build_samples_from_rows(
    rows: list[dict[str, str]],
    sequence_length: int,
    server_a_id: str,
    server_b_id: str,
) -> list[RnnSample]:
    """
    Cria amostras supervisionadas a partir das linhas do CSV.

    Para cada índice `t`, usa as `sequence_length` linhas anteriores como
    entrada e a linha `t` como alvo.

    Args:
        rows: Linhas do CSV.
        sequence_length: Tamanho da janela temporal de entrada.
        server_a_id: Identificador do servidor A.
        server_b_id: Identificador do servidor B.

    Returns:
        Lista de amostras `RnnSample`.

    Raises:
        ValueError: Se não houver linhas suficientes.
    """
    runs: list[list[dict[str, str]]] = split_rows_by_segment_runs(rows)
    if all(len(run_rows) <= sequence_length for run_rows in runs):
        raise ValueError(
            "CSV não possui linhas suficientes para formar sequências. "
            f"Linhas={len(rows)}, sequence_length={sequence_length}"
        )

    samples: list[RnnSample] = []

    for run_rows in runs:
        if len(run_rows) <= sequence_length:
            continue

        feature_rows: list[list[float]] = [
            row_to_feature_vector(
                row=row,
                server_a_id=server_a_id,
                server_b_id=server_b_id,
                startup_segments=sequence_length,
            )
            for row in run_rows
        ]

        targets: list[list[float]] = [row_to_target(row) for row in run_rows]

        for target_index in range(sequence_length, len(run_rows)):
            start_index: int = target_index - sequence_length
            end_index: int = target_index

            x_sequence: list[list[float]] = feature_rows[start_index:end_index]
            y_target: list[float] = targets[target_index]

            samples.append(
                RnnSample(
                    x=x_sequence,
                    y=y_target,
                )
            )

    return samples


def compute_feature_normalizer(samples: list[RnnSample]) -> FeatureNormalizer:
    """
    Calcula média e desvio padrão das features do dataset.

    Args:
        samples: Amostras supervisionadas.

    Returns:
        Normalizador contendo média e desvio padrão de cada feature.

    Raises:
        ValueError: Se a lista de amostras estiver vazia.
    """
    if not samples:
        raise ValueError("Não é possível normalizar dataset sem amostras.")

    feature_size: int = len(samples[0].x[0])

    flattened: list[list[float]] = []

    for sample in samples:
        flattened.extend(sample.x)

    mean: list[float] = []
    std: list[float] = []

    for feature_index in range(feature_size):
        values: list[float] = [
            vector[feature_index]
            for vector in flattened
        ]

        feature_mean: float = sum(values) / len(values)

        variance: float = sum(
            (value - feature_mean) ** 2
            for value in values
        ) / len(values)

        feature_std: float = variance ** 0.5

        if feature_std == 0.0:
            feature_std = 1.0

        mean.append(feature_mean)
        std.append(feature_std)

    return FeatureNormalizer(
        mean=mean,
        std=std,
    )


def load_rnn_samples_from_csv(
    csv_path: Path,
    sequence_length: int,
    server_a_id: str,
    server_b_id: str,
) -> list[RnnSample]:
    """
    Carrega amostras supervisionadas a partir de um CSV.

    Args:
        csv_path: Caminho do CSV de métricas.
        sequence_length: Tamanho da janela temporal da RNN.
        server_a_id: Identificador do servidor A.
        server_b_id: Identificador do servidor B.

    Returns:
        Lista de amostras supervisionadas.
    """
    rows: list[dict[str, str]] = read_csv_rows(csv_path)

    required_columns: set[str] = {
        "server_id",
        "bitrate_kbps",
        "download_time_s",
        "buffer_level_s",
        "rebuffer_event",
        "probe_a_throughput_kbps",
        "probe_a_latency_ms",
        "probe_a_jitter_ms",
        "probe_a_ok",
        "probe_b_throughput_kbps",
        "probe_b_latency_ms",
        "probe_b_jitter_ms",
        "probe_b_ok",
    }

    validate_required_columns(
        rows=rows,
        required_columns=required_columns,
    )

    return build_samples_from_rows(
        rows=rows,
        sequence_length=sequence_length,
        server_a_id=server_a_id,
        server_b_id=server_b_id,
    )


def split_samples(
    samples: list[RnnSample],
    train_ratio: float,
) -> tuple[list[RnnSample], list[RnnSample]]:
    """
    Divide as amostras em treino e validação, preservando a ordem temporal.

    Args:
        samples: Lista de amostras.
        train_ratio: Fração das amostras usada para treino.

    Returns:
        Tupla `(train_samples, val_samples)`.

    Raises:
        ValueError: Se a proporção for inválida.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio deve estar entre 0 e 1.")

    split_index: int = int(len(samples) * train_ratio)

    train_samples: list[RnnSample] = samples[:split_index]
    val_samples: list[RnnSample] = samples[split_index:]

    if not train_samples or not val_samples:
        raise ValueError(
            "Divisão inválida: treino ou validação ficaram vazios. "
            "Use mais dados ou ajuste train_ratio."
        )

    return train_samples, val_samples
