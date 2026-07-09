"""
Constrói datasets supervisionados para treinamento da RNN.

Este módulo lê arquivos CSV de métricas dos experimentos e transforma as linhas
em sequências temporais. Cada amostra contém uma janela de estados recentes como
entrada e três alvos: os probes futuros dos servidores A/B e a vazão real
futura do download.

A RNN é treinada para prever:

    [probe_A_kbps, probe_B_kbps, download_throughput_kbps]

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
    Normaliza features e alvos usando média e desvio padrão.

    Attributes:
        mean: Média de cada feature.
        std: Desvio padrão de cada feature. Valores zero são substituídos por 1.
        target_mean: Média de cada alvo previsto.
        target_std: Desvio padrão de cada alvo previsto.
        target_clip_min: Menor valor observado por alvo.
        target_clip_max: Percentil alto usado como teto de previsão por alvo.
    """

    mean: list[float]
    std: list[float]
    target_mean: list[float] | None = None
    target_std: list[float] | None = None
    target_clip_min: list[float] | None = None
    target_clip_max: list[float] | None = None

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

        Quando estatísticas de alvo estão disponíveis, cada saída é colocada na
        sua própria escala normalizada. Isso evita que a vazão real do download
        domine a perda por estar em escala diferente dos probes.

        Args:
            target: Lista `[probe_A, probe_B, download_throughput]`.

        Returns:
            Alvo normalizado, ou o próprio alvo quando não há estatísticas.
        """
        if self.target_mean is None or self.target_std is None:
            return target

        return [
            (value - self.target_mean[index]) / self.target_std[index]
            for index, value in enumerate(target)
        ]

    def denormalize_target(
        self,
        target: list[float],
        clamp: bool = True,
    ) -> list[float]:
        """
        Retorna previsões normalizadas para a escala original em kbps.

        Args:
            target: Lista de saídas previstas pelo modelo.
            clamp: Quando verdadeiro, limita previsões ao intervalo observado
                no treino para reduzir picos fora da distribuição.

        Returns:
            Lista na escala original, ou a própria lista se não houver
            estatísticas de alvo no checkpoint.
        """
        if self.target_mean is None or self.target_std is None:
            return target

        denormalized = [
            value * self.target_std[index] + self.target_mean[index]
            for index, value in enumerate(target)
        ]

        if not clamp or self.target_clip_min is None or self.target_clip_max is None:
            return denormalized

        return [
            min(
                max(value, self.target_clip_min[index]),
                self.target_clip_max[index],
            )
            for index, value in enumerate(denormalized)
        ]


class StreamingRnnDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset PyTorch para treinamento da RNN de streaming.

    Cada item retornado possui:

        x: Tensor com shape `(sequence_length, feature_size)`.
        y: Tensor com shape `(3,)`, representando os probes futuros dos
           servidores A/B e a vazão real do próximo download.
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

    O alvo combina os probes futuros dos dois servidores, usados para ranquear
    A/B, e a vazão real futura do segmento baixado, usada para escolher
    qualidade.

    Args:
        row: Linha do CSV.

    Returns:
        Lista `[probe_A, probe_B, download_throughput]`.
    """
    return [
        get_float(row, "probe_a_throughput_kbps"),
        get_float(row, "probe_b_throughput_kbps"),
        get_float(row, "throughput_kbps"),
    ]


def median(values: list[float]) -> float:
    """Calcula a mediana de uma lista não vazia."""
    if not values:
        raise ValueError("Não é possível calcular mediana sem valores.")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def rows_to_target(
    rows: list[dict[str, str]],
    target_index: int,
    probe_target_smoothing_window: int,
) -> list[float]:
    """
    Converte linhas futuras no alvo supervisionado.

    Os probes A/B usam mediana em uma janela curta para reduzir rótulos de
    ranking causados por ruído instantâneo. A vazão de download permanece a do
    próximo segmento, pois é a escala usada na decisão imediata de qualidade.
    """
    if probe_target_smoothing_window < 1:
        raise ValueError("probe_target_smoothing_window deve ser positivo.")

    end_index = min(
        len(rows),
        target_index + probe_target_smoothing_window,
    )
    probe_rows = rows[target_index:end_index]
    target_row = rows[target_index]

    return [
        median([
            get_float(row, "probe_a_throughput_kbps")
            for row in probe_rows
        ]),
        median([
            get_float(row, "probe_b_throughput_kbps")
            for row in probe_rows
        ]),
        get_float(target_row, "throughput_kbps"),
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
    probe_target_smoothing_window: int = 1,
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
        probe_target_smoothing_window: Janela de mediana dos probes usados como
            alvo de ranking.

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

        for target_index in range(sequence_length, len(run_rows)):
            start_index: int = target_index - sequence_length
            end_index: int = target_index

            x_sequence: list[list[float]] = feature_rows[start_index:end_index]
            y_target: list[float] = rows_to_target(
                rows=run_rows,
                target_index=target_index,
                probe_target_smoothing_window=probe_target_smoothing_window,
            )

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

    flattened: list[list[float]] = []

    for sample in samples:
        flattened.extend(sample.x)

    mean, std = compute_column_mean_std(flattened)
    target_mean, target_std = compute_column_mean_std(
        [sample.y for sample in samples]
    )
    target_clip_min, target_clip_max = compute_target_clipping_bounds(
        [sample.y for sample in samples]
    )

    return FeatureNormalizer(
        mean=mean,
        std=std,
        target_mean=target_mean,
        target_std=target_std,
        target_clip_min=target_clip_min,
        target_clip_max=target_clip_max,
    )


def compute_column_mean_std(
    values_by_row: list[list[float]],
) -> tuple[list[float], list[float]]:
    """
    Calcula média e desvio padrão por coluna.

    Args:
        values_by_row: Matriz não vazia no formato `(linhas, colunas)`.

    Returns:
        Tupla com listas de médias e desvios. Desvios zero são substituídos por
        1 para evitar divisão por zero na normalização.
    """
    if not values_by_row:
        raise ValueError("Não é possível calcular estatísticas sem valores.")

    column_count: int = len(values_by_row[0])
    mean: list[float] = []
    std: list[float] = []

    for column_index in range(column_count):
        values: list[float] = [
            row[column_index]
            for row in values_by_row
        ]

        column_mean: float = sum(values) / len(values)

        variance: float = sum(
            (value - column_mean) ** 2
            for value in values
        ) / len(values)

        column_std: float = variance ** 0.5

        if column_std == 0.0:
            column_std = 1.0

        mean.append(column_mean)
        std.append(column_std)

    return mean, std


def compute_target_clipping_bounds(
    targets: list[list[float]],
    upper_quantile: float = 0.99,
) -> tuple[list[float], list[float]]:
    """
    Calcula limites de previsão por alvo.

    O limite inferior usa o mínimo observado para preservar casos de falha. O
    limite superior usa um percentil alto para evitar picos artificiais fora da
    distribuição de treino.
    """
    if not targets:
        raise ValueError("Não é possível calcular clipping sem alvos.")
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile deve estar no intervalo (0, 1].")

    target_size = len(targets[0])
    lower: list[float] = []
    upper: list[float] = []
    for target_index in range(target_size):
        values = sorted(row[target_index] for row in targets)
        lower.append(values[0])
        quantile_index = int((len(values) - 1) * upper_quantile)
        upper.append(values[quantile_index])

    return lower, upper


def load_rnn_samples_from_csv(
    csv_path: Path,
    sequence_length: int,
    server_a_id: str,
    server_b_id: str,
    probe_target_smoothing_window: int = 1,
) -> list[RnnSample]:
    """
    Carrega amostras supervisionadas a partir de um CSV.

    Args:
        csv_path: Caminho do CSV de métricas.
        sequence_length: Tamanho da janela temporal da RNN.
        server_a_id: Identificador do servidor A.
        server_b_id: Identificador do servidor B.
        probe_target_smoothing_window: Janela de mediana para probes-alvo.

    Returns:
        Lista de amostras supervisionadas.
    """
    rows: list[dict[str, str]] = read_csv_rows(csv_path)

    required_columns: set[str] = {
        "server_id",
        "bitrate_kbps",
        "throughput_kbps",
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
        probe_target_smoothing_window=probe_target_smoothing_window,
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
