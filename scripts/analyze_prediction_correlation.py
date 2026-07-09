"""Analisa correlacao entre previsoes de vazao e valores observados."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SAFETY_FACTOR  # noqa: E402


@dataclass(frozen=True)
class CorrelationResult:
    """Resumo estatistico de uma serie prevista contra uma serie observada."""

    name: str
    samples: int
    pearson_r: float | None
    spearman_r: float | None
    mae: float | None
    rmse: float | None
    bias: float | None
    predicted_mean: float | None
    observed_mean: float | None


@dataclass(frozen=True)
class CliArgs:
    """Argumentos da analise de correlacao."""

    policy1_csv: Path
    policy3_csv: Path
    output_csv: Path | None
    safety_factor: float


def parse_args() -> CliArgs:
    """Le os caminhos de entrada e parametros da analise."""
    outputs_dir = PROJECT_ROOT / "outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Compara a correlacao das previsoes da RNN com probes medidos "
            "e da Politica 1 com a vazao real medida."
        )
    )
    parser.add_argument(
        "--policy1",
        type=Path,
        default=outputs_dir / "metricas_policy1.csv",
        help="CSV da Politica 1.",
    )
    parser.add_argument(
        "--policy3",
        type=Path,
        default=outputs_dir / "metricas_policy3_rnn.csv",
        help="CSV da Politica 3/RNN.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=outputs_dir / "correlation_analysis.csv",
        help=(
            "CSV de saida com a tabela de correlacao. Use --output '' para "
            "nao gravar arquivo."
        ),
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=SAFETY_FACTOR,
        help=(
            "Fator usado no chute da Politica 1: EWMA anterior * fator. "
            f"Padrao: {SAFETY_FACTOR:g}."
        ),
    )
    args = parser.parse_args()
    output_csv = None if str(args.output).strip() == "" else args.output
    return CliArgs(
        policy1_csv=args.policy1,
        policy3_csv=args.policy3,
        output_csv=output_csv,
        safety_factor=args.safety_factor,
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Carrega linhas de um CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"CSV sem dados: {csv_path}")
    return rows


def optional_float(row: dict[str, str], column: str) -> float | None:
    """Converte coluna numerica vazia/ausente para None."""
    value = row.get(column)
    if value is None or value.strip() == "":
        return None
    return float(value)


def paired_values(
    predicted: list[float | None],
    observed: list[float | None],
) -> tuple[list[float], list[float]]:
    """Remove pares incompletos mantendo alinhamento."""
    clean_predicted: list[float] = []
    clean_observed: list[float] = []
    for predicted_value, observed_value in zip(predicted, observed, strict=True):
        if predicted_value is None or observed_value is None:
            continue
        clean_predicted.append(predicted_value)
        clean_observed.append(observed_value)
    return clean_predicted, clean_observed


def mean(values: list[float]) -> float | None:
    """Media simples."""
    if not values:
        return None
    return sum(values) / len(values)


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    """Calcula a correlacao de Pearson."""
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    centered_x = [value - x_mean for value in x_values]
    centered_y = [value - y_mean for value in y_values]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    denominator_x = math.sqrt(sum(value * value for value in centered_x))
    denominator_y = math.sqrt(sum(value * value for value in centered_y))
    denominator = denominator_x * denominator_y
    if denominator == 0.0:
        return None
    return numerator / denominator


def ranks(values: list[float]) -> list[float]:
    """Converte valores em ranks com media para empates."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _value in indexed[index:end]:
            result[original_index] = rank
        index = end
    return result


def spearman_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    """Calcula a correlacao de Spearman usando ranks."""
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    return pearson_correlation(ranks(x_values), ranks(y_values))


def summarize(
    name: str,
    predicted: list[float | None],
    observed: list[float | None],
) -> CorrelationResult:
    """Resume uma serie prevista contra uma serie observada."""
    predicted_values, observed_values = paired_values(predicted, observed)
    if not predicted_values:
        return CorrelationResult(
            name=name,
            samples=0,
            pearson_r=None,
            spearman_r=None,
            mae=None,
            rmse=None,
            bias=None,
            predicted_mean=None,
            observed_mean=None,
        )

    errors = [
        predicted_value - observed_value
        for predicted_value, observed_value in zip(
            predicted_values,
            observed_values,
            strict=True,
        )
    ]
    return CorrelationResult(
        name=name,
        samples=len(predicted_values),
        pearson_r=pearson_correlation(predicted_values, observed_values),
        spearman_r=spearman_correlation(predicted_values, observed_values),
        mae=sum(abs(error) for error in errors) / len(errors),
        rmse=math.sqrt(sum(error * error for error in errors) / len(errors)),
        bias=sum(errors) / len(errors),
        predicted_mean=mean(predicted_values),
        observed_mean=mean(observed_values),
    )


def rnn_results(policy3_rows: list[dict[str, str]]) -> list[CorrelationResult]:
    """Calcula correlacoes entre probes medidos e previsoes da RNN."""
    predicted_a = [
        optional_float(row, "rnn_predicted_a_throughput_kbps")
        for row in policy3_rows
    ]
    predicted_b = [
        optional_float(row, "rnn_predicted_b_throughput_kbps")
        for row in policy3_rows
    ]
    observed_a = [
        optional_float(row, "probe_a_throughput_kbps")
        for row in policy3_rows
    ]
    observed_b = [
        optional_float(row, "probe_b_throughput_kbps")
        for row in policy3_rows
    ]

    selected_predicted: list[float | None] = []
    selected_observed: list[float | None] = []
    download_predicted = [
        optional_float(row, "rnn_predicted_download_throughput_kbps")
        for row in policy3_rows
    ]
    download_observed = [
        optional_float(row, "throughput_kbps")
        for row in policy3_rows
    ]
    for row in policy3_rows:
        row_predicted_a = optional_float(row, "rnn_predicted_a_throughput_kbps")
        row_predicted_b = optional_float(row, "rnn_predicted_b_throughput_kbps")
        row_predicted_selected = optional_float(
            row,
            "rnn_predicted_selected_throughput_kbps",
        )
        if row_predicted_a is None or row_predicted_b is None:
            selected_predicted.append(None)
            selected_observed.append(None)
            continue
        selected_predicted.append(row_predicted_selected)
        if row_predicted_a >= row_predicted_b:
            selected_observed.append(optional_float(row, "probe_a_throughput_kbps"))
        else:
            selected_observed.append(optional_float(row, "probe_b_throughput_kbps"))

    return [
        summarize("RNN probe A", predicted_a, observed_a),
        summarize("RNN probe B", predicted_b, observed_b),
        summarize(
            "RNN probes A+B",
            predicted_a + predicted_b,
            observed_a + observed_b,
        ),
        summarize("RNN probe selecionado", selected_predicted, selected_observed),
        summarize("RNN vazao download", download_predicted, download_observed),
    ]


def policy1_results(
    policy1_rows: list[dict[str, str]],
    safety_factor: float,
) -> list[CorrelationResult]:
    """Calcula correlacao do chute da Politica 1 contra a vazao real."""
    predicted: list[float | None] = []
    observed: list[float | None] = []
    previous_ewma: float | None = None

    for row in policy1_rows:
        actual_throughput = optional_float(row, "throughput_kbps")
        if previous_ewma is None:
            predicted.append(None)
            observed.append(actual_throughput)
        else:
            predicted.append(previous_ewma * safety_factor)
            observed.append(actual_throughput)
        previous_ewma = optional_float(row, "throughput_ewma_kbps")

    return [
        summarize(
            "Politica 1 EWMA segura",
            predicted,
            observed,
        )
    ]


def format_float(value: float | None) -> str:
    """Formata floats para tabela de terminal."""
    if value is None:
        return "-"
    return f"{value:.4f}"


def print_results(results: list[CorrelationResult]) -> None:
    """Imprime tabela comparativa."""
    print()
    print(
        f"{'serie':<26}"
        f"{'n':>6}"
        f"{'pearson':>11}"
        f"{'spearman':>11}"
        f"{'MAE':>11}"
        f"{'RMSE':>11}"
        f"{'bias':>11}"
        f"{'media pred':>13}"
        f"{'media obs':>13}"
    )
    print("-" * 114)
    for result in results:
        print(
            f"{result.name:<26}"
            f"{result.samples:>6d}"
            f"{format_float(result.pearson_r):>11}"
            f"{format_float(result.spearman_r):>11}"
            f"{format_float(result.mae):>11}"
            f"{format_float(result.rmse):>11}"
            f"{format_float(result.bias):>11}"
            f"{format_float(result.predicted_mean):>13}"
            f"{format_float(result.observed_mean):>13}"
        )
    print()
    print(
        "Nota: na Politica 1, o chute usa throughput_ewma_kbps do segmento "
        "anterior multiplicado pelo fator de seguranca, evitando usar a EWMA "
        "ja atualizada com a vazao real do proprio segmento."
    )


def write_results(output_csv: Path, results: list[CorrelationResult]) -> None:
    """Salva os resultados em CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "series",
                "samples",
                "pearson_r",
                "spearman_r",
                "mae",
                "rmse",
                "bias",
                "predicted_mean",
                "observed_mean",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.samples,
                    "" if result.pearson_r is None else result.pearson_r,
                    "" if result.spearman_r is None else result.spearman_r,
                    "" if result.mae is None else result.mae,
                    "" if result.rmse is None else result.rmse,
                    "" if result.bias is None else result.bias,
                    "" if result.predicted_mean is None else result.predicted_mean,
                    "" if result.observed_mean is None else result.observed_mean,
                ]
            )


def main() -> None:
    """Executa a analise."""
    args = parse_args()
    policy1_rows = read_rows(args.policy1_csv)
    policy3_rows = read_rows(args.policy3_csv)
    results = [
        *rnn_results(policy3_rows),
        *policy1_results(policy1_rows, args.safety_factor),
    ]
    print_results(results)
    if args.output_csv is not None:
        write_results(args.output_csv, results)
        print(f"CSV salvo em: {args.output_csv}")


if __name__ == "__main__":
    main()
