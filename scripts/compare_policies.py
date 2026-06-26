"""
Gera comparações entre as políticas implementadas.

Este script lê os CSVs produzidos pelas políticas 1, 2 e 3, calcula métricas
agregadas e imprime uma tabela comparativa. A geração de gráficos fica
centralizada em `scripts/generate_plots.py`.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


DEFAULT_POLICY1_CSV: Path = PROJECT_ROOT / "outputs" / "metricas_policy1.csv"
DEFAULT_POLICY2_CSV: Path = PROJECT_ROOT / "outputs" / "metricas_policy2.csv"
DEFAULT_POLICY3_CSV: Path = PROJECT_ROOT / "outputs" / "metricas_policy3_rnn.csv"


@dataclass(frozen=True)
class PolicyCsv:
    """Representa o CSV de métricas associado a uma política."""

    name: str
    path: Path


@dataclass(frozen=True)
class PolicySummary:
    """Resumo estatístico de uma política."""

    name: str
    segments: int
    avg_bitrate_kbps: float
    avg_throughput_kbps: float
    avg_buffer_level_s: float
    total_stall_s: float
    rebuffer_events: int
    quality_switches: int
    servers_used: int


@dataclass(frozen=True)
class CliArgs:
    """Argumentos de linha de comando do comparador."""

    policy_csvs: list[PolicyCsv]


def parse_args() -> CliArgs:
    """
    Lê os argumentos de linha de comando.

    Returns:
        Objeto contendo os caminhos dos CSVs das políticas.
    """
    parser = argparse.ArgumentParser(
        description="Compara os CSVs de métricas das políticas 1, 2 e 3.",
    )

    parser.add_argument(
        "--policy1",
        type=Path,
        default=DEFAULT_POLICY1_CSV,
        help=f"CSV da política 1. Padrão: {DEFAULT_POLICY1_CSV}",
    )

    parser.add_argument(
        "--policy2",
        type=Path,
        default=DEFAULT_POLICY2_CSV,
        help=f"CSV da política 2. Padrão: {DEFAULT_POLICY2_CSV}",
    )

    parser.add_argument(
        "--policy3",
        type=Path,
        default=DEFAULT_POLICY3_CSV,
        help=f"CSV da política 3. Padrão: {DEFAULT_POLICY3_CSV}",
    )

    namespace = parser.parse_args()

    return CliArgs(
        policy_csvs=[
            PolicyCsv("policy1", namespace.policy1),
            PolicyCsv("policy2", namespace.policy2),
            PolicyCsv("policy3_rnn", namespace.policy3),
        ],
    )


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """
    Lê as linhas de um CSV de métricas.

    Args:
        csv_path: Caminho do arquivo CSV.

    Returns:
        Lista de linhas representadas como dicionários.

    Raises:
        FileNotFoundError: Se o CSV não existir.
        ValueError: Se o CSV estiver vazio.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: list[dict[str, str]] = list(reader)

    if not rows:
        raise ValueError(f"CSV sem dados: {csv_path}")

    return rows


def to_float(row: dict[str, str], column: str, default: float = 0.0) -> float:
    """
    Converte uma coluna de uma linha para float.

    Args:
        row: Linha do CSV.
        column: Nome da coluna.
        default: Valor usado quando a coluna estiver ausente ou vazia.

    Returns:
        Valor convertido para float.
    """
    value: str | None = row.get(column)

    if value is None or value == "":
        return default

    return float(value)


def to_int(row: dict[str, str], column: str, default: int = 0) -> int:
    """
    Converte uma coluna de uma linha para int.

    Args:
        row: Linha do CSV.
        column: Nome da coluna.
        default: Valor usado quando a coluna estiver ausente ou vazia.

    Returns:
        Valor convertido para int.
    """
    value: str | None = row.get(column)

    if value is None or value == "":
        return default

    return int(float(value))


def count_quality_switches(rows: list[dict[str, str]]) -> int:
    """
    Conta quantas vezes a qualidade escolhida mudou ao longo do experimento.

    Args:
        rows: Linhas do CSV de métricas.

    Returns:
        Número de trocas de qualidade.
    """
    switches: int = 0
    previous_quality: str | None = None

    for row in rows:
        current_quality: str = row.get("quality", "")

        if previous_quality is not None and current_quality != previous_quality:
            switches += 1

        previous_quality = current_quality

    return switches


def summarize_policy(policy_csv: PolicyCsv) -> PolicySummary:
    """
    Calcula métricas agregadas de uma política.

    Args:
        policy_csv: Nome e caminho do CSV da política.

    Returns:
        Resumo estatístico da política.
    """
    rows: list[dict[str, str]] = read_csv_rows(policy_csv.path)

    segments: int = len(rows)

    avg_bitrate_kbps: float = sum(
        to_float(row, "bitrate_kbps") for row in rows
    ) / segments

    avg_throughput_kbps: float = sum(
        to_float(row, "throughput_kbps", to_float(row, "vazao_kbps"))
        for row in rows
    ) / segments

    avg_buffer_level_s: float = sum(
        to_float(row, "buffer_level_s") for row in rows
    ) / segments

    total_stall_s: float = sum(
        to_float(row, "stall_duration_s") for row in rows
    )

    rebuffer_events: int = sum(
        to_int(row, "rebuffer_event") for row in rows
    )

    quality_switches: int = count_quality_switches(rows)

    servers_used: int = len(
        {row.get("server_id", "") for row in rows if row.get("server_id", "")}
    )

    return PolicySummary(
        name=policy_csv.name,
        segments=segments,
        avg_bitrate_kbps=avg_bitrate_kbps,
        avg_throughput_kbps=avg_throughput_kbps,
        avg_buffer_level_s=avg_buffer_level_s,
        total_stall_s=total_stall_s,
        rebuffer_events=rebuffer_events,
        quality_switches=quality_switches,
        servers_used=servers_used,
    )


def print_summary_table(summaries: list[PolicySummary]) -> None:
    """
    Imprime uma tabela comparativa no terminal.

    Args:
        summaries: Lista de resumos das políticas.
    """
    headers: list[str] = [
        "policy",
        "segments",
        "avg bitrate",
        "avg throughput",
        "avg buffer",
        "stall total",
        "rebuffer",
        "switches",
        "servers",
    ]

    print()
    print(
        f"{headers[0]:<14}"
        f"{headers[1]:>10}"
        f"{headers[2]:>16}"
        f"{headers[3]:>18}"
        f"{headers[4]:>14}"
        f"{headers[5]:>14}"
        f"{headers[6]:>10}"
        f"{headers[7]:>10}"
        f"{headers[8]:>10}"
    )

    print("-" * 126)

    for summary in summaries:
        print(
            f"{summary.name:<14}"
            f"{summary.segments:>10d}"
            f"{summary.avg_bitrate_kbps:>16.1f}"
            f"{summary.avg_throughput_kbps:>18.1f}"
            f"{summary.avg_buffer_level_s:>14.2f}"
            f"{summary.total_stall_s:>14.3f}"
            f"{summary.rebuffer_events:>10d}"
            f"{summary.quality_switches:>10d}"
            f"{summary.servers_used:>10d}"
        )

    print()


def main() -> None:
    """Compara as políticas e imprime a tabela de resultados."""
    args: CliArgs = parse_args()

    summaries: list[PolicySummary] = []

    for policy_csv in args.policy_csvs:
        try:
            summary: PolicySummary = summarize_policy(policy_csv)
            summaries.append(summary)
        except FileNotFoundError as exc:
            print(f"Aviso: {exc}")
        except ValueError as exc:
            print(f"Aviso: {exc}")

    if not summaries:
        raise RuntimeError("Nenhum CSV válido foi encontrado para comparação.")

    print_summary_table(summaries)


if __name__ == "__main__":
    main()
