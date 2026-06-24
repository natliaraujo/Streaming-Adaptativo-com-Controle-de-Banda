"""
Gera gráficos para o experimento baseline.

Este script é uma interface de linha de comando para as funções de análise. Ele
lê o CSV da política 1 e produz gráficos, como a relação entre vazão medida e
qualidade escolhida por segmento.

A lógica de plotagem reutilizável fica no pacote `analysis`.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plots import generate_throughput_quality_plot  # noqa: E402


DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "outputs" / "metricas_policy1.csv"
DEFAULT_OUTPUT_PATH: Path = (
    PROJECT_ROOT / "outputs" / "plots" / "grafico_policy1_vazao_qualidade.png"
)


@dataclass(frozen=True)
class CliArgs:
    """Argumentos de linha de comando do script de plotagem."""

    csv_path: Path
    output_path: Path


def parse_args() -> CliArgs:
    """
    Lê os argumentos de linha de comando.

    Returns:
        Objeto contendo o caminho do CSV de entrada e o caminho do PNG de saída.
    """
    parser = argparse.ArgumentParser(
        description="Gera gráfico de vazão e qualidade ao longo dos segmentos.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Caminho do CSV de métricas. Padrão: {DEFAULT_CSV_PATH}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Caminho do PNG gerado. Padrão: {DEFAULT_OUTPUT_PATH}",
    )

    namespace = parser.parse_args()

    return CliArgs(
        csv_path=namespace.csv,
        output_path=namespace.output,
    )


def main() -> None:
    """Gera o gráfico de vazão e qualidade."""
    args = parse_args()

    generate_throughput_quality_plot(
        csv_path=args.csv_path,
        output_path=args.output_path,
    )

    print(f"Gráfico gerado em: {args.output_path}")


if __name__ == "__main__":
    main()