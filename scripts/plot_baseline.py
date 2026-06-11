"""
Script de linha de comando para gerar o gráfico da política baseline.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plots import generate_throughput_quality_plot


DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "metricas_baseline.csv"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "plots" / "grafico_vazao_qualidade.png"
)


@dataclass(frozen=True)
class CliArgs:
    """Argumentos normalizados para a geração do gráfico baseline."""

    csv_path: Path
    """Caminho do CSV de métricas que será lido."""

    output_path: Path
    """Caminho onde o arquivo PNG será gravado."""


def parse_args() -> CliArgs:
    """Lê os argumentos de linha de comando e aplica caminhos padrão."""

    parser = argparse.ArgumentParser(
        description="Gera grafico de vazao e qualidade ao longo dos segmentos."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Caminho do CSV de metricas. Padrao: {DEFAULT_CSV_PATH}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Caminho do PNG gerado. Padrao: {DEFAULT_OUTPUT_PATH}",
    )

    namespace = parser.parse_args()

    return CliArgs(
        csv_path=namespace.csv,
        output_path=namespace.output,
    )


def main() -> None:
    """Executa o fluxo de geração do gráfico a partir da CLI."""

    args = parse_args()

    generate_throughput_quality_plot(
        csv_path=args.csv_path,
        output_path=args.output_path,
    )

    print(f"Grafico gerado: {args.output_path}")


if __name__ == "__main__":
    main()
