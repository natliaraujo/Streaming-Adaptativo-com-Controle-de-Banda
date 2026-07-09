"""Monta um CSV de treino da RNN com dados base e dados on-policy."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.csv_writer import CsvMetricsWriter  # noqa: E402


@dataclass(frozen=True)
class CliArgs:
    """Argumentos para composição do dataset aumentado."""

    base: Path
    extras: list[Path]
    output: Path
    extra_weight: int


def parse_args() -> CliArgs:
    """Lê argumentos de linha de comando."""
    outputs_dir = PROJECT_ROOT / "outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Combina o CSV de treino balanceado com CSVs on-policy para "
            "retreinar a RNN no regime em que a Policy 3 realmente opera."
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=outputs_dir / "rnn_training_data.csv",
        help="CSV base de treinamento.",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="CSV adicional, como metricas_policy3_rnn.csv. Pode repetir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=outputs_dir / "rnn_training_augmented.csv",
        help="CSV aumentado gerado.",
    )
    parser.add_argument(
        "--extra-weight",
        type=int,
        default=4,
        help="Quantas vezes cada CSV extra entra no dataset.",
    )
    args = parser.parse_args()
    return CliArgs(
        base=args.base,
        extras=args.extra,
        output=args.output,
        extra_weight=args.extra_weight,
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Carrega um CSV e valida existência/dados."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"CSV sem dados: {csv_path}")
    return rows


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Escreve linhas no esquema atual do writer de métricas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = CsvMetricsWriter.HEADER
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def main() -> None:
    """Gera o dataset aumentado."""
    args = parse_args()
    if args.extra_weight < 1:
        raise ValueError("--extra-weight deve ser positivo.")

    rows = read_rows(args.base)
    base_count = len(rows)
    extra_counts: list[tuple[Path, int]] = []
    for extra_path in args.extras:
        extra_rows = read_rows(extra_path)
        extra_counts.append((extra_path, len(extra_rows)))
        for _copy_index in range(args.extra_weight):
            rows.extend(extra_rows)

    write_rows(args.output, rows)

    print(f"Dataset aumentado salvo em: {args.output}")
    print(f"Linhas base: {base_count}")
    for extra_path, count in extra_counts:
        print(
            f"Extra: {extra_path} linhas={count} "
            f"peso={args.extra_weight} contribuição={count * args.extra_weight}"
        )
    print(f"Total: {len(rows)}")


if __name__ == "__main__":
    main()
