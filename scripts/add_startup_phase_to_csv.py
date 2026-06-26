"""Adiciona a coluna startup_phase a um CSV de métricas existente."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RNN_STARTUP_SEGMENTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adiciona startup_phase a um CSV antigo, derivando a fase inicial "
            "pelo número do segmento."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rnn_training_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destino opcional. Se omitido, sobrescreve o CSV de entrada.",
    )
    parser.add_argument(
        "--startup-segments",
        type=int,
        default=RNN_STARTUP_SEGMENTS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV não encontrado: {args.csv}")

    with args.csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "segment" not in fieldnames:
        raise ValueError("CSV precisa da coluna segment para derivar startup_phase.")

    if "startup_phase" not in fieldnames:
        insert_at = fieldnames.index("timestamp") + 1 if "timestamp" in fieldnames else 1
        fieldnames.insert(insert_at, "startup_phase")

    for row in rows:
        segment = int(float(row["segment"]))
        row["startup_phase"] = str(int(segment <= args.startup_segments))

    output_path = args.output or args.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV atualizado em: {output_path}")


if __name__ == "__main__":
    main()
