"""Gera todos os gráficos individuais e comparativos da entrega final."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.final_plots import generate_final_plots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera os gráficos finais a partir dos CSVs das três políticas."
    )
    output_dir = PROJECT_ROOT / "outputs"
    parser.add_argument(
        "--policy1",
        type=Path,
        default=output_dir / "metricas_policy1.csv",
    )
    parser.add_argument(
        "--policy2",
        type=Path,
        default=output_dir / "metricas_policy2.csv",
    )
    parser.add_argument(
        "--policy3",
        type=Path,
        default=output_dir / "metricas_policy3_rnn.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_csvs = [
        ("policy1", "Baseline", args.policy1),
        ("policy2", "Política 2", args.policy2),
        ("policy3", "Política 3", args.policy3),
    ]
    generated = generate_final_plots(policy_csvs, args.output_dir)
    print(f"{len(generated)} gráficos gerados em: {args.output_dir}")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
