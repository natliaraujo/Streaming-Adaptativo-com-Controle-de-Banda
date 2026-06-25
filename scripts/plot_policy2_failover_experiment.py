"""Gera o gráfico consolidado do experimento de failover da Política 2."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.final_plots import generate_policy2_failover_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plota vazão, qualidade, buffer e servidor durante o failover."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "policy2_failover_experiment.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "figures"
            / "policy2_failover_experiment.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_policy2_failover_plot(args.csv, args.output)
    print(f"Gráfico de failover gerado em: {args.output}")


if __name__ == "__main__":
    main()
