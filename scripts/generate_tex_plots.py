"""Gera todos os graficos do projeto como fragmentos PGFPlots/TikZ."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.tex_plots import generate_all_tex_plots  # noqa: E402


@dataclass(frozen=True)
class CliArgs:
    """Argumentos da geracao consolidada de graficos TeX."""

    policy1_csv: Path
    policy2_csv: Path
    policy3_csv: Path
    policy2_failover_csv: Path
    output_dir: Path


def parse_args() -> CliArgs:
    """Le caminhos de entrada e diretorio de saida."""
    outputs_dir = PROJECT_ROOT / "outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Gera os graficos finais em arquivos .tex separados. "
            "Cada arquivo contem um unico tikzpicture/axis para uso com PGFPlots."
        )
    )
    parser.add_argument(
        "--policy1",
        type=Path,
        default=outputs_dir / "metricas_policy1.csv",
        help="CSV da Politica 1.",
    )
    parser.add_argument(
        "--policy2",
        type=Path,
        default=outputs_dir / "metricas_policy2.csv",
        help="CSV da Politica 2.",
    )
    parser.add_argument(
        "--policy3",
        type=Path,
        default=outputs_dir / "metricas_policy3_rnn.csv",
        help="CSV da Politica 3/RNN.",
    )
    parser.add_argument(
        "--policy2-failover",
        type=Path,
        default=outputs_dir / "policy2_failover_experiment.csv",
        help="CSV do experimento controlado de failover da Politica 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_dir / "tex",
        help="Diretorio onde os arquivos .tex serao salvos.",
    )
    args = parser.parse_args()
    return CliArgs(
        policy1_csv=args.policy1,
        policy2_csv=args.policy2,
        policy3_csv=args.policy3,
        policy2_failover_csv=args.policy2_failover,
        output_dir=args.output_dir,
    )


def _validate_required_csvs(args: CliArgs) -> None:
    """Garante que os CSVs das tres politicas existam."""
    missing = [
        csv_path
        for csv_path in (args.policy1_csv, args.policy2_csv, args.policy3_csv)
        if not csv_path.exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "CSV obrigatorio nao encontrado. Execute as politicas antes de "
            f"gerar os graficos:\n{formatted}"
        )


def main() -> None:
    """Executa a geracao consolidada de graficos TeX."""
    args = parse_args()
    _validate_required_csvs(args)

    policy_csvs = [
        ("policy1", "Baseline", args.policy1_csv),
        ("policy2", "Política 2", args.policy2_csv),
        ("policy3", "Política 3", args.policy3_csv),
    ]
    failover_csv = args.policy2_failover_csv if args.policy2_failover_csv.exists() else None
    if failover_csv is None:
        print(
            "Aviso: CSV de failover nao encontrado; graficos de failover ignorados: "
            f"{args.policy2_failover_csv}"
        )

    generated = generate_all_tex_plots(
        policy_csvs=policy_csvs,
        policy2_failover_csv=failover_csv,
        output_dir=args.output_dir,
    )

    print(f"{len(generated)} arquivos .tex gerados em: {args.output_dir}")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
