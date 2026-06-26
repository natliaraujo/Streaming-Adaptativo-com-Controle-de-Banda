"""Gera todos os gráficos do projeto a partir dos CSVs de métricas."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.final_plots import (  # noqa: E402
    generate_final_plots,
    generate_policy2_failover_plot,
)
from analysis.plots import (  # noqa: E402
    generate_policy_comparison_plot,
    generate_quality_buffer_comparison_plot,
    generate_throughput_quality_plot,
)


@dataclass(frozen=True)
class CliArgs:
    """Argumentos da geração consolidada de gráficos."""

    policy1_csv: Path
    policy2_csv: Path
    policy3_csv: Path
    policy2_failover_csv: Path
    output_dir: Path


def parse_args() -> CliArgs:
    """Lê caminhos de entrada e diretório de saída."""
    outputs_dir = PROJECT_ROOT / "outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Gera todos os gráficos do projeto: individuais, comparativos, "
            "decisões da RNN, failover da Política 2 e figuras legadas."
        )
    )
    parser.add_argument(
        "--policy1",
        type=Path,
        default=outputs_dir / "metricas_policy1.csv",
        help="CSV da Política 1.",
    )
    parser.add_argument(
        "--policy2",
        type=Path,
        default=outputs_dir / "metricas_policy2.csv",
        help="CSV da Política 2.",
    )
    parser.add_argument(
        "--policy3",
        type=Path,
        default=outputs_dir / "metricas_policy3_rnn.csv",
        help="CSV da Política 3/RNN.",
    )
    parser.add_argument(
        "--policy2-failover",
        type=Path,
        default=outputs_dir / "policy2_failover_experiment.csv",
        help="CSV do experimento controlado de failover da Política 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_dir / "plots",
        help="Diretório onde os PNGs serão salvos.",
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
    """Garante que os CSVs das três políticas existam."""
    missing = [
        csv_path
        for csv_path in (args.policy1_csv, args.policy2_csv, args.policy3_csv)
        if not csv_path.exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "CSV obrigatório não encontrado. Execute as políticas antes de "
            f"gerar os gráficos:\n{formatted}"
        )


def _generate_legacy_plots(args: CliArgs) -> list[Path]:
    """
    Gera figuras mantidas por compatibilidade com nomes anteriores do relatório.

    Esses PNGs mantêm nomes de arquivo anteriores do relatório, sem exigir
    scripts separados.
    """
    generated: list[Path] = []

    baseline_output = args.output_dir / "grafico_policy1_vazao_qualidade.png"
    generate_throughput_quality_plot(
        csv_path=args.policy1_csv,
        output_path=baseline_output,
    )
    generated.append(baseline_output)

    comparison_output = args.output_dir / "comparacao_policy1_policy2.png"
    generate_policy_comparison_plot(
        policy1_csv=args.policy1_csv,
        policy2_csv=args.policy2_csv,
        policy3_csv=args.policy3_csv,
        output_path=comparison_output,
    )
    generated.append(comparison_output)

    quality_buffer_output = args.output_dir / "comparacao_qualidade_buffer.png"
    generate_quality_buffer_comparison_plot(
        policy_csvs=[
            ("Política 1", args.policy1_csv),
            ("Política 2", args.policy2_csv),
            ("Política 3 (RNN)", args.policy3_csv),
        ],
        output_path=quality_buffer_output,
    )
    generated.append(quality_buffer_output)

    return generated


def _generate_failover_plot(args: CliArgs) -> list[Path]:
    """Gera o gráfico de failover quando o CSV correspondente existe."""
    if not args.policy2_failover_csv.exists():
        print(
            "Aviso: CSV de failover não encontrado; gráfico de failover ignorado: "
            f"{args.policy2_failover_csv}"
        )
        return []

    output_path = args.output_dir / "policy2_failover_experiment.png"
    generate_policy2_failover_plot(
        csv_path=args.policy2_failover_csv,
        output_path=output_path,
    )
    return [output_path]


def main() -> None:
    """Executa a geração consolidada de todos os gráficos."""
    args = parse_args()
    _validate_required_csvs(args)

    policy_csvs = [
        ("policy1", "Baseline", args.policy1_csv),
        ("policy2", "Política 2", args.policy2_csv),
        ("policy3", "Política 3", args.policy3_csv),
    ]

    generated = [
        *generate_final_plots(policy_csvs, args.output_dir),
        *_generate_failover_plot(args),
        *_generate_legacy_plots(args),
    ]

    print(f"{len(generated)} gráficos gerados em: {args.output_dir}")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
