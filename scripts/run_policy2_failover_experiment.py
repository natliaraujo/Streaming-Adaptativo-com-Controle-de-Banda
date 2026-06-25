"""Executa a Política 2 com queda temporária controlada do servidor principal."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ABR_HISTORY_SIZE,
    ALPHA_JITTER_EWMA,
    ALPHA_THROUGHPUT_EWMA,
    MANIFEST_URL,
    NUM_SEGMENTS,
    SAFETY_FACTOR,
)
from domain.manifest import ServerInfo  # noqa: E402
from experiment import CsvMetricsWriter, ExperimentRunner  # noqa: E402
from network import load_manifest  # noqa: E402
from policy import ProbeBufferAwarePolicy  # noqa: E402


FAIL_SERVER_ID = "A"
FAIL_AFTER_SEGMENT = 8
FAIL_DURATION_SEGMENTS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa a Política 2 com indisponibilidade temporária do "
            "servidor principal após um segmento configurável."
        )
    )
    parser.add_argument("--manifest-url", default=MANIFEST_URL)
    parser.add_argument("--segments", type=int, default=NUM_SEGMENTS)
    parser.add_argument("--fail-server", default=FAIL_SERVER_ID)
    parser.add_argument(
        "--fail-after-segment",
        type=int,
        default=FAIL_AFTER_SEGMENT,
        help=(
            "Último segmento concluído antes da queda. Com o padrão 8, "
            "a falha e o failover ocorrem no segmento 9."
        ),
    )
    parser.add_argument(
        "--fail-duration-segments",
        type=int,
        default=FAIL_DURATION_SEGMENTS,
        help=(
            "Quantidade de segmentos em que o servidor ficará indisponível. "
            "Com o padrão 4, o servidor volta a partir do segmento 13."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "policy2_failover_experiment.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fail_after_segment < 0:
        raise ValueError("--fail-after-segment não pode ser negativo.")
    if args.fail_duration_segments <= 0:
        raise ValueError("--fail-duration-segments deve ser positivo.")
    recovery_segment = args.fail_after_segment + args.fail_duration_segments + 1
    if args.segments < recovery_segment:
        raise ValueError(
            "--segments deve incluir ao menos um segmento após a recuperação."
        )

    print("Carregando manifest para o experimento de failover da Política 2...")
    manifest = load_manifest(args.manifest_url)
    failed_server = next(
        (
            server
            for server in manifest.servers
            if server.id.casefold() == args.fail_server.casefold()
        ),
        None,
    )
    if failed_server is None:
        available = ", ".join(server.id for server in manifest.servers)
        raise ValueError(
            f"Servidor {args.fail_server!r} não existe no manifest. "
            f"Disponíveis: {available}"
        )
    if not any(server.id != failed_server.id for server in manifest.servers):
        raise ValueError("O experimento exige ao menos um servidor fallback.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = ProbeBufferAwarePolicy(
        history_size=ABR_HISTORY_SIZE,
        safety_factor=SAFETY_FACTOR,
    )

    first_failed_segment = args.fail_after_segment + 1
    last_failed_segment = args.fail_after_segment + args.fail_duration_segments

    def simulated_failure(segment: int, server: ServerInfo) -> bool:
        return (
            server.id == failed_server.id
            and first_failed_segment <= segment <= last_failed_segment
        )

    print(
        f"Servidor {failed_server.id} ficará indisponível do segmento "
        f"{first_failed_segment} ao {last_failed_segment}; fallback esperado "
        f"no segmento {first_failed_segment} e recuperação esperada no "
        f"segmento {recovery_segment}."
    )
    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        csv_writer=CsvMetricsWriter(str(args.output)),
        num_segments=args.segments,
        alpha_jitter_ewma=ALPHA_JITTER_EWMA,
        alpha_throughput_ewma=ALPHA_THROUGHPUT_EWMA,
        simulated_server_failure=simulated_failure,
    )
    runner.run()
    print(f"CSV do experimento gerado em: {args.output}")


if __name__ == "__main__":
    main()
