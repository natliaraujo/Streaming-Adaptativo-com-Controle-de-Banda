"""
Executa o experimento da política 2.

A política 2 usa o servidor primário em condições normais e seleciona a qualidade
com uma estratégia buffer-aware. O servidor secundário é usado apenas em
failover. Ela não usa RNN nem monitoramento concorrente.

O CSV gerado por este script permite comparar a heurística dinâmica com o
baseline e com a política neural.
"""

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

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
from experiment import CsvMetricsWriter, ExperimentRunner  # noqa: E402
from network import load_manifest  # noqa: E402
from policy import ProbeBufferAwarePolicy  # noqa: E402


def main() -> None:
    """Executa a política 2 baseada em probe sequencial e buffer."""
    print("Executando política 2...")
    print("Carregando manifest...")

    manifest = load_manifest(MANIFEST_URL)

    output_dir: Path = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = output_dir / "metricas_policy2.csv"

    policy = ProbeBufferAwarePolicy(
        history_size=ABR_HISTORY_SIZE,
        safety_factor=SAFETY_FACTOR,
    )

    csv_writer = CsvMetricsWriter(str(csv_path))

    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        csv_writer=csv_writer,
        num_segments=NUM_SEGMENTS,
        alpha_jitter_ewma=ALPHA_JITTER_EWMA,
        alpha_throughput_ewma=ALPHA_THROUGHPUT_EWMA,
    )

    runner.run()

    print(f"CSV gerado em: {csv_path}")


if __name__ == "__main__":
    main()
