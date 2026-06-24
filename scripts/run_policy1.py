"""
Executa o experimento da política 1, correspondente ao baseline.

A política 1 usa servidor fixo, normalmente o servidor de maior prioridade do
manifest, e escolhe a qualidade dos segmentos com base apenas na vazão medida
recentemente.

O script gera um CSV de métricas específico para posterior comparação com as
políticas 2 e 3.
"""

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ABR_HISTORY_SIZE,
    ALPHA_EWMA,
    MANIFEST_URL,
    NUM_SEGMENTS,
    SAFETY_FACTOR,
)
from experiment import CsvMetricsWriter, ExperimentRunner  # noqa: E402
from network import load_manifest  # noqa: E402
from policy import RateBasedFixedServerPolicy  # noqa: E402


def main() -> None:
    """Executa a política baseline rate-based com servidor fixo."""
    print("Executando política 1...")
    print("Carregando manifest...")

    manifest = load_manifest(MANIFEST_URL)

    output_dir: Path = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = output_dir / "metricas_policy1.csv"

    policy = RateBasedFixedServerPolicy(
        history_size=ABR_HISTORY_SIZE,
        safety_factor=SAFETY_FACTOR,
    )

    csv_writer = CsvMetricsWriter(str(csv_path))

    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        csv_writer=csv_writer,
        num_segments=NUM_SEGMENTS,
        alpha_ewma=ALPHA_EWMA,
    )

    runner.run()

    print(f"CSV gerado em: {csv_path}")


if __name__ == "__main__":
    main()
