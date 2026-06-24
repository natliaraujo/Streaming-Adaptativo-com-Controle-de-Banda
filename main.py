"""
Ponto de entrada principal do projeto de streaming adaptativo.

Este módulo instancia os componentes centrais do experimento: carregamento do
manifest, política de decisão, escritor de métricas e executor do experimento.
Ele deve conter apenas a composição dos módulos, evitando implementar lógica de
download, buffer, seleção de qualidade ou geração de métricas diretamente.

Para políticas específicas, como baseline, política 2 ou política 3 com RNN,
também podem existir scripts dedicados no pacote `scripts`.
"""

from pathlib import Path

from config import (
    ABR_HISTORY_SIZE,
    ALPHA_EWMA,
    MANIFEST_URL,
    NUM_SEGMENTS,
    SAFETY_FACTOR,
)
from experiment import CsvMetricsWriter, ExperimentRunner
from network import load_manifest
from policy import RateBasedFixedServerPolicy


def main() -> None:
    """Carrega o manifesto, configura a política baseline e executa o experimento."""
    print("Baixando manifesto...")

    manifest = load_manifest(MANIFEST_URL)

    policy = RateBasedFixedServerPolicy(
        history_size=ABR_HISTORY_SIZE,
        safety_factor=SAFETY_FACTOR,
    )

    project_root: Path = Path(__file__).resolve().parent
    output_dir: Path = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = output_dir / "metricas_baseline.csv"

    csv_writer = CsvMetricsWriter(str(csv_path))

    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        csv_writer=csv_writer,
        num_segments=NUM_SEGMENTS,
        alpha_ewma=ALPHA_EWMA,
    )

    runner.run()

    print(f"CSV gerado: {csv_path}")


if __name__ == "__main__":
    main()
