"""Ponto de entrada do experimento baseline de streaming adaptativo."""

import os

from abr.rate_based import RateBasedAbrPolicy
from config import (
    MANIFEST_URL,
    NUM_SEGMENTS,
    ABR_HISTORY_SIZE,
    SAFETY_FACTOR,
    ALPHA_EWMA,
)
from experiment.csv_writer import CsvMetricsWriter
from experiment.runner import ExperimentRunner
from network.manifest_client import load_manifest


def main() -> None:
    """Carrega o manifesto, configura a política ABR e executa o experimento."""

    print("Baixando manifesto...")

    manifest = load_manifest(MANIFEST_URL)

    abr_policy = RateBasedAbrPolicy(
        history_size=ABR_HISTORY_SIZE,
        safety_factor=SAFETY_FACTOR,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "outputs", "metricas_baseline.csv")

    csv_writer = CsvMetricsWriter(csv_path)

    runner = ExperimentRunner(
        manifest=manifest,
        abr_policy=abr_policy,
        csv_writer=csv_writer,
        num_segments=NUM_SEGMENTS,
        alpha_ewma=ALPHA_EWMA,
    )

    runner.run()

    print(f"CSV gerado: {csv_path}")


if __name__ == "__main__":
    main()
