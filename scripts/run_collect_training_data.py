"""
Coleta dados de treinamento para a política RNN.

Este script executa o cliente com monitores concorrentes dos servidores e uma
política exploratória simples, sem RNN. Durante a execução, falhas temporárias
são injetadas nos dois servidores em instantes aleatórios e reproduzíveis.

O CSV gerado por este script deve ser usado como entrada do treinamento:

    python -m models.train \
        --csv outputs/rnn_training_data.csv \
        --output outputs/models/rnn_policy.pt \
        --server-a A \
        --server-b srv-B
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ABR_HISTORY_SIZE,
    ALPHA_EWMA,
    MANIFEST_URL,
    TRAINING_FAULT_MAX_DURATION_S,
    TRAINING_FAULT_MAX_GAP_S,
    TRAINING_FAULT_MIN_DURATION_S,
    TRAINING_FAULT_MIN_GAP_S,
    TRAINING_FAULT_INITIAL_DELAY_S,
    TRAINING_FAULT_SEED,
    TRAINING_FAULTS_PER_SERVER,
    TRAINING_NUM_SEGMENTS,
    SAFETY_FACTOR,
)
from experiment import CsvMetricsWriter, ExperimentRunner  # noqa: E402
from monitoring import (  # noqa: E402
    FaultInjectingObservationStore,
    FaultInjectionConfig,
    FaultWindow,
    build_random_fault_schedule,
)
from monitoring.observation_store import ObservationStore  # noqa: E402
from monitoring.server_monitor import MonitorConfig, ServerMonitor  # noqa: E402
from network import load_manifest  # noqa: E402
from policy.training_collection_policy import TrainingDataCollectionPolicy  # noqa: E402


def main() -> None:
    """Coleta dados com falhas temporárias aleatórias em ambos os servidores."""
    print("Carregando manifest...")

    manifest = load_manifest(MANIFEST_URL)

    if len(manifest.servers) < 2:
        raise ValueError("A coleta de treino espera pelo menos dois servidores.")

    output_dir: Path = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = output_dir / "rnn_training_data.csv"

    real_observation_store = ObservationStore()

    monitor_config = MonitorConfig(
        interval_s=1.0,
        timeout_s=1.0,
        probe_path="/segment/240p?seg=1",
        max_probe_bytes=8192,
        chunk_size=1024,
    )

    monitors: list[ServerMonitor] = [
        ServerMonitor(
            server=server,
            store=real_observation_store,
            config=monitor_config,
        )
        for server in manifest.servers
    ]

    for monitor in monitors:
        monitor.start()

    # Espera os monitores coletarem pelo menos uma amostra antes do experimento.
    time.sleep(1.2)

    monitored_server_ids: list[str] = [
        server.id for server in manifest.servers[:2]
    ]
    fault_windows: tuple[FaultWindow, ...] = build_random_fault_schedule(
        server_ids=monitored_server_ids,
        faults_per_server=TRAINING_FAULTS_PER_SERVER,
        min_initial_delay_s=TRAINING_FAULT_INITIAL_DELAY_S,
        min_healthy_gap_s=TRAINING_FAULT_MIN_GAP_S,
        max_healthy_gap_s=TRAINING_FAULT_MAX_GAP_S,
        min_failure_duration_s=TRAINING_FAULT_MIN_DURATION_S,
        max_failure_duration_s=TRAINING_FAULT_MAX_DURATION_S,
        seed=TRAINING_FAULT_SEED,
    )

    fault_config = FaultInjectionConfig(
        windows=fault_windows,
        failed_latency_ms=10_000.0,
    )

    observation_store = FaultInjectingObservationStore(
        base_store=real_observation_store,
        config=fault_config,
    )

    policy = TrainingDataCollectionPolicy(
        safety_factor=SAFETY_FACTOR,
        history_size=ABR_HISTORY_SIZE,
    )

    csv_writer = CsvMetricsWriter(str(csv_path))

    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        observation_store=observation_store,
        csv_writer=csv_writer,
        num_segments=TRAINING_NUM_SEGMENTS,
        alpha_ewma=ALPHA_EWMA,
    )

    try:
        print(f"Cronograma de falhas (seed={TRAINING_FAULT_SEED}):")
        for window in fault_windows:
            print(
                f"  {window.server_id}: "
                f"{window.start_after_s:.1f}s até {window.end_after_s:.1f}s"
            )

        print("Coletando dados de treino...")

        runner.run()

    finally:
        for monitor in monitors:
            monitor.stop()

        for monitor in monitors:
            monitor.join(timeout=2.0)

    print(f"CSV de treino gerado em: {csv_path}")


if __name__ == "__main__":
    main()
