"""
Coleta dados de treinamento para a política RNN.

Este script executa o cliente com monitores concorrentes dos servidores e uma
política exploratória, sem RNN, que sorteia ciclos balanceados de servidores
saudáveis e qualidades disponíveis. Durante a execução, falhas temporárias são
injetadas nos dois servidores em instantes aleatórios e reproduzíveis.

O CSV atualizado por este script deve ser usado como entrada do treinamento. Se
o arquivo já existir, novas amostras são anexadas ao fim dele:

    python -m models.train \
        --csv outputs/rnn_training_data.csv \
        --output outputs/models/rnn_policy.pt \
        --server-a A \
        --server-b B
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ALPHA_JITTER_EWMA,
    ALPHA_THROUGHPUT_EWMA,
    MANIFEST_URL,
    TRAINING_EXPLORATION_SEED,
    TRAINING_FAULT_MAX_DURATION_S,
    TRAINING_FAULT_MAX_GAP_S,
    TRAINING_FAULT_MIN_DURATION_S,
    TRAINING_FAULT_MIN_GAP_S,
    TRAINING_FAULT_INITIAL_DELAY_S,
    TRAINING_FAULT_SEED,
    TRAINING_FAULTS_PER_SERVER,
    TRAINING_LOW_BUFFER_EXPLORE_PROBABILITY,
    TRAINING_NUM_SEGMENTS,
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


@dataclass(frozen=True)
class CliArgs:
    """Parâmetros de uma rodada de coleta de treinamento."""

    output: Path
    segments: int
    exploration_seed: int
    fault_seed: int
    faults_per_server: int
    low_buffer_explore_probability: float
    preferred_server: str | None
    preferred_server_probability: float
    high_quality_probability: float
    top_quality_count: int
    overwrite: bool


def parse_args() -> CliArgs:
    """Lê opções para variar a coleta sem alterar `config.py`."""
    parser = argparse.ArgumentParser(
        description="Coleta dados de treinamento para a política RNN.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rnn_training_data.csv",
        help="CSV de saída da coleta.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=TRAINING_NUM_SEGMENTS,
        help="Quantidade de segmentos coletados.",
    )
    parser.add_argument(
        "--exploration-seed",
        type=int,
        default=TRAINING_EXPLORATION_SEED,
        help="Seed da política exploratória.",
    )
    parser.add_argument(
        "--fault-seed",
        type=int,
        default=TRAINING_FAULT_SEED,
        help="Seed do cronograma de falhas sintéticas.",
    )
    parser.add_argument(
        "--faults-per-server",
        type=int,
        default=TRAINING_FAULTS_PER_SERVER,
        help="Quantidade de falhas sintéticas por servidor.",
    )
    parser.add_argument(
        "--low-buffer-explore-probability",
        type=float,
        default=TRAINING_LOW_BUFFER_EXPLORE_PROBABILITY,
        help="Probabilidade de explorar quando o buffer está baixo.",
    )
    parser.add_argument(
        "--preferred-server",
        type=str,
        default=None,
        help="Servidor preferido para coleta enviesada, por exemplo A ou B.",
    )
    parser.add_argument(
        "--preferred-server-probability",
        type=float,
        default=0.0,
        help="Probabilidade de usar o servidor preferido quando saudável.",
    )
    parser.add_argument(
        "--high-quality-probability",
        type=float,
        default=0.0,
        help="Probabilidade de amostrar uma das maiores qualidades.",
    )
    parser.add_argument(
        "--top-quality-count",
        type=int,
        default=2,
        help="Quantidade de maiores qualidades consideradas no viés.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve o CSV de saída em vez de anexar.",
    )
    args = parser.parse_args()
    return CliArgs(
        output=args.output,
        segments=args.segments,
        exploration_seed=args.exploration_seed,
        fault_seed=args.fault_seed,
        faults_per_server=args.faults_per_server,
        low_buffer_explore_probability=args.low_buffer_explore_probability,
        preferred_server=args.preferred_server,
        preferred_server_probability=args.preferred_server_probability,
        high_quality_probability=args.high_quality_probability,
        top_quality_count=args.top_quality_count,
        overwrite=args.overwrite,
    )


def main() -> None:
    """Coleta dados com falhas temporárias aleatórias em ambos os servidores."""
    args = parse_args()
    print("Carregando manifest...")

    manifest = load_manifest(MANIFEST_URL)

    if len(manifest.servers) < 2:
        raise ValueError("A coleta de treino espera pelo menos dois servidores.")

    output_dir: Path = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = args.output
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

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
        faults_per_server=args.faults_per_server,
        min_initial_delay_s=TRAINING_FAULT_INITIAL_DELAY_S,
        min_healthy_gap_s=TRAINING_FAULT_MIN_GAP_S,
        max_healthy_gap_s=TRAINING_FAULT_MAX_GAP_S,
        min_failure_duration_s=TRAINING_FAULT_MIN_DURATION_S,
        max_failure_duration_s=TRAINING_FAULT_MAX_DURATION_S,
        seed=args.fault_seed,
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
        seed=args.exploration_seed,
        low_buffer_explore_probability=args.low_buffer_explore_probability,
        preferred_server_id=args.preferred_server,
        preferred_server_probability=args.preferred_server_probability,
        high_quality_probability=args.high_quality_probability,
        top_quality_count=args.top_quality_count,
    )

    appending: bool = (
        not args.overwrite and csv_path.exists() and csv_path.stat().st_size > 0
    )
    csv_writer = CsvMetricsWriter(str(csv_path), append=appending)

    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        observation_store=observation_store,
        csv_writer=csv_writer,
        num_segments=args.segments,
        alpha_jitter_ewma=ALPHA_JITTER_EWMA,
        alpha_throughput_ewma=ALPHA_THROUGHPUT_EWMA,
    )

    try:
        print(
            f"Exploração (seed={args.exploration_seed}, "
            "probabilidade de explorar com buffer baixo="
            f"{args.low_buffer_explore_probability:.2f})."
        )
        if args.preferred_server is not None or args.high_quality_probability > 0.0:
            print(
                "Viés de coleta: "
                f"servidor preferido={args.preferred_server or '-'} "
                f"p={args.preferred_server_probability:.2f}; "
                f"alta qualidade p={args.high_quality_probability:.2f} "
                f"top={args.top_quality_count}."
            )
        print(f"Cronograma de falhas (seed={args.fault_seed}):")
        for window in fault_windows:
            print(
                f"  {window.server_id}: "
                f"{window.start_after_s:.1f}s até {window.end_after_s:.1f}s"
            )

        if appending:
            print(f"Anexando novas amostras em: {csv_path}")
        else:
            print(f"Criando CSV de treino em: {csv_path}")
        print("Coletando dados de treino...")

        runner.run()

    finally:
        for monitor in monitors:
            monitor.stop()

        for monitor in monitors:
            monitor.join(timeout=2.0)

    print(f"CSV de treino atualizado em: {csv_path}")


if __name__ == "__main__":
    main()
