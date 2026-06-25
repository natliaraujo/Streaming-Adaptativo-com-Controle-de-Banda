"""
Executa o experimento da política 3 baseada em RNN pré-treinada.

O script carrega o checkpoint salvo por `models.train`, inicializa os monitores
dos servidores, instancia a política RNN e executa o experimento. Durante a
execução, a rede é usada apenas em modo de inferência.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALPHA_JITTER_EWMA, ALPHA_THROUGHPUT_EWMA, MANIFEST_URL, NUM_SEGMENTS, SAFETY_FACTOR  # noqa: E402
from experiment import CsvMetricsWriter, ExperimentRunner  # noqa: E402
from models.checkpoint import LoadedRnnModel, load_rnn_checkpoint  # noqa: E402
from monitoring.feature_builder import FeatureConfig, FeatureHistory  # noqa: E402
from monitoring.observation_store import ObservationStore  # noqa: E402
from monitoring.server_monitor import MonitorConfig, ServerMonitor  # noqa: E402
from network import load_manifest  # noqa: E402
from policy.rnn import RnnStreamingPolicy  # noqa: E402


def main() -> None:
    """Executa a política 3 com modelo RNN pré-treinado."""
    print("Executando política 3...")
    print("Carregando manifest...")

    manifest = load_manifest(MANIFEST_URL)

    if len(manifest.servers) < 2:
        raise ValueError("A política RNN espera pelo menos dois servidores no manifest.")

    checkpoint_path: Path = PROJECT_ROOT / "outputs" / "models" / "rnn_policy.pt"

    loaded_model: LoadedRnnModel = load_rnn_checkpoint(checkpoint_path)

    output_dir: Path = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Path = output_dir / "metricas_policy3_rnn.csv"

    observation_store = ObservationStore()

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
            store=observation_store,
            config=monitor_config,
        )
        for server in manifest.servers
    ]

    for monitor in monitors:
        monitor.start()

    time.sleep(1.2)

    feature_config = FeatureConfig(
        sequence_length=loaded_model.sequence_length,
        server_a_id=loaded_model.server_a_id,
        server_b_id=loaded_model.server_b_id,
    )

    feature_history = FeatureHistory(
        sequence_length=loaded_model.sequence_length,
        feature_size=loaded_model.feature_size,
    )

    policy = RnnStreamingPolicy(
        model=loaded_model.model,
        feature_config=feature_config,
        feature_history=feature_history,
        normalizer=loaded_model.normalizer,
        safety_factor=SAFETY_FACTOR,
    )

    csv_writer = CsvMetricsWriter(str(csv_path))

    runner = ExperimentRunner(
        manifest=manifest,
        policy=policy,
        observation_store=observation_store,
        csv_writer=csv_writer,
        num_segments=NUM_SEGMENTS,
        alpha_jitter_ewma=ALPHA_JITTER_EWMA,
        alpha_throughput_ewma=ALPHA_THROUGHPUT_EWMA,
    )

    try:
        runner.run()
    finally:
        for monitor in monitors:
            monitor.stop()

        for monitor in monitors:
            monitor.join(timeout=2.0)

    print(f"CSV gerado em: {csv_path}")


if __name__ == "__main__":
    main()
