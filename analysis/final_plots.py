"""Gráficos finais e comparáveis dos experimentos de streaming."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import BUFFER_MAX_S, BUFFER_TARGET_S


@dataclass(frozen=True)
class PlotMetric:
    """Amostra normalizada lida de um CSV de experimento."""

    segment: int
    server_id: str
    quality: str
    bitrate_kbps: float
    throughput_kbps: float
    jitter_ewma_ms: float
    buffer_level_s: float
    rebuffer_event: int
    failover_event: int


POLICY_COLORS: tuple[str, ...] = ("#2563eb", "#d97706", "#059669")


def _pyplot() -> Any:
    """Importa matplotlib com backend adequado para execução sem interface."""
    try:
        cache_dir = Path(
            os.environ.setdefault(
                "MPLCONFIGDIR",
                str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependência ausente: instale matplotlib para gerar os gráficos "
            "(`python -m pip install matplotlib`)."
        ) from exc
    return plt


def _float_value(row: dict[str, str], column: str, default: float = 0.0) -> float:
    value = row.get(column)
    if value is None or value.strip() == "":
        return default
    return float(value)


def read_plot_metrics(csv_path: Path) -> list[PlotMetric]:
    """Lê e valida as colunas necessárias aos gráficos finais."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "segment",
            "server_id",
            "quality",
            "bitrate_kbps",
            "buffer_level_s",
            "jitter_ewma_ms",
        }
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"CSV {csv_path} sem colunas obrigatórias: "
                + ", ".join(sorted(missing))
            )

        if "throughput_kbps" in fieldnames:
            throughput_column = "throughput_kbps"
        elif "vazao_kbps" in fieldnames:
            throughput_column = "vazao_kbps"
        else:
            raise ValueError(
                f"CSV {csv_path} sem throughput_kbps (ou vazao_kbps)."
            )

        metrics = [
            PlotMetric(
                segment=int(float(row["segment"])),
                server_id=row["server_id"],
                quality=row["quality"],
                bitrate_kbps=_float_value(row, "bitrate_kbps"),
                throughput_kbps=_float_value(row, throughput_column),
                jitter_ewma_ms=_float_value(row, "jitter_ewma_ms"),
                buffer_level_s=_float_value(row, "buffer_level_s"),
                rebuffer_event=int(_float_value(row, "rebuffer_event")),
                failover_event=int(_float_value(row, "failover_event")),
            )
            for row in reader
        ]

    if not metrics:
        raise ValueError(f"CSV sem dados: {csv_path}")
    return metrics


def _event_segments(metrics: list[PlotMetric], attribute: str) -> list[int]:
    return [
        metric.segment
        for metric in metrics
        if int(getattr(metric, attribute)) == 1
    ]


def _draw_failovers(axis: Any, metrics: list[PlotMetric]) -> None:
    for index, segment in enumerate(_event_segments(metrics, "failover_event")):
        axis.axvline(
            segment,
            color="#dc2626",
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label="Failover" if index == 0 else None,
        )


def _annotate_quality_changes(axis: Any, metrics: list[PlotMetric]) -> None:
    previous: str | None = None
    for metric in metrics:
        if metric.quality != previous:
            axis.annotate(
                metric.quality,
                (metric.segment, metric.bitrate_kbps),
                xytext=(3, 7),
                textcoords="offset points",
                fontsize=8,
                color="#7c2d12",
            )
        previous = metric.quality


def _save(
    fig: Any,
    output_path: Path,
    *,
    bottom_margin: float = 0.0,
    right_margin: float = 1.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, bottom_margin, right_margin, 1.0))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    _pyplot().close(fig)


def plot_throughput_quality(metrics: list[PlotMetric], output_path: Path, title: str) -> None:
    """Plota vazão e bitrate em eixos compatíveis, com rótulos de qualidade."""
    plt = _pyplot()
    fig, throughput_axis = plt.subplots(figsize=(12, 6))
    bitrate_axis = throughput_axis.twinx()
    segments = [metric.segment for metric in metrics]

    throughput_axis.plot(
        segments,
        [metric.throughput_kbps for metric in metrics],
        color="#2563eb",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Vazão medida",
    )
    bitrate_axis.step(
        segments,
        [metric.bitrate_kbps for metric in metrics],
        where="mid",
        color="#d97706",
        linewidth=2,
        label="Bitrate selecionado",
    )
    bitrate_axis.scatter(
        segments,
        [metric.bitrate_kbps for metric in metrics],
        color="#d97706",
        s=13,
        zorder=3,
    )
    _annotate_quality_changes(bitrate_axis, metrics)
    _draw_failovers(throughput_axis, metrics)

    throughput_axis.set_title(title)
    throughput_axis.set_xlabel("Número do segmento")
    throughput_axis.set_ylabel("Vazão medida (kbps)")
    bitrate_axis.set_ylabel("Bitrate selecionado (kbps)")
    throughput_axis.grid(True, linestyle="--", alpha=0.28)
    lines1, labels1 = throughput_axis.get_legend_handles_labels()
    lines2, labels2 = bitrate_axis.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )
    _save(fig, output_path, bottom_margin=0.12)


def plot_buffer(metrics: list[PlotMetric], output_path: Path, title: str) -> None:
    """Plota o buffer, seus limites e os eventos de rebuffering/failover."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(12, 5.5))
    segments = [metric.segment for metric in metrics]
    levels = [metric.buffer_level_s for metric in metrics]
    axis.plot(segments, levels, color="#059669", linewidth=2, label="Buffer")
    axis.axhline(
        BUFFER_TARGET_S,
        color="#7c3aed",
        linestyle="--",
        label=f"BUFFER_TARGET_S ({BUFFER_TARGET_S:g} s)",
    )
    axis.axhline(
        BUFFER_MAX_S,
        color="#4b5563",
        linestyle=":",
        label=f"BUFFER_MAX_S ({BUFFER_MAX_S:g} s)",
    )
    rebuffer_metrics = [metric for metric in metrics if metric.rebuffer_event]
    if rebuffer_metrics:
        axis.scatter(
            [metric.segment for metric in rebuffer_metrics],
            [metric.buffer_level_s for metric in rebuffer_metrics],
            color="#dc2626",
            marker="x",
            s=65,
            linewidths=2,
            zorder=4,
            label="Rebuffering",
        )
    _draw_failovers(axis, metrics)
    axis.set_title(title)
    axis.set_xlabel("Número do segmento")
    axis.set_ylabel("Nível do buffer (s)")
    axis.set_ylim(bottom=0)
    axis.grid(True, linestyle="--", alpha=0.28)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(labels)),
    )
    _save(fig, output_path, bottom_margin=0.14)


def plot_jitter(metrics: list[PlotMetric], output_path: Path, title: str) -> None:
    """Plota a EWMA do jitter ao longo dos segmentos."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(
        [metric.segment for metric in metrics],
        [metric.jitter_ewma_ms for metric in metrics],
        color="#7c3aed",
        linewidth=2,
        label="Jitter EWMA",
    )
    _draw_failovers(axis, metrics)
    axis.set_title(title)
    axis.set_xlabel("Número do segmento")
    axis.set_ylabel("Jitter EWMA (ms)")
    axis.set_ylim(bottom=0)
    axis.grid(True, linestyle="--", alpha=0.28)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
    )
    _save(fig, output_path, bottom_margin=0.14)


def _plot_comparison(
    datasets: list[tuple[str, list[PlotMetric]]],
    output_path: Path,
    title: str,
    ylabel: str,
    value: Callable[[PlotMetric], float],
    *,
    use_steps: bool = False,
    show_buffer_thresholds: bool = False,
    show_rebuffers: bool = False,
) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(12, 6))
    for index, (label, metrics) in enumerate(datasets):
        segments = [metric.segment for metric in metrics]
        values = [value(metric) for metric in metrics]
        color = POLICY_COLORS[index % len(POLICY_COLORS)]
        if use_steps:
            axis.step(
                segments,
                values,
                where="mid",
                color=color,
                linewidth=2,
                label=label,
            )
        else:
            axis.plot(segments, values, color=color, linewidth=1.9, label=label)
        if show_rebuffers:
            events = [metric for metric in metrics if metric.rebuffer_event]
            if events:
                axis.scatter(
                    [metric.segment for metric in events],
                    [value(metric) for metric in events],
                    color=color,
                    marker="x",
                    s=55,
                    linewidths=2,
                )

    if show_buffer_thresholds:
        axis.axhline(
            BUFFER_TARGET_S,
            color="#7c3aed",
            linestyle="--",
            alpha=0.65,
            label="BUFFER_TARGET_S",
        )
        axis.axhline(
            BUFFER_MAX_S,
            color="#4b5563",
            linestyle=":",
            alpha=0.65,
            label="BUFFER_MAX_S",
        )
    axis.set_title(title)
    axis.set_xlabel("Número do segmento")
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0)
    axis.grid(True, linestyle="--", alpha=0.28)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(labels)),
    )
    _save(fig, output_path, bottom_margin=0.14)


def generate_final_plots(
    policy_csvs: list[tuple[str, str, Path]],
    output_dir: Path,
) -> list[Path]:
    """Gera os nove gráficos individuais e os quatro comparativos finais."""
    datasets: list[tuple[str, list[PlotMetric]]] = []
    generated: list[Path] = []
    for slug, label, csv_path in policy_csvs:
        metrics = read_plot_metrics(csv_path)
        datasets.append((label, metrics))
        plots = (
            (
                output_dir / f"{slug}_throughput_quality.png",
                plot_throughput_quality,
                f"{label}: vazão e qualidade selecionada",
            ),
            (
                output_dir / f"{slug}_buffer.png",
                plot_buffer,
                f"{label}: nível do buffer",
            ),
            (
                output_dir / f"{slug}_jitter_ewma.png",
                plot_jitter,
                f"{label}: jitter EWMA",
            ),
        )
        for output_path, plotter, title in plots:
            plotter(metrics, output_path, title)
            generated.append(output_path)

    comparisons = (
        (
            output_dir / "compare_policies_quality.png",
            "Qualidade selecionada: comparação das políticas",
            "Bitrate selecionado (kbps)",
            lambda metric: metric.bitrate_kbps,
            True,
            False,
            False,
        ),
        (
            output_dir / "compare_policies_buffer.png",
            "Nível do buffer: comparação das políticas",
            "Nível do buffer (s)",
            lambda metric: metric.buffer_level_s,
            False,
            True,
            True,
        ),
        (
            output_dir / "compare_policies_throughput.png",
            "Vazão medida: comparação das políticas",
            "Vazão medida (kbps)",
            lambda metric: metric.throughput_kbps,
            False,
            False,
            False,
        ),
        (
            output_dir / "compare_policies_jitter_ewma.png",
            "Jitter EWMA: comparação das políticas",
            "Jitter EWMA (ms)",
            lambda metric: metric.jitter_ewma_ms,
            False,
            False,
            False,
        ),
    )
    for output_path, title, ylabel, value, steps, thresholds, rebuffers in comparisons:
        _plot_comparison(
            datasets,
            output_path,
            title,
            ylabel,
            value,
            use_steps=steps,
            show_buffer_thresholds=thresholds,
            show_rebuffers=rebuffers,
        )
        generated.append(output_path)
    return generated


def generate_policy2_failover_plot(csv_path: Path, output_path: Path) -> None:
    """Gera a figura consolidada do experimento controlado de failover."""
    metrics = read_plot_metrics(csv_path)
    failover_segments = _event_segments(metrics, "failover_event")
    if not failover_segments:
        raise ValueError(f"CSV sem evento de failover: {csv_path}")

    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    throughput_axis = axes[0]
    bitrate_axis = throughput_axis.twinx()
    segments = [metric.segment for metric in metrics]

    throughput_axis.plot(
        segments,
        [metric.throughput_kbps for metric in metrics],
        color="#2563eb",
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Vazão medida",
    )
    bitrate_axis.step(
        segments,
        [metric.bitrate_kbps for metric in metrics],
        where="mid",
        color="#d97706",
        linewidth=2,
        label="Bitrate selecionado",
    )
    _annotate_quality_changes(bitrate_axis, metrics)
    throughput_axis.set_ylabel("Vazão (kbps)")
    bitrate_axis.set_ylabel("Bitrate (kbps)")
    throughput_axis.set_title("Política 2: vazão e qualidade durante o failover")
    lines1, labels1 = throughput_axis.get_legend_handles_labels()
    lines2, labels2 = bitrate_axis.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center right",
        bbox_to_anchor=(0.995, 0.82),
    )

    axes[1].plot(
        segments,
        [metric.buffer_level_s for metric in metrics],
        color="#059669",
        linewidth=2,
        label="Buffer",
    )
    axes[1].axhline(
        BUFFER_TARGET_S,
        color="#7c3aed",
        linestyle="--",
        label="BUFFER_TARGET_S",
    )
    axes[1].axhline(
        BUFFER_MAX_S,
        color="#4b5563",
        linestyle=":",
        label="BUFFER_MAX_S",
    )
    rebuffers = [metric for metric in metrics if metric.rebuffer_event]
    if rebuffers:
        axes[1].scatter(
            [metric.segment for metric in rebuffers],
            [metric.buffer_level_s for metric in rebuffers],
            color="#dc2626",
            marker="x",
            s=70,
            linewidths=2,
            label="Rebuffering",
            zorder=4,
        )
    axes[1].set_ylabel("Buffer (s)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Nível do buffer e rebuffering")
    buffer_handles, buffer_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        buffer_handles,
        buffer_labels,
        loc="center right",
        bbox_to_anchor=(0.995, 0.50),
    )

    server_ids = list(dict.fromkeys(metric.server_id for metric in metrics))
    server_index = {server_id: index for index, server_id in enumerate(server_ids)}
    axes[2].step(
        segments,
        [server_index[metric.server_id] for metric in metrics],
        where="mid",
        color="#374151",
        linewidth=2,
        label="Servidor efetivo",
    )
    axes[2].scatter(
        segments,
        [server_index[metric.server_id] for metric in metrics],
        color="#374151",
        s=15,
    )
    axes[2].set_yticks(range(len(server_ids)), server_ids)
    axes[2].set_ylabel("Servidor")
    axes[2].set_xlabel("Número do segmento")
    axes[2].set_title("Servidor usado por segmento")
    server_handles, server_labels = axes[2].get_legend_handles_labels()
    fig.legend(
        server_handles,
        server_labels,
        loc="center right",
        bbox_to_anchor=(0.995, 0.18),
    )

    for axis in axes:
        _draw_failovers(axis, metrics)
        axis.grid(True, linestyle="--", alpha=0.28)
    for segment in failover_segments:
        axes[0].annotate(
            f"Failover no segmento {segment}",
            xy=(segment, 1),
            xycoords=("data", "axes fraction"),
            xytext=(8, -22),
            textcoords="offset points",
            color="#991b1b",
            fontsize=9,
        )
    _save(fig, output_path, right_margin=0.82)
