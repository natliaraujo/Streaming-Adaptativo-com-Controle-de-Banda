"""Gera graficos PGFPlots/TikZ a partir dos CSVs dos experimentos."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from config import BUFFER_MAX_S, BUFFER_TARGET_S
from analysis.final_plots import (
    RNN_DECISION_COLUMNS,
    PlotMetric,
    RnnDecisionMetric,
    read_plot_metrics,
    read_rnn_decision_metrics,
)


PGFPLOTS_HEADER = r"""% Gerado automaticamente. No preambulo do documento use:
% \usepackage{pgfplots}
% \pgfplotsset{compat=1.18}
\definecolor{plotblue}{HTML}{2563EB}
\definecolor{plotorange}{HTML}{D97706}
\definecolor{plotgreen}{HTML}{059669}
\definecolor{plotpurple}{HTML}{7C3AED}
\definecolor{plotgray}{HTML}{374151}
\definecolor{plotred}{HTML}{DC2626}
\definecolor{plotteal}{HTML}{0F766E}
\definecolor{plotslate}{HTML}{64748B}
"""


POLICY_COLORS = ("plotblue", "plotorange", "plotgreen")
TEX_WIDTH = "14cm"
TEX_HEIGHT = "7cm"
LINE_STYLE = "line width=0.9pt, mark=none"
EMPHASIS_LINE_STYLE = "line width=1.05pt, mark=none"
STEP_STYLE = "const plot, line width=1.05pt, mark=none"
THRESHOLD_STYLE = "line width=0.55pt, mark=none"
EVENT_LINE_STYLE = "line width=0.65pt, densely dashed, mark=none"
REBUFFER_STYLE = (
    "only marks, mark=triangle*, mark size=1.8pt, "
    "plotred, mark options={solid, fill=plotred!70, draw=plotred}"
)


def _tex_escape(text: str) -> str:
    """Escapa caracteres especiais comuns em texto de legenda/titulo."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _number(value: float) -> str:
    return f"{value:.6g}"


def _maybe_number(value: float | None) -> str | None:
    if value is None:
        return None
    return _number(value)


def _coordinates(
    x_values: list[int],
    y_values: list[float | None],
) -> str:
    points = []
    for x_value, y_value in zip(x_values, y_values, strict=True):
        y_text = _maybe_number(y_value)
        if y_text is None:
            continue
        points.append(f"({_number(float(x_value))},{y_text})")
    return "\n".join(points)


def _write_tex(output_path: Path, body: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{PGFPLOTS_HEADER}\n\\begin{{tikzpicture}}\n{body}\\end{{tikzpicture}}\n",
        encoding="utf-8",
    )


def _axis_begin(options: list[str]) -> str:
    return "\\begin{axis}[\n" + ",\n".join(f"    {option}" for option in options) + "\n]\n"


def _axis_end() -> str:
    return "\\end{axis}\n"


def _base_axis_options(
    title: str,
    ylabel: str,
    *,
    xlabel: str = "Número do segmento",
    ymin: float | None = 0.0,
    legend_columns: int = -1,
) -> list[str]:
    options = [
        f"title={{{_tex_escape(title)}}}",
        f"xlabel={{{_tex_escape(xlabel)}}}",
        f"ylabel={{{_tex_escape(ylabel)}}}",
        f"width={TEX_WIDTH}",
        f"height={TEX_HEIGHT}",
        "grid=major",
        "grid style={line width=.12pt, draw=gray!18}",
        "axis line style={draw=gray!55}",
        "tick style={draw=gray!55}",
        "tick align=outside",
        "title style={font=\\small}",
        "label style={font=\\small}",
        "tick label style={font=\\scriptsize}",
        "legend cell align={left}",
        (
            "legend style={at={(0.5,-0.20)}, anchor=north, "
            f"legend columns={legend_columns}, draw=none, font=\\scriptsize}}"
        ),
    ]
    if ymin is not None:
        options.append(f"ymin={_number(ymin)}")
    return options


def _add_plot(
    options: str,
    coordinates: str,
    legend: str | None = None,
) -> str:
    if not coordinates.strip():
        return ""
    text = f"\\addplot+[{options}] coordinates {{\n{coordinates}\n}};\n"
    if legend is not None:
        text += f"\\addlegendentry{{{_tex_escape(legend)}}}\n"
    return text


def _hline(
    x_values: list[int],
    y_value: float,
    color: str,
    style: str,
    label: str,
) -> str:
    if not x_values:
        return ""
    xmin = min(x_values)
    xmax = max(x_values)
    text = (
        f"\\addplot+[{color}, {style}, {THRESHOLD_STYLE}, forget plot] coordinates "
        f"{{({_number(float(xmin))},{_number(y_value)}) "
        f"({_number(float(xmax))},{_number(y_value)})}};\n"
    )
    if label:
        text += (
            f"\\node[anchor=east, {color}, font=\\scriptsize\\itshape] at "
            f"(axis cs:{_number(float(xmax))},{_number(y_value)}) "
            f"{{{_tex_escape(label)}}};\n"
        )
    return text


def _vlines(
    segments: list[int],
    ymin: float,
    ymax: float,
) -> str:
    lines = []
    for segment in segments:
        lines.append(
            f"\\addplot+[plotred, {EVENT_LINE_STYLE}, forget plot] coordinates "
            f"{{({_number(float(segment))},{_number(ymin)}) "
            f"({_number(float(segment))},{_number(ymax)})}};"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _quality_levels(metrics: list[PlotMetric] | list[RnnDecisionMetric]) -> list[tuple[float, str]]:
    by_bitrate: dict[float, str] = {}
    for metric in metrics:
        by_bitrate.setdefault(metric.bitrate_kbps, metric.quality)
    return sorted(by_bitrate.items())


def _quality_ticks(metrics: list[PlotMetric] | list[RnnDecisionMetric]) -> list[str]:
    levels = _quality_levels(metrics)
    if not levels:
        return []
    ticks = ",".join(_number(bitrate) for bitrate, _quality in levels)
    labels = ",".join(
        f"{{{_tex_escape(quality)}}}"
        for bitrate, quality in levels
    )
    return [f"ytick={{{ticks}}}", f"yticklabels={{{labels}}}"]


def _event_segments(metrics: list[PlotMetric] | list[RnnDecisionMetric], attribute: str) -> list[int]:
    return [
        metric.segment
        for metric in metrics
        if int(getattr(metric, attribute)) == 1
    ]


def _series_max(*series: list[float | None]) -> float:
    values = [
        value
        for items in series
        for value in items
        if value is not None
    ]
    if not values:
        return 1.0
    return max(values) * 1.08


def _write_throughput_quality(
    metrics: list[PlotMetric] | list[RnnDecisionMetric],
    output_path: Path,
    title: str,
) -> None:
    segments = [metric.segment for metric in metrics]
    throughputs = [metric.throughput_kbps for metric in metrics]
    bitrates = [metric.bitrate_kbps for metric in metrics]
    predicted_downloads = [
        getattr(metric, "predicted_download_throughput_kbps", None)
        for metric in metrics
    ]
    ymax = _series_max(throughputs, bitrates, predicted_downloads)
    body = _axis_begin(
        _base_axis_options(title, "Vazão / bitrate (kbps)", legend_columns=3)
        + [f"ymax={_number(ymax)}"]
    )
    body += _add_plot(
        f"plotblue, {LINE_STYLE}, opacity=0.9",
        _coordinates(segments, throughputs),
        "Vazão real do download",
    )
    body += _add_plot(
        f"plotpurple, dashed, {EMPHASIS_LINE_STYLE}",
        _coordinates(segments, predicted_downloads),
        "Vazão real prevista" if any(
            value is not None for value in predicted_downloads
        ) else None,
    )
    body += _add_plot(
        f"plotorange, {STEP_STYLE}",
        _coordinates(segments, bitrates),
        "Representação escolhida",
    )
    body += _vlines(_event_segments(metrics, "failover_event"), 0.0, ymax)
    body += _axis_end()
    _write_tex(output_path, body)


def _write_buffer(
    metrics: list[PlotMetric],
    output_path: Path,
    title: str,
) -> None:
    segments = [metric.segment for metric in metrics]
    buffers = [metric.buffer_level_s for metric in metrics]
    body = _axis_begin(
        _base_axis_options(title, "Buffer (s)", legend_columns=3)
        + [f"ymax={_number(BUFFER_MAX_S + 1.0)}"]
    )
    body += _add_plot(
        f"plotgreen, {EMPHASIS_LINE_STYLE}",
        _coordinates(segments, buffers),
        "Buffer",
    )
    body += _hline(segments, BUFFER_TARGET_S, "plotpurple", "dashed", f"target {BUFFER_TARGET_S:g}s")
    body += _hline(segments, BUFFER_MAX_S, "plotgray", "dotted", f"max {BUFFER_MAX_S:g}s")
    rebuffer_metrics = [metric for metric in metrics if metric.rebuffer_event]
    body += _add_plot(
        REBUFFER_STYLE,
        _coordinates(
            [metric.segment for metric in rebuffer_metrics],
            [metric.buffer_level_s for metric in rebuffer_metrics],
        ),
        "Rebuffering" if rebuffer_metrics else None,
    )
    body += _vlines(_event_segments(metrics, "failover_event"), 0.0, BUFFER_MAX_S + 1.0)
    body += _axis_end()
    _write_tex(output_path, body)


def _write_jitter(
    metrics: list[PlotMetric],
    output_path: Path,
    title: str,
) -> None:
    segments = [metric.segment for metric in metrics]
    jitters = [metric.jitter_ewma_ms for metric in metrics]
    ymax = _series_max(jitters)
    body = _axis_begin(
        _base_axis_options(title, "Jitter EWMA (ms)", legend_columns=2)
        + [f"ymax={_number(ymax)}"]
    )
    body += _add_plot(
        f"plotpurple, {EMPHASIS_LINE_STYLE}",
        _coordinates(segments, jitters),
        "Jitter EWMA",
    )
    body += _vlines(_event_segments(metrics, "failover_event"), 0.0, ymax)
    body += _axis_end()
    _write_tex(output_path, body)


def _write_comparison(
    datasets: list[tuple[str, list[PlotMetric]]],
    output_path: Path,
    title: str,
    ylabel: str,
    value: Callable[[PlotMetric], float],
    *,
    use_steps: bool = False,
    quality_ticks: bool = False,
    show_buffer_thresholds: bool = False,
    show_rebuffers: bool = False,
) -> None:
    all_values = [
        value(metric)
        for _label, metrics in datasets
        for metric in metrics
    ]
    ymax = (max(all_values) * 1.08) if all_values else 1.0
    options = _base_axis_options(title, ylabel, legend_columns=min(3, len(datasets)))
    options.append(f"ymax={_number(ymax)}")
    if quality_ticks:
        all_metrics = [
            metric
            for _label, metrics in datasets
            for metric in metrics
        ]
        options.extend(_quality_ticks(all_metrics))
    body = _axis_begin(options)
    for index, (label, metrics) in enumerate(datasets):
        style = STEP_STYLE if use_steps else LINE_STYLE
        body += _add_plot(
            f"{POLICY_COLORS[index % len(POLICY_COLORS)]}, {style}",
            _coordinates(
                [metric.segment for metric in metrics],
                [value(metric) for metric in metrics],
            ),
            label,
        )
        if show_rebuffers:
            rebuffer_metrics = [metric for metric in metrics if metric.rebuffer_event]
            body += _add_plot(
                f"{REBUFFER_STYLE}, forget plot",
                _coordinates(
                    [metric.segment for metric in rebuffer_metrics],
                    [value(metric) for metric in rebuffer_metrics],
                ),
            )
    if show_buffer_thresholds:
        segments = [
            metric.segment
            for _label, metrics in datasets
            for metric in metrics
        ]
        body += _hline(segments, BUFFER_TARGET_S, "plotpurple", "dashed", f"target {BUFFER_TARGET_S:g}s")
        body += _hline(segments, BUFFER_MAX_S, "plotgray", "dotted", f"max {BUFFER_MAX_S:g}s")
    body += _axis_end()
    _write_tex(output_path, body)


def _csv_has_columns(csv_path: Path, columns: set[str]) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return columns.issubset(set(reader.fieldnames or []))


def _predicted_target(metric: RnnDecisionMetric) -> float | None:
    if (
        metric.predicted_a_throughput_kbps is None
        or metric.predicted_b_throughput_kbps is None
    ):
        return None
    if metric.predicted_a_throughput_kbps >= metric.predicted_b_throughput_kbps:
        return metric.probe_a_throughput_kbps
    return metric.probe_b_throughput_kbps


def _error_summary(predicted: list[float | None], actual: list[float | None]) -> str | None:
    errors = [
        predicted_value - actual_value
        for predicted_value, actual_value in zip(predicted, actual, strict=True)
        if predicted_value is not None and actual_value is not None
    ]
    if not errors:
        return None
    mae = sum(abs(error) for error in errors) / len(errors)
    bias = sum(errors) / len(errors)
    return f"MAE={mae:.1f} kbps | bias={bias:.1f} kbps"


def _summary_node(summary: str | None, x: int, y: float) -> str:
    if summary is None:
        return ""
    return (
        f"\\node[anchor=north east, font=\\scriptsize, plotgray] at "
        f"(axis cs:{_number(float(x))},{_number(y)}) "
        f"{{{_tex_escape(summary)}}};\n"
    )


def _write_probe_prediction(
    metrics: list[RnnDecisionMetric],
    output_path: Path,
    server_label: str,
    color: str,
    actual: list[float | None],
    predicted: list[float | None],
) -> None:
    segments = [metric.segment for metric in metrics]
    ymax = _series_max(actual, predicted)
    body = _axis_begin(
        _base_axis_options(
            f"Política 3: probe {server_label} real vs previsão",
            "Vazão do probe (kbps)",
            legend_columns=2,
        )
        + [f"ymax={_number(ymax)}"]
    )
    body += _add_plot(
        f"{color}, {LINE_STYLE}, opacity=0.72",
        _coordinates(segments, actual),
        f"Probe {server_label} real",
    )
    body += _add_plot(
        f"{color}, dashed, {EMPHASIS_LINE_STYLE}",
        _coordinates(segments, predicted),
        f"Probe {server_label} previsto",
    )
    body += _summary_node(_error_summary(predicted, actual), max(segments), ymax)
    body += _vlines(_event_segments(metrics, "failover_event"), 0.0, ymax)
    body += _axis_end()
    _write_tex(output_path, body)


def _write_rnn_prediction_error(
    metrics: list[RnnDecisionMetric],
    output_path: Path,
) -> None:
    error_points = [
        (metric, target)
        for metric in metrics
        if metric.predicted_selected_throughput_kbps is not None
        for target in [_predicted_target(metric)]
        if target is not None
    ]
    segments = [metric.segment for metric, _target in error_points]
    errors = [
        metric.predicted_selected_throughput_kbps - target
        for metric, target in error_points
        if metric.predicted_selected_throughput_kbps is not None
    ]
    max_abs = max([abs(error) for error in errors], default=1.0) * 1.12
    body = _axis_begin(
        _base_axis_options(
            "Política 3: erro da previsão usada contra o probe alvo",
            "Erro (kbps)",
            ymin=-max_abs,
            legend_columns=1,
        )
        + [f"ymax={_number(max_abs)}"]
    )
    body += _add_plot(
        "ybar, bar width=1.1pt, mark=none, fill=plotgreen!70, draw=plotgreen!70",
        _coordinates(
            segments,
            [error if error < 0 else None for error in errors],
        ),
    )
    body += _add_plot(
        "ybar, bar width=1.1pt, mark=none, fill=plotred!70, draw=plotred!70",
        _coordinates(
            segments,
            [error if error >= 0 else None for error in errors],
        ),
    )
    body += _hline(segments, 0.0, "plotgray", "solid", "")
    if errors:
        mae = sum(abs(error) for error in errors) / len(errors)
        bias = sum(errors) / len(errors)
        body += _summary_node(
            f"MAE={mae:.1f} kbps | bias={bias:.1f} kbps",
            max(segments),
            max_abs,
        )
    body += _vlines(_event_segments(metrics, "failover_event"), -max_abs, max_abs)
    body += _axis_end()
    _write_tex(output_path, body)


def _write_server_buffer(
    metrics: list[RnnDecisionMetric],
    output_path: Path,
    title: str,
) -> None:
    segments = [metric.segment for metric in metrics]
    server_ids = list(dict.fromkeys(metric.server_id for metric in metrics))
    server_index = {server_id: index for index, server_id in enumerate(server_ids)}
    ytick = ",".join(str(index) for index in range(len(server_ids)))
    yticklabels = ",".join(f"{{{_tex_escape(server_id)}}}" for server_id in server_ids)

    server_axis_options = _base_axis_options(
        title,
        "Servidor",
        legend_columns=2,
        ymin=None,
    ) + [
        "axis y line*=left",
        "axis x line*=bottom",
        f"ytick={{{ytick}}}",
        f"yticklabels={{{yticklabels}}}",
        "ymin=-0.1",
        f"ymax={_number(max(len(server_ids) - 1, 1) + 0.1)}",
    ]
    body = _axis_begin(server_axis_options)
    body += _add_plot(
        f"plotslate, {STEP_STYLE}",
        _coordinates(
            segments,
            [float(server_index[metric.server_id]) for metric in metrics],
        ),
        "Servidor usado",
    )
    body += _vlines(_event_segments(metrics, "failover_event"), -0.1, max(len(server_ids) - 1, 1) + 0.1)
    body += _axis_end()

    buffer_options = [
        f"width={TEX_WIDTH}",
        f"height={TEX_HEIGHT}",
        "axis y line*=right",
        "axis x line=none",
        "ylabel={Buffer (s)}",
        "ymin=0",
        f"ymax={_number(BUFFER_MAX_S + 1.0)}",
        "axis line style={draw=gray!55}",
        "tick style={draw=gray!55}",
        "label style={font=\\small}",
        "tick label style={font=\\scriptsize}",
        "legend cell align={left}",
        "legend style={at={(0.5,-0.32)}, anchor=north, legend columns=3, draw=none, font=\\scriptsize}",
    ]
    body += _axis_begin(buffer_options)
    body += _add_plot(
        f"plotgreen, {EMPHASIS_LINE_STYLE}",
        _coordinates(segments, [metric.buffer_level_s for metric in metrics]),
        "Buffer",
    )
    body += _hline(segments, BUFFER_TARGET_S, "plotpurple", "dashed", f"target {BUFFER_TARGET_S:g}s")
    body += _hline(segments, BUFFER_MAX_S, "plotgray", "dotted", f"max {BUFFER_MAX_S:g}s")
    rebuffers = [metric for metric in metrics if metric.rebuffer_event]
    body += _add_plot(
        REBUFFER_STYLE,
        _coordinates(
            [metric.segment for metric in rebuffers],
            [metric.buffer_level_s for metric in rebuffers],
        ),
        "Rebuffering" if rebuffers else None,
    )
    body += _axis_end()
    _write_tex(output_path, body)


def _rnn_output_paths(output_dir: Path) -> dict[str, Path]:
    base = output_dir / "policy3_rnn_decisions.tex"
    return {
        "throughput_quality": base,
        "probe_a": output_dir / "policy3_rnn_decisions_probe_a.tex",
        "probe_b": output_dir / "policy3_rnn_decisions_probe_b.tex",
        "prediction_error": output_dir / "policy3_rnn_decisions_prediction_error.tex",
        "server_buffer": output_dir / "policy3_rnn_decisions_server_buffer.tex",
    }


def generate_rnn_tex_plots(csv_path: Path, output_dir: Path) -> list[Path]:
    """Gera os cinco graficos da Politica 3/RNN em arquivos separados."""
    metrics = read_rnn_decision_metrics(csv_path)
    paths = _rnn_output_paths(output_dir)
    generated: list[Path] = []

    _write_throughput_quality(
        metrics,
        paths["throughput_quality"],
        "Política 3: vazão real, previsão e representação",
    )
    generated.append(paths["throughput_quality"])

    _write_probe_prediction(
        metrics,
        paths["probe_a"],
        "A",
        "plotteal",
        [metric.probe_a_throughput_kbps for metric in metrics],
        [metric.predicted_a_throughput_kbps for metric in metrics],
    )
    generated.append(paths["probe_a"])

    _write_probe_prediction(
        metrics,
        paths["probe_b"],
        "B",
        "plotpurple",
        [metric.probe_b_throughput_kbps for metric in metrics],
        [metric.predicted_b_throughput_kbps for metric in metrics],
    )
    generated.append(paths["probe_b"])

    _write_rnn_prediction_error(metrics, paths["prediction_error"])
    generated.append(paths["prediction_error"])

    _write_server_buffer(
        metrics,
        paths["server_buffer"],
        "Política 3: servidor escolhido, buffer e eventos",
    )
    generated.append(paths["server_buffer"])

    return generated


def generate_policy_tex_plots(
    slug: str,
    label: str,
    csv_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Gera graficos individuais de uma politica em TeX."""
    metrics = read_plot_metrics(csv_path)
    paths = [
        output_dir / f"{slug}_throughput_quality.tex",
        output_dir / f"{slug}_buffer.tex",
        output_dir / f"{slug}_jitter_ewma.tex",
    ]
    _write_throughput_quality(
        metrics,
        paths[0],
        f"{label}: vazão e qualidade selecionada",
    )
    _write_buffer(metrics, paths[1], f"{label}: nível do buffer")
    _write_jitter(metrics, paths[2], f"{label}: jitter EWMA")
    return paths


def generate_comparison_tex_plots(
    policy_csvs: list[tuple[str, str, Path]],
    output_dir: Path,
) -> list[Path]:
    """Gera comparativos das politicas, um grafico por arquivo."""
    datasets = [
        (label, read_plot_metrics(csv_path))
        for _slug, label, csv_path in policy_csvs
    ]
    plots = [
        (
            output_dir / "compare_policies_quality.tex",
            "Qualidade selecionada: comparação das políticas",
            "Bitrate selecionado (kbps)",
            lambda metric: metric.bitrate_kbps,
            True,
            True,
            False,
            False,
        ),
        (
            output_dir / "compare_policies_buffer.tex",
            "Nível do buffer: comparação das políticas",
            "Nível do buffer (s)",
            lambda metric: metric.buffer_level_s,
            False,
            False,
            True,
            True,
        ),
        (
            output_dir / "compare_policies_throughput.tex",
            "Vazão medida: comparação das políticas",
            "Vazão medida (kbps)",
            lambda metric: metric.throughput_kbps,
            False,
            False,
            False,
            False,
        ),
        (
            output_dir / "compare_policies_jitter_ewma.tex",
            "Jitter EWMA: comparação das políticas",
            "Jitter EWMA (ms)",
            lambda metric: metric.jitter_ewma_ms,
            False,
            False,
            False,
            False,
        ),
    ]
    generated: list[Path] = []
    for (
        path,
        title,
        ylabel,
        value,
        steps,
        ticks,
        thresholds,
        rebuffers,
    ) in plots:
        _write_comparison(
            datasets,
            path,
            title,
            ylabel,
            value,
            use_steps=steps,
            quality_ticks=ticks,
            show_buffer_thresholds=thresholds,
            show_rebuffers=rebuffers,
        )
        generated.append(path)
    return generated


def generate_policy2_failover_tex_plots(
    csv_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Gera o experimento de failover da Politica 2 em graficos separados."""
    if not csv_path.exists():
        return []
    metrics = read_plot_metrics(csv_path)
    generated: list[Path] = []

    throughput_path = output_dir / "policy2_failover_experiment_throughput_quality.tex"
    _write_throughput_quality(
        metrics,
        throughput_path,
        "Política 2: vazão e qualidade durante o failover",
    )
    generated.append(throughput_path)

    buffer_path = output_dir / "policy2_failover_experiment_buffer.tex"
    _write_buffer(metrics, buffer_path, "Política 2: nível do buffer durante o failover")
    generated.append(buffer_path)

    server_path = output_dir / "policy2_failover_experiment_server.tex"
    segments = [metric.segment for metric in metrics]
    server_ids = list(dict.fromkeys(metric.server_id for metric in metrics))
    server_index = {server_id: index for index, server_id in enumerate(server_ids)}
    ytick = ",".join(str(index) for index in range(len(server_ids)))
    yticklabels = ",".join(f"{{{_tex_escape(server_id)}}}" for server_id in server_ids)
    body = _axis_begin(
        _base_axis_options(
            "Política 2: servidor usado por segmento",
            "Servidor",
            legend_columns=1,
            ymin=None,
        )
        + [
            f"ytick={{{ytick}}}",
            f"yticklabels={{{yticklabels}}}",
            "ymin=-0.1",
            f"ymax={_number(max(len(server_ids) - 1, 1) + 0.1)}",
        ]
    )
    body += _add_plot(
        f"plotslate, {STEP_STYLE}",
        _coordinates(
            segments,
            [float(server_index[metric.server_id]) for metric in metrics],
        ),
        "Servidor efetivo",
    )
    body += _vlines(_event_segments(metrics, "failover_event"), -0.1, max(len(server_ids) - 1, 1) + 0.1)
    body += _axis_end()
    _write_tex(server_path, body)
    generated.append(server_path)

    return generated


def generate_all_tex_plots(
    policy_csvs: list[tuple[str, str, Path]],
    policy2_failover_csv: Path | None,
    output_dir: Path,
) -> list[Path]:
    """Gera todos os graficos finais em TeX, sempre um grafico por arquivo."""
    generated: list[Path] = []
    for slug, label, csv_path in policy_csvs:
        generated.extend(generate_policy_tex_plots(slug, label, csv_path, output_dir))

    generated.extend(generate_comparison_tex_plots(policy_csvs, output_dir))

    policy3_candidates = [
        csv_path
        for slug, _label, csv_path in policy_csvs
        if slug == "policy3" and _csv_has_columns(csv_path, RNN_DECISION_COLUMNS)
    ]
    if policy3_candidates:
        generated.extend(generate_rnn_tex_plots(policy3_candidates[0], output_dir))

    if policy2_failover_csv is not None:
        generated.extend(
            generate_policy2_failover_tex_plots(policy2_failover_csv, output_dir)
        )

    return generated
