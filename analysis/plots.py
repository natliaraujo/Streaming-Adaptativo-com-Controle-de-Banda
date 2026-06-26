"""
Gera gráficos a partir dos CSVs de métricas dos experimentos.

Este módulo contém funções reutilizáveis para visualizar vazão, qualidade
escolhida, nível do buffer, eventos de rebuffering, jitter e uso dos servidores
ao longo dos segmentos.

Ele não executa experimentos; apenas realiza pós-processamento dos arquivos CSV
gerados pelo pacote `experiment`.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentMetric:
    """
    Representa as métricas de um segmento usadas no gráfico.

    Attributes:
        - `segment`: Número sequencial do segmento baixado.
        - `quality`: Rótulo textual da qualidade escolhida, como `240p` ou `720p`.
        - `bitrate_kbps`: Bitrate nominal da representação escolhida.
        - `throughput_kbps`: Vazão medida durante o download do segmento.
    """

    segment: int
    quality: str
    bitrate_kbps: float
    throughput_kbps: float
    buffer_level_s: float
    failover_event: int
    rebuffer_event: int


def quality_levels(metrics: list[SegmentMetric]) -> list[tuple[float, str]]:
    """Retorna níveis únicos de qualidade ordenados por bitrate."""
    by_bitrate: dict[float, str] = {}
    for metric in metrics:
        by_bitrate.setdefault(metric.bitrate_kbps, metric.quality)
    return sorted(by_bitrate.items())


def set_quality_ticks(axis, metrics: list[SegmentMetric]) -> None:
    """Mostra as representações no eixo, sem rótulos repetidos na série."""
    levels = quality_levels(metrics)
    if not levels:
        return
    axis.set_yticks([bitrate for bitrate, _quality in levels])
    axis.set_yticklabels(
        [
            f"{quality}\n{bitrate:g} kbps"
            for bitrate, quality in levels
        ],
        fontsize=8,
    )


def markevery(total_points: int, target_markers: int = 28) -> int:
    """Espaça marcadores em séries longas."""
    return max(1, total_points // target_markers)


def dedup_legend(handles, labels):
    """Remove entradas repetidas de legenda preservando a ordem."""
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels, strict=True):
        if not label or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    return unique_handles, unique_labels


def choose_throughput_column(fieldnames: list[str]) -> str:
    """
    Seleciona a coluna do CSV que contém a vazão medida.

    O cliente atual escreve a coluna como `vazao_kbps`, enquanto outros módulos
    podem usar `throughput_kbps`. Esta função aceita ambos os nomes.

    Args:
        fieldnames: Colunas disponíveis no cabeçalho do CSV.

    Returns:
        Nome da primeira coluna de vazão reconhecida.

    Raises:
        ValueError: Se nenhuma coluna compatível estiver presente.
    """

    for column in ("vazao_kbps", "throughput_kbps"):
        if column in fieldnames:
            return column

    raise ValueError(
        "CSV sem coluna de vazao: esperado 'vazao_kbps' ou 'throughput_kbps'"
    )


def read_metrics(csv_path: Path) -> list[SegmentMetric]:
    """
    Carrega do CSV as métricas necessárias para plotar vazão e qualidade.

    Campos novos de buffer e eventos aceitam zero como fallback para manter
    compatibilidade com resultados antigos do projeto.

    Args:
        csv_path: Arquivo de métricas produzido por uma política.

    Returns:
        Métricas na ordem original dos segmentos.

    Raises:
        ValueError: Se o CSV não possuir colunas obrigatórias ou estiver vazio.
    """

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])

        required_columns = {"segment", "quality", "bitrate_kbps"}
        missing_columns = required_columns - set(fieldnames)

        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV sem colunas obrigatorias: {columns}")

        throughput_column = choose_throughput_column(fieldnames)

        metrics: list[SegmentMetric] = []

        for row in reader:
            metrics.append(
                SegmentMetric(
                    segment=int(row["segment"]),
                    quality=row["quality"],
                    bitrate_kbps=float(row["bitrate_kbps"]),
                    throughput_kbps=float(row[throughput_column]),
                    buffer_level_s=float(row.get("buffer_level_s") or 0.0),
                    failover_event=int(float(row.get("failover_event") or 0)),
                    rebuffer_event=int(float(row.get("rebuffer_event") or 0)),
                )
            )

    if not metrics:
        raise ValueError(f"CSV sem dados: {csv_path}")

    return metrics


def plot_throughput_and_quality(
    metrics: list[SegmentMetric],
    output_path: Path,
) -> None:
    """
    Gera um gráfico PNG com vazão medida e qualidade escolhida por segmento.

    Args:
        metrics: Amostras já carregadas e ordenadas por segmento.
        output_path: Destino do arquivo PNG.

    Raises:
        SystemExit: Se ``matplotlib`` não estiver instalado no ambiente.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia ausente: instale matplotlib para gerar os graficos "
            "(`python3 -m pip install matplotlib`)."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = [metric.segment for metric in metrics]
    throughputs = [metric.throughput_kbps for metric in metrics]
    bitrates = [metric.bitrate_kbps for metric in metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    bitrate_axis = ax.twinx()

    ax.plot(
        segments,
        throughputs,
        marker="o",
        markersize=3,
        markevery=markevery(len(segments)),
        linewidth=1.8,
        label="Vazao medida",
    )

    bitrate_axis.step(
        segments,
        bitrates,
        where="mid",
        linewidth=2,
        color="#d97706",
        label="Bitrate selecionado",
    )
    set_quality_ticks(bitrate_axis, metrics)

    ax.set_title("Vazao e qualidade por segmento")
    ax.set_xlabel("Segmento")
    ax.set_ylabel("Vazao medida (kbps)")
    bitrate_axis.set_ylabel("Representacao selecionada")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = bitrate_axis.get_legend_handles_labels()
    handles, labels = dedup_legend(lines1 + lines2, labels1 + labels2)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_throughput_quality_plot(
    csv_path: Path,
    output_path: Path,
) -> None:
    """
    Fluxo reutilizável para gerar o gráfico de vazão e qualidade.

    Lê as métricas do CSV informado e salva a imagem no caminho de saída.

    Args:
        csv_path: CSV de uma política.
        output_path: Destino do gráfico PNG.
    """

    metrics = read_metrics(csv_path)
    plot_throughput_and_quality(metrics, output_path)


def generate_policy_comparison_plot(
    policy1_csv: Path,
    policy2_csv: Path,
    output_path: Path,
    policy3_csv: Path | None = None,
) -> None:
    """Compara bitrate, throughput e buffer das políticas disponíveis.

    A função sobrepõe as séries de cada política em três painéis compartilhados.
    Failovers são marcados no painel de bitrate e rebuffers no painel de buffer.

    Args:
        policy1_csv: CSV da política rate-based.
        policy2_csv: CSV da política buffer-aware.
        output_path: Destino do PNG comparativo.
        policy3_csv: CSV opcional da política RNN.

    Raises:
        SystemExit: Se ``matplotlib`` não estiver instalado.
        ValueError: Se algum CSV estiver vazio ou tiver esquema incompatível.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependência ausente: instale matplotlib para gerar os gráficos."
        ) from exc

    datasets = [
        ("Política 1", read_metrics(policy1_csv), "#2563eb"),
        ("Política 2", read_metrics(policy2_csv), "#d97706"),
    ]
    if policy3_csv is not None:
        datasets.append(
            ("Política 3 (RNN)", read_metrics(policy3_csv), "#059669")
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for label, metrics, color in datasets:
        segments = [metric.segment for metric in metrics]
        axes[0].step(
            segments,
            [metric.bitrate_kbps for metric in metrics],
            where="mid",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[1].plot(
            segments,
            [metric.throughput_kbps for metric in metrics],
            linewidth=1.8,
            marker="o",
            markersize=3,
            markevery=markevery(len(segments)),
            label=label,
            color=color,
        )
        axes[2].plot(
            segments,
            [metric.buffer_level_s for metric in metrics],
            linewidth=2,
            label=label,
            color=color,
        )

        for metric in metrics:
            if metric.failover_event:
                axes[0].axvline(
                    metric.segment,
                    color="#dc2626",
                    linestyle="--",
                    alpha=0.45,
                )
            if metric.rebuffer_event:
                axes[2].scatter(
                    metric.segment,
                    metric.buffer_level_s,
                    color="#111827",
                    marker="x",
                    s=45,
                    zorder=4,
                )

    axes[0].set_title("Qualidade selecionada")
    axes[0].set_ylabel("Representação")
    axes[1].set_title("Vazão medida")
    axes[1].set_ylabel("Throughput (kbps)")
    axes[2].set_title("Nível do buffer")
    axes[2].set_ylabel("Buffer (s)")
    axes[2].set_xlabel("Segmento")
    all_metrics = [
        metric
        for _label, metrics, _color in datasets
        for metric in metrics
    ]
    set_quality_ticks(axes[0], all_metrics)

    for axis in axes:
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = dedup_legend(handles, labels)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(3, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_quality_buffer_comparison_plot(
    policy_csvs: list[tuple[str, Path]],
    output_path: Path,
) -> None:
    """Compara bitrate escolhido e buffer disponível em cada política.

    Cada política recebe um painel próprio com dois eixos verticais: bitrate à
    esquerda e buffer à direita. Os painéis compartilham a escala de segmentos e
    exibem os limiares mínimo e confortável para relacionar cada decisão à reserva
    disponível naquela amostra.

    Args:
        policy_csvs: Pares de rótulo de exibição e caminho do CSV.
        output_path: Destino do gráfico PNG.

    Raises:
        SystemExit: Se ``matplotlib`` não estiver instalado.
        ValueError: Se nenhuma política for informada ou algum CSV for inválido.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependência ausente: instale matplotlib para gerar os gráficos."
        ) from exc

    from config import BUFFER_MAX_S, BUFFER_MIN_S, BUFFER_TARGET_S

    datasets = [
        (label, read_metrics(csv_path))
        for label, csv_path in policy_csvs
    ]
    if not datasets:
        raise ValueError("Nenhuma política foi informada para o gráfico.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(13, 4 * len(datasets)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    quality_colors = ["#2563eb", "#d97706", "#059669"]

    for index, (label, metrics) in enumerate(datasets):
        quality_axis = axes[index][0]
        buffer_axis = quality_axis.twinx()
        segments = [metric.segment for metric in metrics]

        quality_axis.step(
            segments,
            [metric.bitrate_kbps for metric in metrics],
            where="mid",
            linewidth=2.2,
            color=quality_colors[index % len(quality_colors)],
            label="Bitrate escolhido",
        )
        buffer_axis.plot(
            segments,
            [metric.buffer_level_s for metric in metrics],
            linewidth=1.8,
            color="#374151",
            alpha=0.85,
            label="Buffer disponível",
        )
        buffer_axis.axhline(
            BUFFER_MIN_S,
            color="#dc2626",
            linestyle=":",
            alpha=0.55,
            label="Buffer mínimo",
        )
        buffer_axis.axhline(
            BUFFER_TARGET_S,
            color="#7c3aed",
            linestyle="--",
            alpha=0.55,
            label="Buffer confortável",
        )

        quality_axis.set_title(label)
        quality_axis.set_ylabel("Representação")
        set_quality_ticks(quality_axis, metrics)
        quality_axis.grid(True, axis="y", linestyle="--", alpha=0.25)
        quality_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        buffer_axis.set_ylabel("Buffer (s)")
        buffer_axis.set_ylim(0.0, BUFFER_MAX_S + 1.0)

    axes[-1][0].set_xlabel("Segmento")
    legend_handles = []
    legend_labels = []
    for axis in fig.axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        legend_handles.extend(axis_handles)
        legend_labels.extend(axis_labels)
    legend_handles, legend_labels = dedup_legend(legend_handles, legend_labels)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(legend_labels)),
        frameon=False,
    )
    fig.suptitle("Qualidade escolhida e buffer por política", fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
