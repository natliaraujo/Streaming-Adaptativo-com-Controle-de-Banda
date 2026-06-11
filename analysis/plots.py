"""
Funções para gerar gráficos a partir dos CSVs de métricas do cliente ABR.

Este módulo não executa experimentos nem define argumentos de linha de comando.
Ele apenas lê métricas e gera arquivos de imagem.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentMetric:
    """
    Representa as métricas de um segmento usadas no gráfico.

    Attributes:
        segment: Número sequencial do segmento baixado.
        quality: Rótulo textual da qualidade escolhida, como `240p` ou `720p`.
        bitrate_kbps: Bitrate nominal da representação escolhida.
        throughput_kbps: Vazão medida durante o download do segmento.
    """

    segment: int
    quality: str
    bitrate_kbps: float
    throughput_kbps: float


def choose_throughput_column(fieldnames: list[str]) -> str:
    """
    Seleciona a coluna do CSV que contém a vazão medida.

    O cliente atual escreve a coluna como `vazao_kbps`, enquanto outros módulos
    podem usar `throughput_kbps`. Esta função aceita ambos os nomes.
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

    Raises:
        ValueError: Se o CSV não possuir colunas obrigatórias ou não tiver dados.
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

    Raises:
        SystemExit: Se `matplotlib` não estiver instalado no ambiente.
    """

    try:
        import matplotlib.pyplot as plt
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

    ax.plot(
        segments,
        throughputs,
        marker="o",
        linewidth=2,
        label="Vazao medida",
    )

    ax.step(
        segments,
        bitrates,
        where="mid",
        linewidth=2,
        label="Qualidade escolhida",
    )

    ax.scatter(segments, bitrates, s=32, zorder=3)

    for metric in metrics:
        ax.annotate(
            metric.quality,
            (metric.segment, metric.bitrate_kbps),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.set_title("Vazao e qualidade por segmento")
    ax.set_xlabel("Segmento")
    ax.set_ylabel("kbps")
    ax.set_xticks(segments)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_throughput_quality_plot(
    csv_path: Path,
    output_path: Path,
) -> None:
    """
    Fluxo reutilizável para gerar o gráfico de vazão e qualidade.

    Lê as métricas do CSV informado e salva a imagem no caminho de saída.
    """

    metrics = read_metrics(csv_path)
    plot_throughput_and_quality(metrics, output_path)
