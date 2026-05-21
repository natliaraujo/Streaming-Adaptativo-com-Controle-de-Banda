"""
Gera gráficos da política baseline a partir do CSV de métricas.

O gráfico principal cruza duas informações centrais do cliente ABR: a vazão
medida durante o download de cada segmento e a qualidade escolhida para o
segmento. Como a qualidade é categórica (240p, 360p, 720p etc.), ela é
representada no eixo em kbps pelo bitrate nominal da representação escolhida.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV_PATH: Path = Path(__file__).with_name("metricas_baseline.csv")
DEFAULT_OUTPUT_PATH: Path = Path(__file__).with_name("grafico_vazao_qualidade.png")


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


@dataclass(frozen=True)
class CliArgs:
    """
    Agrupa os argumentos de linha de comando já convertidos para Path.

    Separar os argumentos em uma dataclass evita espalhar `argparse.Namespace`
    pelo restante do código, deixando as funções seguintes mais fáceis de
    tipar e testar.
    """

    csv_path: Path
    output_path: Path


def parse_args() -> CliArgs:
    """
    Lê os argumentos de linha de comando do gerador de gráficos.

    Se nenhum argumento for informado, usa o CSV e o PNG padrão na mesma pasta
    deste módulo. Isso permite rodar o script diretamente após executar o
    cliente baseline.

    Returns:
        Caminhos do CSV de entrada e do arquivo PNG de saída.
    """
    parser = argparse.ArgumentParser(
        description="Gera grafico de vazao e qualidade ao longo dos segmentos."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Caminho do CSV de metricas. Padrao: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Caminho do PNG gerado. Padrao: {DEFAULT_OUTPUT_PATH}",
    )
    namespace = parser.parse_args()
    return CliArgs(csv_path=namespace.csv, output_path=namespace.output)


def read_metrics(csv_path: Path) -> list[SegmentMetric]:
    """
    Carrega do CSV as métricas necessárias para plotar vazão e qualidade.

    O CSV é tratado como a fonte de verdade dos experimentos. Esta função faz
    uma validação mínima das colunas obrigatórias antes de converter os valores
    para tipos numéricos usados pelo matplotlib.

    Args:
        csv_path: Caminho do arquivo CSV gerado pelo cliente baseline.

    Returns:
        Lista de métricas, uma por segmento.

    Raises:
        ValueError: Se o CSV estiver vazio ou não tiver as colunas esperadas.
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


def choose_throughput_column(fieldnames: list[str]) -> str:
    """
    Seleciona a coluna do CSV que contém a vazão medida.

    O cliente atual escreve a coluna como `vazao_kbps`, enquanto o README
    também menciona `throughput_kbps`. Esta função aceita os dois nomes para
    manter compatibilidade com ambos os formatos.

    Args:
        fieldnames: Lista de nomes de colunas presentes no CSV.

    Returns:
        Nome da coluna de vazão encontrada.

    Raises:
        ValueError: Se nenhuma coluna de vazão conhecida existir no CSV.
    """
    for column in ("vazao_kbps", "throughput_kbps"):
        if column in fieldnames:
            return column
    raise ValueError("CSV sem coluna de vazao: esperado 'vazao_kbps' ou 'throughput_kbps'")


def plot_throughput_and_quality(metrics: list[SegmentMetric], output_path: Path) -> None:
    """
    Gera um gráfico PNG com vazão medida e qualidade escolhida por segmento.

    A linha azul mostra a vazão observada pelo cliente em cada download. A
    linha verde mostra a decisão do algoritmo ABR, usando o bitrate nominal da
    qualidade escolhida para permitir comparação no mesmo eixo. Os rótulos
    textuais (`240p`, `720p` etc.) são adicionados sobre a linha verde para
    facilitar a leitura na apresentação e no relatório.

    Args:
        metrics: Métricas carregadas do CSV.
        output_path: Caminho onde o arquivo PNG será salvo.

    Raises:
        SystemExit: Se o matplotlib não estiver instalado.
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
        color="#1565c0",
        marker="o",
        linewidth=2,
        label="Vazao medida",
    )
    ax.step(
        segments,
        bitrates,
        where="mid",
        color="#2e7d32",
        linewidth=2,
        label="Qualidade escolhida",
    )
    ax.scatter(segments, bitrates, color="#2e7d32", s=32, zorder=3)

    for metric in metrics:
        ax.annotate(
            metric.quality,
            (metric.segment, metric.bitrate_kbps),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#1b5e20",
        )

    ax.set_title("Vazao e Qualidade por Segmento")
    ax.set_xlabel("Segmento")
    ax.set_ylabel("kbps")
    ax.set_xticks(segments)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """
    Executa o fluxo completo via linha de comando.

    O fluxo lê argumentos, carrega o CSV, gera o gráfico e informa o caminho do
    arquivo criado. Essa função existe para manter o módulo utilizável tanto
    como script quanto como biblioteca importável por outros experimentos.
    """
    args = parse_args()
    metrics = read_metrics(args.csv_path)
    plot_throughput_and_quality(metrics, args.output_path)
    print(f"Grafico gerado: {args.output_path}")


if __name__ == "__main__":
    main()
