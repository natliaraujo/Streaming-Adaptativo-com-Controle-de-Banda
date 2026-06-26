"""Valida o checkpoint salvo da RNN contra um CSV de métricas."""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.checkpoint import load_rnn_checkpoint  # noqa: E402
from models.dataset import (  # noqa: E402
    RnnSample,
    load_rnn_samples_from_csv,
    split_samples,
)


@dataclass(frozen=True)
class PredictionRow:
    """Previsão e alvo de uma amostra temporal."""

    index: int
    predicted_a: float
    predicted_b: float
    target_a: float
    target_b: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a acurácia das previsões do modelo RNN salvo."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rnn_training_data.csv",
        help="CSV usado para validação.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "rnn_policy.pt",
        help="Checkpoint treinado a ser avaliado.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "train", "all"),
        default="validation",
        help="Parte do CSV usada na avaliação.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Mesma fração usada no treinamento para separar treino/validação.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="CSV opcional com previsão, alvo e erro por amostra.",
    )
    return parser.parse_args()


def select_split(
    samples: list[RnnSample],
    split: str,
    train_ratio: float,
) -> list[RnnSample]:
    """Seleciona a partição de amostras que será avaliada."""
    if split == "all":
        return samples

    train_samples, val_samples = split_samples(
        samples=samples,
        train_ratio=train_ratio,
    )
    if split == "train":
        return train_samples
    return val_samples


def predict_samples(
    samples: list[RnnSample],
    model: torch.nn.Module,
    normalizer,
) -> list[PredictionRow]:
    """Executa inferência do modelo para todas as amostras."""
    rows: list[PredictionRow] = []
    model.eval()

    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            normalized_sequence = normalizer.normalize_sequence(sample.x)
            x = torch.tensor(
                [normalized_sequence],
                dtype=torch.float32,
            )
            prediction = model(x)[0]
            rows.append(
                PredictionRow(
                    index=index,
                    predicted_a=float(prediction[0].item()),
                    predicted_b=float(prediction[1].item()),
                    target_a=float(sample.y[0]),
                    target_b=float(sample.y[1]),
                )
            )

    return rows


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def rmse(errors: list[float]) -> float:
    return math.sqrt(mean([error * error for error in errors]))


def mape(predicted: list[float], target: list[float]) -> float:
    percentages = [
        abs(pred - actual) / actual * 100.0
        for pred, actual in zip(predicted, target, strict=True)
        if actual > 0.0
    ]
    return mean(percentages)


def print_metrics(rows: list[PredictionRow], split: str, train_ratio: float) -> None:
    """Imprime métricas agregadas de previsão."""
    predicted_a = [row.predicted_a for row in rows]
    predicted_b = [row.predicted_b for row in rows]
    target_a = [row.target_a for row in rows]
    target_b = [row.target_b for row in rows]
    errors_a = [
        row.predicted_a - row.target_a
        for row in rows
    ]
    errors_b = [
        row.predicted_b - row.target_b
        for row in rows
    ]
    errors_all = errors_a + errors_b

    correct_best_server = sum(
        1
        for row in rows
        if (row.predicted_a >= row.predicted_b)
        == (row.target_a >= row.target_b)
    )

    print(f"Split avaliado: {split} (train_ratio={train_ratio:g})")
    print(f"Amostras avaliadas: {len(rows)}")
    print()
    print("Erro de vazão prevista:")
    print(
        "  Servidor A: "
        f"MAE={mean([abs(error) for error in errors_a]):.2f} kbps | "
        f"RMSE={rmse(errors_a):.2f} kbps | "
        f"bias={mean(errors_a):.2f} kbps | "
        f"MAPE={mape(predicted_a, target_a):.2f}%"
    )
    print(
        "  Servidor B: "
        f"MAE={mean([abs(error) for error in errors_b]):.2f} kbps | "
        f"RMSE={rmse(errors_b):.2f} kbps | "
        f"bias={mean(errors_b):.2f} kbps | "
        f"MAPE={mape(predicted_b, target_b):.2f}%"
    )
    print(
        "  Geral:      "
        f"MAE={mean([abs(error) for error in errors_all]):.2f} kbps | "
        f"RMSE={rmse(errors_all):.2f} kbps | "
        f"bias={mean(errors_all):.2f} kbps"
    )
    print()
    print(
        "Acurácia na escolha do melhor servidor: "
        f"{correct_best_server / max(len(rows), 1) * 100.0:.2f}% "
        f"({correct_best_server}/{len(rows)})"
    )
    print("Bias positivo indica superestimação; bias negativo indica subestimação.")


def write_predictions(rows: list[PredictionRow], output_path: Path) -> None:
    """Salva previsões por amostra em CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sample",
                "predicted_a_kbps",
                "target_a_kbps",
                "error_a_kbps",
                "predicted_b_kbps",
                "target_b_kbps",
                "error_b_kbps",
                "predicted_best_server",
                "actual_best_server",
            ]
        )
        for row in rows:
            predicted_best = "A" if row.predicted_a >= row.predicted_b else "B"
            actual_best = "A" if row.target_a >= row.target_b else "B"
            writer.writerow(
                [
                    row.index,
                    round(row.predicted_a, 3),
                    round(row.target_a, 3),
                    round(row.predicted_a - row.target_a, 3),
                    round(row.predicted_b, 3),
                    round(row.target_b, 3),
                    round(row.predicted_b - row.target_b, 3),
                    predicted_best,
                    actual_best,
                ]
            )


def main() -> None:
    args = parse_args()
    loaded_model = load_rnn_checkpoint(args.model)
    samples = load_rnn_samples_from_csv(
        csv_path=args.csv,
        sequence_length=loaded_model.sequence_length,
        server_a_id=loaded_model.server_a_id,
        server_b_id=loaded_model.server_b_id,
    )
    sample_feature_size = len(samples[0].x[0])
    if loaded_model.feature_size != sample_feature_size:
        raise ValueError(
            "Checkpoint incompatível com o vetor de features do CSV: "
            f"modelo tem {loaded_model.feature_size}, CSV gera {sample_feature_size}. "
            "Treine novamente o modelo com a versão atual das features."
        )
    selected_samples = select_split(
        samples=samples,
        split=args.split,
        train_ratio=args.train_ratio,
    )
    rows = predict_samples(
        samples=selected_samples,
        model=loaded_model.model,
        normalizer=loaded_model.normalizer,
    )

    print(f"Modelo: {args.model}")
    print(f"CSV: {args.csv}")
    print_metrics(rows, args.split, args.train_ratio)

    if args.predictions_output is not None:
        write_predictions(rows, args.predictions_output)
        print(f"Previsões por amostra salvas em: {args.predictions_output}")


if __name__ == "__main__":
    main()
