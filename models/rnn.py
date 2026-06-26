"""
Define o modelo de rede neural recorrente usado pela política 3.

O modelo recebe uma sequência temporal de features representando o histórico
recente dos servidores e do player. A saída prevista corresponde à vazão futura
esperada para cada servidor disponível.

A arquitetura recomendada usa GRU por ser simples, eficiente e suficiente para
séries temporais curtas no contexto do experimento.
"""

import torch
import torch.nn as nn


class StreamingRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar no intervalo [0, 1).")

        self.rnn: nn.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x)
        last_output: torch.Tensor = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)
