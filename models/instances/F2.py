from torch import nn

from models.abs_model import AbstractModel


class F2(AbstractModel):
    """Сеть F2 для признаков TIMIT по приложению B.2 Keskar et al.

    Семь скрытых слоёв по 512 нейронов: Linear -> BatchNorm -> ReLU.
    Вход: (N, 360). Выход: logits (N, 1973).
    В train-режиме N >= 2. Используются стандартные BatchNorm и
    инициализация PyTorch; softmax из описания статьи вынесен в loss.
    """

    def __init__(self):
        super().__init__()
        layers = []
        in_features = 360
        for _ in range(7):
            layers.extend([
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ])
            in_features = 512
        layers.append(nn.Linear(512, 1973))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
