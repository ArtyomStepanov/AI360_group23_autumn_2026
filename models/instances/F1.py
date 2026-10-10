from torch import nn

from models.abs_model import AbstractModel


class F1(AbstractModel):
    """Сеть F1 для MNIST по приложению B.1 Keskar et al.

    Пять скрытых слоёв по 512 нейронов: Linear -> BatchNorm -> ReLU.
    Вход: (N, 784) или изображения (N, 1, 28, 28).
    Выход: logits (N, 10), совместимые с CrossEntropyLoss.
    В train-режиме N >= 2. Используются стандартные BatchNorm и
    инициализация PyTorch; softmax из описания статьи вынесен в loss.
    """

    def __init__(self):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = 784
        for _ in range(5):
            layers.extend([
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ])
            in_features = 512
        layers.append(nn.Linear(512, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
