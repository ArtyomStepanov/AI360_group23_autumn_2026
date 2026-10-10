from .ShallowNN import ShallowNN


class C1(ShallowNN):
    """Сеть C1 для CIFAR-10 на основе авторского shallownet.

    Вход: (N, 3, 32, 32). Выход: logits (N, 10).
    Две свёртки 64x5x5, pooling 3x3, dense 384 -> 192 -> 10.
    Свёртки имеют stride=1 по Keras-коду авторов; в приложении B.3
    статьи указан stride=2. В train-режиме N >= 2.
    Реализация наследует ShallowNN: стандартные BatchNorm, инициализация
    PyTorch и симметричный padding pooling. Полная численная эквивалентность
    старому Keras BatchNormalization(mode=2) и SAME pooling не заявляется.
    Softmax не применяется: выход предназначен для CrossEntropyLoss.
    """

    def __init__(self):
        super().__init__(nb_classes=10)
