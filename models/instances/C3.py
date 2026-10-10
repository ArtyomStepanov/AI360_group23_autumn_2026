from .ShallowNN import ShallowNN


class C3(ShallowNN):
    """Сеть C3 для CIFAR-100 на основе авторского shallownet.

    Архитектура C1 с выходом на 100 классов.
    Вход: (N, 3, 32, 32). Выход: logits (N, 100).
    Свёртки имеют stride=1 по Keras-коду авторов, вместо stride=2
    из описания статьи. В train-режиме N >= 2.
    Реализация наследует ShallowNN: стандартные BatchNorm, инициализация
    PyTorch и симметричный padding pooling. Полная численная эквивалентность
    старому Keras BatchNormalization(mode=2) и SAME pooling не заявляется.
    Softmax не применяется: выход предназначен для CrossEntropyLoss.
    """

    def __init__(self):
        super().__init__(nb_classes=100)
