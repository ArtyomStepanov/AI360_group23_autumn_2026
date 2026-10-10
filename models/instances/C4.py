from .DNN import DNN


class C4(DNN):
    """Сеть C4 для CIFAR-100 на основе авторского deepnet.

    Архитектура C2 с выходом на 100 классов.
    Вход: (N, 3, 32, 32). Выход: logits (N, 100).
    В train-режиме N >= 2. Наследует DNN со стандартными BatchNorm,
    инициализацией PyTorch и Dropout по авторскому Keras-коду.
    Семантика старого Keras BatchNormalization(mode=2) не воспроизводится
    специально. Softmax не применяется: выход предназначен для CrossEntropyLoss.
    """

    def __init__(self):
        super().__init__(nb_classes=100)
