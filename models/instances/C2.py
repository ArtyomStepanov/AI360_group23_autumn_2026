from .DNN import DNN


class C2(DNN):
    """Сеть C2 для CIFAR-10 на основе авторского deepnet.

    Вход: (N, 3, 32, 32). Выход: logits (N, 10).
    Блоки: 2x64, 2x128, 3x256, 3x512, 3x512 свёрток 3x3,
    pooling 2x2 после каждого блока; dense 512 -> 10.
    В train-режиме N >= 2. Наследует DNN со стандартными BatchNorm,
    инициализацией PyTorch и Dropout по авторскому Keras-коду.
    Семантика старого Keras BatchNormalization(mode=2) не воспроизводится
    специально. Softmax не применяется: выход предназначен для CrossEntropyLoss.
    """

    def __init__(self):
        super().__init__(nb_classes=10)
