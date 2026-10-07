import torch

from .base import BaseMetric


class Accuracy(BaseMetric):
    """Доля правильных ответов при многоклассовой классификации.

    update принимает outputs формы (N, C) и targets формы (N,) с индексами
    классов. Размеры батчей должны совпадать. Предсказания определяются через
    argmax(dim=1); неверные формы вызывают ValueError.

    compute возвращает число правильных ответов, делённое на общее число
    примеров, от 0 до 1. При отсутствии примеров вызывает ValueError.
    reset очищает оба счётчика. При создании счётчики равны нулю.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if outputs.ndim != 2 or targets.ndim != 1:
            raise ValueError("Accuracy expects outputs (N, C) and targets (N,)")
        if outputs.shape[0] != targets.shape[0]:
            raise ValueError("Outputs and targets must have the same batch size")
        self.correct += (outputs.argmax(dim=1) == targets).sum().item()
        self.total += targets.numel()

    def compute(self) -> float:
        if self.total == 0:
            raise ValueError("No samples available to compute accuracy")
        return self.correct / self.total


class BinaryAccuracy(Accuracy):
    """Доля правильных ответов при бинарной классификации с одним logit.

    Args:
        threshold: Порог в logits; 0 соответствует вероятности 0.5.

    update принимает logits и метки 0/1 формы (N,) или (N, 1) с одинаковым N.
    Тензоры разворачиваются в одномерные; предсказание равно logit >= threshold.
    Неверные формы вызывают ValueError. Значения меток на принадлежность
    множеству {0, 1} не проверяются.

    compute возвращает долю правильных ответов от 0 до 1 и вызывает ValueError
    при отсутствии примеров. reset очищает счётчики, изначально равные нулю.
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        for tensor in (outputs, targets):
            if tensor.ndim != 1 and not (tensor.ndim == 2 and tensor.shape[1] == 1):
                raise ValueError("BinaryAccuracy expects tensors (N,) or (N, 1)")
        if outputs.shape[0] != targets.shape[0]:
            raise ValueError("Outputs and targets must have the same batch size")
        predictions = outputs.reshape(-1) >= self.threshold
        self.correct += (predictions == targets.reshape(-1)).sum().item()
        self.total += targets.numel()
