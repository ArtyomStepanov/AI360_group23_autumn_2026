import torch

from .base import BaseMetric


class Accuracy(BaseMetric):
    """Multiclass accuracy for outputs (N, C) and class indices (N,)."""

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
    """Binary accuracy for logits and 0/1 targets shaped (N,) or (N, 1).

    threshold is expressed in logits: 0 corresponds to probability 0.5.
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
