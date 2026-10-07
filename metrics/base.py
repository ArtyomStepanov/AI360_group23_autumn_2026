from abc import ABC, abstractmethod

import torch


class BaseMetric(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear state before evaluating a new dataset."""
        pass

    @abstractmethod
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate results from one batch."""
        pass

    @abstractmethod
    def compute(self) -> float:
        """Compute the metric across all accumulated batches."""
        pass
