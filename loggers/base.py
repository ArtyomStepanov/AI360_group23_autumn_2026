from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class BaseLogger(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated history."""
        pass

    @abstractmethod
    def __call__(self, model: nn.Module, loss: float) -> None:
        """Record the current model and batch loss after an optimizer step."""
        pass

    @abstractmethod
    def get_history(self) -> Any:
        """Return accumulated history in the implementation's format."""
        pass
