import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Any


SchedulerInstance = torch.optim.lr_scheduler._LRScheduler


class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class BaseLogger(ABC):
    @abstractmethod
    def __call__(self, model: AbstractModel, loss: float) -> None:
        pass

    @abstractmethod
    def get_history(self) -> Any:
        pass


@dataclass(kw_only=True)
class TrainConfig:
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    seed: int
    batch_size: int
    epoch_count: int
    scheduler: Optional[SchedulerInstance] = None
    logger: Optional[BaseLogger] = None
    warm_start: bool = False
