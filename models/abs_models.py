import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    logger: Optional[BaseLogger] = None
    batch_size: int
    epoch_count: int
    is_warm_start: bool = False
     

class AbstractModel(nn.Module):
    @abstractmethod
    def __init__(self, nb_classes=10):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


def fit_model(model: AbstractModel, config: TrainConfig):
        pass

def load_model(model: AbstractModel):
    pass

def save_model(model: AbstractModel):
    pass

class BaseLogger(ABC):
    @abstractmethod
    def __call__(self, model: AbstractModel, global_step: int) -> None:
        pass

    @abstractmethod
    def get_history(self) -> Any:
        pass
