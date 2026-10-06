import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class TrainConfig:
    optimizer: torch.optim.Optimizer
    batch_size: int
    epoch_count: int
    is_warm_start: bool
     

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


