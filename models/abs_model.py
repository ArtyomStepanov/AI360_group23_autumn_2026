import torch
import torch.nn as nn
from abc import abstractmethod, ABC


SchedulerInstance = torch.optim.lr_scheduler._LRScheduler


class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass
