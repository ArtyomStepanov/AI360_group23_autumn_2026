import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from loggers import BaseLogger
from models.abs_model import SchedulerInstance


@dataclass(kw_only=True)
class TrainConfig:
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    batch_size: int
    epoch_count: int
    seed: Optional[int] = 42
    scheduler: Optional[SchedulerInstance] = None
    logger: Optional[BaseLogger] = None
    warm_start: bool = False


def fit(
    model: nn.Module,
    config: TrainConfig,
    dataset: Dataset,
    device: str,
) -> nn.Module:
    if config.epoch_count < 1:
        raise ValueError("epoch_count должен быть положительным")
    if config.batch_size < 2:
        raise ValueError("Для моделей с BatchNorm1d batch_size должен быть >= 2")
    if len(dataset) < config.batch_size:
        raise ValueError(
            "При drop_last=True размер датасета должен быть >= batch_size"
        )

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    device = torch.device(device)
    model.to(device)
    config.criterion.to(device)

    if not config.warm_start:
        def reset_parameters(module: nn.Module):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

        model.apply(reset_parameters)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    for _ in range(config.epoch_count):
        model.train()

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            config.optimizer.zero_grad(set_to_none=True)

            prediction = model(x)
            loss = config.criterion(prediction, y)

            loss.backward()
            config.optimizer.step()

            loss_value = loss.detach().item()

            if config.logger is not None:
                config.logger(model, loss_value)

        if config.scheduler is not None:
            config.scheduler.step()

    return model
