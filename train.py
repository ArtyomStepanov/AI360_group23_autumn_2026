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
    workers: int = 0,      # Контроль потоков CPU для загрузки данных
    optimize: bool = True  # Контроль аппаратного ускорения (AMP)
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

    device_obj = torch.device(device)
    model.to(device_obj)
    config.criterion.to(device_obj)

    if not config.warm_start:
        def reset_parameters(module: nn.Module):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

        model.apply(reset_parameters)

    use_cuda = device_obj.type == "cuda"
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=(workers > 0)
    )

    use_amp = optimize and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for _ in range(config.epoch_count):
        model.train()

        for x, y in loader:
            x = x.to(device_obj, non_blocking=use_cuda)
            y = y.to(device_obj, non_blocking=use_cuda)

            config.optimizer.zero_grad(set_to_none=True)

            if optimize:
                with torch.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=use_amp):
                    prediction = model(x)
                    loss = config.criterion(prediction, y)

                scaler.scale(loss).backward()
                scaler.step(config.optimizer)
                scaler.update()
            else:
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
