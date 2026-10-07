from abs_model import AbstractModel, TrainConfig, BaseLogger, SchedulerInstance
from pathlib import Path
from typing import Optional
import torch

def load_model(model: AbstractModel, load_path: Path | str) -> AbstractModel:
    """Load parameters and buffers into an existing compatible model."""
    state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def save_model(model: AbstractModel, save_path: Path | str) -> None:
    """Save parameters and buffers without training or optimizer state."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def config_generator(model: AbstractModel, prompt: str, *, batch_size: int, epoch_count: int, seed: int = 42, lr: float = 0.0, logger: Optional[BaseLogger] = None, warm_start: bool = False,
optimizer: Optional[torch.optim.Optimizer] = None, criterion: Optional[torch.nn.Module] = None, scheduler: Optional[SchedulerInstance]=None) -> TrainConfig:    
    prompt = prompt.lower().strip().split()

    for entry in prompt:
        if entry == "ce":
            criterion = torch.nn.CrossEntropyLoss()
        elif entry == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif entry == "mse":
            criterion = torch.nn.MSELoss()

        if entry == "adamw":
            lr = lr if lr > 0 else 1e-3
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        elif entry == "adam":
            lr = lr if lr > 0 else 1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif entry == "sgd":
            lr = lr if lr > 0 else 1e-1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    if optimizer is None:
        raise ValueError(
            "optimizer либо не содержится в промте, либо не могу распарсить"
        )
    if criterion is None:
        raise ValueError(
            "criterion либо не содержится в промте, либо не могу распарсить"
        )

    for entry in prompt:
        if entry == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif entry == "msteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
        elif entry == "explr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif entry == "calr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        elif entry == "linearlr":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

    config = TrainConfig(batch_size=batch_size, 
            epoch_count=epoch_count,
            seed=seed,
            logger=logger, 
            warm_start=warm_start,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler)

    if config.scheduler is None:
        raise ValueError(
            "scheduler либо не содержится в промте, либо не могу распарсить"
        )
    
    return config
