# Использование models и utility

config_generator:
    supported optimizers:
        -AdamW (default lr=1e-3, weight_decay=0.01)
        -Adam (default lr=1e-3)
        -SGD (default lr=1e-1, weight_decay=5e-4)
    supported criterions:
        -CrossEntropyLoss (as CE)
        -BCEWithLogitsLoss (as BCE)
        -MSELoss (as MSE)
    supported schedulers:
        -StepLR (step_size=30, gamma=0.1)
        -MultiStepLR (as MStepLR) (milestones=[30, 60, 90], gamma=0.1)
        -ExponentialLR (as expLR) (gamma=0.95)
        -CosineAnnealingLR (as CALR) (T_max=100, eta_min=1e-6)
        -LinearLR (start_factor=0.01, end_factor=1.0, total_iters=10) - Warmup
    parameters:
        model: AbstractModel
        prompt: str
        batch_size: int
        epoch_count: int
        seed: int = 42
        lr: float = 0.0
        warm_start: bool = False
        optimizer: Optional[torch.optim.Optimizer] = None
        criterion: Optional[torch.nn.Module] = None
        scheduler: Optional[SchedulerInstance] = None
        logger: Optional[BaseLogger] = None