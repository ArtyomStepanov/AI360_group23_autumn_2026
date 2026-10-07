import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from metrics import Accuracy, BaseMetric


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    criterion: nn.Module,
    device: str,
    batch_size: int = 128,
    *,
    metrics: dict[str, BaseMetric] | None = None,
) -> dict[str, float]:
    """Evaluate mean loss per sample and accumulated metrics.

    criterion must return a scalar mean over samples in each batch.
    metrics=None defaults to multiclass accuracy; {} evaluates only loss.
    Metric instances are reset on every call. The name 'loss' is reserved.
    """
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    if len(dataset) == 0:
        raise ValueError("dataset не должен быть пустым")
    metrics = {"accuracy": Accuracy()} if metrics is None else metrics
    if "loss" in metrics:
        raise ValueError("Имя метрики 'loss' зарезервировано")
    for metric in metrics.values():
        metric.reset()

    device = torch.device(device)
    model.to(device)
    criterion.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_samples = 0

    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                current_batch_size = y.shape[0]
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size
                for metric in metrics.values():
                    metric.update(outputs, y)
    finally:
        model.train(was_training)

    result = {name: metric.compute() for name, metric in metrics.items()}
    result["loss"] = total_loss / total_samples
    return result
