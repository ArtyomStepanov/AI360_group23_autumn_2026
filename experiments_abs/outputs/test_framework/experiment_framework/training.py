"""Контракты обучения. Функции намеренно не реализованы."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Protocol

from .contracts import TrainConfig, TrainingResult

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


class EpochScheduler(Protocol):
    """Минимальный контракт scheduler, вызываемого после каждой эпохи."""
    def step(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    config: TrainConfig,
    output_dir: Path,
    eval_loaders: Mapping[str, DataLoader],
    scheduler: EpochScheduler | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    """Обучить одну стадию, изменяя переданные model/optimizer/scheduler.

    epochs — полный целевой бюджет, включая эпохи до resume.
    Resume восстанавливает веса, optimizer, scheduler, RNG, генератор loader
    и счётчики на границе эпох; проверяет совместимость конфигурации.
    При resume записи метрик после checkpoint необходимо согласовать с ним.
    Test не используется для выбора best. Scheduler.step вызывается после эпохи.
    Результаты сохраняются по контракту README; ошибки пробрасываются вызывающему.
    """
    raise NotImplementedError("Implement the shared training loop in training.py")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: str,
) -> dict[str, float]:
    """Вернуть loss/accuracy, усреднённые по объектам, и num_examples.

    Временно включить eval, отключить градиенты и восстановить режим модели.
    Loss предполагается скаляром со средним по батчу; последний батч учитывается.
    """
    raise NotImplementedError("Implement evaluation in training.py")


def load_weights(model: nn.Module, checkpoint: Path) -> None:
    """Загрузить model_state строго по ключам, включая buffers.

    Optimizer, scheduler, RNG и счётчики не переносятся: это warm-start.
    """
    raise NotImplementedError("Implement checkpoint weight loading in training.py")
