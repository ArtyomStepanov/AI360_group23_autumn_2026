"""Общие структуры данных без логики обучения."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    train: Dataset
    train_eval: Dataset
    validation: Dataset | None
    test: Dataset
    input_shape: tuple[int, ...]
    num_classes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata: JSON-совместимые имя/версия, preprocessing, split и его checksum.
    # train и train_eval содержат одни объекты; train_eval без аугментации.


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    device: str = "cpu"
    save_every_epochs: int = 1
    selection_metric: str | None = None
    selection_mode: Literal["min", "max"] = "min"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.save_every_epochs < 1:
            raise ValueError("save_every_epochs must be positive")
        if self.selection_mode not in ("min", "max"):
            raise ValueError("selection_mode must be min or max")


@dataclass(frozen=True)
class TrainingResult:
    stage_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path | None
    metrics_file: Path
    completed_epochs: int
    optimizer_steps: int
    examples_seen: int
