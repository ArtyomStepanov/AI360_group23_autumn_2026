from pathlib import Path
from ..contracts import DatasetBundle


def build_dataset(
    root: Path, *, split_file: Path, augment: bool = True,
) -> DatasetBundle:
    """Построить CIFAR-10 datasets по сохранённым индексам split.

    Проверить непересечение train/validation. train_eval использует те же
    индексы, что train, и детерминированный preprocessing. Loaders создаёт сценарий.
    """
    raise NotImplementedError("Implement CIFAR-10 loading and preprocessing")
