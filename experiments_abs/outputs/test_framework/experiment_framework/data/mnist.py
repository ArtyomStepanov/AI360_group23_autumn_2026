from pathlib import Path
from ..contracts import DatasetBundle


def build_dataset(
    root: Path, *, split_file: Path, augment: bool = False,
) -> DatasetBundle:
    """Построить MNIST datasets с фиксированным split и train_eval."""
    raise NotImplementedError("Implement MNIST loading and preprocessing")
