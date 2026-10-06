"""Публичные интерфейсы. Для импорта каркаса PyTorch не требуется."""
from .contracts import DatasetBundle, TrainConfig, TrainingResult
from .storage import RunPaths, StagePaths, create_run, read_result, write_result
from .training import evaluate, fit, load_weights

__all__ = [
    "DatasetBundle", "TrainConfig", "TrainingResult", "RunPaths", "StagePaths",
    "create_run", "read_result", "write_result", "fit", "evaluate", "load_weights",
]
