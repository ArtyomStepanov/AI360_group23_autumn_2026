"""Самостоятельный анализ checkpoints без базового Analyzer."""
from pathlib import Path


def measure_sharpness(
    checkpoint: Path, *, epsilon: float, seed: int, output_dir: Path,
) -> dict[str, float]:
    """Загрузить модель и данные из описания run; сохранить параметры и оценку.

    Перед реализацией уточнить область возмущений, выборку и бюджет оптимизации.
    Исходный checkpoint не изменять.
    """
    raise NotImplementedError("Implement the agreed sharpness protocol")
