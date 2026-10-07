from torch import nn

from .base import BaseLogger


class LossLogger(BaseLogger):
    """История loss каждого батча в порядке вызовов.

    Создаётся с пустой историей. __call__(model, loss) добавляет float(loss);
    модель и её параметры не сохраняются.

    get_history возвращает копию списка list[float]. Изменение этой копии
    не влияет на внутреннее состояние. reset очищает историю; ранее
    возвращённые копии сохраняются.
    """

    def __init__(self):
        self._history: list[float] = []

    def reset(self) -> None:
        self._history.clear()

    def __call__(self, model: nn.Module, loss: float) -> None:
        self._history.append(float(loss))

    def get_history(self) -> list[float]:
        """Return a copy of the recorded losses."""
        return self._history.copy()
