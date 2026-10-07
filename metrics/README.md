# Метрики

Метрики наследуют `BaseMetric` из `base.py`. Реализация хранит состояние между
батчами и вычисляет итоговое значение для всего датасета.

## Интерфейс

| Метод | Назначение |
| --- | --- |
| `reset() -> None` | Очистить накопленные данные |
| `update(outputs, targets) -> None` | Учесть один батч |
| `compute() -> float` | Вернуть итоговую метрику |

`evaluate` вызывает `reset` перед оценкой, `update` для каждого батча внутри
`torch.no_grad()` и `compute` после прохода. В `update` поступают выходы модели
без преобразований и метки на выбранном устройстве. Преобразование выходов в
предсказания выполняет сама метрика. Не изменяйте входные тензоры: они могут
использоваться другими метриками.

## Готовые метрики

| Класс | Выходы модели | Метки | Правило |
| --- | --- | --- | --- |
| `Accuracy()` | `(N, C)` | Индексы классов `(N,)` | `argmax(dim=1)` |
| `BinaryAccuracy(threshold=0.0)` | Logits `(N,)` или `(N, 1)` | 0/1, `(N,)` или `(N, 1)` | `logit >= threshold` |

Обе метрики возвращают долю правильных ответов от 0 до 1. У BinaryAccuracy порог
задаётся в logits: 0 соответствует вероятности 0.5. Форма меток для loss должна
совпадать с выходом модели, даже если сама метрика допускает обе формы.

## Использование

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset
from eval import evaluate
from metrics import Accuracy

dataset = TensorDataset(
    torch.tensor([[3., 0.], [0., 3.], [2., 0.]]),
    torch.tensor([0, 1, 1]),
)
results = evaluate(
    model=nn.Identity(),
    dataset=dataset,
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    batch_size=2,
    metrics={"accuracy": Accuracy()},
)
print(results)  # accuracy = 2 / 3; loss также включён
```

Ключ словаря задаёт имя результата. Имя `loss` зарезервировано. Если metrics
не переданы или равны `None`, используется Accuracy. Пустой словарь `{}`
включает оценку только loss. Для бинарной задачи передайте BinaryAccuracy и
подходящий criterion, например BCEWithLogitsLoss.

`evaluate` использует `model.eval()`, сохраняет последний неполный батч и
восстанавливает исходный режим модели в конце, включая случай исключения в
цикле. Функция переносит модель и criterion на device; обратно устройство
не переключается.

Loss вычисляется как среднее по примерам всего датасета. Поэтому criterion
должен возвращать скалярное среднее по примерам батча. Взвешенный
CrossEntropyLoss, ignore_index и reduction="sum" требуют отдельного учёта
знаменателя; текущий расчёт не поддерживает их в общем случае.

## Новая метрика

Создайте модуль в этой папке, наследуйте BaseMetric и экспортируйте класс из
`__init__.py`, если нужен импорт `from metrics import ...`.

```python
from metrics import BaseMetric


class SampleCount(BaseMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0

    def update(self, outputs, targets):
        self.total += targets.shape[0]

    def compute(self):
        return float(self.total)
```

Накапливайте величины, необходимые для глобального результата. Для accuracy
это число правильных ответов и примеров; для F1 — TP, FP и FN с учётом способа
усреднения. Простое среднее метрик батчей обычно не равно метрике датасета.
Не храните все предсказания, если достаточно счётчиков.

Проверяйте результат на известных примерах, независимость от batch_size,
очистку через reset и обработку неправильных форм. Текущие проверки запускаются
из корня проекта: `.venv/bin/python -m unittest discover -s tests -v`.
