# Модели

`abs_model.py` определяет интерфейсы `AbstractModel`, `BaseLogger` и конфигурацию
`TrainConfig`. Реализации моделей размещаются в `instances/`, сохранённые веса —
в `weights/`. Запускайте скрипты и примеры из корня проекта.

## Создание модели

Наследуйте `AbstractModel`, реализуйте конструктор и `forward`. В конструкторе
вызовите `super().__init__()` и зарегистрируйте слои как атрибуты модели.

```python
from torch import nn
from models.abs_model import AbstractModel


class LinearClassifier(AbstractModel):
    def __init__(self, in_features: int, nb_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_features, nb_classes)

    def forward(self, x):
        return self.classifier(x)
```

`forward` принимает батч тензоров и возвращает тензор, совместимый с выбранными
criterion и метриками. Для многоклассовой классификации это logits `(N, C)`.
Перед `CrossEntropyLoss` не применяйте softmax. Для бинарной классификации с
`BCEWithLogitsLoss` возвращайте один logit на пример; форма выхода и меток должна
совпадать: `(N,)` либо `(N, 1)`.

Текущие `CNN`, `DNN` и `ShallowNN` рассчитаны на RGB-изображения `(N, 3, 32, 32)`.
Число выходных классов задаётся аргументом `nb_classes`.

## Обучение

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset
from models.abs_model import TrainConfig
from models.instances.CNN import CNN
from train import fit

dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
model = CNN(nb_classes=10)
config = TrainConfig(
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    batch_size=4,
    epoch_count=2,
)
fit(model, config, dataset, device="cpu")
```

`fit` изменяет переданную модель и возвращает её. Параметры `TrainConfig`
передаются по имени. Обязательны optimizer, criterion, batch_size и epoch_count.
Scheduler вызывается через `step()` после каждой эпохи. Logger вызывается после
каждого шага optimizer и получает модель и скалярный loss этого батча; наследник
`BaseLogger` также должен реализовать `get_history()`.

Для каждого нового запуска создавайте свежие optimizer, scheduler и logger:
`fit` самостоятельно не очищает их состояние. Optimizer должен ссылаться на
параметры обучаемой модели.

- `warm_start=False`: у подмодулей вызывается `reset_parameters()`, если он есть.
  Стандартные слои переинициализируют параметры; BatchNorm сбрасывает статистики.
  Для собственных обучаемых слоёв реализуйте этот метод.
- `warm_start=True`: параметры и буферы переданной модели сохраняются перед
  обучением. Это режим продолжения обучения после загрузки весов.

В текущем `fit` batch_size должен быть не меньше 2, epoch_count — не меньше 1,
а размер датасета — не меньше batch_size. Данные перемешиваются, последний
неполный батч отбрасывается из-за BatchNorm в моделях проекта.

## Сохранение и загрузка

```python
from models.instances.CNN import CNN
from models.utility import load_model, save_model

model = CNN(nb_classes=10)
save_model(model, "models/weights/cnn.pt")
restored = load_model(CNN(nb_classes=10), "models/weights/cnn.pt")
```

Сохраняется `state_dict`: параметры и буферы, включая статистики BatchNorm.
Архитектура, optimizer, scheduler и история logger не сохраняются. Для загрузки
создайте совместимую модель с теми же размерами слоёв. Загрузка строгая и
возвращает переданный экземпляр; содержимое файла сначала загружается на CPU.
Для предсказаний вызовите `model.eval()` и отключите вычисление градиентов.

Правила оценки описаны в [README метрик](../metrics/README.md), формат входных
данных — в [README датасетов](../datasets/README.md).
