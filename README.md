# AI360_group23_autumn_2026

Код для обучения и оценки моделей в экспериментах.

| Раздел | Содержание |
| --- | --- |
| [models](models/README.md) | Интерфейс моделей, сохранение состояния |
| [loggers](loggers/README.md) | Интерфейс логгера обучения |
| [metrics](metrics/README.md) | Интерфейс и реализации метрик |
| [datasets](datasets/README.md) | Контракт датасетов и форматы примеров |
| `train.py` | TrainConfig и fit |
| `eval.py` | evaluate |
| `experiments/` | Материалы экспериментов |
| [scripts](scripts/README.md) | Первый эксперимент: размер батча, качество и sharpness |
| [exp2](scripts/exp2_README.md) | Learning rate × размер батча: 400 моделей C1 |

## Пример обучения и оценки

Пример запускается из корня проекта с установленными зависимостями из
`requirements.txt`. Используются синтетические данные; значения метрик служат
только проверкой запуска.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset

from eval import evaluate
from metrics import Accuracy
from loggers import LossLogger
from models.instances.CNN import CNN
from models.utils import load_model, save_model
from train import TrainConfig, fit

# Seed для создания данных; seed в TrainConfig действует внутри fit.
torch.manual_seed(42)
train_dataset = TensorDataset(
    torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,)),
)
test_dataset = TensorDataset(
    torch.rand(5, 3, 32, 32), torch.randint(0, 10, (5,)),
)
device = "cpu"
model = CNN(nb_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
config = TrainConfig(
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=criterion,
    batch_size=4,
    epoch_count=2,
    seed=42,
    logger=LossLogger(),
)
fit(model, config, train_dataset, device, workers=0, optimize=True)
results = evaluate(
    model, test_dataset, criterion, device,
    batch_size=4, metrics={"accuracy": Accuracy()},
)
print(results)
print(config.logger.get_history())

save_model(model, "models/weights/cnn.pt")
restored = load_model(CNN(nb_classes=10), "models/weights/cnn.pt")
```

## TrainConfig и fit

TrainConfig — dataclass с именованными аргументами.

| Поле | Контракт / значение по умолчанию |
| --- | --- |
| optimizer | Обязательный optimizer для параметров переданной модели |
| criterion | Обязательный nn.Module, возвращающий скалярный loss |
| batch_size | Обязательное целое число, не меньше 2 |
| epoch_count | Обязательное целое число, не меньше 1 |
| seed | 42; None отключает установку seed |
| scheduler | None; при наличии вызывается step() после каждой эпохи |
| logger | None; при наличии вызывается после каждого шага optimizer |
| warm_start | False; управляет сбросом параметров модели |

`fit(model, config, dataset, device, workers=0, optimize=True)` обучает и возвращает переданный экземпляр
модели. Размер dataset должен быть не меньше batch_size. Примеры перемешиваются,
последний неполный батч отбрасывается. После обучения модель остаётся в train-режиме.
Workers задаёт число процессов загрузки данных; при workers > 0 они сохраняются
между эпохами. На CUDA используется закреплённая память и неблокирующий перенос.
Optimize=True включает AMP float16 и GradScaler только на CUDA; на CPU AMP
отключён. Optimize=False использует обычный forward/backward.

При warm_start=False fit вызывает reset_parameters у подмодулей, имеющих этот
метод. У стандартных слоёв переинициализируются параметры, у BatchNorm также
сбрасываются статистики. При warm_start=True этот сброс пропускается.
Optimizer, scheduler и logger внутри fit не сбрасываются: независимые запуски
предполагают отдельные экземпляры этих объектов.

Seed устанавливается для Python, NumPy и PyTorch до сброса параметров и создания
DataLoader. Данные, созданные до fit, этим seed не управляются. Совпадение seed
не гарантирует одинаковые результаты между устройствами и для
недетерминированных GPU-операций.

## evaluate

`evaluate(model, dataset, criterion, device, batch_size=128, *, metrics=None)`
возвращает словарь с loss и результатами переданных метрик. Датасет непустой,
batch_size положительный. Примеры не перемешиваются; учитываются все батчи.

- metrics=None: используется Accuracy с именем accuracy.
- metrics={}: вычисляется только loss.
- В словаре метрик ключ задаёт имя результата; имя loss зарезервировано.

Criterion должен возвращать скалярное среднее по примерам батча. Итоговый loss
взвешивается по размерам батчей. Этот расчёт не поддерживает в общем случае
reduction="sum", веса классов и ignore_index без дополнительного учёта знаменателя.

Оценка выполняется в eval-режиме без вычисления градиентов. Исходный режим
модели восстанавливается, включая исключение внутри цикла. Модель и criterion
переносятся на device; исходное устройство не восстанавливается.

## Sharpness

В [metrics/sharpness.py](metrics/sharpness.py) доступна функция
compute_sharpness. Она получает модель и датасет и требует вычисления
градиентов, поэтому вызывается отдельно от evaluate.

```python
from metrics.sharpness import compute_sharpness

sharpness = compute_sharpness(
    restored, test_dataset, device,
    criterion=criterion, batch_size=4, subspace_dim=1,
)
print(sharpness)
```

Результат — приближённая оценка, а не гарантированный максимум. Функция использует
границы из Metric 2.1 статьи Keskar et al. и LBFGS с tanh-репараметризацией.
Maxiter ограничивает итерации оптимизатора, но не число проходов по датасету.
Контракт функции описан в [README метрик](metrics/README.md).
