# Модели

`abs_model.py` объявляет AbstractModel и SchedulerInstance.
Реализации моделей находятся в `instances/`, файлы состояния — в `weights/`.
TrainConfig и fit объявлены в [train.py](../train.py).

## AbstractModel

AbstractModel наследует nn.Module и ABC. Абстрактные методы:

| Метод | Контракт |
| --- | --- |
| __init__ | Инициализация nn.Module и слоёв конкретной модели |
| forward(x) | Принимает батч тензоров, возвращает выход модели |

Форма выхода определяется задачей и должна быть совместима с criterion и
метриками. Для CrossEntropyLoss ожидаются logits `(N, C)`, для бинарного
BCEWithLogitsLoss — logits `(N,)` или `(N, 1)` той же формы, что метки.

Реализации: [CNN](instances/CNN.py), [DNN](instances/DNN.py),
[ShallowNN](instances/ShallowNN.py). Параметры, формы входов и выходов и
особенности каждой модели описаны в docstring её класса.

## SchedulerInstance

SchedulerInstance — псевдоним типа torch.optim.lr_scheduler._LRScheduler.

## Сохранение и загрузка

`save_model(model, save_path) -> None` сохраняет state_dict и создаёт родительские
каталоги пути. `load_model(model, load_path) -> AbstractModel` загружает state_dict
в переданный экземпляр и возвращает его. Оба пути принимают Path или str.

State_dict содержит параметры и буферы, включая статистики BatchNorm.
Архитектура и состояние optimizer, scheduler и logger не сохраняются.
Загрузка использует weights_only=True, map_location="cpu" и strict=True;
имена и размеры параметров должны совпадать. Режим train/eval загрузка не меняет.

Общий пример использования приведён в [корневом README](../README.md).

## Генератор конфигураций

### Поддерживаемые оптимизаторы

- **AdamW** — по умолчанию `lr=1e-3`, `weight_decay=0.01`
- **Adam** — по умолчанию `lr=1e-3`
- **SGD** — по умолчанию `lr=1e-1`, `weight_decay=5e-4`

### Поддерживаемые функции потерь

- **CrossEntropyLoss** (алиас: `CE`)
- **BCEWithLogitsLoss** (алиас: `BCE`)
- **MSELoss** (алиас: `MSE`)

### Поддерживаемые планировщики learning rate

- **StepLR** — `step_size=30`, `gamma=0.1`
- **MultiStepLR** (алиас: `MStepLR`) — `milestones=[30, 60, 90]`, `gamma=0.1`
- **ExponentialLR** (алиас: `expLR`) — `gamma=0.95`
- **CosineAnnealingLR** (алиас: `CALR`) — `T_max=100`, `eta_min=1e-6`
- **LinearLR** — `start_factor=0.01`, `end_factor=1.0`, `total_iters=10` (используется для warmup)

### Параметры

| Параметр | Тип | Описание |
|---|---|---|
| `model` | `AbstractModel` | Модель для обучения |
| `prompt` | `str` | Названия и алиасы, разделённые пробелами; регистр не учитывается |
| `batch_size` | `int` | Размер батча |
| `epoch_count` | `int` | Количество эпох |
| `seed` | `Optional[int]` | Seed (по умолчанию `42`); `None` отключает установку |
| `lr` | `float` | Learning rate (по умолчанию `0.0`) |
| `warm_start` | `bool` | Флаг тёплого старта (по умолчанию `False`) |
| `optimizer` | `Optional[torch.optim.Optimizer]` | Оптимизатор (по умолчанию `None`) |
| `criterion` | `Optional[torch.nn.Module]` | Функция потерь (по умолчанию `None`) |
| `scheduler` | `Optional[SchedulerInstance]` | Планировщик learning rate (по умолчанию `None`) |
| `logger` | `Optional[BaseLogger]` | Логгер (по умолчанию `None`) |

Scheduler необязателен. Optimizer и criterion должны быть заданы в prompt
или переданы аргументами.
