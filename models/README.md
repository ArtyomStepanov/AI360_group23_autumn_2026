# Использование моделей и утилит

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
| `prompt` | `str` | Текстовый промпт |
| `batch_size` | `int` | Размер батча |
| `epoch_count` | `int` | Количество эпох |
| `seed` | `int` | Сид генератора случайных чисел (по умолчанию `42`) |
| `lr` | `float` | Learning rate (по умолчанию `0.0`) |
| `warm_start` | `bool` | Флаг тёплого старта (по умолчанию `False`) |
| `optimizer` | `Optional[torch.optim.Optimizer]` | Оптимизатор (по умолчанию `None`) |
| `criterion` | `Optional[torch.nn.Module]` | Функция потерь (по умолчанию `None`) |
| `scheduler` | `Optional[SchedulerInstance]` | Планировщик learning rate (по умолчанию `None`) |
| `logger` | `Optional[BaseLogger]` | Логгер (по умолчанию `None`) |