# Модели

`abs_model.py` объявляет AbstractModel и SchedulerInstance.
Реализации моделей находятся в [instances](instances/README.md), файлы состояния —
в [weights](weights/README.md). Утилиты объявлены в [utils.py](utils.py).
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

`config_generator` из [utils.py](utils.py) возвращает TrainConfig для переданной
модели. Обязательные аргументы: model, prompt, batch_size и epoch_count.
Prompt содержит разделённые пробелами названия или алиасы; регистр не учитывается.

| Объекты | Поддерживаемые токены |
| --- | --- |
| Optimizer | adamw, adam, sgd |
| Criterion | ce / crossentropyloss, bce / bcewithlogitsloss, mse / mseloss |
| Scheduler | steplr, msteplr / multisteplr, explr / exponentiallr, calr / cosineannealinglr, linearlr |

Optimizer и criterion должны быть заданы токенами или переданными объектами.
Scheduler необязателен. Соответствующий токен заменяет переданный объект;
неизвестные токены игнорируются. Положительный lr переопределяет стандартный
lr создаваемого optimizer. Seed по умолчанию 42, warm_start=False, logger=None.
Настройки создаваемых объектов приведены непосредственно в реализации.
