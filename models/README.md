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
