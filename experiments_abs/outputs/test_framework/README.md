# Импортируемый каркас экспериментов

Это набросок интерфейсов, а не работающий фреймворк обучения. Реализованы структуры данных, создание каталогов и чтение/запись `TrainingResult`. Фабрики моделей/датасетов, `fit`, `evaluate`, загрузка весов и sharpness явно выбрасывают `NotImplementedError`.

## Использование

Из этой папки можно импортировать пакет без установки и без PyTorch:

```python
from pathlib import Path
from experiment_framework import TrainConfig, create_run

config = TrainConfig(epochs=10)
run = create_run(Path("results"), "batch_comparison")
stage = run.create_stage("train")
print(stage.root)
```

Для импорта из другого проекта: `python -m pip install -e /absolute/path/to/test_framework`. Для будущей реализации обучения доступны зависимости `python -m pip install -e '.[training]'`. Сейчас установка зависимостей не требуется для проверки импортов. Python 3.10+.

`experiments/warm_start.py` показывает два вызова `fit`; после заполнения заглушек его можно запускать из корня через `python -m experiments.warm_start`. YAML демонстрирует будущий формат: пример пока использует параметры прямо в Python и не читает YAML. В аннотациях используются типы PyTorch под TYPE_CHECKING, чтобы каркас импортировался без него.

## Ответственность файлов

- `contracts.py`: DatasetBundle, TrainConfig, TrainingResult.
- `data/<dataset>.py`: фабрика datasets; loaders создаёт сценарий.
- `models/<model>.py`: фабрика nn.Module, возвращающего logits.
- `training.py`: единый цикл обучения, evaluation и загрузка весов.
- `storage.py`: небольшие файловые функции, без ArtifactStore.
- `experiments/`: свободный код конкретного исследования.

## Хранение

```text
datasets/
  cifar10/                       # общий кеш исходных данных
  splits/cifar10.json            # индексы train/validation, seed и версия
results/<experiment>/
  experiment.yaml                # сценарий и ожидаемые условия sweep
  runs/<run_id>/
    config.yaml                  # фактические параметры с defaults
    metadata.json                # seed, condition, attempt, версии, данные
    initialization.pt            # при общей инициализации SB/LB
    stages/<stage>/
      config.yaml
      status.json
      metrics.jsonl
      result.json
      checkpoints/
        epoch_0010.pt
        index.json               # last/best указывают на конкретный файл
    analyses/<analysis_id>/
      config.yaml                # конкретные checkpoints и параметры
      status.json
      metrics.json
      curve.csv
  summary/
    runs.csv
    aggregate.csv
```

`create_run`/`create_stage`/`create_analysis` создают каталоги. Остальные файлы должен записывать соответствующий сценарий или будущая реализация `fit`. Существующая стадия повторно не создаётся: для resume передаётся её текущий путь. `write_result` записывает result.json атомарно; один процесс владеет одной стадией. Ссылки result.json относительны папке стадии, остальные межстадийные ссылки следует хранить относительно корня эксперимента.

### Checkpoint

Словарь `.pt`: schema_version, run_id, stage_id, model_state, optimizer_state, scheduler_state, completed_epochs, optimizer_steps, examples_seen, rng_state, loader_generator_state, config_hash, parent_checkpoint. Сохранять state_dict, включая buffers, а не целый объект модели. Файл checkpoint записывается атомарно до обновления index.json.

Resume продолжает ту же стадию на границе эпох и восстанавливает все состояния. Warm-start через load_weights переносит только модель в новую стадию. Для воспроизводимого resume нужно согласовать seed workers и генератор DataLoader; persistent workers пока вне контракта.

### Метрики

Одна JSONL-строка на split и эпоху:

```json
{"epoch": 1, "optimizer_steps": 196, "examples_seen": 50000, "split": "validation", "loss": 0.42, "accuracy": 0.85, "num_examples": 5000}
```

Счётчики локальны для стадии. train_online и train_eval разделяются. При resume записи после восстановленного checkpoint нужно архивировать или удалить. Test не используется для выбора best. Для сводок сохраняются mean, выборочное std, n успешных и n ожидаемых; при n=1 std отсутствует. Повторные попытки seed не являются независимыми наблюдениями.

## Порядок заполнения

1. Реализовать одну модель и датасет по согласованному протоколу.
2. Реализовать evaluate, затем fit и checkpoint/resume.
3. Дополнить сценарии сохранением конфигураций и метаданных.
4. Добавлять анализы обычными функциями по мере необходимости.
