# Небольшой фреймворк экспериментов: интерфейсы и хранение

**Статус:** черновик для командного обсуждения  
**Дата:** 6 октября 2026  
**Контекст:** эксперименты по Keskar et al., [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836).

## 1. Принятое упрощение

Для каждого датасета и модели есть отдельный модуль с фабрикой. Общий цикл обучения находится в `training.py`. Сценарии сравнения batch size, warm-start, sharpness, интерполяции и Hessian — обычные Python-скрипты: они создают компоненты, вызывают обучение и обрабатывают его результаты.

На старте не вводим ExperimentRunner, общий Analyzer, registry и универсальный pipeline. Общими остаются контракты данных, обучения и файловых результатов. Приведённые сигнатуры — эскиз интерфейсов для PyTorch и задач классификации; это не готовая реализация. Численные параметры в примерах иллюстративные.

## 2. Структура проекта

```text
project/
  configs/
    batch_comparison.yaml
    warm_start.yaml
  data/
    cifar10.py
    mnist.py
  models/
    fully_connected.py
    convnet.py
  contracts.py              # небольшие структуры данных
  training.py               # обучение, evaluation, checkpoints
  experiments/
    batch_comparison.py
    warm_start.py
    sharpness.py
    interpolation.py
    hessian.py
  datasets/                 # скачанные данные и сохранённые splits
  results/                  # результаты запусков
```

Скрипты используют прямые импорты фабрик. YAML читается конкретным сценарием; глобальная система разрешения конфигураций пока не нужна. Повторяющийся код анализа можно позднее вынести в функции.

## 3. Данные

Фабрика датасета возвращает datasets, а не готовые DataLoaders: batch size и порядок выборки относятся к конкретному обучению. Для оценки train loss нужна отдельная версия train dataset без случайной аугментации, но с теми же объектами.

```python
@dataclass
class DatasetBundle:
    train: Dataset
    train_eval: Dataset
    validation: Dataset | None
    test: Dataset
    input_shape: tuple[int, ...]
    num_classes: int
    metadata: dict  # JSON-совместимое описание данных и split

# data/cifar10.py
def build_dataset(
    root: Path,
    *,
    split_file: Path,
    augment: bool = True,
) -> DatasetBundle: ...
```

`split_file` содержит фиксированные индексы train/validation. Подготовку split можно сделать отдельной функцией один раз; seed split хранится отдельно от seed обучения. Если validation не используется, файл всё равно фиксирует состав train.

`metadata` включает имя/версию датасета, размер каждой части, описание preprocessing, идентификатор и checksum split. При создании bundle проверяются непересечение train/validation и соответствие `train`/`train_eval`. Сами исходные данные общие для всех runs и не копируются в результаты.

DataLoaders создаёт скрипт эксперимента. Он сохраняет их параметры: batch size, shuffle, drop_last, num_workers и seed генератора. Для train_eval/validation/test используются детерминированные преобразования, `shuffle=False` и `drop_last=False`. Метрики усредняются по объектам, включая последний неполный батч.

## 4. Модели

Каждый модуль экспортирует фабрику обычного `nn.Module`:

```python
# models/convnet.py
def build_model(
    *,
    input_shape: tuple[int, ...],
    num_classes: int,
    channels: tuple[int, ...] = (32, 64),
) -> nn.Module: ...
```

Фабрика не скачивает данные и не запускает обучение. Для классификации `forward(x)` возвращает logits формы `[batch, num_classes]`. Seed задаётся до создания модели. Имя фабрики и все её аргументы сохраняются в конфигурации, чтобы модель можно было восстановить перед загрузкой `state_dict`.

Для парного сравнения SB/LB одного seed недостаточно как явного свидетельства общей инициализации: скрипт сохраняет исходный `state_dict` и загружает его в обе модели. Buffers входят в это состояние вместе с параметрами.

## 5. Абстракция обучения

Достаточно функции `fit`; класс Trainer можно добавить, если понадобится долгоживущий объект. Оптимизатор, loss и loaders создаются снаружи. Это позволяет экспериментам менять их без расширения общего класса.

```python
@dataclass(frozen=True)
class TrainConfig:
    epochs: int                 # полный целевой бюджет стадии
    device: str = "cpu"
    save_every_epochs: int = 1
    selection_metric: str | None = None  # например validation.loss
    selection_mode: Literal["min", "max"] = "min"

@dataclass(frozen=True)
class TrainingResult:
    stage_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path | None
    metrics_file: Path
    completed_epochs: int
    optimizer_steps: int
    examples_seen: int

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    loss_fn: Callable,
    optimizer: Optimizer,
    config: TrainConfig,
    output_dir: Path,
    eval_loaders: Mapping[str, DataLoader],
    scheduler: object | None = None,
    resume_from: Path | None = None,
) -> TrainingResult: ...

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: Callable,
    device: str,
) -> dict[str, float]: ...  # loss, accuracy, num_examples
```

`fit` изменяет переданную модель и состояние optimizer/scheduler. Модель остаётся доступна скрипту после вызова; `TrainingResult` содержит небольшое описание и ссылки на файлы, а не дублирует веса в памяти.

`evaluate` временно включает evaluation mode, отключает вычисление градиентов и восстанавливает предыдущий режим. Поддержка других задач или дополнительных метрик добавляется при реальной необходимости.

Правила `fit`:

- Сохраняет метрики и checkpoint на границе завершённой эпохи. `last` — последнее сохранённое состояние, которое может отставать от текущего вычисления при редком сохранении.
- Если `selection_metric` задана, проверяет наличие соответствующего evaluation loader и сохраняет лучший checkpoint. Test не используется для выбора состояния.
- `epochs` при resume означает полный бюджет стадии, а не число дополнительных эпох. При сохранённых 7 эпохах и `epochs=10` выполняются ещё 3.
- Scheduler в MVP вызывается после эпохи; варианты, требующие другого порядка или метрики, должны получить явную поддержку перед использованием.
- При ошибке оставляет доступные checkpoints и метрики, отмечает стадию как failed и пробрасывает исключение.

## 6. Загрузка, resume и warm-start

```python
def load_weights(model: nn.Module, checkpoint: Path) -> None: ...

def read_result(stage_dir: Path) -> TrainingResult: ...
```

`load_weights` переносит только model state, включая buffers. `fit(resume_from=...)` восстанавливает model, optimizer, scheduler, счётчики и случайные состояния. Скрипт заранее создаёт совместимые объекты; конфигурация стадии проверяется перед восстановлением.

Для MVP resume поддерживается на границе эпох. Нужно сохранять Python/NumPy/PyTorch RNG и состояние генератора train DataLoader. При случайных преобразованиях в workers воспроизводимое продолжение требует согласованного посева по эпохам; режим persistent workers требует отдельной проработки. Побитовая идентичность между разными устройствами не предполагается.

Warm-start выражается двумя вызовами, без pipeline engine:

```python
sb = fit(model, sb_loader, optimizer=sb_optimizer,
         config=sb_config, output_dir=run_dir / "stages/sb", ...)

load_weights(model, sb.last_checkpoint)
lb_optimizer = Adam(model.parameters(), lr=lb_lr)
lb = fit(model, lb_loader, optimizer=lb_optimizer,
         config=lb_config, output_dir=run_dir / "stages/lb", ...)
```

Это пример со сбросом optimizer и локальных счётчиков стадии. Если протокол требует переноса Adam moments, добавляется отдельная явная операция загрузки optimizer state. Такая передача не должна маскироваться под resume предыдущей стадии. В metadata LB сохраняются родительский checkpoint и политика переноса состояния.

## 7. Организация результатов на диске

Различаем набор экспериментов, run с конкретными условиями/seed и stage обучения. Даже одностадийный run использует `stages/train`, чтобы warm-start не требовал другого формата.

```text
results/<experiment_name>/
  experiment.yaml                  # параметры сценария, оси sweep
  runs/<run_id>/
    config.yaml                    # все фактические параметры этого run
    metadata.json                  # seed, статус, версии, время, устройство
    initialization.pt              # при необходимости общего старта
    stages/
      train/                       # либо sb/ и lb/
        config.yaml                # фактические параметры стадии
        status.json
        metrics.jsonl
        result.json                # сериализованный TrainingResult
        checkpoints/
          epoch_0005.pt
          epoch_0010.pt
          index.json               # last/best → конкретный файл
    analyses/<analysis_id>/
      config.yaml                  # входные checkpoints и параметры анализа
      status.json
      metrics.json                 # скалярные результаты
      curve.csv                    # если есть кривая
      figures/
  summary/
    runs.csv                       # значения отдельных runs
    aggregate.csv                  # mean, std, n
    figures/
```

`run_id` уникален для попытки; одинаковые конфигурации не перезаписываются. Условие, seed и номер попытки — отдельные поля metadata. `analysis_id` также различает повторные анализы с разными параметрами. Пути в JSON/YAML хранятся относительно корня набора экспериментов: каталог можно перенести целиком.

`metadata.json` содержит schema_version, run_id, condition_id, seed, статус, времена начала/окончания, Git commit и признак локальных изменений, версии библиотек, устройство и метаданные датасета. Полная конфигурация сохраняется после подстановки defaults и вычисления фактического batch size. При необходимости воспроизводить незакоммиченный код сохраняется patch или snapshot.

Для записи checkpoint используется временный файл с последующим атомарным переименованием; указатель `last` обновляется после успешной записи. `index.json` позволяет ссылаться на неизменяемое имя checkpoint. Анализ сохраняет конкретный путь, а не подвижный указатель `last`.

## 8. Форматы файлов

### Метрики обучения: JSONL

Одна строка — метрики одного split на завершённой эпохе:

```json
{"epoch": 5, "optimizer_steps": 980, "examples_seen": 250000, "split": "validation", "loss": 0.42, "accuracy": 0.85, "num_examples": 5000}
```

Счётчики локальны для стадии. `epoch` — число завершённых эпох, начиная с 1. `train_online` обозначает метрики во время обучения; `train_eval` — оценку итогового состояния эпохи без случайной аугментации. Их нельзя смешивать при расчёте generalization gap. При resume лог дописывается от восстановленного состояния; записи после его счётчика удаляются или архивируются, чтобы не возникали дубликаты.

### Checkpoint: словарь в `.pt`

```python
{
    "schema_version": 1,
    "run_id": "...",
    "stage_id": "sb",
    "model_state": {...},
    "optimizer_state": {...},
    "scheduler_state": None,
    "completed_epochs": 5,
    "optimizer_steps": 980,
    "examples_seen": 250000,
    "rng_state": {...},
    "loader_generator_state": ...,
    "config_hash": "...",
    "parent_checkpoint": None,
}
```

Сохраняются state_dict, а не сериализованный объект модели. Для анализа достаточно model_state; остальные поля нужны для resume и происхождения. Если позже появится mixed precision, checkpoint расширяется состоянием scaler.

### Результаты анализа

Общий класс не нужен. Каждый скрипт сохраняет свою конфигурацию, скаляры и таблицы:

- Sharpness: метод, epsilon, подпространство, seed, выборка, бюджет оптимизации и достигнутая оценка.
- Interpolation: два checkpoints, правило обработки buffers, строки `alpha, split, loss, accuracy`.
- Hessian: checkpoint, выборка, метод, допуски, seed и найденные оценки.
- Generalization: checkpoint, сопоставимые train_eval/test метрики и явное определение gap.

Это соглашение о хранении, не требование одинаковых сигнатур. Анализ загружает отдельную модель и не изменяет исходный checkpoint. Неуспешный анализ получает собственный failed status, не меняя успешный статус обучения.

## 9. Sweeps и сводки

В `batch_comparison.py` достаточно обычного цикла по batch size и seed. Скрипт заранее сохраняет ожидаемый список условий, создаёт отдельные runs, а затем читает результаты и строит таблицы. Долю train set переводит в целый batch size по явно заданному правилу; сохраняет долю и полученное число.

`runs.csv` содержит condition_id, run_id, seed, attempt, статус, путь checkpoint и метрики. `aggregate.csv` группирует по условию и содержит имя метрики, mean, выборочное std, число успешных и ожидаемых runs. При одном наблюдении std отсутствует; failed runs не заменяются нулями. Повторные попытки одного seed не считаются независимыми наблюдениями.

Состав condition задаёт скрипт: модель, датасет/split, batch size, optimizer, бюджет и параметры анализа. Разные протоколы анализа не объединяются. Для парного SB/LB сравнения можно отдельно агрегировать разность по совпадающим seeds.

## 10. Что согласовать перед реализацией

1. Список первых моделей/датасетов и точный протокол воспроизведения статьи.
2. Бюджет в эпохах для MVP или необходимость сразу поддерживать optimizer steps.
3. Правило выбора checkpoint и периодичность сохранения.
4. Сброс или перенос optimizer state при SB→LB.
5. Формат фиксированного split и требования к resume.
6. Нужна ли общая исходная инициализация для каждой пары SB/LB.

Критерий достаточности этой архитектуры: новый эксперимент можно написать отдельным скриптом, используя готовые фабрики и `fit`; сохранённую модель можно проанализировать без повторного обучения; по файлам результатов понятны условия, происхождение checkpoint и число наблюдений в сводке.
