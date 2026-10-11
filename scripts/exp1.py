import csv
import sys
from itertools import product
from pathlib import Path

import torch
from torch import nn

# Добавляем корень репозитория в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import evaluate
from metrics import Accuracy
from metrics.sharpness import compute_sharpness
from models.instances.C1 import C1
from models.instances.C2 import C2
from models.instances.C3 import C3
from models.instances.C4 import C4
from train import TrainConfig, fit

# Сетапы C1-C4 (Keskar et al.)
SETUPS = {
    "C1": {"dataset": "CIFAR10", "model_cls": C1},
    "C2": {"dataset": "CIFAR10", "model_cls": C2},
    "C3": {"dataset": "CIFAR100", "model_cls": C3},
    "C4": {"dataset": "CIFAR100", "model_cls": C4},
}


def get_batch_sizes(dataset_len: int) -> list[int]:
    """
    Степени двойки от 64 до максимальной <= dataset_len
    + 10 линейных долей: 1%, 2%, ..., 10% от размера выборки.
    """
    powers_of_2 = []
    bs = 64
    while bs <= dataset_len:
        powers_of_2.append(bs)
        bs *= 2

    # Линейные доли от 1% до 10% с шагом 1%
    linear_fractions = [int(dataset_len * (i / 100.0)) for i in range(1, 11)]

    all_sizes = set(powers_of_2 + linear_fractions)
    return sorted([b for b in all_sizes if 64 <= b <= dataset_len])


def load_dataset(dataset_name: str):
    from torchvision import datasets, transforms

    data_dir = PROJECT_ROOT / "datasets"
    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        train_ds = datasets.CIFAR100(root=str(data_dir), train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")
    return train_ds, test_ds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [67, 69, 228, 360, 1488]
    epochs_list = [5, 10, 20]
    lr = 1e-3
    workers = 8

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "c1_c4_results.csv"

    # Чтение ранее завершенных запусков для безопасного перезапуска
    completed = set()
    if results_csv.exists() and results_csv.stat().st_size > 0:
        with results_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            expected_fields = [
                "setup", "dataset", "model", "epochs", "batch_size", "seed",
                "loss", "accuracy", "sharpness",
            ]
            if reader.fieldnames != expected_fields:
                raise ValueError(f"Несовместимый заголовок CSV: {results_csv}")
            for r in reader:
                completed.add((
                    r["setup"], r["dataset"], r["model"],
                    int(r["epochs"]), int(r["batch_size"]), int(r["seed"]),
                ))
        print(f"Найдено {len(completed)} завершенных прогонов в {results_csv.name}. Они будут пропущены.")

    for setup_name, cfg in SETUPS.items():
        ds_name = cfg["dataset"]
        model_cls = cfg["model_cls"]

        train_dataset, test_dataset = load_dataset(ds_name)
        batch_sizes = get_batch_sizes(len(train_dataset))

        print(f"\nЗапуск сетапа {setup_name} ({model_cls.__name__} на {ds_name})")
        print(f"Сетка батчей ({len(batch_sizes)} шт.): {batch_sizes}")

        for epochs, bs, seed in product(epochs_list, batch_sizes, seeds):
            run_key = (setup_name, ds_name, model_cls.__name__, epochs, bs, seed)
            if run_key in completed:
                continue

            print(f"[{setup_name}] Epochs: {epochs:2d} | Batch: {bs:5d} | Seed: {seed:4d}")

            # Инициализация модели строго по сигнатуре проекта
            model = model_cls().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            config = TrainConfig(
                optimizer=optimizer,
                criterion=criterion,
                batch_size=bs,
                epoch_count=epochs,
                seed=seed,
            )

            # 1. Обучение
            fit(model, config, train_dataset, device, workers=workers)

            # 2. Оценка test loss и accuracy
            eval_bs = min(len(test_dataset), 2048)
            eval_metrics = evaluate(
                model=model,
                dataset=test_dataset,
                criterion=criterion,
                device=device,
                batch_size=eval_bs,
                metrics={"accuracy": Accuracy()},
            )
            test_loss = float(eval_metrics["loss"])
            test_acc = float(eval_metrics["accuracy"])

            # 3. Вычисление Sharpness на обучающей выборке
            sharpness_val = compute_sharpness(
                model=model,
                dataset=train_dataset,
                device=device,
                criterion=criterion,
                batch_size=eval_bs,
                seed=seed,
                workers=workers,
            )

            # 4. Инкрементальная запись в CSV
            record = {
                "setup": setup_name,
                "dataset": ds_name,
                "model": model_cls.__name__,
                "epochs": epochs,
                "batch_size": bs,
                "seed": seed,
                "loss": test_loss,
                "accuracy": test_acc,
                "sharpness": float(sharpness_val),
            }

            write_header = not results_csv.exists() or results_csv.stat().st_size == 0
            with results_csv.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                if write_header:
                    writer.writeheader()
                writer.writerow(record)
            completed.add(run_key)

            print(f"   -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Sharpness: {float(sharpness_val):.4f}")


if __name__ == "__main__":
    main()
