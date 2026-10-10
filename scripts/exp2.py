"""C1: 20 learning rates × 20 batch sizes, 10 epochs per model."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import evaluate
from metrics import Accuracy
from metrics.sharpness import compute_sharpness
from models.instances.C1 import C1
from scripts.exp1 import load_dataset
from train import TrainConfig, fit

EPOCHS = 10
GRID_SIZE = 20
DEFAULT_BATCH_SIZES = [
    64, 128, 192, 256, 320, 384, 448, 512, 640, 768,
    896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
]
DEFAULT_LEARNING_RATES = [
    0.000125, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006,
    0.008, 0.012, 0.016, 0.024, 0.032, 0.04, 0.048, 0.064, 0.096, 0.128,
]
FIELDS = [
    "batch_size", "lr", "seed", "epochs", "status", "val_loss", "val_accuracy",
    "test_loss", "test_accuracy", "sharpness", "train_seconds", "eval_seconds",
]
CRITERIA = {"accuracy": "val_accuracy", "loss": "val_loss", "sharpness": "sharpness"}


def source_digest():
    """Prevent resumed runs from mixing different training/model implementations."""
    paths = [Path(__file__).resolve()] + [PROJECT_ROOT / name for name in (
        "train.py", "eval.py", "scripts/exp1.py", "models/abs_model.py",
        "models/instances/C1.py", "models/instances/ShallowNN.py",
        "metrics/__init__.py", "metrics/base.py", "metrics/accuracy.py", "metrics/sharpness.py",
    )]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_grid(min_batch=64, max_batch=2048,
              min_lr=DEFAULT_LEARNING_RATES[0], max_lr=DEFAULT_LEARNING_RATES[-1]):
    """Readable default grids, or geometric custom ranges: exactly 400 pairs."""
    if not (2 <= min_batch < max_batch and 0 < min_lr < max_lr
            and math.isfinite(min_lr) and math.isfinite(max_lr)):
        raise ValueError("Нужны 2 <= min_batch < max_batch и 0 < min_lr < max_lr")
    batches = (DEFAULT_BATCH_SIZES.copy() if (min_batch, max_batch) == (64, 2048)
               else np.rint(np.geomspace(min_batch, max_batch, GRID_SIZE)).astype(int).tolist())
    lrs = (DEFAULT_LEARNING_RATES.copy()
           if (min_lr, max_lr) == (DEFAULT_LEARNING_RATES[0], DEFAULT_LEARNING_RATES[-1])
           else np.geomspace(min_lr, max_lr, GRID_SIZE).tolist())
    if len(set(batches)) != GRID_SIZE or len(set(lrs)) != GRID_SIZE:
        raise ValueError("Диапазон слишком узкий для 20 различных значений")
    return batches, lrs


def read_results(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Несовместимый заголовок CSV: {path}")
        rows = []
        keys = set()
        for row in reader:
            for key in ("batch_size", "seed", "epochs"):
                row[key] = int(row[key])
            for key in FIELDS:
                if key not in ("batch_size", "seed", "epochs", "status"):
                    row[key] = float(row[key])
            key = (row["batch_size"], row["lr"])
            if key in keys or row["status"] not in ("ok", "nonfinite"):
                raise ValueError(f"Повтор или некорректный статус в CSV: {key}")
            keys.add(key)
            rows.append(row)
    return rows


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def linear_fits(batches, lrs):
    """Compare lr=a*batch+b and the stronger hypothesis lr=k*batch."""
    x, y = np.asarray(batches, dtype=float), np.asarray(lrs, dtype=float)
    if len(x) < 2:
        return None
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    k = float(x @ y / (x @ x))
    total = 0.0 if np.all(y == y[0]) else float(np.sum((y - y.mean()) ** 2))

    def r_squared(prediction):
        return None if total == 0 else float(1 - np.sum((y - prediction) ** 2) / total)

    return {
        "slope": float(slope), "intercept": float(intercept),
        "r2_linear": r_squared(slope * x + intercept),
        "proportionality_k": k, "r2_proportional": r_squared(k * x),
        "log_log_slope": float(np.polyfit(np.log(x), np.log(y), 1)[0]),
    }


def analyze(rows, config, output_dir):
    """Save separate optima, fits, and all grid values without forcing a trend."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    batches, lrs = config["batch_sizes"], config["learning_rates"]
    valid = [r for r in rows if r["status"] == "ok"]
    complete_batches = [
        batch for batch in batches
        if {r["lr"] for r in rows if r["batch_size"] == batch} == set(lrs)
    ]
    best = []
    report = {
        "planned": 400, "recorded": len(rows), "valid": len(valid),
        "complete_batches": complete_batches,
        "incomplete_batches": [batch for batch in batches if batch not in complete_batches],
        "criteria": {},
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    heatmaps, heat_axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, heat_axis, (criterion, field) in zip(axes, heat_axes, CRITERIA.items()):
        metric_rows = [r for r in rows if math.isfinite(r[field])]
        selected = []
        matrix = np.full((len(lrs), len(batches)), np.nan)
        for row in metric_rows:
            matrix[lrs.index(row["lr"]), batches.index(row["batch_size"])] = row[field]
        for batch in batches:
            if batch not in complete_batches:
                continue
            candidates = [r for r in metric_rows if r["batch_size"] == batch]
            if not candidates:
                continue
            # Accuracy ties: prefer lower validation loss, then lower lr.
            if criterion == "accuracy":
                winner = min(candidates, key=lambda r: (
                    -r[field], r["val_loss"] if math.isfinite(r["val_loss"]) else float("inf"), r["lr"],
                ))
            else:
                winner = min(candidates, key=lambda r: (r[field], r["lr"]))
            entry = {
                "criterion": criterion, "batch_size": batch, "lr": winner["lr"],
                "value": winner[field], "lr_per_batch": winner["lr"] / batch,
                "boundary": winner["lr"] in (lrs[0], lrs[-1]),
                "ties": sum(r[field] == winner[field] for r in candidates),
            }
            selected.append(entry)
            best.append(entry)
        unambiguous = [
            r for r in selected if not r["boundary"] and r["ties"] == 1
            and sum(row["batch_size"] == r["batch_size"] for row in metric_rows) == len(lrs)
        ]
        x = np.array([r["batch_size"] for r in unambiguous])
        y = np.array([r["lr"] for r in unambiguous])
        fits = linear_fits(x, y) if len(x) >= 3 else None
        report["criteria"][criterion] = {
            "selection_field": field, "boundary_optima": sum(r["boundary"] for r in selected),
            "batches_with_ties": sum(r["ties"] > 1 for r in selected),
            "fit_batches": x.tolist(), "finite_models": len(metric_rows),
            "zero_values": sum(r[field] == 0 for r in metric_rows), "fits": fits,
        }
        axis.scatter([r["batch_size"] for r in selected], [r["lr"] for r in selected],
                     color="gray", marker="x", label="All best tested lr")
        axis.scatter(x, y, label="Unambiguous interior optima")
        if fits is not None:
            axis.plot(x, fits["slope"] * x + fits["intercept"], label="lr = a B + b")
            axis.plot(x, fits["proportionality_k"] * x, "--", label="lr = k B")
        axis.set(title=criterion, xlabel="Training batch size", ylabel="Best lr")
        anchors = [b for b in batches if b & (b - 1) == 0]
        axis.set_xticks(anchors if anchors else batches[::3])
        axis.grid(alpha=0.3)
        axis.legend()
        # The grid is discrete; labels show actual values, rather than interpolated optima.
        plot = heat_axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest")
        batch_ticks = [batches.index(b) for b in anchors] if anchors else list(range(0, len(batches), 3))
        heat_axis.set_xticks(batch_ticks, [str(batches[i]) for i in batch_ticks], rotation=45)
        heat_axis.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs], fontsize=8)
        heat_axis.set(title=criterion, xlabel="Training batch size", ylabel="Learning rate")
        heatmaps.colorbar(plot, ax=heat_axis)
    figure.savefig(output_dir / "best_lr.png", dpi=160)
    heatmaps.savefig(output_dir / "grid.png", dpi=160)
    plt.close(figure)
    plt.close(heatmaps)
    write_csv(output_dir / "best_lr.csv", [
        "criterion", "batch_size", "lr", "value", "lr_per_batch", "boundary", "ties",
    ], best)
    (output_dir / "linearity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run_model(batch, lr, args, train_dataset, val_dataset, test_dataset, checkpoint):
    model = C1().to(args.device)
    criterion = nn.CrossEntropyLoss()
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        train_seconds = saved["train_seconds"]
        del saved
        print("  Загружена уже обученная модель; повторяем только оценки.", flush=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        config = TrainConfig(
            optimizer=optimizer, criterion=criterion, batch_size=batch,
            epoch_count=EPOCHS, seed=args.seed,
        )
        started = time.perf_counter()
        fit(model, config, train_dataset, args.device, workers=args.workers)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        optimizer.zero_grad(set_to_none=True)
        del optimizer, config
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "train_seconds": train_seconds,
        }, temporary)
        temporary.replace(checkpoint)

    started = time.perf_counter()
    val = evaluate(model, val_dataset, criterion, args.device, args.eval_batch_size, metrics={"accuracy": Accuracy()})
    test = evaluate(model, test_dataset, criterion, args.device, args.eval_batch_size, metrics={"accuracy": Accuracy()})
    sharpness = float("nan")
    if all(math.isfinite(v) for v in (*val.values(), *test.values())):
        sharpness = compute_sharpness(
            model, train_dataset, args.device, criterion=criterion,
            batch_size=args.eval_batch_size, seed=args.seed, workers=args.workers,
            n_starts=args.sharpness_starts,
        )
    status = "ok" if math.isfinite(sharpness) else "nonfinite"
    return {
        "batch_size": batch, "lr": lr, "seed": args.seed, "epochs": EPOCHS,
        "status": status, "val_loss": val["loss"], "val_accuracy": val["accuracy"],
        "test_loss": test["loss"], "test_accuracy": test["accuracy"], "sharpness": sharpness,
        "train_seconds": train_seconds, "eval_seconds": time.perf_counter() - started,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-batch", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=2048)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_LEARNING_RATES[0])
    parser.add_argument("--max-lr", type=float, default=DEFAULT_LEARNING_RATES[-1])
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--sharpness-starts", type=int, default=3,
                        help="Число стартов поиска sharpness для каждой модели")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "exp2")
    parser.add_argument("--limit", type=int, help="Выполнить N незавершённых прогонов; сетка остаётся 400")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.workers < 0 or args.eval_batch_size < 1 or args.sharpness_starts < 1
            or not 0 <= args.seed < 2**32 or (args.limit is not None and args.limit < 1)):
        raise ValueError("Некорректные workers, eval_batch_size, sharpness_starts, seed или limit")
    output_dir = args.output_dir
    results_csv = output_dir / "results.csv"
    manifest = output_dir / "config.json"
    if args.analyze_only:
        config = json.loads(manifest.read_text())
        analyze(read_results(results_csv), config, output_dir)
        return

    batches, lrs = make_grid(args.min_batch, args.max_batch, args.min_lr, args.max_lr)
    dataset, test_dataset = load_dataset("CIFAR10")
    val_size = len(dataset) // 10
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    if val_size == 0 or max(batches) > len(train_dataset):
        raise ValueError("Датасет слишком мал для выбранной сетки")
    config = {
        "setup": "C1", "dataset": "CIFAR10", "epochs": EPOCHS, "seed": args.seed,
        "optimizer": "Adam", "batch_sizes": batches, "learning_rates": lrs,
        "train_size": len(train_dataset), "val_size": val_size, "test_size": len(test_dataset),
        "eval_batch_size": args.eval_batch_size,
        "device_type": torch.device(args.device).type,
        "train_amp": torch.device(args.device).type == "cuda",
        "torch_version": str(torch.__version__),
        "source_digest": source_digest(),
        "sharpness": {
            "epsilon": 5e-4, "maxiter": 10, "subspace_dim": 100, "optimize": False,
            "n_starts": args.sharpness_starts,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Конфигурация изменилась: укажите новый --output-dir")
    else:
        if results_csv.exists() or (output_dir / "checkpoints").exists():
            raise ValueError("Результаты без config.json: укажите новый --output-dir")
        manifest.write_text(json.dumps(config, indent=2) + "\n")
    rows = read_results(results_csv)
    planned = set(product(batches, lrs))
    if any((r["batch_size"], r["lr"]) not in planned or r["seed"] != args.seed or r["epochs"] != EPOCHS for r in rows):
        raise ValueError("CSV не соответствует config.json")
    completed = {(r["batch_size"], r["lr"]) for r in rows}
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    print(f"C1, {EPOCHS} эпох, 20 × 20 = {len(planned)} моделей; завершено {len(rows)}", flush=True)
    print(f"Device: {args.device}; batches: {batches}", flush=True)
    print(f"LR: {lrs}", flush=True)
    executed = 0
    for batch, (lr_index, lr) in product(batches, enumerate(lrs)):
        if (batch, lr) in completed:
            continue
        print(f"[{len(rows) + 1}/400] batch={batch}, lr={lr:.6g}", flush=True)
        checkpoint = checkpoints / f"batch_{batch}_lr_{lr_index:02d}.pt"
        try:
            row = run_model(batch, lr, args, train_dataset, val_dataset, test_dataset, checkpoint)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "Недостаточно VRAM. Сетка не изменена автоматически. Если ошибка на оценке, "
                "повторите с меньшим --eval-batch-size в новом --output-dir; "
                "для обучения используйте GPU с большей памятью или новую сетку --max-batch."
            ) from error
        header = not results_csv.exists() or results_csv.stat().st_size == 0
        with results_csv.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if header:
                writer.writeheader()
            writer.writerow(row)
        rows.append(row)
        print(f"  {row['status']}: val_accuracy={row['val_accuracy']:.4f}, "
              f"val_loss={row['val_loss']:.4f}, sharpness={row['sharpness']:.4f}; "
              f"train={row['train_seconds']:.1f}s, eval={row['eval_seconds']:.1f}s", flush=True)
        executed += 1
        if args.limit is not None and executed >= args.limit:
            break
    analyze(rows, config, output_dir)
    print(f"Записано {len(rows)}/400, успешных {sum(r['status'] == 'ok' for r in rows)}. "
          f"Результаты: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
