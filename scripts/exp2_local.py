"""Local exp2 runs: C4 by default, 400 models, 10 epochs per model."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import exp2 as experiment
from scripts.exp1 import SETUPS, load_dataset
from eval import evaluate
from metrics import Accuracy
from metrics.sharpness import compute_sharpness
from train import TrainConfig, fit

EPOCHS = experiment.EPOCHS
FIELDS = experiment.FIELDS
DEFAULT_LEARNING_RATES = experiment.DEFAULT_LEARNING_RATES
read_results = experiment.read_results
analyze = experiment.analyze


def make_grid(min_batch=8, max_batch=256,
              min_lr=DEFAULT_LEARNING_RATES[0], max_lr=DEFAULT_LEARNING_RATES[-1]):
    batches, lrs = experiment.make_grid(min_batch, max_batch, min_lr, max_lr)
    if max_batch == 32 * min_batch:
        batches = [b // 64 * min_batch for b in experiment.DEFAULT_BATCH_SIZES]
    return batches, lrs


def source_digest():
    digest = hashlib.sha256(experiment.source_digest().encode())
    paths = [Path(__file__).resolve()] + [PROJECT_ROOT / name for name in (
        "models/instances/C2.py", "models/instances/C3.py",
        "models/instances/C4.py", "models/instances/DNN.py",
    )]
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_model(batch, lr, args, train_dataset, val_dataset, test_dataset, checkpoint):
    model = SETUPS[args.setup]["model_cls"]().to(args.device)
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
            torch.cuda.synchronize(torch.device(args.device))
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
            model, train_dataset, args.sharpness_device, criterion=criterion,
            batch_size=args.eval_batch_size, seed=args.seed, workers=args.workers,
            n_starts=args.sharpness_starts,
            subspace_dim=args.sharpness_subspace_dim,
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
    parser.add_argument("--setup", choices=sorted(SETUPS), default="C4")
    parser.add_argument("--min-batch", type=int)
    parser.add_argument("--max-batch", type=int)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_LEARNING_RATES[0])
    parser.add_argument("--max-lr", type=float, default=DEFAULT_LEARNING_RATES[-1])
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--sharpness-starts", type=int, default=3,
                        help="Число стартов поиска sharpness для каждой модели")
    parser.add_argument("--sharpness-subspace-dim", type=int, default=100,
                        help="Размерность подпространства sharpness; влияет на память и оценку")
    parser.add_argument("--sharpness-device", help="Устройство sharpness; по умолчанию устройство обучения")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Выполнить N незавершённых прогонов; сетка остаётся 400")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    deep = args.setup in ("C2", "C4")
    if args.min_batch is None:
        args.min_batch = 8 if deep else 64
    if args.max_batch is None:
        args.max_batch = 256 if deep else 2048
    if args.eval_batch_size is None:
        args.eval_batch_size = 64 if deep else 256
    if args.sharpness_device is None:
        args.sharpness_device = args.device
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / f"exp2_local_{args.setup}"
    return args

def main(argv=None):
    args = parse_args(argv)
    if (args.workers < 0 or args.eval_batch_size < 1 or args.sharpness_starts < 1
            or args.sharpness_subspace_dim < 1
            or not 0 <= args.seed < 2**32 or (args.limit is not None and args.limit < 1)):
        raise ValueError("Некорректные workers, eval_batch_size, sharpness_starts, sharpness_subspace_dim, seed или limit")
    output_dir = args.output_dir
    results_csv = output_dir / "results.csv"
    manifest = output_dir / "config.json"
    if args.analyze_only:
        config = json.loads(manifest.read_text())
        analyze(read_results(results_csv), config, output_dir)
        return

    setup = SETUPS[args.setup]
    if manifest.exists() and json.loads(manifest.read_text()).get("setup") != args.setup:
        raise ValueError("В каталоге результаты другой конфигурации: укажите новый --output-dir")

    batches, lrs = make_grid(args.min_batch, args.max_batch, args.min_lr, args.max_lr)
    dataset, test_dataset = load_dataset(setup["dataset"])
    val_size = len(dataset) // 10
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    if val_size == 0 or max(batches) > len(train_dataset):
        raise ValueError("Датасет слишком мал для выбранной сетки")
    config = {
        "setup": args.setup, "dataset": setup["dataset"], "epochs": EPOCHS, "seed": args.seed,
        "optimizer": "Adam", "batch_sizes": batches, "learning_rates": lrs,
        "train_size": len(train_dataset), "val_size": val_size, "test_size": len(test_dataset),
        "eval_batch_size": args.eval_batch_size,
        "device_type": torch.device(args.device).type,
        "train_amp": torch.device(args.device).type == "cuda",
        "torch_version": str(torch.__version__),
        "source_digest": source_digest(),
        "sharpness": {
            "epsilon": 5e-4, "maxiter": 10, "subspace_dim": args.sharpness_subspace_dim, "optimize": False,
            "n_starts": args.sharpness_starts,
            "device_type": torch.device(args.sharpness_device).type,
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
    print(f"{args.setup} ({setup['dataset']}), {EPOCHS} эпох, 20 × 20 = {len(planned)} моделей; завершено {len(rows)}", flush=True)
    print(f"Device: {args.device}; batches: {batches}", flush=True)
    print(f"LR: {lrs}", flush=True)
    print(f"Sharpness: device={args.sharpness_device}, subspace_dim={args.sharpness_subspace_dim}, "
          f"starts={args.sharpness_starts}", flush=True)
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
                "если на sharpness — используйте --sharpness-device cpu или уменьшите "
                "--sharpness-subspace-dim (это меняет оценку sharpness); "
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
