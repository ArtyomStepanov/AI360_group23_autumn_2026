"""Small/large batch trajectories: 3 paired runs, 50 epochs, selectable models."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import evaluate
from loggers import BaseLogger
from metrics import Accuracy
from scripts.exp1 import SETUPS as CLASSIFIER_SETUPS
from train import TrainConfig, fit

EPOCHS = 50
NUM_RUNS = 3
LR = 1e-3
BATCHES = {"SB": 265, "LB": 1024}
LOG_EVERY = {"SB": 10, "LB": 1}


class RegularizedCNN(nn.Module):
    """The two-convolution BatchNorm/Dropout model from draft/oldcode.py."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(64 * 8 * 8, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


SETUPS = {
    "RegularizedCNN": {"model_cls": RegularizedCNN, "dataset": "CIFAR10"},
    **CLASSIFIER_SETUPS,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class TrajectoryLogger(BaseLogger):
    """Batch gradients before the update and distance after the update.

    fit calls this logger after Adam.step; the batch gradients are still present.
    Distances include trainable parameters, excluding BatchNorm running buffers.
    Epoch gradient norms are means of batch norms, not full-dataset gradients.
    """

    def __init__(self, model, log_every):
        self.initial_weights = parameters_to_vector(model.parameters()).detach().clone()
        self.log_every = log_every
        self.reset()

    def reset(self):
        self.history = {
            "step": {name: [] for name in ("steps", "grad_norm", "distance", "train_loss")},
            "epoch": {name: [] for name in (
                "steps", "grad_norm", "distance", "train_loss", "test_loss", "test_accuracy",
                "train_seconds", "eval_seconds", "peak_cuda_gib",
            )},
        }
        self.global_step = 0
        self.on_epoch_start()

    def on_epoch_start(self):
        self.grad_sum = self.loss_sum = 0.0
        self.batch_count = 0

    @torch.no_grad()
    def distance(self, model):
        return torch.linalg.vector_norm(
            parameters_to_vector(model.parameters()) - self.initial_weights
        ).item()

    @torch.no_grad()
    def __call__(self, model, loss):
        grad_norm = torch.sqrt(sum(
            p.grad.square().sum() for p in model.parameters() if p.grad is not None
        )).item()
        self.global_step += 1
        self.batch_count += 1
        self.grad_sum += grad_norm
        self.loss_sum += loss
        if self.global_step == 1 or self.global_step % self.log_every == 0:
            for name, value in (
                ("steps", self.global_step), ("grad_norm", grad_norm),
                ("distance", self.distance(model)), ("train_loss", loss),
            ):
                self.history["step"][name].append(value)

    def on_epoch_end(self, model, test, train_seconds, eval_seconds, peak_cuda_gib):
        # All training batches have equal size because fit uses drop_last=True.
        values = {
            "steps": self.global_step,
            "grad_norm": self.grad_sum / self.batch_count,
            "distance": self.distance(model),
            "train_loss": self.loss_sum / self.batch_count,
            "test_loss": test["loss"], "test_accuracy": test["accuracy"] * 100,
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "peak_cuda_gib": peak_cuda_gib,
        }
        for name, value in values.items():
            self.history["epoch"][name].append(value)

    def get_history(self):
        return self.history


def run_training(train_dataset, test_dataset, initial_state, seed, batch, log_every, args, history_path):
    set_seed(seed)
    device = torch.device(args.device)
    model = SETUPS[args.setup]["model_cls"]().to(device)
    model.load_state_dict(initial_state)
    logger = TrajectoryLogger(model, log_every)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    config = TrainConfig(
        optimizer=optimizer, criterion=criterion, batch_size=batch, epoch_count=1,
        seed=None, logger=logger, warm_start=True,
    )
    use_cuda = device.type == "cuda"
    for epoch in range(args.epochs):
        logger.on_epoch_start()
        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        fit(model, config, train_dataset, str(device), workers=args.workers, optimize=False)
        if use_cuda:
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        test = evaluate(
            model, test_dataset, criterion, str(device), batch_size=args.eval_batch_size,
            metrics={"accuracy": Accuracy()},
        )
        if use_cuda:
            torch.cuda.synchronize(device)
        eval_seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if use_cuda else None
        logger.on_epoch_end(model, test, train_seconds, eval_seconds, peak)
        history = logger.get_history()
        save_json(history_path, {
            "setup": args.setup, "seed": seed, "batch_size": batch, "history": history,
        })
        last = history["epoch"]
        memory = f"; peak VRAM={peak:.2f} GiB" if peak is not None else ""
        print(
            f"  epoch {epoch + 1}/{args.epochs}: train_loss={last['train_loss'][-1]:.4f}, "
            f"test_accuracy={last['test_accuracy'][-1]:.2f}%, "
            f"distance={last['distance'][-1]:.3f}; "
            f"train={train_seconds:.1f}s, eval={eval_seconds:.1f}s{memory}", flush=True,
        )
    return logger.get_history()


def load_data(dataset_name):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    root = str(PROJECT_ROOT / "datasets")
    dataset_cls = getattr(datasets, dataset_name)
    return (
        dataset_cls(root=root, train=True, download=True, transform=transform),
        dataset_cls(root=root, train=False, download=True, transform=transform),
    )


def smooth(runs, weight):
    values = np.asarray(runs, dtype=float).copy()
    for column in range(1, values.shape[1]):
        values[:, column] = weight * values[:, column - 1] + (1 - weight) * values[:, column]
    return values


def generate_plots(results, output_dir, setup="RegularizedCNN", batches=None):
    if batches is None:
        batches = BATCHES
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, section, field, title, smoothing=None):
        for kind, color in (("SB", "blue"), ("LB", "red")):
            histories = results[kind]
            values = np.asarray([h[section][field] for h in histories])
            if smoothing is not None:
                values = smooth(values, smoothing)
            x = (np.arange(1, values.shape[1] + 1) if section == "epoch"
                 else histories[0]["step"]["steps"])
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(x, mean, color=color, label=f"{kind}: batch {batches[kind]}")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        if section == "step":
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Epoch" if section == "epoch" else "Optimizer steps (log scale)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, (
        ("distance", "Distance from initial parameters"),
        ("grad_norm", "Mean batch gradient norm"),
        ("train_loss", "Mean training batch loss"),
        ("test_loss", "Test loss"), ("test_accuracy", "Test accuracy (%)"),
        ("steps", "Cumulative optimizer steps"),
    )):
        draw(ax, "epoch", field, title)
    fig.suptitle(f"{setup}: epoch trajectories, mean ± std across paired runs")
    fig.savefig(output_dir / "epochs.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, (field, title, weight) in zip(axes, (
        ("grad_norm", "Batch gradient norm (EMA)", 0.85),
        ("distance", "Distance from initial parameters (EMA)", 0.1),
        ("train_loss", "Training batch loss (EMA)", 0.85),
    )):
        draw(ax, "step", field, title, smoothing=weight)
    fig.suptitle(f"{setup}: step trajectories, mean ± std of smoothed runs")
    fig.savefig(output_dir / "steps.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=sorted(SETUPS), default="RegularizedCNN",
                        help="Архитектура; по умолчанию исходная модель из draft/oldcode.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--small-batch", type=int, default=BATCHES["SB"])
    parser.add_argument("--large-batch", type=int, default=BATCHES["LB"])
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Одна пара запусков по одной эпохе")
    args = parser.parse_args(argv)
    args.epochs, args.runs = (1, 1) if args.smoke_test else (EPOCHS, NUM_RUNS)
    if args.output_dir is None:
        suffix = "_smoke" if args.smoke_test else ""
        args.output_dir = PROJECT_ROOT / "results" / f"exp3_{args.setup}{suffix}"
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.workers < 0 or args.eval_batch_size < 1:
        raise ValueError("Нужны workers >= 0 и eval_batch_size >= 1")
    if not 2 <= args.small_batch < args.large_batch:
        raise ValueError("Нужны 2 <= small_batch < large_batch")
    batches = {"SB": args.small_batch, "LB": args.large_batch}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Каталог результатов непустой: укажите новый --output-dir")
    setup = SETUPS[args.setup]
    train_dataset, test_dataset = load_data(setup["dataset"])
    if len(train_dataset) < args.large_batch:
        raise ValueError(f"Недостаточно train-примеров для батча {args.large_batch}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(43, 43 + args.runs))
    steps = {kind: len(train_dataset) // batch * args.epochs for kind, batch in batches.items()}
    save_json(args.output_dir / "config.json", {
        "setup": args.setup, "model": setup["model_cls"].__name__, "dataset": setup["dataset"],
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "epochs": args.epochs, "seeds": seeds, "batch_sizes": batches,
        "optimizer": "Adam", "lr": LR, "amp": False, "drop_last": True,
        "steps_per_run": steps, "train_size": len(train_dataset), "test_size": len(test_dataset),
        "device": args.device, "workers": args.workers, "eval_batch_size": args.eval_batch_size,
        "torch_version": str(torch.__version__), "smoke_test": args.smoke_test,
    })
    print(f"{args.setup} ({setup['dataset']}), device: {args.device}; "
          f"{args.runs} pairs × {args.epochs} epochs; steps per run: {steps}", flush=True)
    results = {kind: [] for kind in batches}
    for run, seed in enumerate(seeds, start=1):
        set_seed(seed)
        base_model = setup["model_cls"]()
        initial_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        del base_model
        for kind, batch in batches.items():
            print(f"Run {run}/{args.runs}, seed={seed}, {kind}, batch={batch}", flush=True)
            try:
                history = run_training(
                    train_dataset, test_dataset, initial_state, seed, batch, LOG_EVERY[kind], args,
                    args.output_dir / f"run_{seed}_{kind}.json",
                )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"Батч {batch} не поместился в VRAM. Истории завершённых эпох сохранены. "
                    "Батч и float32 автоматически не меняются: это изменило бы эксперимент."
                ) from error
            results[kind].append(history)
    generate_plots(results, args.output_dir, args.setup, batches)
    distances = {kind: float(np.mean([h["epoch"]["distance"][-1] for h in histories]))
                 for kind, histories in results.items()}
    ratio = distances["SB"] / distances["LB"] if distances["LB"] > 0 else None
    save_json(args.output_dir / "summary.json", {
        "mean_final_distance": distances, "distance_ratio_SB_LB": ratio,
    })
    print(f"Final mean distances: {distances}; SB/LB={ratio}. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
