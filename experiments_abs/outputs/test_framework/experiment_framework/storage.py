"""Минимальные работающие операции с каталогами и TrainingResult."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .contracts import TrainingResult


def _component(value: str) -> str:
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("Expected a single directory name")
    return value


@dataclass(frozen=True)
class StagePaths:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def create_stage(self, name: str) -> StagePaths:
        paths = StagePaths(self.root / "stages" / _component(name))
        paths.root.mkdir(parents=True, exist_ok=False)
        paths.checkpoints.mkdir()
        return paths

    def create_analysis(self, name: str) -> Path:
        path = self.root / "analyses" / _component(name)
        path.mkdir(parents=True, exist_ok=False)
        return path


def create_run(results_root: Path, experiment_name: str) -> RunPaths:
    """Создать новый run, не перезаписывая предыдущие попытки."""
    root = results_root / _component(experiment_name) / "runs" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return RunPaths(root)


def write_result(result: TrainingResult) -> Path:
    """Сохранить result.json; файловые ссылки относительны каталогу стадии."""
    stage = result.stage_dir.resolve()

    def relative(path: Path) -> str:
        return path.resolve().relative_to(stage).as_posix()

    payload = {
        "schema_version": 1,
        "last_checkpoint": relative(result.last_checkpoint),
        "best_checkpoint": relative(result.best_checkpoint)
            if result.best_checkpoint is not None else None,
        "metrics_file": relative(result.metrics_file),
        "completed_epochs": result.completed_epochs,
        "optimizer_steps": result.optimizer_steps,
        "examples_seen": result.examples_seen,
    }
    target = stage / "result.json"
    temporary = stage / f".result-{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_result(stage_dir: Path) -> TrainingResult:
    """Прочитать результат, в том числе после переноса всего каталога."""
    stage = stage_dir.resolve()
    payload = json.loads((stage / "result.json").read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported result schema version")

    def local(value: str) -> Path:
        path = (stage / value).resolve()
        path.relative_to(stage)
        return path

    return TrainingResult(
        stage_dir=stage,
        last_checkpoint=local(payload["last_checkpoint"]),
        best_checkpoint=local(payload["best_checkpoint"])
            if payload["best_checkpoint"] is not None else None,
        metrics_file=local(payload["metrics_file"]),
        completed_epochs=payload["completed_epochs"],
        optimizer_steps=payload["optimizer_steps"],
        examples_seen=payload["examples_seen"],
    )
