"""Эскиз композиции двух стадий; запускается после реализации заглушек."""
from pathlib import Path

from experiment_framework import TrainConfig, create_run, fit, load_weights
from experiment_framework.data.cifar10 import build_dataset
from experiment_framework.models.convnet import build_model


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(1)
    data = build_dataset(Path("datasets"), split_file=Path("datasets/splits/cifar10.json"))
    model = build_model(input_shape=data.input_shape, num_classes=data.num_classes)
    run = create_run(Path("results"), "warm_start")
    sb_stage = run.create_stage("sb")
    lb_stage = run.create_stage("lb")
    evaluation = {"train_eval": DataLoader(data.train_eval, batch_size=256)}
    if data.validation is not None:
        evaluation["validation"] = DataLoader(data.validation, batch_size=256)

    # Перед fit сценарий должен сохранить конфигурацию и метаданные run/stages.
    loss = torch.nn.CrossEntropyLoss()
    sb = fit(
        model, DataLoader(data.train, batch_size=256, shuffle=True),
        loss_fn=loss, optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        config=TrainConfig(epochs=10), output_dir=sb_stage.root,
        eval_loaders=evaluation,
    )
    load_weights(model, sb.last_checkpoint)
    # Новый optimizer явно сбрасывает Adam moments.
    fit(
        model, DataLoader(data.train, batch_size=4096, shuffle=True),
        loss_fn=loss, optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        config=TrainConfig(epochs=20), output_dir=lb_stage.root,
        eval_loaders=evaluation,
    )


if __name__ == "__main__":
    main()
