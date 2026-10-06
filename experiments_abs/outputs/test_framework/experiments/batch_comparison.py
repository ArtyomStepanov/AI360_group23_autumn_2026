"""Сценарий сам определяет условия и собирает результаты."""
from itertools import product


def conditions():
    """Простой sweep без отдельного движка."""
    for batch_size, seed in product((256, 4096), (1, 2, 3, 4, 5)):
        yield {"batch_size": batch_size, "seed": seed}


def main() -> None:
    # Для каждого seed сохранить общую инициализацию для парного SB/LB.
    # Для условия: создать run, данные, модель, optimizer и вызвать fit.
    # Сохранить строки отдельных runs, затем mean/std/n по условиям.
    raise NotImplementedError("Implement the batch comparison scenario")


if __name__ == "__main__":
    main()
