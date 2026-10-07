import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from eval import evaluate
from metrics import Accuracy, BaseMetric, BinaryAccuracy


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.outputs = torch.tensor([[3., 0.], [0., 3.], [2., 0.], [0., 2.], [0., 4.]])
        self.targets = torch.tensor([0, 1, 1, 1, 0])
        self.dataset = TensorDataset(self.outputs, self.targets)

    def test_multiclass_and_reset_across_calls(self):
        metric = Accuracy()
        for batch_size in (1, 2, 128):
            result = evaluate(
                nn.Identity(), self.dataset, nn.CrossEntropyLoss(), "cpu",
                batch_size, metrics={"accuracy": metric},
            )
            self.assertEqual(result["accuracy"], 0.6)
            self.assertEqual(metric.total, 5)
            self.assertAlmostEqual(
                result["loss"], nn.CrossEntropyLoss()(self.outputs, self.targets).item(),
                places=6,
            )

    def test_binary_logits(self):
        for shape in ((5,), (5, 1)):
            outputs = torch.tensor([-2., 3., 1., -1., 0.]).reshape(shape)
            targets = torch.tensor([0., 1., 0., 0., 1.]).reshape(shape)
            result = evaluate(
                nn.Identity(), TensorDataset(outputs, targets),
                nn.BCEWithLogitsLoss(), "cpu", 2,
                metrics={"accuracy": BinaryAccuracy()},
            )
            self.assertEqual(result["accuracy"], 0.8)
            self.assertAlmostEqual(
                result["loss"], nn.BCEWithLogitsLoss()(outputs, targets).item(), places=6,
            )

    def test_default_and_loss_only(self):
        args = (nn.Identity(), self.dataset, nn.CrossEntropyLoss(), "cpu")
        self.assertEqual(evaluate(*args)["accuracy"], 0.6)
        self.assertEqual(set(evaluate(*args, metrics={})), {"loss"})
        with self.assertRaises(ValueError):
            evaluate(*args, metrics={"loss": Accuracy()})

    def test_mode_restoration_on_metric_failure(self):
        class FailingMetric(BaseMetric):
            def reset(self): pass
            def update(self, outputs, targets):
                if torch.is_grad_enabled():
                    raise AssertionError("Gradients must be disabled")
                raise RuntimeError("metric failure")
            def compute(self): return 0.

        for training in (False, True):
            model = nn.Identity().train(training)
            with self.assertRaisesRegex(RuntimeError, "metric failure"):
                evaluate(model, self.dataset, nn.CrossEntropyLoss(), "cpu",
                         metrics={"custom": FailingMetric()})
            self.assertEqual(model.training, training)

    def test_accuracy_rejects_broadcasting(self):
        with self.assertRaises(ValueError):
            Accuracy().update(self.outputs, self.targets[:, None])
        with self.assertRaises(ValueError):
            BinaryAccuracy().update(self.outputs, self.targets)


if __name__ == "__main__":
    unittest.main()
