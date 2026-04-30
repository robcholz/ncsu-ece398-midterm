from __future__ import annotations

import unittest

import numpy as np
import torch

from benchmark.host import (
    apply_task,
    apply_train_normalization,
    augment_batch,
    tune_binary_threshold,
)
from benchmark.ensemble import parse_weights
from model.cnn import build_model
from model.dataset import BenchmarkSplit, LABELS


class BenchmarkHostTests(unittest.TestCase):
    def test_binary_cough_task_maps_cough_to_positive(self) -> None:
        split = BenchmarkSplit(
            x_train=np.zeros((3, 3, 4), dtype=np.float32),
            y_train=np.asarray([0, LABELS.index("Cough"), LABELS.index("Speech")]),
            x_val=np.zeros((2, 3, 4), dtype=np.float32),
            y_val=np.asarray([LABELS.index("Cough"), LABELS.index("Laugh")]),
            class_names=LABELS,
            train_subjects=("001",),
            val_subjects=("002",),
        )

        mapped = apply_task(split, "binary-cough")

        self.assertEqual(mapped.class_names, ("non_cough", "cough"))
        self.assertEqual(mapped.y_train.tolist(), [0, 1, 0])
        self.assertEqual(mapped.y_val.tolist(), [1, 0])

    def test_train_normalization_uses_train_statistics_for_val(self) -> None:
        split = BenchmarkSplit(
            x_train=np.asarray(
                [
                    [[1, 3], [10, 14]],
                    [[5, 7], [18, 22]],
                ],
                dtype=np.float32,
            ),
            y_train=np.asarray([0, 1]),
            x_val=np.asarray([[[3, 5], [14, 18]]], dtype=np.float32),
            y_val=np.asarray([0]),
            class_names=("a", "b"),
            train_subjects=("001",),
            val_subjects=("002",),
        )

        normalized = apply_train_normalization(split)

        self.assertTrue(np.allclose(normalized.x_train.mean(axis=(0, 2)), 0.0))
        self.assertTrue(np.allclose(normalized.x_train.std(axis=(0, 2)), 1.0))
        self.assertTrue(np.allclose(normalized.x_val[0, 0], [-0.4472136, 0.4472136]))

    def test_tune_binary_threshold_improves_macro_f1(self) -> None:
        y_true = np.asarray([0, 0, 0, 1, 1])
        positive_probability = np.asarray([0.10, 0.20, 0.60, 0.55, 0.90])

        threshold, report = tune_binary_threshold(
            y_true,
            positive_probability,
            ("non_cough", "cough"),
            "macro-f1",
        )

        self.assertGreater(threshold, 0.20)
        self.assertLessEqual(threshold, 0.56)
        self.assertGreater(report["macro_f1"], 0.75)

    def test_augment_batch_is_noop_when_disabled(self) -> None:
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        augmented = augment_batch(
            x,
            noise_std=0.0,
            scale_std=0.0,
            max_time_shift=0,
        )

        self.assertTrue(torch.equal(augmented, x))

    def test_augment_batch_can_time_shift(self) -> None:
        torch.manual_seed(398)
        x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)

        augmented = augment_batch(
            x,
            noise_std=0.0,
            scale_std=0.0,
            max_time_shift=1,
        )

        self.assertEqual(augmented.shape, x.shape)
        self.assertCountEqual(augmented.reshape(-1).tolist(), x.reshape(-1).tolist())

    def test_convgru_model_returns_class_logits(self) -> None:
        model = build_model("convgru", in_channels=4, num_classes=8, dropout=0.1)
        x = torch.zeros(2, 4, 100)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 8))

    def test_statsmlp_model_returns_class_logits(self) -> None:
        model = build_model("statsmlp", in_channels=4, num_classes=8, dropout=0.1)
        x = torch.zeros(2, 4, 100)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 8))

    def test_parse_ensemble_weights_normalizes_values(self) -> None:
        self.assertEqual(parse_weights("9,1", 2), [0.9, 0.1])


if __name__ == "__main__":
    unittest.main()
