from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.dataset import LABELS, WindowConfig, build_windows, discover_recordings


class DatasetWindowTests(unittest.TestCase):
    def test_discovers_recordings_and_applies_sync_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(Path(tmp))

            recordings = discover_recordings(root)

            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0].subject, "005")
            self.assertAlmostEqual(recordings[0].events[0].start, 11.0)
            self.assertAlmostEqual(recordings[0].events[0].end, 12.0)

    def test_build_windows_labels_overlapping_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(Path(tmp))

            x, y, metadata = build_windows(
                root,
                WindowConfig(stride_seconds=1.0, max_background_ratio=None),
            )

            self.assertEqual(x.shape[1:], (3, 200))
            self.assertIn(LABELS.index("Cough"), set(y.tolist()))
            cough_idx = int(np.where(y == LABELS.index("Cough"))[0][0])
            self.assertEqual(metadata[cough_idx]["subject"], "005")

    def test_event_centered_sampling_keeps_event_and_background_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(Path(tmp))

            x, y, metadata = build_windows(
                root,
                WindowConfig(
                    sampling_strategy="event-centered",
                    event_windows_per_event=2,
                    background_windows_per_event=2,
                    background_exclusion_seconds=0.25,
                    max_background_ratio=None,
                ),
            )

            self.assertEqual(x.shape, (4, 3, 200))
            self.assertEqual(y.tolist().count(LABELS.index("Cough")), 2)
            self.assertEqual(y.tolist().count(LABELS.index("background")), 2)
            self.assertEqual({item["source"] for item in metadata}, {"event", "background"})

    def _make_dataset(self, root: Path) -> Path:
        dataset_root = root / "Multimodal Cough Dataset"
        trial = dataset_root / "005" / "Trial_1_No_Talking"
        trial.mkdir(parents=True)
        (dataset_root / "005" / "sync_time.txt").write_text(
            "trial 1  data_label_start 5.0  imu_start 10.0\n",
            encoding="utf-8",
        )
        with (trial / "Accelerometer.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoc (ms)",
                    "timestamp (-0400)",
                    "elapsed (s)",
                    "x-axis (g)",
                    "y-axis (g)",
                    "z-axis (g)",
                ]
            )
            for i in range(1500):
                t = i / 100
                writer.writerow([i * 10, "", f"{t:.3f}", "1.0", "0.0", "0.0"])
        (dataset_root / "DataAnnotation.json").write_text(
            json.dumps(
                [
                    {
                        "original_filename": "005_No_Talking_In.wav",
                        "segmentations": [
                            {
                                "start_time": 6.0,
                                "end_time": 7.0,
                                "annotations": {"Cough": {}},
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        return dataset_root


if __name__ == "__main__":
    unittest.main()
