from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.detector import OpenWorldDetector


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="openworld-detector-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "fixture"
        self.path.write_bytes(b"fixture")
        self.model = MagicMock()
        box = SimpleNamespace(cls=np.array([0]), conf=np.array([.9]), xyxy=np.array([[1., 2., 3., 4.]]))
        self.model.predict.return_value = [SimpleNamespace(names={0: "person"}, boxes=[box])]
        with patch("src.detector.YOLOWorld", return_value=self.model):
            self.detector = OpenWorldDetector(self.path)

    def test_reuses_classes_when_only_threshold_changes(self):
        first = self.detector.detect(self.path, ["person"], .25)
        self.detector.detect(self.path, ("person",), .75)
        self.model.set_classes.assert_called_once_with(["person"])
        self.assertEqual(self.model.predict.call_count, 2)
        self.assertEqual(first, [{"class_name": "person", "confidence": .9, "bbox": [1., 2., 3., 4.]}])
        self.detector.detect(self.path, ["person", "bus"])
        self.assertEqual(self.model.set_classes.call_count, 2)

    def test_failed_class_update_is_retried(self):
        self.model.set_classes.side_effect = [RuntimeError("text encoder failed"), None]
        with self.assertRaises(RuntimeError):
            self.detector.detect(self.path, ["person"])
        self.detector.detect(self.path, ["person"])
        self.assertEqual(self.model.set_classes.call_count, 2)

    def test_invalid_threshold_never_reaches_model(self):
        for value in (None, "0.5", True, -.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.detector.detect(self.path, ["person"], value)
        self.model.set_classes.assert_not_called()
        self.model.predict.assert_not_called()

    def test_invalid_classes_never_reach_model(self):
        for value in ([], (), "person", [""], ["  "], [123]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.detector.detect(self.path, value)
        self.model.predict.assert_not_called()
