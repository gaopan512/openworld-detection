from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.image_io import read_image, write_image
from src.visualizer import draw_detections


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="openworld-image-")
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "中文目录/测试图片.png"
        self.image = np.zeros((80, 160, 3), dtype=np.uint8)
        write_image(self.source, self.image)

    def test_unicode_path_roundtrip(self):
        self.assertTrue(np.array_equal(read_image(self.source), self.image))

    def test_empty_detections_preserve_pixels(self):
        self.assertTrue(np.array_equal(draw_detections(self.source, []), self.image))

    def test_long_label_keeps_confidence_and_fits(self):
        with patch("src.visualizer.cv2.putText", wraps=cv2.putText) as text:
            result = draw_detections(self.source, [{
                "class_name": "long object category " * 1000,
                "confidence": .95, "bbox": [100, 0, 159, 60],
            }])
        label = text.call_args.args[1]
        self.assertTrue(label.endswith("95%"))
        self.assertIn("...", label)
        self.assertLessEqual(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 2)[0][0], 150)
        self.assertEqual(result.shape, self.image.shape)

    def test_tiny_image_retains_box_without_overflowing_label(self):
        tiny = self.source.with_name("tiny.png")
        write_image(tiny, np.zeros((10, 160, 3), dtype=np.uint8))
        with patch("src.visualizer.cv2.putText", wraps=cv2.putText) as text:
            image = draw_detections(tiny, [{"class_name": "person", "confidence": .9, "bbox": [1, 1, 8, 8]}])
        text.assert_not_called()
        self.assertTrue(image.any())

    def test_empty_or_unreadable_file_reports_error(self):
        for content in (b"", b"not an image"):
            self.source.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "无法读取图片"):
                read_image(self.source)
