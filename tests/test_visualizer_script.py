"""验证可视化脚本在没有历史输出目录时也能保存图片。"""

from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestVisualizerScript(unittest.TestCase):
    def test_creates_missing_output_directory(self):
        with TemporaryDirectory(prefix="openworld-visualizer-") as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "data/demo").mkdir(parents=True)
            source = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "data/demo/bus.jpg"), source))
            script = root / "scripts/test_visualizer.py"
            script.write_bytes((PROJECT_ROOT / "scripts/test_visualizer.py").read_bytes())
            self.assertFalse((root / "outputs").exists())
            with patch.object(sys, "path", [str(PROJECT_ROOT), *sys.path]):
                runpy.run_path(str(script), run_name="__main__")
            result = cv2.imread(str(root / "outputs/detections/visualizer_test.jpg"))
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, source.shape)
            self.assertTrue(np.any(result != source))


if __name__ == "__main__":
    unittest.main()
