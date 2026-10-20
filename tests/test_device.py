from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.device import resolve_device


class TestDevice(unittest.TestCase):
    def test_auto_selects_available_backend_on_each_platform(self):
        for has_mps, has_cuda, expected in ((True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")):
            with self.subTest(mps=has_mps, cuda=has_cuda):
                torch = SimpleNamespace(
                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: has_mps)),
                    cuda=SimpleNamespace(is_available=lambda: has_cuda),
                )
                with patch.dict("sys.modules", {"torch": torch}):
                    self.assertEqual(resolve_device("auto"), expected)

    def test_auto_handles_platform_without_mps_backend(self):
        torch = SimpleNamespace(backends=SimpleNamespace(), cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_explicit_device_is_preserved_and_invalid_device_rejected(self):
        for device in ("cpu", "mps", "cuda"):
            self.assertEqual(resolve_device(device), device)
        with self.assertRaises(ValueError):
            resolve_device("unknown")
