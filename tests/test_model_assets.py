import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.model_assets import Asset, DEFAULT_MODEL, ensure_asset, prepare_default_assets


class Response(BytesIO):
    def geturl(self):
        return "https://example.test/weights.pt"


class TestModelAssets(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="openworld-assets-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "模型" / "weights.pt"
        self.data = b"official fixture model"
        self.asset = Asset("weights.pt", "https://example.test/weights.pt",
                           hashlib.sha256(self.data).hexdigest(), len(self.data))
        sleeper = patch("src.model_assets.time.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def test_download_then_reuse_without_network(self):
        messages = []
        with patch("src.model_assets.urlopen", return_value=Response(self.data)) as request:
            self.assertEqual(ensure_asset(self.path, self.asset, messages.append), self.path)
            self.assertEqual(self.path.read_bytes(), self.data)
            ensure_asset(self.path, self.asset)
            request.assert_called_once()
            self.assertEqual(request.call_args.kwargs["timeout"], 20)
        self.assertTrue(any("100%" in message for message in messages))
        self.assertFalse(list(self.path.parent.glob("*.part")))

    def test_corrupt_existing_file_replaced_only_after_verified_download(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"x" * len(self.data))

        def download(*args, **kwargs):
            self.assertEqual(self.path.read_bytes(), b"x" * len(self.data))
            return Response(self.data)

        with patch("src.model_assets.urlopen", side_effect=download):
            ensure_asset(self.path, self.asset)
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_network_retry_recovers(self):
        with patch("src.model_assets.urlopen", side_effect=[URLError("offline"), Response(self.data)]) as request:
            ensure_asset(self.path, self.asset)
        self.assertEqual(request.call_count, 2)

    def test_bad_responses_never_become_model_and_leave_no_partial_file(self):
        for payload in (b"", self.data[:-1], b"x" * len(self.data), self.data + b"x"):
            with self.subTest(payload=payload):
                with patch("src.model_assets.urlopen", side_effect=lambda *a, **k: Response(payload)) as request:
                    with self.assertRaisesRegex(RuntimeError, "已尝试 3 次"):
                        ensure_asset(self.path, self.asset)
                self.assertEqual(request.call_count, 3)
                self.assertFalse(self.path.exists())
                self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_failed_download_preserves_previous_file(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"existing")
        with patch("src.model_assets.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "手动下载到"):
                ensure_asset(self.path, self.asset)
        self.assertEqual(self.path.read_bytes(), b"existing")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_interrupted_read_is_retried(self):
        from http.client import IncompleteRead

        with patch("src.model_assets.urlopen", side_effect=[IncompleteRead(b"half"), Response(self.data)]):
            ensure_asset(self.path, self.asset)
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_http_redirect_is_rejected(self):
        response = Response(self.data)
        response.geturl = lambda: "http://example.test/insecure"
        with patch("src.model_assets.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "HTTPS"):
                ensure_asset(self.path, self.asset, attempts=1)
        self.assertFalse(self.path.exists())

    def test_custom_model_does_not_trigger_managed_download(self):
        with patch("src.model_assets.ensure_asset") as download:
            prepare_default_assets(self.path)
        download.assert_not_called()

    def test_default_prepares_both_assets(self):
        with patch("src.model_assets.ensure_asset") as download, patch(
                "src.model_assets.clip_path", return_value=self.path):
            prepare_default_assets(DEFAULT_MODEL)
        self.assertEqual([call.args[0] for call in download.call_args_list], [DEFAULT_MODEL, self.path])


if __name__ == "__main__":
    unittest.main()
