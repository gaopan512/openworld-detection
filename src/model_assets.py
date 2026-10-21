"""Pinned official model downloads; no model is loaded before verification."""

import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models/yolov8s-worldv2.pt"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int


# SHA-256 independently checked against the official v8.3.0 download.
YOLO = Asset(
    "yolov8s-worldv2.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
    "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792",
    25923032,
)
# Hash is also embedded in the URL used by the pinned CLIP dependency.
CLIP = Asset(
    "ViT-B-32.pt",
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    353976522,
)


def verified(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset.sha256


def ensure_asset(path, asset, status=lambda message: None, *, attempts=3):
    """Reuse verified files, otherwise download with bounded retries and atomic replace.

    Each attempt starts over in a unique temporary file; an existing destination
    is preserved until the full replacement passes both size and SHA-256 checks.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    path = Path(path)
    status(f"正在校验 {asset.name}……")
    if verified(path, asset):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(1, attempts + 1):
        temporary = None
        try:
            status(f"正在下载 {asset.name}（第 {attempt}/{attempts} 次）……")
            request = Request(asset.url, headers={"User-Agent": "openworld-detection/1"})
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{asset.name}.", suffix=".part", delete=False
            ) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                received = 0
                last_percent = -1
                with urlopen(request, timeout=20) as source:
                    if not source.geturl().startswith("https://"):
                        raise ValueError("下载重定向必须使用 HTTPS")
                    while chunk := source.read(1024 * 1024):
                        received += len(chunk)
                        if received > asset.size:
                            raise ValueError("下载文件超过预期大小")
                        digest.update(chunk)
                        output.write(chunk)
                        percent = received * 100 // asset.size
                        if percent // 5 != last_percent // 5:
                            status(f"下载 {asset.name}：{percent}%")
                            last_percent = percent
                if received != asset.size or digest.hexdigest() != asset.sha256:
                    raise ValueError("文件大小或 SHA-256 校验失败")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            status(f"{asset.name} 下载并校验完成")
            return path
        except (OSError, URLError, HTTPException, ValueError) as caught:
            error = caught
            if attempt < attempts:
                status(f"{asset.name} 下载失败：{caught}；即将重试……")
                time.sleep(attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"{asset.name} 获取失败（已尝试 {attempts} 次）：{error}。"
        f"请检查网络或从 {asset.url} 手动下载到 {path} 后重试。"
    ) from error


def clip_path():
    # Match the cache used by the pinned Ultralytics text encoder.
    from ultralytics.utils import WEIGHTS_DIR

    return Path(WEIGHTS_DIR) / "clip" / CLIP.name


def prepare_default_assets(model_path, status=lambda message: None):
    """Only manage the bundled default selection; custom models stay user-managed."""
    if Path(model_path).resolve() == DEFAULT_MODEL:
        ensure_asset(DEFAULT_MODEL, YOLO, status)
        ensure_asset(clip_path(), CLIP, status)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="下载并校验默认 YOLO-World 和 CLIP 权重")
    parser.parse_args()
    try:
        print(f"YOLO 权重位置：{DEFAULT_MODEL}", flush=True)
        print(f"CLIP 权重位置：{clip_path()}", flush=True)
        prepare_default_assets(DEFAULT_MODEL, print)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print("默认模型和文本编码权重准备完成。可运行 python -B main.py")


if __name__ == "__main__":
    main()
