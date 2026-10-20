"""通过字节解码/编码处理图片，兼容 Windows 下的中文文件路径。"""

from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error) as error:
        raise ValueError(f"无法读取图片：{path}") from error
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, data = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"图片编码失败：{path}")
    data.tofile(path)
