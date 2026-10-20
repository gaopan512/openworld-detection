"""真实权重 → Qt 后台线程 → GUI 结果 → PNG 保存的无窗口验证。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "data/demo/bus.jpg")
    parser.add_argument("--classes", default="person, bus")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from src.class_parser import parse_classes
    from src.gui.main_window import MainWindow, read_pixmap

    parse_classes(args.classes)
    app = QApplication([])
    if read_pixmap(args.image).isNull():
        parser.error(f"无法读取图片：{args.image}")
    window = MainWindow()
    window.load_image(str(args.image.resolve()))
    window.classes_input.setText(args.classes)
    window.device_combo.setCurrentIndex(window.device_combo.findData(args.device))
    errors, metadata = [], []
    # 验证失败时记录错误，避免弹出对话框阻塞命令行验证。
    window.worker.failed.disconnect(window.show_error)
    window.worker.failed.connect(errors.append)
    window.worker.result_ready.connect(lambda image, message, boxes, meta: metadata.append(meta))
    window.worker.finished.connect(app.quit)
    QTimer.singleShot(0, window.start_detection)
    app.exec()
    window.worker.wait()
    if errors:
        window.close()
        raise RuntimeError(errors[0])
    if window.busy or window.result_image is None or not metadata:
        raise RuntimeError("后台任务结束后未收到有效 GUI 结果")
    if window.result_image.size() != window.original_pixmap.size():
        raise RuntimeError("原图与检测结果尺寸不同")

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or PROJECT_ROOT / "outputs/verification" / stamp
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "result.png"
    if not window.result_image.save(str(image_path), "PNG"):
        raise RuntimeError("无法保存验证图片")
    if QImage(str(image_path)).size() != window.result_image.size():
        raise RuntimeError("保存后重新读取的图片尺寸不匹配")
    record = {
        "image": str(args.image.resolve()), "classes": parse_classes(args.classes),
        "meta": metadata[-1], "detections": window.detections,
        "width": window.result_image.width(), "height": window.result_image.height(),
        "result_image": str(image_path.resolve()),
    }
    (output / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: {len(window.detections)} 个目标，设备 {metadata[-1]['device']}，记录：{output}")
    window.close()


if __name__ == "__main__":
    main()
