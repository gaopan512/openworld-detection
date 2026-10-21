import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from src.device import resolve_device
from src.model_assets import prepare_default_assets


class DetectionWorker(QThread):
    # 结果图片、提示信息、检测明细列表、元信息（耗时/设备/模型/阈值）
    result_ready = pyqtSignal(QImage, str, list, dict)
    failed = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None, device="auto"):
        super().__init__(parent)
        self.device = device

        project_root = Path(__file__).resolve().parents[2]
        self.model_path = project_root / "models/yolov8s-worldv2.pt"

        self.detector = None
        self._detector_key = None
        self.image_path = None
        self.classes = []
        self.confidence = 0.25

    def _detect(self, device):
        from src.detector import OpenWorldDetector

        key = (str(self.model_path), device)
        if self.detector is None or self._detector_key != key:
            self.detector = OpenWorldDetector(self.model_path, device=device)
            self._detector_key = key
        return self.detector.detect(
            image_path=self.image_path, classes=self.classes, confidence=self.confidence
        )

    def run(self):
        """在后台线程中加载模型、推理和绘图。"""
        try:
            import cv2

            from src.visualizer import draw_detections

            device = resolve_device(self.device)
            fallback_reason = None
            # 耗时包含权重准备、加载、文本编码、推理及 CPU 重试，不包含绘图。
            started = time.perf_counter()
            key = (str(self.model_path), device)
            if self.detector is None or self._detector_key != key:
                # Network failures must not be mistaken for accelerator failures.
                prepare_default_assets(self.model_path, self.status_changed.emit)
            self.status_changed.emit("正在加载模型并检测……")
            try:
                detections = self._detect(device)
            except Exception as error:
                if self.device != "auto" or device == "cpu":
                    raise
                fallback_reason = f"{device.upper()} 失败：{type(error).__name__}: {error}"
                self.status_changed.emit(f"{fallback_reason}；正在用 CPU 重试一次……")
                self.detector = None
                self._detector_key = None
                device = "cpu"
                try:
                    detections = self._detect(device)
                except Exception as cpu_error:
                    raise RuntimeError(
                        f"{fallback_reason}；CPU 重试也失败：{type(cpu_error).__name__}: {cpu_error}"
                    ) from cpu_error
            elapsed = time.perf_counter() - started

            image = draw_detections(self.image_path, detections)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb_image.shape
            result_image = QImage(
                rgb_image.data,
                width,
                height,
                rgb_image.strides[0],
                QImage.Format.Format_RGB888
            ).copy()

            if detections:
                message = "检测完成"
            else:
                message = "检测完成，当前提示词和阈值下未检测到目标"
            if fallback_reason:
                message += f"（已回退 CPU；{fallback_reason}）"

            meta = {
                "elapsed": elapsed,
                "device": device,
                "model": Path(self.model_path).stem,
                "confidence": self.confidence,
                "fallback_reason": fallback_reason,
            }
            self.result_ready.emit(result_image, message, list(detections), meta)

        except Exception as error:
            self.detector = None
            self._detector_key = None
            self.failed.emit(f"{type(error).__name__}: {error}")
