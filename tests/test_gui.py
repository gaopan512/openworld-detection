"""GUI 回归测试：使用真实 Qt 线程和图片编码，替换模型以避免加载权重。

在项目根目录运行：python -B -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# 必须在导入 Qt 之前设置；测试不弹出桌面窗口。
os.environ["QT_QPA_PLATFORM"] = "offscreen"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from PIL import Image
from PyQt6.QtGui import QCloseEvent, QImage
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QMimeData, QUrl

from src.gui.main_window import MainWindow


class TestGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="openworld-gui-")
        self.directory = Path(self.temporary.name)
        self.image = QImage(810, 1080, QImage.Format.Format_RGB888)
        self.image.fill(0xFF204060)
        self.source = self.directory / "source.png"
        self.assertTrue(self.image.save(str(self.source)))
        self.window = MainWindow()

    def tearDown(self):
        self.assertTrue(self.window.worker.wait(10000), "后台线程未结束")
        self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def select_source(self):
        with patch("src.gui.main_window.QFileDialog.getOpenFileName",
                   return_value=(str(self.source), "")):
            self.window.select_image()

    def prepare_result(self):
        self.select_source()
        self.window.show_result(self.image.copy(), "检测完成")
        self.window.on_detection_finished()

    def save_to(self, path, selected_filter="PNG 图片 (*.png)"):
        with patch("src.gui.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(path), selected_filter)):
            # 默认目录的创建另行测试，避免测试写入项目输出目录。
            with patch("src.gui.main_window.Path.mkdir"):
                self.window.save_result()

    def test_select_cancel_and_unreadable_image(self):
        self.assertFalse(self.window.detect_button.isEnabled())
        self.assertFalse(self.window.save_button.isEnabled())
        self.select_source()
        self.assertTrue(self.window.detect_button.isEnabled())
        for path in ("", str(self.directory / "missing.png")):
            with self.subTest(path=path):
                with patch("src.gui.main_window.QFileDialog.getOpenFileName",
                           return_value=(path, "")), patch(
                               "src.gui.main_window.QMessageBox.warning"):
                    self.window.select_image()
                self.assertEqual(self.window.image_path, str(self.source))
                self.assertFalse(self.window.original_pixmap.isNull())

    def test_invalid_input_never_starts_model(self):
        self.select_source()
        for text in ("", "汽车", ", ,"):
            with self.subTest(text=text):
                self.window.classes_input.setText(text)
                with patch.object(self.window.worker, "start") as start, patch(
                        "src.gui.main_window.QMessageBox.warning") as warning:
                    self.window.start_detection()
                start.assert_not_called()
                warning.assert_called_once()
                self.assertFalse(self.window.busy)

    def test_select_dialog_pauses_auto_detection_and_loads_chosen_image(self):
        self.select_source()
        self.window.classes_input.setText("person")
        self.window.auto_detect.setChecked(True)
        replacement = self.directory / "replacement.png"
        self.assertTrue(self.image.save(str(replacement)))

        def dialog(*args, **kwargs):
            self.assertFalse(self.window.auto_timer.isActive())
            QTest.qWait(1000)
            self.window.auto_timer.timeout.emit()
            start.assert_not_called()
            return str(replacement), ""

        with patch.object(self.window.worker, "start") as start, patch(
                "src.gui.main_window.QFileDialog.getOpenFileName", side_effect=dialog):
            self.window.select_image()
        self.assertEqual(self.window.image_path, str(replacement))
        self.assertTrue(self.window.auto_timer.isActive())
        self.assertFalse(self.window._selecting)

    def test_failed_task_does_not_show_previous_parameters(self):
        self.prepare_result()
        self.window.classes_input.setText("bus")
        with patch.object(self.window.worker, "start"):
            self.window.start_detection()
        self.assertIn("正在检测", self.window.result_parameters.text())
        with patch("src.gui.main_window.QMessageBox.critical"):
            self.window.show_error("模拟错误")
        self.window.on_detection_finished()
        self.assertIn("检测失败", self.window.result_parameters.text())
        self.assertIsNone(self.window.result_image)

    def test_exif_orientation_matches_detection_before_and_after_start(self):
        # 非对称色块同时验证旋转和镜像，避免只核对宽高而漏掉方向错误。
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        pixels[:20, :40] = (240, 20, 20)
        pixels[:20, 40:] = (20, 240, 20)
        pixels[20:, :40] = (20, 20, 240)
        pixels[20:, 40:] = (240, 240, 20)
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                photo = self.directory / f"orientation_{orientation}.jpg"
                image = Image.fromarray(pixels)
                exif = image.getexif()
                exif[274] = orientation
                image.save(photo, exif=exif, quality=95)
                expected = cv2.cvtColor(cv2.imread(str(photo)), cv2.COLOR_BGR2RGB)
                self.window.load_image(str(photo))
                self.window.classes_input.setText("person")

                def check_preview():
                    actual = self.window.original_pixmap.toImage()
                    height, width = expected.shape[:2]
                    self.assertEqual((actual.width(), actual.height()), (width, height))
                    for x, y in ((5, 5), (width-6, 5), (5, height-6), (width-6, height-6)):
                        for got, want in zip(actual.pixelColor(x, y).getRgb()[:3], expected[y, x]):
                            self.assertAlmostEqual(got, int(want), delta=3)

                check_preview()
                with patch.object(self.window.worker, "start"):
                    self.window.start_detection()
                check_preview()
                self.window.on_detection_finished()

    def test_thread_result_reuse_failure_and_retry(self):
        self.select_source()
        detector = MagicMock()
        detector.detect.return_value = []
        constructor = MagicMock(return_value=detector)
        bgr = np.zeros((12, 24, 3), dtype=np.uint8)
        bgr[:] = (10, 20, 30)
        modules = {
            "src.detector": SimpleNamespace(OpenWorldDetector=constructor),
            "src.visualizer": SimpleNamespace(
                draw_detections=lambda *args: bgr.copy()),
        }

        def run_task(text):
            self.window.classes_input.setText(text)
            finished = QSignalSpy(self.window.worker.finished)
            with patch.object(self.window.worker, "start",
                              wraps=self.window.worker.start) as start:
                self.window.start_detection()
                self.assertTrue(self.window.busy)
                self.assertFalse(self.window.select_button.isEnabled())
                self.assertFalse(self.window.save_button.isEnabled())
                event = QCloseEvent()
                self.window.closeEvent(event)
                self.assertFalse(event.isAccepted())
                self.window.start_detection()
                self.assertEqual(start.call_count, 1)
            self.assertTrue(finished.wait(10000), "检测结束信号超时")
            self.assertTrue(self.window.worker.wait(10000))
            self.app.processEvents()
            self.assertFalse(self.window.busy)
            self.assertTrue(self.window.detect_button.isEnabled())

        with patch.dict(sys.modules, modules):
            run_task("Person, traffic light, PERSON")
            self.assertEqual(detector.detect.call_args.kwargs["classes"],
                             ["Person", "traffic light"])
            self.assertEqual(self.window.result_image.pixelColor(0, 0).getRgb()[:3],
                             (30, 20, 10))
            self.assertTrue(self.window.save_button.isEnabled())
            self.assertIn("未检测到目标", self.window.statusBar().currentMessage())
            run_task("bus")
            self.assertEqual(constructor.call_count, 1)

            detector.detect.side_effect = ValueError("模拟推理失败")
            with patch("src.gui.main_window.QMessageBox.critical") as error:
                run_task("person")
                error.assert_called_once()
            self.assertIsNone(self.window.result_image)
            self.assertIsNone(self.window.worker.detector)
            self.assertFalse(self.window.save_button.isEnabled())

            detector.detect.side_effect = None
            detector.detect.return_value = [{"class_name": "person"}]
            run_task("person")
            self.assertEqual(constructor.call_count, 2)
            self.assertEqual(self.window.statusBar().currentMessage(), "检测完成")

    def test_default_directory_is_project_output(self):
        self.prepare_result()
        with patch("src.gui.main_window.Path.mkdir") as mkdir, patch(
                "src.gui.main_window.QFileDialog.getSaveFileName",
                return_value=("", "")) as dialog:
            self.window.save_result()
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(Path(dialog.call_args.args[2]),
                         PROJECT_ROOT / "outputs/detections/source_detected.png")
        self.assertEqual(list(self.directory.iterdir()), [self.source])

    def test_directory_creation_error(self):
        self.prepare_result()
        with patch("src.gui.main_window.Path.mkdir", side_effect=PermissionError), patch(
                "src.gui.main_window.QMessageBox.critical") as error, patch(
                "src.gui.main_window.QFileDialog.getSaveFileName") as dialog:
            self.window.save_result()
        error.assert_called_once()
        dialog.assert_not_called()

    def test_save_formats_preserve_full_dimensions(self):
        self.prepare_result()
        for name, selected, expected in (
            ("result", "PNG 图片 (*.png)", "result.png"),
            ("结果", "JPEG 图片 (*.jpg *.jpeg)", "结果.jpg"),
            ("upper.PNG", "PNG 图片 (*.png)", "upper.PNG"),
        ):
            with self.subTest(name=name):
                self.save_to(self.directory / name, selected)
                saved = QImage(str(self.directory / expected))
                self.assertFalse(saved.isNull())
                self.assertEqual(saved.size(), self.image.size())
                if expected.lower().endswith(".png"):
                    self.assertEqual(saved.pixelColor(0, 0), self.image.pixelColor(0, 0))

    def test_save_pauses_auto_detection_and_exports_snapshot(self):
        self.prepare_result()
        self.window.auto_detect.setChecked(True)
        self.window.classes_input.setText("person")
        self.assertTrue(self.window.auto_timer.isActive())
        target = self.directory / "snapshot.png"

        def dialog(*args, **kwargs):
            self.assertFalse(self.window.auto_timer.isActive())
            # 模拟保存对话框的事件循环，包括已经排队的超时信号。
            QTest.qWait(1000)
            self.window.auto_timer.timeout.emit()
            start.assert_not_called()
            # 即使当前结果被其他代码替换，导出仍使用进入保存流程时的快照。
            self.window.result_image.fill(0xFFABCDEF)
            return str(target), "PNG 图片 (*.png)"

        with patch.object(self.window.worker, "start") as start, patch(
                "src.gui.main_window.QFileDialog.getSaveFileName", side_effect=dialog), patch(
                    "src.gui.main_window.Path.mkdir"):
            self.window.save_result()
        saved = QImage(str(target))
        self.assertEqual(saved.size(), self.image.size())
        self.assertEqual(saved.pixelColor(0, 0), self.image.pixelColor(0, 0))
        self.assertFalse(self.window._saving)
        self.assertTrue(self.window.auto_timer.isActive())

    def test_cancel_save_resumes_pending_auto_detection(self):
        self.prepare_result()
        self.window.auto_detect.setChecked(True)
        self.window.classes_input.setText("person")
        with patch("src.gui.main_window.QFileDialog.getSaveFileName",
                   return_value=("", "")), patch("src.gui.main_window.Path.mkdir"):
            self.window.save_result()
        self.assertFalse(self.window._saving)
        self.assertTrue(self.window.auto_timer.isActive())
        with patch.object(self.window.worker, "start") as start:
            QTest.qWait(1050)
            start.assert_called_once()
        self.window.on_detection_finished()

    def test_original_and_unsupported_format_are_not_overwritten(self):
        self.prepare_result()
        original = self.source.read_bytes()
        for target in (self.source, self.directory / "result.gif"):
            with self.subTest(target=target):
                with patch("src.gui.main_window.QMessageBox.warning") as warning:
                    self.save_to(target)
                warning.assert_called_once()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse((self.directory / "result.gif").exists())

    def test_overwrite_confirmation_and_failed_write_preserve_file(self):
        self.prepare_result()
        target = self.directory / "existing.png"
        target.write_bytes(b"original contents")
        with patch("src.gui.main_window.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.No):
            self.save_to(target)
        self.assertEqual(target.read_bytes(), b"original contents")

        with patch("src.gui.main_window.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.Yes):
            self.save_to(target)
        self.assertFalse(QImage(str(target)).isNull())
        saved_contents = target.read_bytes()

        with patch("src.gui.main_window.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.Yes), patch(
                       "src.gui.main_window.QImageWriter") as writer, patch(
                           "src.gui.main_window.QMessageBox.critical") as error:
            writer.return_value.write.return_value = False
            writer.return_value.errorString.return_value = "模拟写入失败"
            self.save_to(target)
        error.assert_called_once()
        self.assertEqual(target.read_bytes(), saved_contents)

    def test_save_requires_current_result(self):
        with patch("src.gui.main_window.QFileDialog.getSaveFileName") as dialog:
            self.window.save_result()
        dialog.assert_not_called()
        self.prepare_result()
        self.window.busy = True
        self.window.set_controls_enabled(False)
        self.assertFalse(self.window.save_button.isEnabled())
        with patch("src.gui.main_window.QFileDialog.getSaveFileName") as dialog:
            self.window.save_result()
        dialog.assert_not_called()
        self.window.on_detection_finished()
        self.assertTrue(self.window.save_button.isEnabled())
        self.select_source()
        self.assertIsNone(self.window.result_image)
        self.assertFalse(self.window.save_button.isEnabled())

    def test_original_result_and_class_highlight_are_separate(self):
        self.select_source()
        self.window.classes_input.setText("person, bus, fire extinguisher")
        self.window.last_classes = ["person", "bus", "fire extinguisher"]
        self.window.last_confidence = 0.25
        boxes = [{"class_name": "person", "confidence": 0.91, "bbox": [10, 20, 70, 90]},
                 {"class_name": "bus", "confidence": 0.86, "bbox": [80, 20, 200, 300]}]
        result = self.image.copy()
        result.fill(0xFF102030)
        self.window.show_result(result, "检测完成", boxes, {"confidence": 0.25, "device": "cpu"})
        self.window.on_detection_finished()
        self.assertEqual(self.window.original_pixmap.toImage().pixelColor(0, 0), self.image.pixelColor(0, 0))
        self.assertNotEqual(self.window.result_image.pixelColor(0, 0), self.image.pixelColor(0, 0))
        self.assertIn("未检出", self.window.chip_buttons[2].text())
        self.window.chip_buttons[0].click()
        self.assertEqual(self.window.focused_class, "person")
        self.assertEqual(len(self.window.result_preview._highlights), 2)
        self.window.chip_buttons[0].click()
        self.assertIsNone(self.window.focused_class)
        self.assertEqual(len(self.window.result_preview._highlights), 0)
        self.window.classes_input.setText("helmet")
        self.assertIn("待检测", self.window.chip_buttons[0].text())
        self.assertIn("person, bus, fire extinguisher", self.window.result_parameters.text())

    def test_auto_detection_is_optional_debounced_and_cancelled(self):
        self.select_source()
        self.assertFalse(self.window.auto_detect.isChecked())
        self.window.classes_input.setText("person")
        self.assertFalse(self.window.auto_timer.isActive())
        self.window.auto_detect.setChecked(True)
        self.window.classes_input.setText("person, bus")
        self.assertTrue(self.window.auto_timer.isActive())
        self.window.classes_input.setText("bus")
        self.assertEqual(self.window.auto_timer.interval(), 900)
        with patch.object(self.window.worker, "start") as start:
            self.window.auto_timer.timeout.emit()
        start.assert_called_once()
        self.assertTrue(self.window.busy)
        self.assertFalse(self.window.auto_detect.isEnabled())
        self.window.on_detection_finished()
        self.window.classes_input.setText("helmet")
        self.assertTrue(self.window.auto_timer.isActive())
        self.window.auto_detect.setChecked(False)
        self.assertFalse(self.window.auto_timer.isActive())
        with patch("src.gui.main_window.QMessageBox.warning") as warning:
            self.window.classes_input.setText("中文")
            self.window.start_detection(automatic=True)
        warning.assert_not_called()
        self.assertFalse(self.window.busy)

    def test_preview_zoom_and_fit(self):
        self.select_source()
        preview = self.window.original_preview
        before = preview.transform().m11()
        preview.zoom_by(1.2)
        self.assertAlmostEqual(preview.transform().m11(), before * 1.2)
        self.assertFalse(preview._fit)
        preview.fit_image()
        self.assertTrue(preview._fit)
        self.assertEqual(preview.dragMode(), preview.DragMode.ScrollHandDrag)

    def test_slider_numeric_sync_and_drop_filter(self):
        self.window.confidence_slider.setValue(45)
        self.assertEqual(self.window.confidence_input.value(), 0.45)
        self.window.confidence_input.setValue(0.75)
        self.assertEqual(self.window.confidence_slider.value(), 75)
        mime = QMimeData()
        mime.setUrls([QUrl("https://example.com/test.png"), QUrl.fromLocalFile(str(self.source))])
        self.assertEqual(self.window.dropped_image_path(mime), str(self.source))
        mime.setUrls([QUrl.fromLocalFile(str(self.directory / "text.txt"))])
        self.assertIsNone(self.window.dropped_image_path(mime))

    def test_actual_timer_debounces_and_invalid_input_cancels(self):
        self.select_source()
        self.window.auto_detect.setChecked(True)
        with patch.object(self.window.worker, "start") as start:
            self.window.classes_input.setText("person")
            QTest.qWait(500)
            self.window.classes_input.setText("person, bus")
            QTest.qWait(500)
            start.assert_not_called()
            QTest.qWait(550)
            start.assert_called_once()
            self.assertEqual(self.window.worker.classes, ["person", "bus"])
        self.window.on_detection_finished()
        self.window.classes_input.setText("helmet")
        self.assertTrue(self.window.auto_timer.isActive())
        self.window.classes_input.setText("中文")
        self.assertFalse(self.window.auto_timer.isActive())

    def test_table_highlight_and_stale_result_do_not_modify_export(self):
        self.select_source()
        self.window.classes_input.setText("person, bus")
        self.window.last_classes = ["person", "bus"]
        boxes = [{"class_name": "person", "confidence": .9, "bbox": [1, 2, 40, 80]}]
        result = self.image.copy()
        self.window.show_result(result, "完成", boxes, {"elapsed": .5})
        self.window.table.selectRow(0)
        self.assertEqual(self.window.result_preview._highlights, [(boxes[0]["bbox"], True)])
        self.window.chip_buttons[0].click()
        self.assertEqual(self.window.focused_class, "person")
        self.assertEqual(self.window.result_image, result)
        self.window.classes_input.setText("helmet")
        self.assertIn("待检测", self.window.chip_buttons[0].text())
        self.assertFalse(self.window.result_preview._highlights)
        self.assertEqual(self.window.result_image, result)
        self.assertEqual(self.window.table.rowCount(), 1)

    def test_worker_reloads_on_device_or_model_change(self):
        self.select_source()
        detector = MagicMock()
        detector.detect.return_value = []
        constructor = MagicMock(return_value=detector)
        bgr = np.zeros((12, 24, 3), dtype=np.uint8)
        modules = {
            "src.detector": SimpleNamespace(OpenWorldDetector=constructor),
            "src.visualizer": SimpleNamespace(draw_detections=lambda *args: bgr.copy()),
        }
        worker = self.window.worker
        worker.image_path = str(self.source)
        worker.classes = ["person"]
        worker.device = "cpu"
        with patch.dict(sys.modules, modules):
            worker.run()
            worker.run()
            self.assertEqual(constructor.call_count, 1)
            worker.model_path = self.directory / "different.pt"
            worker.run()
            self.assertEqual(constructor.call_count, 2)
            worker.device = "mps"
            worker.run()
            self.assertEqual(constructor.call_count, 3)

    def test_auto_device_retries_cpu_once_and_reports_reason(self):
        worker = self.window.worker
        worker.device = "auto"
        ready, failed, status = QSignalSpy(worker.result_ready), QSignalSpy(worker.failed), QSignalSpy(worker.status_changed)
        bgr = np.zeros((12, 24, 3), dtype=np.uint8)
        with patch("src.gui.worker.resolve_device", return_value="cuda"), patch.object(
                worker, "_detect", side_effect=[RuntimeError("GPU unavailable"), []]) as detect, patch(
                    "src.visualizer.draw_detections", return_value=bgr):
            worker.run()
        self.assertEqual([call.args for call in detect.call_args_list], [("cuda",), ("cpu",)])
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(status), 1)
        self.assertEqual(ready[0][3]["device"], "cpu")
        self.assertIn("GPU unavailable", ready[0][3]["fallback_reason"])
        self.assertIn("回退 CPU", ready[0][1])

    def test_manual_device_never_falls_back_and_double_failure_stops(self):
        worker = self.window.worker
        worker.failed.disconnect(self.window.show_error)
        for requested, resolved, expected_calls in (("mps", "mps", 1), ("auto", "cpu", 1), ("auto", "mps", 2)):
            with self.subTest(requested=requested, resolved=resolved):
                worker.device = requested
                failed, ready = QSignalSpy(worker.failed), QSignalSpy(worker.result_ready)
                with patch("src.gui.worker.resolve_device", return_value=resolved), patch.object(
                        worker, "_detect", side_effect=RuntimeError("inference error")) as detect:
                    worker.run()
                self.assertEqual(detect.call_count, expected_calls)
                self.assertEqual(len(failed), 1)
                self.assertEqual(len(ready), 0)
                self.assertIsNone(worker.detector)
                if expected_calls == 2:
                    self.assertIn("CPU 重试也失败", failed[0][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
