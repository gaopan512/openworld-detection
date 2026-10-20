import hashlib
import sys
from pathlib import Path

from PyQt6.QtCore import QIODevice, QRectF, QSaveFile, Qt, QTimer
from PyQt6.QtGui import QColor, QImageReader, QImageWriter, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QGraphicsView,
    QGraphicsScene,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.class_parser import parse_classes
from src.gui.style import APP_STYLE
from src.gui.worker import DetectionWorker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_pixmap(image_path):
    """应用 EXIF 方向，与检测和绘图使用的图片方向保持一致。"""
    reader = QImageReader(str(image_path))
    reader.setAutoTransform(True)
    return QPixmap.fromImage(reader.read())


def class_color_rgb(class_name):
    """与 src/visualizer.py 的 get_class_color 使用同一算法，返回 RGB。

    不直接导入 visualizer，避免界面启动时加载 OpenCV。
    """
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    return (55 + digest[2] % 201, 55 + digest[1] % 201, 55 + digest[0] % 201)


class ImagePreview(QGraphicsView):
    """原图坐标中的预览与高亮；缩放和高亮不修改保存图片。"""

    def __init__(self, placeholder):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._item = self.scene().addPixmap(QPixmap())
        self._placeholder = self.scene().addText(placeholder)
        self._placeholder.setDefaultTextColor(QColor("#C9CED8"))
        self._fit = True
        self._highlights = []
        self._highlight_items = []
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setBackgroundBrush(QColor("#1B1F29"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, pixmap):
        self.set_highlights([])
        self._item.setPixmap(pixmap)
        self._placeholder.setVisible(pixmap.isNull())
        self.setSceneRect(self._item.boundingRect())
        self.fit_image()

    def set_highlights(self, highlights):
        for item in self._highlight_items:
            self.scene().removeItem(item)
        self._highlight_items = []
        self._highlights = list(highlights)
        for bbox, selected in self._highlights:
            x1, y1, x2, y2 = bbox
            pen = QPen(QColor("#FFFFFF" if selected else "#596170"), 3)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine)
            item = self.scene().addRect(QRectF(x1, y1, x2-x1, y2-y1), pen)
            item.setZValue(1)
            self._highlight_items.append(item)

    def set_highlight(self, bbox):
        self.set_highlights([(bbox, True)] if bbox else [])

    def fit_image(self):
        self._fit = True
        self.resetTransform()
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_by(self, factor):
        if self._item.pixmap().isNull() or factor <= 0:
            return
        scale = self.transform().m11() * factor
        if 0.01 <= scale <= 100:
            self._fit = False
            self.scale(factor, factor)

    def wheelEvent(self, event):
        self.zoom_by(1.2 ** (event.angleDelta().y() / 120))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit_image()


def make_section(title):
    """带小标题的分组：返回 (容器, 内部纵向布局)。"""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    label = QLabel(title)
    label.setObjectName("sectionTitle")
    layout.addWidget(label)
    return box, layout


def make_stat_card(title):
    """摘要卡片：返回 (卡片, 数值标签)。"""
    card = QFrame()
    card.setObjectName("statCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(2)
    title_label = QLabel(title)
    title_label.setObjectName("statTitle")
    value_label = QLabel("—")
    value_label.setObjectName("statValue")
    layout.addWidget(title_label)
    layout.addWidget(value_label)
    return card, value_label


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO-World 开放词汇目标检测")
        self.resize(1280, 800)
        self.setMinimumSize(1000, 640)
        self.setStyleSheet(APP_STYLE)
        self.setAcceptDrops(True)

        # original_pixmap 始终是原图；检测结果单独保存，便于对比
        self.image_path = None
        self.original_pixmap = None
        self.result_image = None
        self.detections = []
        self.last_classes = []
        self.busy = False
        self._saving = False
        self._selecting = False
        self.focused_class = None
        self.last_confidence = None
        self._result_key = None
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.setInterval(900)
        self.auto_timer.timeout.connect(lambda: self.start_detection(automatic=True))

        self.worker = DetectionWorker(self)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.status_changed.connect(self.statusBar().showMessage)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 660, 340])
        root.addWidget(splitter, 1)

        # 状态栏：检测中显示不确定进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(140)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("就绪，请选择图片")
        self.classes_input.textChanged.connect(self.inputs_changed)
        self.confidence_input.valueChanged.connect(self.inputs_changed)
        self.device_combo.currentIndexChanged.connect(self.inputs_changed)
        self.model_combo.currentIndexChanged.connect(self.inputs_changed)

    # ---------- 界面搭建 ----------

    def _build_top_bar(self):
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(52)
        row = QHBoxLayout(bar)
        row.setContentsMargins(16, 0, 16, 0)

        title = QLabel("YOLO-World 开放词汇目标检测")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_button = QPushButton("打开图片")
        self.select_button.clicked.connect(self.select_image)
        self.save_button = QPushButton("保存结果")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_result)

        buttons = QWidget()
        button_row = QHBoxLayout(buttons)
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch()
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.save_button)

        # 左右两侧都忽略自身尺寸、按相同拉伸系数分空间，标题才会严格居中
        spacer = QWidget()
        for side in (spacer, buttons):
            side.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
        row.addWidget(spacer, 1)
        row.addWidget(title, 0)
        row.addWidget(buttons, 1)
        return bar

    def _build_left_panel(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)

        # 图片
        section, box = make_section("图片")
        card = QFrame()
        card.setObjectName("imageCard")
        card_row = QHBoxLayout(card)
        card_row.setContentsMargins(12, 12, 12, 12)
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(44, 44)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        self.path_label = QLabel("尚未选择图片")
        self.image_info = QLabel("可拖拽图片到窗口")
        self.image_info.setObjectName("hint")
        text_col.addWidget(self.path_label)
        text_col.addWidget(self.image_info)
        card_row.addWidget(self.thumb_label)
        card_row.addLayout(text_col, 1)
        box.addWidget(card)
        layout.addWidget(section)

        # 检测目标
        section, box = make_section("检测目标")
        hint = QLabel("英文类别，逗号分隔")
        hint.setObjectName("hint")
        self.classes_input = QLineEdit()
        self.classes_input.setPlaceholderText("例如 person, bus, traffic light")
        box.addWidget(hint)
        box.addWidget(self.classes_input)
        self.chip_buttons = []
        chips = QWidget()
        self.chips_layout = QVBoxLayout(chips)
        self.chips_layout.setContentsMargins(0, 0, 0, 0)
        self.chips_layout.setSpacing(4)
        chip_scroll = QScrollArea()
        chip_scroll.setWidgetResizable(True)
        chip_scroll.setWidget(chips)
        chip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        chip_scroll.setMaximumHeight(140)
        box.addWidget(chip_scroll)
        layout.addWidget(section)

        # 参数
        section, box = make_section("参数")
        conf_label = QLabel("置信度阈值")
        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0.0, 1.0)
        self.confidence_input.setDecimals(2)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setValue(0.25)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(25)
        # 滑块与数值框双向联动
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_input.setValue(v / 100)
        )
        self.confidence_input.valueChanged.connect(
            lambda v: self.confidence_slider.setValue(round(v * 100))
        )
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.confidence_slider, 1)
        conf_row.addWidget(self.confidence_input)

        self.model_combo = QComboBox()
        models = sorted((PROJECT_ROOT / "models").glob("*.pt"))
        for path in models:
            self.model_combo.addItem(path.stem, str(path))
        if not models:
            self.model_combo.addItem(
                "yolov8s-worldv2", str(PROJECT_ROOT / "models/yolov8s-worldv2.pt")
            )

        self.device_combo = QComboBox()
        self.device_combo.addItem("自动（MPS / CUDA / CPU）", "auto")
        self.device_combo.addItem("MPS", "mps")
        self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")

        box.addWidget(conf_label)
        box.addLayout(conf_row)
        box.addWidget(QLabel("模型"))
        box.addWidget(self.model_combo)
        box.addWidget(QLabel("计算设备"))
        box.addWidget(self.device_combo)
        layout.addWidget(section)

        self.auto_detect = QCheckBox("输入停顿后自动检测")
        self.auto_detect.toggled.connect(self.schedule_auto_detection)
        layout.addWidget(self.auto_detect)
        layout.addStretch()
        self.detect_button = QPushButton("开始检测")
        self.detect_button.setObjectName("primaryButton")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(self.start_detection)
        layout.addWidget(self.detect_button)
        return panel

    def _build_center_panel(self):
        panel = QWidget()
        panel.setObjectName("centerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)

        self.result_preview = ImagePreview("请选择图片")
        self.original_preview = ImagePreview("请选择图片")
        self.tabs = QTabWidget()
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.addTab(self.result_preview, "检测结果")
        self.tabs.addTab(self.original_preview, "原图")
        layout.addWidget(self.tabs)
        self.result_parameters = QLabel("尚未检测 · 滚轮缩放，拖动平移，双击适应窗口")
        self.result_parameters.setWordWrap(True)
        layout.addWidget(self.result_parameters)
        return panel

    def _build_right_panel(self):
        panel = QFrame()
        panel.setObjectName("resultPanel")
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)

        # 摘要
        section, box = make_section("摘要")
        grid = QGridLayout()
        grid.setSpacing(8)
        count_card, self.count_value = make_stat_card("目标数量")
        time_card, self.time_value = make_stat_card("检测耗时")
        device_card, self.device_value = make_stat_card("设备")
        conf_card, self.conf_value = make_stat_card("阈值")
        grid.addWidget(count_card, 0, 0)
        grid.addWidget(time_card, 0, 1)
        grid.addWidget(device_card, 1, 0)
        grid.addWidget(conf_card, 1, 1)
        box.addLayout(grid)
        layout.addWidget(section)

        # 检测明细
        section, box = make_section("检测明细")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "类别", "置信度", "x1,y1,x2,y2"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 18)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        box.addWidget(self.table, 1)
        layout.addWidget(section, 1)

        # 类别统计
        section, box = make_section("类别统计")
        self.stats_label = QLabel("—")
        box.addWidget(self.stats_label)
        layout.addWidget(section)
        return panel

    # ---------- 图片加载与拖拽 ----------

    def select_image(self):
        """选择图片，读取成功后更新预览。"""
        if self.busy or self._selecting or self._saving:
            return

        resume_auto = self.auto_timer.isActive()
        self.auto_timer.stop()
        self._selecting = True
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", str(Path.home()),
                "图片文件 (*.jpg *.jpeg *.png *.bmp)"
            )
        finally:
            self._selecting = False
            if resume_auto:
                self.schedule_auto_detection()
        # 用户取消时，保留当前图片。
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        if self.busy:
            return
        pixmap = read_pixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(
                self, "读取失败", "无法读取该图片，请选择其他文件。"
            )
            return

        self.image_path = file_path
        self.original_pixmap = pixmap
        self.result_image = None
        self.detect_button.setEnabled(True)
        self.save_button.setEnabled(False)

        self.path_label.setText(Path(file_path).name)
        self.path_label.setToolTip(file_path)
        self.image_info.setText(f"{pixmap.width()} × {pixmap.height()} · 可拖拽替换")
        self.thumb_label.setPixmap(
            pixmap.scaled(
                44, 44,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.show_source_previews()
        self.clear_results()
        self._result_key = None
        self.result_parameters.setText("尚未检测 · 滚轮缩放，拖动平移，双击适应窗口")
        self.result_parameters.setToolTip("")
        self.inputs_changed()
        self.statusBar().showMessage("图片已加载")

    def show_source_previews(self):
        self.result_preview.set_image(self.original_pixmap)
        self.original_preview.set_image(self.original_pixmap)

    @staticmethod
    def dropped_image_path(mime):
        for url in mime.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):
        if not self.busy and self.dropped_image_path(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self.dropped_image_path(event.mimeData())
        if path and not self.busy:
            self.load_image(path)

    # ---------- 检测 ----------

    def input_key(self):
        try:
            classes = tuple(parse_classes(self.classes_input.text()))
        except (TypeError, ValueError):
            classes = ()
        return (self.image_path, classes, self.confidence_input.value(),
                self.model_combo.currentData(), self.device_combo.currentData())

    def inputs_changed(self, *_):
        self.focused_class = None
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.result_preview.set_highlights([])
        self.refresh_chips()
        self.schedule_auto_detection()

    def schedule_auto_detection(self, *_):
        self.auto_timer.stop()
        if self.busy or self._saving or self._selecting or not self.auto_detect.isChecked() or not self.image_path:
            return
        try:
            parse_classes(self.classes_input.text())
        except (TypeError, ValueError):
            return
        self.auto_timer.start()

    def refresh_chips(self):
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            item.widget().deleteLater()
        self.chip_buttons = []
        try:
            classes = parse_classes(self.classes_input.text())
        except (TypeError, ValueError):
            return
        current = self.result_image is not None and self.input_key() == self._result_key
        for name in classes:
            found = any(d.get("class_name") == name for d in self.detections)
            text = name if current and found else f"{name} · {'未检出' if current else '待检测'}"
            chip = QPushButton(text)
            chip.setToolTip(name)
            chip.setCheckable(True)
            chip.setChecked(name == self.focused_class)
            chip.setEnabled(not self.busy)
            r, g, b = class_color_rgb(name)
            border = "solid" if current and found else "dashed"
            chip.setStyleSheet(f"QPushButton {{border: 1px {border} rgb({r},{g},{b});"
                              f"border-radius: 6px; text-align: left; padding: 4px;}}"
                              "QPushButton:checked {background: #E7F0FB;}")
            chip.clicked.connect(lambda _=False, n=name: self.toggle_class(n))
            self.chips_layout.addWidget(chip)
            self.chip_buttons.append(chip)

    def toggle_class(self, name):
        if self.result_image is None or self.input_key() != self._result_key:
            return
        focused_class = None if self.focused_class == name else name
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.focused_class = focused_class
        highlights = [(det["bbox"], det.get("class_name") == self.focused_class)
                      for det in self.detections if det.get("bbox")]
        self.result_preview.set_highlights(highlights if self.focused_class else [])
        self.tabs.setCurrentIndex(0)
        self.refresh_chips()

    def start_detection(self, checked=False, *, automatic=False):
        """检查输入并启动后台检测。"""
        if self.busy or self._saving or self._selecting:
            return
        self.auto_timer.stop()

        if not self.image_path:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return

        try:
            classes = parse_classes(self.classes_input.text())
        except (TypeError, ValueError) as error:
            if not automatic:
                QMessageBox.warning(self, "输入错误", str(error))
            return

        # 重新读取原图，避免把上一次的检测框带进本次结果
        pixmap = read_pixmap(self.image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "读取失败", "图片已无法读取，请重新选择。")
            return

        self.worker.image_path = self.image_path
        self.worker.classes = classes
        self.worker.confidence = self.confidence_input.value()
        self.worker.device = self.device_combo.currentData()
        self.worker.model_path = Path(self.model_combo.currentData())
        self.last_classes = classes
        self.last_confidence = self.confidence_input.value()
        self._submitted_key = self.input_key()

        self.result_image = None
        self.original_pixmap = pixmap
        self.show_source_previews()
        self.clear_results()
        self.result_parameters.setText("正在检测，请等待结果……")
        self.result_parameters.setToolTip("")

        self.busy = True
        self.refresh_chips()
        self.set_controls_enabled(False)
        self.progress.show()
        self.statusBar().showMessage("正在检测，首次运行需要加载模型……")
        self.worker.start()

    def set_controls_enabled(self, enabled):
        """检测期间禁止更换图片、改参数和重复提交任务。"""
        self.auto_detect.setEnabled(enabled)
        self.select_button.setEnabled(enabled)
        self.classes_input.setEnabled(enabled)
        self.confidence_input.setEnabled(enabled)
        self.confidence_slider.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        for chip in self.chip_buttons:
            chip.setEnabled(enabled)
        self.detect_button.setEnabled(enabled and self.image_path is not None)
        self.save_button.setEnabled(enabled and self.result_image is not None)

    def show_result(self, image, message, detections=None, meta=None):
        """在主线程中显示结果，保留完整分辨率的图片。"""
        self.result_image = image
        self.result_preview.set_image(QPixmap.fromImage(image))
        self.tabs.setCurrentIndex(0)
        self.populate_results(detections or [], meta or {})
        self._result_key = getattr(self, "_submitted_key", self.input_key())
        self.result_parameters.setText(
            f"类别：{', '.join(self.last_classes)} · 阈值：{self.last_confidence} · "
            f"模型：{(meta or {}).get('model', '—')} · "
            f"设备：{(meta or {}).get('device', '—')}")
        self.result_parameters.setToolTip((meta or {}).get("fallback_reason") or "")
        self.focused_class = None
        self.refresh_chips()
        self.statusBar().showMessage(message)

    # ---------- 右侧结果面板 ----------

    def clear_results(self):
        self.detections = []
        self.table.setRowCount(0)
        for label in (self.count_value, self.time_value, self.device_value, self.conf_value):
            label.setText("—")
        self.stats_label.setText("—")

    def populate_results(self, detections, meta):
        self.clear_results()
        self.detections = list(detections)

        counts = {}
        for det in self.detections:
            name = str(det.get("class_name", "?"))
            counts[name] = counts.get(name, 0) + 1
            confidence = det.get("confidence")
            bbox = det.get("bbox")

            row = self.table.rowCount()
            self.table.insertRow(row)
            swatch = QTableWidgetItem()
            swatch.setBackground(QColor(*class_color_rgb(name)))
            self.table.setItem(row, 0, swatch)
            self.table.setItem(row, 1, QTableWidgetItem(name))
            self.table.setItem(
                row, 2,
                QTableWidgetItem("—" if confidence is None else f"{confidence:.2f}"),
            )
            self.table.setItem(
                row, 3,
                QTableWidgetItem(",".join(str(int(v)) for v in bbox) if bbox else "—"),
            )

        self.count_value.setText(str(len(self.detections)))
        if "elapsed" in meta:
            self.time_value.setText(f"{meta['elapsed']:.2f} s")
        if "device" in meta:
            self.device_value.setText(str(meta["device"]).upper())
        if "confidence" in meta:
            self.conf_value.setText(f"{meta['confidence']:.2f}")

        lines = [f"{name}    ×{count}" for name, count in counts.items()]
        lines += [f"{name}    未检测到" for name in self.last_classes if name not in counts]
        self.stats_label.setText("\n".join(lines) if lines else "—")

    def on_row_selected(self):
        """点击明细行：在结果图上高亮对应检测框。"""
        rows = self.table.selectionModel().selectedRows()
        bbox = None
        if rows and rows[0].row() < len(self.detections):
            bbox = self.detections[rows[0].row()].get("bbox")
        self.focused_class = None
        self.result_preview.set_highlight(bbox)
        self.refresh_chips()
        if bbox:
            self.tabs.setCurrentIndex(0)

    # ---------- 保存 ----------

    def save_result(self):
        """保存完整分辨率的检测结果，默认使用无损 PNG。"""
        if self.busy or self._saving or self.result_image is None:
            return

        # 对话框会运行嵌套事件循环：暂停自动任务，并固定本次导出的图片。
        result_image = self.result_image.copy()
        source_path = Path(self.image_path)
        resume_auto = self.auto_timer.isActive()
        self.auto_timer.stop()
        self._saving = True
        try:
            self._save_result(result_image, source_path)
        finally:
            self._saving = False
            if resume_auto:
                self.schedule_auto_detection()

    def _save_result(self, result_image, source_path):
        """保存进入对话框之前取得的结果快照。"""
        output_dir = PROJECT_ROOT / "outputs/detections"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.critical(self, "保存失败", f"无法创建结果目录：{error}")
            return

        default_path = output_dir / f"{source_path.stem}_detected.png"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存检测结果",
            str(default_path),
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)",
            options=QFileDialog.Option.DontConfirmOverwrite
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if not output_path.suffix:
            suffix = ".jpg" if selected_filter.startswith("JPEG") else ".png"
            output_path = output_path.with_suffix(suffix)

        suffix = output_path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg"):
            QMessageBox.warning(self, "格式不支持", "请使用 .png、.jpg 或 .jpeg 后缀。")
            return

        if output_path.resolve() == source_path.resolve():
            QMessageBox.warning(self, "无法保存", "请选择其他文件名，保留原始图片。")
            return

        if output_path.exists():
            answer = QMessageBox.question(
                self,
                "确认覆盖",
                f"文件已存在：\n{output_path}\n是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # 先写临时文件，成功后再替换目标，避免保存失败损坏旧结果
        output_file = QSaveFile(str(output_path))
        if not output_file.open(QIODevice.OpenModeFlag.WriteOnly):
            QMessageBox.critical(self, "保存失败", output_file.errorString())
            return

        image_format = b"png" if suffix == ".png" else b"jpeg"
        writer = QImageWriter(output_file, image_format)
        if not writer.write(result_image):
            output_file.cancelWriting()
            QMessageBox.critical(self, "保存失败", writer.errorString())
            return

        if not output_file.commit():
            QMessageBox.critical(self, "保存失败", output_file.errorString())
            return

        self.statusBar().showMessage(f"结果已保存：{output_path}")

    # ---------- 收尾 ----------

    def show_error(self, message):
        self.result_parameters.setText("检测失败，请检查输入、模型或计算设备后重试。")
        self.result_parameters.setToolTip(message)
        self.statusBar().showMessage("检测失败")
        QMessageBox.critical(self, "检测失败", message)

    def on_detection_finished(self):
        """成功和失败都会触发线程结束，统一恢复控件。"""
        self.busy = False
        self.progress.hide()
        self.set_controls_enabled(True)

    def closeEvent(self, event):
        """检测结束后再关闭，避免销毁仍在运行的线程。"""
        if self.busy or self.worker.isRunning():
            self.statusBar().showMessage("检测仍在进行，请完成后再关闭窗口")
            event.ignore()
            return

        self.auto_timer.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
