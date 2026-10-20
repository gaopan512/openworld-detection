"""全局 QSS 样式：浅色主题，主色 #2F5BEA。"""

APP_STYLE = """
QWidget { font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei"; font-size: 13px; color: #1B1F29; }
QMainWindow, QWidget#centerPanel { background: #F4F5F7; }

#topBar { background: #FFFFFF; border-bottom: 1px solid #E1E4EA; }
#appTitle { font-size: 20px; font-weight: 700; }
#sidePanel { background: #FFFFFF; border-right: 1px solid #E1E4EA; }
#resultPanel { background: #FFFFFF; border-left: 1px solid #E1E4EA; }
#sectionTitle { font-size: 12px; font-weight: 700; color: #5B6272; }
#hint { font-size: 12px; color: #5B6272; }

QPushButton { border: 1px solid #C9CED8; background: #FFFFFF; border-radius: 6px; padding: 7px 14px; }
QPushButton:hover { background: #F4F5F7; }
QPushButton:disabled { color: #9AA3B5; background: #F4F5F7; }
QPushButton#chip { border-radius: 14px; padding: 4px 10px; font-size: 12px; }
QPushButton#primaryButton { background: #2F5BEA; color: #FFFFFF; border: none; border-radius: 8px;
                            padding: 12px; font-size: 15px; font-weight: 700; }
QPushButton#primaryButton:hover { background: #2549C4; }
QPushButton#primaryButton:disabled { background: #AAB8F0; color: #FFFFFF; }

QLineEdit, QComboBox, QDoubleSpinBox { border: 1px solid #C9CED8; border-radius: 6px; padding: 7px 10px; background: #FFFFFF; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid #2F5BEA; }
QSlider::groove:horizontal { height: 4px; background: #E6E9EF; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2F5BEA; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #2F5BEA; }

#imageCard { border: 1.5px dashed #9AA3B5; border-radius: 8px; }
#statCard { background: #F4F5F7; border-radius: 8px; }
#statTitle { font-size: 11px; color: #5B6272; }
#statValue { font-size: 18px; font-weight: 700; }

QTabWidget::pane { border: none; }
QTabWidget::tab-bar { left: 0px; }
QTabBar::tab { background: #E6E9EF; color: #5B6272; padding: 7px 14px; margin-right: 4px; border-radius: 6px; }
QTabBar::tab:selected { background: #FFFFFF; color: #1B1F29; font-weight: 700; }
QLabel#preview { background: #1B1F29; color: #C9CED8; border-radius: 10px; }

QTableWidget { border: none; gridline-color: transparent; selection-background-color: #E7F0FB; selection-color: #1B1F29; }
QHeaderView::section { background: #FFFFFF; border: none; color: #5B6272; font-size: 11px; padding: 4px; }

QStatusBar { background: #FFFFFF; border-top: 1px solid #E1E4EA; }
QProgressBar { background: #E6E9EF; border: none; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #2F5BEA; border-radius: 3px; }
"""
