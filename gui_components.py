"""Minimalist design system: palette, typography, spacing, and widget builders."""
import os
import logging
from PySide6.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QFileDialog,
    QMessageBox, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QFrame, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
)
from PySide6.QtCore import Qt
from config import Config

logger = logging.getLogger(__name__)

# ── Design tokens ──
PALETTE = {
    'bg':          '#0d0d0f',
    'surface':     '#151518',
    'surface_2':   '#1d1d21',
    'surface_3':   '#25252a',
    'border':      '#2a2a30',
    'border_hov':  '#3a3a42',
    'text':        '#c8c8cc',
    'text_2':      '#909096',
    'text_3':      '#5e5e64',
    'accent':      '#7a8a9e',
    'accent_hov':  '#8c9cb0',
    'accent_dim':  '#3a4250',
    'success':     '#6a8a72',
    'warning':     '#9a8a5e',
    'error':       '#9a6262',
}

SP = {'xs': 4, 'sm': 8, 'md': 12, 'lg': 16, 'xl': 24}

STYLES = {
    'card': f"""
        QFrame#card {{
            background-color: {PALETTE['surface_2']};
            border: 1px solid {PALETTE['border']};
            border-radius: 4px;
        }}
    """,
    'slider': f"""
        QSlider::groove:horizontal {{
            border: none; height: 3px;
            background: {PALETTE['surface_3']}; border-radius: 1px;
        }}
        QSlider::handle:horizontal {{
            background: {PALETTE['text_2']}; border: none;
            width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
        }}
        QSlider::handle:horizontal:hover {{ background: {PALETTE['accent']}; }}
        QSlider::sub-page:horizontal {{
            background: {PALETTE['accent_dim']}; border-radius: 1px;
        }}
    """,
    'dropdown': f"""
        QComboBox {{
            background-color: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 3px; padding: 5px 10px;
            color: {PALETTE['text']}; min-height: 22px;
        }}
        QComboBox:hover {{ border-color: {PALETTE['border_hov']}; }}
        QComboBox:focus {{ border-color: {PALETTE['accent_dim']}; }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox::down-arrow {{
            image: none; border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {PALETTE['text_3']}; margin-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            selection-background-color: {PALETTE['surface_3']};
            color: {PALETTE['text']}; padding: 2px;
        }}
    """,
    'spinbox': f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 3px; padding: 3px 6px;
            color: {PALETTE['text']}; min-height: 22px;
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {PALETTE['border_hov']}; }}
        QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {PALETTE['accent_dim']}; }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: transparent; border: none; width: 16px;
        }}
    """,
    'button_primary': f"""
        QPushButton {{
            background-color: {PALETTE['accent']};
            color: #0d0d0f; border: none; border-radius: 3px;
            padding: 7px 16px; font-size: 12px; font-weight: 500;
        }}
        QPushButton:hover {{ background-color: {PALETTE['accent_hov']}; }}
        QPushButton:pressed {{ background-color: {PALETTE['accent_dim']}; color: {PALETTE['text']}; }}
        QPushButton:disabled {{ background-color: {PALETTE['surface_3']}; color: {PALETTE['text_3']}; }}
    """,
    'button_secondary': f"""
        QPushButton {{
            background-color: {PALETTE['surface_2']};
            color: {PALETTE['text_2']};
            border: 1px solid {PALETTE['border']};
            border-radius: 3px; padding: 6px 12px; font-size: 11px;
        }}
        QPushButton:hover {{
            background-color: {PALETTE['surface_3']};
            border-color: {PALETTE['border_hov']}; color: {PALETTE['text']};
        }}
        QPushButton:pressed {{ background-color: {PALETTE['surface']}; }}
        QPushButton:disabled {{ color: {PALETTE['text_3']}; border-color: {PALETTE['surface_3']}; }}
    """,
    'button_danger': f"""
        QPushButton {{
            background-color: transparent; color: {PALETTE['error']};
            border: 1px solid {PALETTE['error']}; border-radius: 3px;
            padding: 5px 10px; font-size: 11px;
        }}
        QPushButton:hover {{ background-color: {PALETTE['error']}; color: #fff; }}
        QPushButton:disabled {{ color: {PALETTE['text_3']}; border-color: {PALETTE['surface_3']}; }}
    """,
    'line_edit': f"""
        QLineEdit {{
            background-color: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 3px; padding: 5px 8px;
            color: {PALETTE['text']}; selection-background-color: {PALETTE['accent_dim']};
        }}
        QLineEdit:hover {{ border-color: {PALETTE['border_hov']}; }}
        QLineEdit:focus {{ border-color: {PALETTE['accent_dim']}; }}
    """,
    'progress_bar': f"""
        QProgressBar {{
            border: 1px solid {PALETTE['border']}; border-radius: 2px;
            text-align: center; background-color: {PALETTE['surface']};
            color: {PALETTE['text_2']}; height: 18px; font-size: 10px;
        }}
        QProgressBar::chunk {{
            background: {PALETTE['accent_dim']}; border-radius: 1px;
        }}
    """,
    'checkbox': f"""
        QCheckBox {{ color: {PALETTE['text_2']}; spacing: 7px; font-size: 11px; }}
        QCheckBox::indicator {{
            width: 14px; height: 14px;
            border: 1px solid {PALETTE['border_hov']};
            border-radius: 2px; background-color: {PALETTE['surface']};
        }}
        QCheckBox::indicator:hover {{ border-color: {PALETTE['accent']}; }}
        QCheckBox::indicator:checked {{
            background-color: {PALETTE['accent']}; border-color: {PALETTE['accent']};
        }}
    """,
    'tooltip': f"""
        QToolTip {{
            background-color: {PALETTE['surface']};
            color: {PALETTE['text_2']};
            border: 1px solid {PALETTE['border']};
            border-radius: 3px; padding: 5px 8px; font-size: 10px;
        }}
    """,
}


def apply_global_styles(app):
    app.setStyleSheet(f"""
        QMainWindow, QDialog {{ background-color: {PALETTE['bg']}; }}
        QWidget {{
            color: {PALETTE['text_2']};
            font-family: 'Segoe UI', 'SF Pro Text', 'Arial', sans-serif;
            font-size: 11px;
        }}
        QLabel {{ color: {PALETTE['text_2']}; background: transparent; }}
        QTextEdit {{
            background-color: {PALETTE['bg']};
            border: 1px solid {PALETTE['border']};
            border-radius: 3px; padding: 8px;
            color: {PALETTE['text_2']};
            selection-background-color: {PALETTE['accent_dim']};
        }}
        QTextEdit:focus {{ border-color: {PALETTE['border_hov']}; }}
        QSplitter::handle {{ background-color: {PALETTE['border']}; width: 1px; }}
        QScrollBar:vertical {{
            background-color: transparent; width: 8px; border-radius: 4px; margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {PALETTE['surface_3']};
            border-radius: 4px; min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover {{ background-color: {PALETTE['border_hov']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{
            background-color: transparent; height: 8px; border-radius: 4px; margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {PALETTE['surface_3']};
            border-radius: 4px; min-width: 24px;
        }}
        QScrollArea {{ border: none; background: transparent; }}
        QScrollArea > QWidget > QWidget {{ background: transparent; }}
    """ + STYLES['tooltip'] + STYLES['checkbox'])


# ── Layout helpers ──
def section_header(text: str) -> QFrame:
    wrap = QFrame()
    wrap.setStyleSheet("background: transparent;")
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, SP['sm'], 0, SP['xs'])
    lay.setSpacing(SP['sm'])
    bar = QFrame()
    bar.setFixedWidth(2)
    bar.setFixedHeight(12)
    bar.setStyleSheet(f"background-color: {PALETTE['accent_dim']}; border-radius: 1px;")
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {PALETTE['text_3']}; font-size: 9px; "
        "font-weight: 600; letter-spacing: 1.5px;"
    )
    lay.addWidget(bar)
    lay.addWidget(lbl)
    lay.addStretch()
    return wrap


def card(widget=None) -> QFrame:
    c = QFrame()
    c.setObjectName("card")
    c.setStyleSheet(STYLES['card'])
    if widget is not None:
        lay = QVBoxLayout(c)
        lay.setContentsMargins(SP['md'], SP['sm'], SP['md'], SP['sm'])
        lay.addWidget(widget)
    return c


def hline() -> QFrame:
    ln = QFrame()
    ln.setFixedHeight(1)
    ln.setStyleSheet(f"background-color: {PALETTE['border']}; border: none;")
    return ln


# ── Settings dialog ──
class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(520, 340)
        self.config = config
        self.setStyleSheet(f"background-color: {PALETTE['bg']};")

        root = QVBoxLayout(self)
        root.setContentsMargins(SP['lg'], SP['lg'], SP['lg'], SP['lg'])
        root.setSpacing(SP['sm'])

        root.addWidget(section_header("System Paths"))

        rows = [
            ("Model Cache", config.cache_dir, "cache"),
            ("Checkpoints", config.ckpt_dir, "ckpt"),
            ("HeartLib Repo", config.heartlib_repo_dir, "repo"),
            ("Output Directory", config.output_dir, "output"),
        ]
        for label, val, attr in rows:
            root.addWidget(self._path_row(label, val, attr))

        root.addStretch()
        root.addWidget(hline())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(STYLES['button_primary'])
        buttons.button(QDialogButtonBox.Cancel).setStyleSheet(STYLES['button_secondary'])
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _path_row(self, label_text, default_value, attr_suffix):
        frame = card()
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(SP['md'], SP['sm'], SP['md'], SP['sm'])
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(110)
        lbl.setStyleSheet(f"color: {PALETTE['text_2']};")
        lay.addWidget(lbl)
        edit = QLineEdit(default_value)
        edit.setStyleSheet(STYLES['line_edit'])
        lay.addWidget(edit)
        btn = QPushButton("Browse")
        btn.setStyleSheet(STYLES['button_secondary'])
        btn.clicked.connect(lambda: self._browse(edit))
        lay.addWidget(btn)
        setattr(self, f"{attr_suffix}_edit", edit)
        return frame

    def _browse(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

    def _save(self):
        paths = {
            "Model Cache": self.cache_edit.text(),
            "Checkpoints": self.ckpt_edit.text(),
            "HeartLib Repo": self.repo_edit.text(),
            "Output Directory": self.output_edit.text(),
        }
        
        missing = []
        for name, path in paths.items():
            if path.strip() and not os.path.exists(path):
                missing.append(f"{name}: {path}")
                
        if missing:
            reply = QMessageBox.question(self, "Paths Not Found",
                "The following paths do not exist:\n\n" + "\n".join(missing) + 
                "\n\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.config.set_system_path("cache_dir", self.cache_edit.text())
        self.config.set_system_path("ckpt_dir", self.ckpt_edit.text())
        self.config.set_system_path("heartlib_repo_dir", self.repo_edit.text())
        self.config.set_output_dir(self.output_edit.text())
        
        try:
            self.config.save()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config:\n{e}")


# ── Dynamic settings widget ──
class SettingWidget:
    def __init__(self, setting_id, widget, value_label=None, default=None, step=None):
        self.id = setting_id
        self.widget = widget
        self.value_label = value_label
        self.default = default
        self.step = step

    def get_value(self):
        if isinstance(self.widget, QSlider):
            scale = 1.0 / self.step if self.step else 1.0
            val = self.widget.value() / scale
            if self.step and self.step >= 1:
                return int(round(val))
            return round(val, 6)
        elif isinstance(self.widget, QComboBox):
            return self.widget.currentText()
        elif isinstance(self.widget, QLineEdit):
            return self.widget.text()
        elif isinstance(self.widget, (QSpinBox, QDoubleSpinBox)):
            return self.widget.value()
        return None

    def set_value(self, value):
        if isinstance(self.widget, QSlider):
            scale = 1.0 / self.step if self.step else 1.0
            self.widget.setValue(int(round(float(value) * scale)))
            if self.value_label:
                self.value_label.setText(self._format_value(value))
        elif isinstance(self.widget, QComboBox):
            idx = self.widget.findText(str(value))
            if idx >= 0:
                self.widget.setCurrentIndex(idx)
        elif isinstance(self.widget, QLineEdit):
            self.widget.setText(str(value))
        elif isinstance(self.widget, (QSpinBox, QDoubleSpinBox)):
            self.widget.setValue(float(value))

    def _format_value(self, value):
        if self.step and self.step >= 1:
            return str(int(round(value)))
        step_str = str(self.step) if self.step else "0.1"
        if "." in step_str:
            prec = len(step_str.split(".")[1].rstrip("0"))
        else:
            prec = 1
        return f"{float(value):.{prec}f}"

    def reset_to_default(self):
        if self.default is not None:
            self.set_value(self.default)


def _browse_directory(line_edit):
    folder = QFileDialog.getExistingDirectory(None, "Select Directory")
    if folder:
        folder = folder.replace("/", "\\")
        line_edit.setText(folder)


def build_settings_widgets(schema, layout, settings_widgets, saved_values=None):
    categories: dict = {}
    uncategorized = []
    for item in schema:
        cat = item.get('category')
        if cat:
            categories.setdefault(cat, []).append(item)
        else:
            uncategorized.append(item)

    for cat_name, items in categories.items():
        layout.addWidget(section_header(cat_name))
        for item in items:
            _build_single_setting(item, layout, settings_widgets, saved_values)

    if uncategorized:
        if categories:
            layout.addWidget(section_header("Parameters"))
        for item in uncategorized:
            _build_single_setting(item, layout, settings_widgets, saved_values)


def _build_single_setting(item, layout, settings_widgets, saved_values):
    setting_id = item['id']
    saved_value = saved_values.get(setting_id, item.get('default')) if saved_values else item.get('default')

    c = card()
    c_layout = QHBoxLayout(c)
    c_layout.setContentsMargins(SP['md'], SP['xs'], SP['md'], SP['xs'])
    c_layout.setSpacing(SP['sm'])

    label = QLabel(item.get('label', setting_id))
    label.setMinimumWidth(90)
    label.setStyleSheet(f"color: {PALETTE['text_2']};")
    if item.get('description'):
        label.setToolTip(item['description'])
    c_layout.addWidget(label)
    c_layout.addStretch()

    wtype = item['type']
    if wtype == 'slider':
        widget, value_label = _build_slider(item, saved_value)
        c_layout.addWidget(widget, 1)
        c_layout.addWidget(value_label)
        settings_widgets[setting_id] = SettingWidget(
            setting_id, widget, value_label,
            item.get('default'), item.get('step'))
    elif wtype in ('dropdown', 'select'):
        widget = _build_dropdown(item, saved_value)
        widget.setMinimumWidth(120)
        c_layout.addWidget(widget)
        settings_widgets[setting_id] = SettingWidget(setting_id, widget, default=item.get('default'))
    elif wtype in ('text', 'string'):
        path_container = QWidget()
        path_layout = QHBoxLayout(path_container)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(SP['xs'])
        
        widget = _build_text_input(item, saved_value)
        path_layout.addWidget(widget, 1)
        
        if setting_id.endswith('_path') or setting_id.endswith('_dir'):
            btn = QPushButton("...")
            btn.setStyleSheet(STYLES["button_secondary"])
            btn.setFixedWidth(30)
            btn.setToolTip("Browse for directory")
            btn.clicked.connect(lambda checked, le=widget: _browse_directory(le))
            path_layout.addWidget(btn)
            
        c_layout.addWidget(path_container, 1)
        settings_widgets[setting_id] = SettingWidget(setting_id, widget, default=item.get('default'))
    elif wtype in ('spinbox', 'float', 'int'):
        widget = _build_spinbox(item, saved_value)
        c_layout.addWidget(widget)
        settings_widgets[setting_id] = SettingWidget(setting_id, widget, default=item.get('default'))
    else:
        logger.warning("Unknown setting type: %s", wtype)
        return

    layout.addWidget(c)


def _build_slider(item, saved_value):
    min_val = float(item.get('min', 0))
    max_val = float(item.get('max', 100))
    step = float(item.get('step', 0.1))
    value = float(saved_value) if saved_value is not None else float(item.get('default', min_val))

    scale = 1.0 / step
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(int(round(min_val * scale)))
    slider.setMaximum(int(round(max_val * scale)))
    slider.setValue(int(round(value * scale)))
    slider.setStyleSheet(STYLES["slider"])

    step_str = f"{step:.10f}".rstrip('0').rstrip('.')
    prec = len(step_str.split(".")[1]) if "." in step_str else 0

    def format_val(v):
        scaled = v / scale
        if prec == 0:
            return str(int(round(scaled)))
        return f"{scaled:.{prec}f}"

    value_label = QLabel(format_val(slider.value()))
    value_label.setStyleSheet(
        f"color: {PALETTE['text']}; font-family: 'JetBrains Mono', 'Consolas', monospace; "
        "font-size: 10px; min-width: 42px;"
    )
    value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    slider.valueChanged.connect(lambda v: value_label.setText(format_val(v)))
    return slider, value_label


def _build_dropdown(item, saved_value):
    combo = QComboBox()
    combo.addItems(item.get('options', []))
    if saved_value is not None:
        idx = combo.findText(str(saved_value))
        if idx >= 0:
            combo.setCurrentIndex(idx)
    combo.setStyleSheet(STYLES["dropdown"])
    if item.get('description'):
        combo.setToolTip(item['description'])
    return combo


def _build_text_input(item, saved_value):
    edit = QLineEdit(str(saved_value) if saved_value is not None else item.get('default', ''))
    edit.setStyleSheet(STYLES["line_edit"])
    edit.setPlaceholderText(item.get('placeholder', ''))
    if item.get('description'):
        edit.setToolTip(item['description'])
    return edit


def _build_spinbox(item, saved_value):
    min_val = item.get('min', 0)
    max_val = item.get('max', 1000)
    step = item.get('step', 1)
    value = saved_value if saved_value is not None else item.get('default', min_val)

    if step >= 1:
        spinbox = QSpinBox()
        spinbox.setMinimum(int(min_val))
        spinbox.setMaximum(int(max_val))
        spinbox.setSingleStep(int(step))
        spinbox.setValue(int(value))
    else:
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(float(min_val))
        spinbox.setMaximum(float(max_val))
        spinbox.setSingleStep(float(step))
        step_str = str(step)
        decimals = len(step_str.split(".")[1].rstrip("0")) if "." in step_str else 2
        spinbox.setDecimals(max(decimals, 1))
        spinbox.setValue(float(value))
    spinbox.setStyleSheet(STYLES["spinbox"])
    return spinbox


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.setParent(None)
            widget.deleteLater()
        sub_layout = item.layout()
        if sub_layout:
            clear_layout(sub_layout)


def create_reset_button(settings_widgets, layout):
    btn = QPushButton("↺  Reset to Defaults")
    btn.setStyleSheet(STYLES["button_secondary"])
    btn.clicked.connect(lambda: [w.reset_to_default() for w in settings_widgets.values()])
    layout.addWidget(btn)