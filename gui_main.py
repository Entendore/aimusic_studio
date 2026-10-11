"""Main application window — refined minimalist single-column layout."""
import sys
import os
import logging
import subprocess
import platform
from typing import Optional, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QPushButton, QTextEdit, QLineEdit,
    QMessageBox, QProgressBar, QComboBox, QScrollArea, QFrame,
    QCheckBox, QInputDialog, QSizePolicy, QSlider,
)
from PySide6.QtCore import Qt, QThread, Slot, QUrl
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from core import AppContext, BackendError
from gui_components import (
    SettingsDialog, build_settings_widgets, clear_layout,
    apply_global_styles, STYLES, PALETTE, SP,
    SettingWidget, create_reset_button, section_header, card, hline,
)
from gui_workers import GenerationWorker, GenerationRequest
from backend import BackendState, BackendCapabilities

logger = logging.getLogger(__name__)

class MusicStudioApp(QMainWindow):
    IDLE = "idle"
    GENERATING = "generating"

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self._ui_state = self.IDLE
        self._thread: Optional[QThread] = None
        self._worker: Optional[GenerationWorker] = None
        self._settings_widgets: dict = {}
        self._last_saved_path: Optional[str] = None
        self._queue: List[GenerationRequest] = []
        self._metadata = ctx.backend_metadata
        self._capabilities = ctx.backend_capabilities

        self.setWindowTitle("AI Music Studio")
        self.resize(480, 760)
        self._build_ui()
        self._setup_media_player()
        self._setup_shortcuts()
        self._setup_initial_backend()

        geo = self.ctx.config.get("settings", {}).get("window_geometry", "")
        if geo:
            self.restoreGeometry(bytes.fromhex(geo))

    # ════════════════════════════════════════════════════════════
    #  UI Construction
    # ════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(SP['sm'], SP['sm'], SP['sm'], SP['sm'])
        root.setSpacing(SP['sm'])

        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(SP['md'], SP['md'], SP['md'], SP['md'])
        self.control_layout.setSpacing(SP['sm'])
        self.control_scroll.setWidget(self.control_panel)
        root.addWidget(self.control_scroll)

        self._build_header()
        self._build_backend_selector()
        self.control_layout.addWidget(hline())
        self._build_input_section()
        self._build_settings_section()
        self._build_preset_section()
        self.control_layout.addWidget(hline())
        self._build_controls_section()
        self._build_playback_section()
        self._build_analysis_section()
        self.control_layout.addStretch()

    def _build_header(self):
        row = QHBoxLayout()
        row.setSpacing(SP['sm'])
        title = QLabel("AI Music Studio")
        title.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        title.setStyleSheet(f"color: {PALETTE['text']}; letter-spacing: 0.5px;")
        row.addWidget(title)
        row.addStretch()
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setCurrentText(self.ctx.config.device)
        self.device_combo.setStyleSheet(STYLES["dropdown"])
        self.device_combo.setFixedWidth(80)
        self.device_combo.setToolTip("Select compute device. Requires 'Reload Engine' to apply.")
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        row.addWidget(self.device_combo)
        
        self.device_label = QLabel("")
        self.device_label.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 10px; "
            f"font-family: 'JetBrains Mono', 'Consolas', monospace;"
        )
        row.addWidget(self.device_label)
        self.control_layout.addLayout(row)

    def _build_backend_selector(self):
        wrap = card()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(SP['md'], SP['sm'], SP['md'], SP['sm'])
        lay.setSpacing(SP['xs'])

        head = QHBoxLayout()
        head.setSpacing(SP['sm'])
        ind = QFrame()
        ind.setFixedSize(8, 8)
        ind.setStyleSheet(f"background-color: {PALETTE['text_3']}; border-radius: 4px;")
        self.backend_indicator = ind
        head.addWidget(ind)
        head.addWidget(QLabel("ENGINE"))
        head.addStretch()
        lay.addLayout(head)

        self.backend_combo = QComboBox()
        for bid in self.ctx.available_backends:
            info = self._metadata.get(bid, {})
            caps = self._capabilities.get(bid, BackendCapabilities())
            label = info.get('name', bid)
            if caps.min_vram_gb > 0:
                label += f"  ·  {caps.min_vram_gb:.0f}GB"
            self.backend_combo.addItem(label, bid)
        self.backend_combo.setStyleSheet(STYLES["dropdown"])
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        lay.addWidget(self.backend_combo)

        self.backend_status_label = QLabel("")
        self.backend_status_label.setStyleSheet(f"color: {PALETTE['text_3']}; font-size: 10px;")
        self.backend_status_label.setWordWrap(True)
        lay.addWidget(self.backend_status_label)
        self.control_layout.addWidget(wrap)

    def _build_input_section(self):
        head = QHBoxLayout()
        head.setSpacing(SP['sm'])
        head.addWidget(section_header("Prompt"))
        head.addStretch()
        
        self.examples_combo = QComboBox()
        self.examples_combo.setStyleSheet(STYLES["dropdown"])
        self.examples_combo.setFixedWidth(150)
        self.examples_combo.addItem("Load Example...")
        self.examples_combo.currentIndexChanged.connect(self._on_example_selected)
        head.addWidget(self.examples_combo)
        
        btn_clear = QPushButton("Clear")
        btn_clear.setStyleSheet(STYLES["button_secondary"])
        btn_clear.setFixedWidth(60)
        btn_clear.clicked.connect(lambda: self.prompt_input.clear())
        head.addWidget(btn_clear)
        self.control_layout.addLayout(head)

        self.prompt_input = QTextEdit()
        self.prompt_input.setMinimumHeight(90)
        self.prompt_input.setMaximumHeight(140)
        self.prompt_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: {PALETTE['surface']};
                border: 1px solid {PALETTE['border']};
                border-radius: 3px; padding: 8px;
                color: {PALETTE['text']}; selection-background-color: {PALETTE['accent_dim']};
            }}
            QTextEdit:focus {{ border-color: {PALETTE['accent_dim']}; }}
        """)
        self.prompt_input.textChanged.connect(self._update_generate_state)
        self.control_layout.addWidget(self.prompt_input)

        hist_row = QHBoxLayout()
        hist_row.setSpacing(SP['sm'])
        hist_label = QLabel("History")
        hist_label.setStyleSheet(f"color: {PALETTE['text_3']}; font-size: 10px;")
        hist_label.setMinimumWidth(60)
        hist_row.addWidget(hist_label)
        self.history_combo = QComboBox()
        self.history_combo.setStyleSheet(STYLES["dropdown"])
        self._refresh_history()
        self.history_combo.currentIndexChanged.connect(self._on_history_selected)
        hist_row.addWidget(self.history_combo, 1)
        
        self.btn_clear_history = QPushButton("Clear")
        self.btn_clear_history.setStyleSheet(STYLES["button_secondary"])
        self.btn_clear_history.setFixedWidth(60)
        self.btn_clear_history.clicked.connect(self._clear_history)
        hist_row.addWidget(self.btn_clear_history)
        
        self.control_layout.addLayout(hist_row)

        self.prompt_desc_label = QLabel("")
        self.prompt_desc_label.setWordWrap(True)
        self.prompt_desc_label.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 10px; padding: 2px 0;")
        self.control_layout.addWidget(self.prompt_desc_label)

        row = QHBoxLayout()
        row.setSpacing(SP['md'])
        row.addWidget(self._labeled_spin("Duration", "duration_spin", 5, 600, 30, "s"))
        row.addWidget(self._labeled_spin("Variations", "variations_spin", 1, 10, 1, ""))
        row.addStretch()
        self.control_layout.addLayout(row)

    def _labeled_spin(self, label_text, attr, mn, mx, val, suffix):
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lbl = QLabel(label_text.upper())
        lbl.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 9px; "
            "font-weight: 600; letter-spacing: 1px;")
        lay.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(mn, mx)
        spin.setValue(val)
        if suffix: spin.setSuffix(suffix)
        spin.setStyleSheet(STYLES["spinbox"])
        spin.setFixedWidth(90)
        lay.addWidget(spin)
        setattr(self, attr, spin)
        return wrap

    def _build_settings_section(self):
        self.control_layout.addWidget(section_header("Parameters"))
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setStyleSheet(f"""
            QScrollArea {{ border: 1px solid {PALETTE['border']};
                          border-radius: 4px; background-color: {PALETTE['surface']}; }}
            QScrollArea > QWidget > QWidget {{ background-color: {PALETTE['surface']}; }}
        """)
        container = QWidget()
        self.settings_layout = QVBoxLayout(container)
        self.settings_layout.setContentsMargins(SP['sm'], SP['sm'], SP['sm'], SP['sm'])
        self.settings_layout.setSpacing(SP['xs'])
        self.settings_scroll.setWidget(container)
        self.control_layout.addWidget(self.settings_scroll)

    def _build_preset_section(self):
        self.control_layout.addWidget(section_header("Presets"))
        row = QHBoxLayout()
        row.setSpacing(SP['xs'])
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(STYLES["dropdown"])
        self.preset_combo.setMinimumWidth(120)
        row.addWidget(self.preset_combo, 1)

        self.btn_load_preset = QPushButton("Load")
        self.btn_load_preset.setStyleSheet(STYLES["button_secondary"])
        self.btn_load_preset.clicked.connect(self._load_preset)
        row.addWidget(self.btn_load_preset)

        self.btn_save_preset = QPushButton("Save")
        self.btn_save_preset.setStyleSheet(STYLES["button_secondary"])
        self.btn_save_preset.clicked.connect(self._save_preset)
        row.addWidget(self.btn_save_preset)

        self.btn_delete_preset = QPushButton("Delete")
        self.btn_delete_preset.setStyleSheet(STYLES["button_danger"])
        self.btn_delete_preset.clicked.connect(self._delete_preset)
        row.addWidget(self.btn_delete_preset)
        self.control_layout.addLayout(row)

    def _build_controls_section(self):
        gen_row = QHBoxLayout()
        gen_row.setSpacing(SP['sm'])
        self.btn_generate = QPushButton("▶  Generate")
        self.btn_generate.setStyleSheet(STYLES["button_primary"])
        self.btn_generate.clicked.connect(self._start_generation)
        gen_row.addWidget(self.btn_generate, 2)

        self.btn_queue = QPushButton("Queue")
        self.btn_queue.setStyleSheet(STYLES["button_secondary"])
        self.btn_queue.clicked.connect(self._add_to_queue)
        gen_row.addWidget(self.btn_queue, 1)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(STYLES["button_danger"])
        self.btn_stop.clicked.connect(self._stop_generation)
        gen_row.addWidget(self.btn_stop, 1)
        self.control_layout.addLayout(gen_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(STYLES["progress_bar"])
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.control_layout.addWidget(self.progress_bar)

        self.phase_label = QLabel("")
        self.phase_label.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 10px; padding: 0 2px;")
        self.control_layout.addWidget(self.phase_label)

        opts = QHBoxLayout()
        opts.setSpacing(SP['md'])
        self.chk_auto_play = QCheckBox("Auto-play")
        self.chk_auto_play.setChecked(self.ctx.config.auto_play)
        self.chk_auto_play.stateChanged.connect(self._on_auto_play_toggled)
        opts.addWidget(self.chk_auto_play)

        self.chk_organize = QCheckBox("Organize")
        self.chk_organize.setChecked(self.ctx.config.organize_output)
        self.chk_organize.stateChanged.connect(self._on_organize_toggled)
        opts.addWidget(self.chk_organize)
        
        self.chk_auto_reveal = QCheckBox("Auto-reveal")
        self.chk_auto_reveal.setChecked(self.ctx.config.get("settings", {}).get("auto_reveal", False))
        self.chk_auto_reveal.stateChanged.connect(self._on_auto_reveal_toggled)
        opts.addWidget(self.chk_auto_reveal)

        opts.addStretch()
        self.queue_label = QLabel("Queue: 0")
        self.queue_label.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 10px; "
            f"font-family: 'JetBrains Mono', 'Consolas', monospace;")
        opts.addWidget(self.queue_label)
        
        self.btn_clear_queue = QPushButton("Clear Queue")
        self.btn_clear_queue.setStyleSheet(STYLES["button_secondary"])
        self.btn_clear_queue.clicked.connect(self._clear_queue)
        self.btn_clear_queue.setEnabled(False)
        opts.addWidget(self.btn_clear_queue)
        
        opts.addWidget(QLabel("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["WAV", "FLAC", "OGG"])
        self.format_combo.setCurrentText(self.ctx.config.export_format)
        self.format_combo.setStyleSheet(STYLES["dropdown"])
        self.format_combo.setFixedWidth(80)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        opts.addWidget(self.format_combo)
        self.control_layout.addLayout(opts)

        bot = QHBoxLayout()
        bot.setSpacing(SP['sm'])
        self.btn_open_output = QPushButton("Open Output")
        self.btn_open_output.setStyleSheet(STYLES["button_secondary"])
        self.btn_open_output.clicked.connect(self._open_output_dir)
        bot.addWidget(self.btn_open_output)

        self.btn_reload = QPushButton("Reload Engine")
        self.btn_reload.setStyleSheet(STYLES["button_secondary"])
        self.btn_reload.clicked.connect(self._reload_backend)
        bot.addWidget(self.btn_reload)

        self.btn_settings = QPushButton("Paths")
        self.btn_settings.setStyleSheet(STYLES["button_secondary"])
        self.btn_settings.clicked.connect(self._open_settings)
        bot.addWidget(self.btn_settings)
        
        self.btn_open_log = QPushButton("Log")
        self.btn_open_log.setStyleSheet(STYLES["button_secondary"])
        self.btn_open_log.clicked.connect(self._open_log_file)
        bot.addWidget(self.btn_open_log)
        
        bot.addStretch()
        self.control_layout.addLayout(bot)

    def _build_playback_section(self):
        self.control_layout.addWidget(section_header("Playback"))
        row = QHBoxLayout()
        row.setSpacing(SP['sm'])
        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setEnabled(False)
        self.btn_play.setStyleSheet(STYLES["button_primary"])
        self.btn_play.clicked.connect(self._play_audio)
        row.addWidget(self.btn_play)

        self.btn_stop_play = QPushButton("■")
        self.btn_stop_play.setEnabled(False)
        self.btn_stop_play.setStyleSheet(STYLES["button_secondary"])
        self.btn_stop_play.setFixedWidth(40)
        self.btn_stop_play.clicked.connect(self._stop_audio)
        row.addWidget(self.btn_stop_play)
        self.control_layout.addLayout(row)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setStyleSheet(STYLES["slider"])
        self.seek_slider.sliderMoved.connect(self._on_seek_changed)
        self.control_layout.addWidget(self.seek_slider)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 10px; "
            f"font-family: 'JetBrains Mono', 'Consolas', monospace;")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(self.time_label)

        vol_row = QHBoxLayout()
        vol_row.setSpacing(SP['sm'])
        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet(f"color: {PALETTE['text_3']}; font-size: 12px;")
        vol_row.addWidget(vol_icon)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setStyleSheet(STYLES["slider"])
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.volume_slider)
        self.control_layout.addLayout(vol_row)

        self.lbl_now_playing = QLabel("No audio loaded")
        self.lbl_now_playing.setStyleSheet(
            f"color: {PALETTE['text_3']}; font-size: 10px; padding: 2px 2px;")
        self.control_layout.addWidget(self.lbl_now_playing)

    def _build_analysis_section(self):
        self.control_layout.addWidget(section_header("Analysis"))
        self.analysis_label = QLabel("—")
        self.analysis_label.setStyleSheet(
            f"color: {PALETTE['text_2']}; "
            f"font-family: 'JetBrains Mono', 'Consolas', monospace; "
            f"font-size: 10px; background-color: {PALETTE['surface']}; "
            f"padding: 10px; border-radius: 3px; border: 1px solid {PALETTE['border']};")
        self.analysis_label.setWordWrap(True)
        self.control_layout.addWidget(self.analysis_label)

    # ════════════════════════════════════════════════════════════
    #  Setup helpers
    # ════════════════════════════════════════════════════════════
    def _setup_media_player(self):
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.playbackStateChanged.connect(self._update_playback_state)
        self.player.positionChanged.connect(self._update_position)
        self.player.durationChanged.connect(self._update_duration)
        self.audio_output.setVolume(self.volume_slider.value() / 100.0)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._start_generation)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_playback)

    def _setup_initial_backend(self):
        available = self.ctx.available_backends
        if not available:
            self._set_indicator('error')
            self.backend_status_label.setText(
                "No backends available. Check log for missing dependencies (e.g. torch, librosa)."
            )
            self.btn_generate.setEnabled(False)
            return

        start_name = self.ctx.config.active_plugin
        start_idx = available.index(start_name) if start_name in available else 0

        for i in range(start_idx, len(available)):
            name = available[i]
            self.ctx.config.set_active_plugin(name)
            
            self.backend_combo.blockSignals(True)
            for j in range(self.backend_combo.count()):
                if self.backend_combo.itemData(j) == name:
                    self.backend_combo.setCurrentIndex(j)
                    break
            self.backend_combo.blockSignals(False)
            
            try:
                if self._load_current_backend():
                    self.ctx.config.save()
                    return
            except Exception:
                continue
        
        self._set_indicator('error')
        self.backend_status_label.setText("No backends could be loaded. Check logs.")
        self.btn_generate.setEnabled(False)

    # ════════════════════════════════════════════════════════════
    #  Backend lifecycle
    # ════════════════════════════════════════════════════════════
    def _set_indicator(self, state: str):
        color = {
            'idle': PALETTE['text_3'],
            'loading': PALETTE['warning'],
            'ready': PALETTE['success'],
            'error': PALETTE['error'],
        }.get(state, PALETTE['text_3'])
        self.backend_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;")

    def _load_current_backend(self) -> bool:
        name = self.ctx.config.active_plugin
        if not name:
            self._set_indicator('idle')
            self.backend_status_label.setText("No backend selected")
            return False
        info = self._metadata.get(name, {})
        self._set_indicator('loading')
        self.backend_status_label.setText("Loading…")
        self.btn_generate.setEnabled(False)
        try:
            self.ctx.backend_service.load(name)
            self._populate_settings()
            self._refresh_presets()
            caps = self.ctx.backend_service.capabilities
            self.duration_spin.setMaximum(caps.max_duration_sec)
            
            self.variations_spin.setEnabled(caps.supports_batch)
            if not caps.supports_batch:
                self.variations_spin.setValue(1)
                
            try:
                import torch
                if self.ctx.config.device == "cuda" and torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if caps.min_vram_gb > vram_gb:
                        QMessageBox.warning(self, "VRAM Warning",
                            f"{info.get('name', name)} requires {caps.min_vram_gb:.0f}GB of VRAM, "
                            f"but your GPU only has {vram_gb:.1f}GB.\n\n"
                            "Generation may fail or crash your system.")
            except ImportError:
                pass

            self.btn_generate.setEnabled(True)
            self._set_indicator('ready')
            status = f"{info.get('name', name)} ready"
            if caps.description:
                status += f" · {caps.description}"
            self.backend_status_label.setText(status)
            self.device_label.setText(
                f"{self.ctx.config.device.upper()}  ·  "
                f"{len(self.ctx.available_backends)} engines")
            
            self._update_generate_state()
            return True
        except BackendError as e:
            logger.error("Failed to load backend: %s", e)
            self._set_indicator('error')
            self.backend_status_label.setText(f"Failed: {e}")
            return False

    def _reload_backend(self):
        self._save_backend_settings()
        self._load_current_backend()

    def _on_backend_changed(self, index: int):
        name = self.backend_combo.itemData(index)
        if not name: return
        if self._ui_state == self.GENERATING:
            QMessageBox.warning(self, "Cannot Switch", "A generation is currently running.")
            return
        self.ctx.config.set_active_plugin(name)
        self.ctx.config.save()
        self._load_current_backend()

    def _on_device_changed(self, device: str):
        self.ctx.config.raw.setdefault("hardware", {})["device"] = device
        self.ctx.config.save()
        self.device_label.setText(f"Set to {device.upper()} (Reload to apply)")

    # ════════════════════════════════════════════════════════════
    #  Settings + Presets
    # ════════════════════════════════════════════════════════════
    def _populate_settings(self):
        clear_layout(self.settings_layout)
        self._settings_widgets = {}
        backend = self.ctx.backend_service.active
        if not backend:
            empty = QLabel("No backend loaded")
            empty.setStyleSheet(f"color: {PALETTE['text_3']}; padding: 10px;")
            self.settings_layout.addWidget(empty)
            self.prompt_desc_label.hide()
            return

        prompt_schema = type(backend).get_prompt_schema()
        self.prompt_desc_label.show()
        self.prompt_desc_label.setText(prompt_schema.get("description", ""))
        self.prompt_input.setPlaceholderText(prompt_schema.get("placeholder", ""))

        self.examples_combo.blockSignals(True)
        self.examples_combo.clear()
        self.examples_combo.addItem("Load Example...")
        examples = type(backend).get_examples()
        for ex in examples:
            self.examples_combo.addItem(ex.get("name", "Unnamed"), ex)
        self.examples_combo.blockSignals(False)

        schema = type(backend).get_settings_schema()
        saved = self.ctx.config.get_backend_settings(self.ctx.backend_service.active_name)
        build_settings_widgets(schema, self.settings_layout, self._settings_widgets, saved)
        create_reset_button(self._settings_widgets, self.settings_layout)

    def _save_backend_settings(self):
        if not self._settings_widgets: return
        name = self.ctx.backend_service.active_name
        if not name: return
        for key, widget in self._settings_widgets.items():
            val = widget.get_value()
            if val is not None:
                self.ctx.config.update_backend_setting(name, key, val)
        self.ctx.config.save()

    def _refresh_presets(self):
        self.preset_combo.clear()
        name = self.ctx.backend_service.active_name
        presets = self.ctx.config.get_presets_for_backend(name)
        for pname in sorted(presets.keys()):
            self.preset_combo.addItem(pname)

    def _load_preset(self):
        pname = self.preset_combo.currentText()
        if not pname: return
        name = self.ctx.backend_service.active_name
        preset = self.ctx.config.get_presets_for_backend(name).get(pname, {})
        for key, widget in self._settings_widgets.items():
            if key in preset: widget.set_value(preset[key])
        logger.info("Loaded preset: %s", pname)

    def _save_preset(self):
        pname, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not pname.strip(): return
        self._save_backend_settings()
        name = self.ctx.backend_service.active_name
        settings = self.ctx.config.get_backend_settings(name)
        self.ctx.config.save_preset(name, pname, settings)
        self.ctx.config.save()
        self._refresh_presets()
        self.preset_combo.setCurrentText(pname)
        logger.info("Saved preset: %s", pname)

    def _delete_preset(self):
        pname = self.preset_combo.currentText()
        if not pname: return
        name = self.ctx.backend_service.active_name
        self.ctx.config.delete_preset(name, pname)
        self.ctx.config.save()
        self._refresh_presets()
        logger.info("Deleted preset: %s", pname)

    def _open_settings(self): SettingsDialog(self.ctx.config, self).exec()

    def _open_output_dir(self):
        path = self.ctx.config.output_dir
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _open_log_file(self):
        log_file = os.path.join(os.path.expanduser("~"), ".aimusic", "aimusic_studio.log")
        if not os.path.exists(log_file):
            QMessageBox.information(self, "Log", "No log file found yet.")
            return
            
        if platform.system() == "Windows":
            os.startfile(log_file)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", log_file])
        else:
            subprocess.Popen(["xdg-open", log_file])

    def _refresh_history(self):
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("—")
        for p in self.ctx.config.prompt_history:
            label = p[:40].replace("\n", " ") + ("…" if len(p) > 40 else "")
            self.history_combo.addItem(label, p)
            idx = self.history_combo.count() - 1
            self.history_combo.setItemData(idx, p, Qt.ToolTipRole)
        self.history_combo.blockSignals(False)

    def _on_history_selected(self, index: int):
        if index <= 0: return
        data = self.history_combo.itemData(index)
        if data:
            self.prompt_input.setPlainText(data)
            self.history_combo.blockSignals(True)
            self.history_combo.setCurrentIndex(0)
            self.history_combo.blockSignals(False)

    def _clear_history(self):
        self.ctx.config.clear_prompt_history()
        self.ctx.config.save()
        self._refresh_history()

    def _on_example_selected(self, index: int):
        if index <= 0: 
            return
        example = self.examples_combo.itemData(index)
        if not example:
            return
            
        if "prompt" in example:
            self.prompt_input.setPlainText(example["prompt"])
            
        if "params" in example:
            for key, val in example["params"].items():
                if key in self._settings_widgets:
                    self._settings_widgets[key].set_value(val)
                    
        self.examples_combo.blockSignals(True)
        self.examples_combo.setCurrentIndex(0)
        self.examples_combo.blockSignals(False)

    # ════════════════════════════════════════════════════════════
    #  Generation
    # ════════════════════════════════════════════════════════════
    def _collect_params(self) -> dict:
        return {k: w.get_value() for k, w in self._settings_widgets.items() if w.get_value() is not None}

    def _validate_prompt(self, prompt: str) -> bool:
        backend = self.ctx.backend_service.active
        if backend:
            schema = type(backend).get_prompt_schema()
            if schema.get("required", True) and not prompt.strip():
                QMessageBox.warning(self, "Empty Prompt", "Prompt is required for this backend.")
                return False
        return True

    def _update_generate_state(self):
        has_text = bool(self.prompt_input.toPlainText().strip())
        is_ready = self.ctx.backend_service.active is not None and self._ui_state == self.IDLE
        self.btn_generate.setEnabled(has_text and is_ready)

    def _build_request(self, prompt: str) -> GenerationRequest:
        return GenerationRequest(
            prompt=prompt,
            duration=self.duration_spin.value(),
            params=self._collect_params(),
            output_dir=self.ctx.config.output_dir,
            variations=self.variations_spin.value(),
            export_format=self.ctx.config.export_format,
            organize_output=self.ctx.config.organize_output,
        )

    def _start_generation(self):
        if self._ui_state == self.GENERATING: return
        self._stop_audio()
        self._save_backend_settings()
        prompt = self.prompt_input.toPlainText()
        if not self._validate_prompt(prompt): return
        self.ctx.config.add_prompt_to_history(prompt)
        self.ctx.config.save()
        self._refresh_history()
        request = self._build_request(prompt)
        self._queue = [request]
        self._run_queue()

    def _add_to_queue(self):
        prompt = self.prompt_input.toPlainText()
        if not self._validate_prompt(prompt): return
        self._save_backend_settings()
        request = self._build_request(prompt)
        self._queue.append(request)
        self._update_queue_label()
        logger.info("Queued (position %d)", len(self._queue))

    def _run_queue(self):
        if not self._queue: return
        requests = self._queue
        self._queue = []
        self._update_queue_label()

        self._thread = QThread()
        self._worker = GenerationWorker(self.ctx.generation_service, requests)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.success.connect(self._on_generation_success)
        self._worker.error.connect(self._on_generation_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.queue_progress.connect(self._on_queue_progress)
        self._thread.start()
        self._set_ui_state(self.GENERATING)

    def _stop_generation(self):
        if self._worker: self._worker.stop()
        self._queue.clear()
        self._update_queue_label()
        logger.info("Stopping…")

    def _clear_queue(self):
        self._queue.clear()
        self._update_queue_label()

    def _set_ui_state(self, state: str):
        self._ui_state = state
        if state == self.GENERATING:
            self.btn_generate.setEnabled(False)
            self.btn_generate.setText("Generating…")
            self.btn_stop.setEnabled(True)
            self.btn_queue.setEnabled(False)
            self.progress_bar.setValue(0)
            self.phase_label.setText("Starting…")
        else:
            self.btn_generate.setText("▶  Generate")
            self.btn_stop.setEnabled(False)
            self.btn_queue.setEnabled(True)
            self.phase_label.setText("")
            self._update_generate_state()

    def _update_queue_label(self):
        count = len(self._queue)
        self.queue_label.setText(f"Queue: {count}")
        self.btn_clear_queue.setEnabled(count > 0)

    def _on_auto_play_toggled(self, val: int):
        self.ctx.config.set_auto_play(bool(val))
        self.ctx.config.save()

    def _on_auto_reveal_toggled(self, val: int):
        self.ctx.config.raw.setdefault("settings", {})["auto_reveal"] = bool(val)
        self.ctx.config.save()

    def _on_format_changed(self, fmt: str):
        self.ctx.config.set_export_format(fmt)
        self.ctx.config.save()

    def _on_organize_toggled(self, val: int):
        self.ctx.config.set_organize_output(bool(val))
        self.ctx.config.save()

    # ════════════════════════════════════════════════════════════
    #  Signal handlers
    # ════════════════════════════════════════════════════════════
    @Slot(int, str)
    def _on_progress(self, value: int, phase: str):
        self.progress_bar.setValue(value)
        self.phase_label.setText(phase)

    @Slot(int, int)
    def _on_queue_progress(self, current: int, total: int):
        logger.info("Processing %d/%d", current, total)

    @Slot(str, dict)
    def _on_generation_success(self, path: str, analysis: dict):
        self._last_saved_path = path
        self.btn_play.setEnabled(True)
        logger.info("Saved: %s", path)
        a = analysis
        self.analysis_label.setText(
            f"Duration   {a.get('duration_sec', '?'):>6} s\n"
            f"Peak       {a.get('peak_db', '?'):>6} dB\n"
            f"RMS        {a.get('rms_db', '?'):>6} dB\n"
            f"DR         {a.get('dynamic_range_db', '?'):>6} dB\n"
            f"Crest      {a.get('crest_factor_db', '?'):>6} dB\n"
            f"Samples    {a.get('n_samples', '?'):>6}\n"
            f"Rate       {a.get('sample_rate', '?'):>6} Hz"
        )
        
        if self.chk_auto_reveal.isChecked():
            self._reveal_file(path)
            
        if not self.ctx.config.auto_play:
            reply = QMessageBox.question(self, "Generation Complete",
                f"Audio saved successfully.\n{os.path.basename(path)}\n\nOpen output folder?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._reveal_file(path)
                
        if self.ctx.config.auto_play:
            self._play_audio()

    @Slot(str)
    def _on_generation_error(self, msg: str):
        logger.error("Error: %s", msg)
        QMessageBox.critical(self, "Error", f"Generation failed:\n\n{msg}")
        self.progress_bar.setValue(0)
        self.phase_label.setText("Failed")

    @Slot()
    def _on_worker_finished(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
            self._worker.deleteLater()
            self._thread = None
            self._worker = None
        if self._queue:
            logger.info("Continuing queue…")
            self._run_queue()
        else:
            self._set_ui_state(self.IDLE)

    # ════════════════════════════════════════════════════════════
    #  Playback
    # ════════════════════════════════════════════════════════════
    def _format_time(self, ms: int) -> str:
        sec = ms // 1000
        return f"{sec // 60:02d}:{sec % 60:02d}"

    def _update_position(self, pos: int):
        self.seek_slider.setValue(pos)
        self._update_time_label(pos, self.player.duration())

    def _update_duration(self, dur: int):
        self.seek_slider.setRange(0, dur)
        self._update_time_label(self.player.position(), dur)

    def _update_time_label(self, pos: int, dur: int):
        self.time_label.setText(f"{self._format_time(pos)} / {self._format_time(dur)}")

    def _on_seek_changed(self, pos: int):
        self.player.setPosition(pos)

    def _on_volume_changed(self, val: int):
        self.audio_output.setVolume(val / 100.0)

    def _toggle_playback(self):
        if isinstance(self.focusWidget(), QTextEdit) or isinstance(self.focusWidget(), QLineEdit):
            return
            
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._stop_audio()
        elif self._last_saved_path is not None:
            self._play_audio()

    def _play_audio(self):
        if self._last_saved_path:
            self.player.stop()
            self.player.setPosition(0)
            self.player.setSource(QUrl.fromLocalFile(self._last_saved_path))
            self.player.play()

    def _stop_audio(self): self.player.stop()

    def _update_playback_state(self, state):
        is_playing = state == QMediaPlayer.PlayingState.PlayingState
        has_audio = self._last_saved_path is not None
        self.btn_play.setEnabled(not is_playing and has_audio)
        self.btn_stop_play.setEnabled(is_playing)
        if is_playing and self._last_saved_path:
            self.lbl_now_playing.setText(f"▶  {os.path.basename(self._last_saved_path)}")
        elif has_audio:
            self.lbl_now_playing.setText("Paused")
        else:
            self.lbl_now_playing.setText("No audio loaded")

    def _reveal_file(self, path: str):
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])

    # ════════════════════════════════════════════════════════════
    #  Lifecycle
    # ════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        if self._ui_state == self.GENERATING:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "A generation is currently running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        self._save_backend_settings()
        self._stop_audio()

        geo = self.saveGeometry().data().hex()
        self.ctx.config.raw.setdefault("settings", {})["window_geometry"] = geo
        self.ctx.config.save()

        if self._thread and self._thread.isRunning():
            if self._worker: self._worker.stop()
            self._thread.quit()
            self._thread.wait(5000)

        try:
            self.ctx.shutdown()
        except Exception as e:
            logger.warning("Error during shutdown: %s", e)

        event.accept()

def launch_gui(ctx: AppContext):
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_global_styles(app)
    window = MusicStudioApp(ctx)
    window.show()
    sys.exit(app.exec())