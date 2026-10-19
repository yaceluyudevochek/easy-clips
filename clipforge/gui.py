"""Desktop GUI (PySide6): pick a video, run the pipeline, preview and manage clips, edit settings."""
from __future__ import annotations

import copy
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFontComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QSlider, QSpinBox, QSplitter, QStyle, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from .config import Config, load_config, save_config
from .progress import STAGE_NAMES, Cancelled, Reporter

log = logging.getLogger("clipforge.gui")

# Share of the overall progress bar per stage.
STAGE_RANGE = {
    "download": (0, 5), "audio": (5, 8), "transcribe": (8, 40), "llm_start": (40, 45),
    "highlights": (45, 70), "rerank": (70, 75), "render": (75, 100), "done": (100, 100),
}
LANGUAGES = [("Автоопределение", None), ("Русский", "ru"), ("English", "en"), ("Українська", "uk"),
             ("Deutsch", "de"), ("Español", "es"), ("Français", "fr")]
VIDEO_FILTER = "Видео (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;Все файлы (*)"


# ---------------------------------------------------------------- background work

class LogBridge(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.message.emit(self.format(record))
        except RuntimeError:  # widget already destroyed on exit
            pass


class PipelineThread(QThread):
    progress = Signal(str, float, float)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, cfg: Config, max_clips: Optional[int], force: set):
        super().__init__()
        self.source, self.cfg, self.max_clips, self.force = source, cfg, max_clips, force
        self.reporter = Reporter(lambda s, d, t: self.progress.emit(s, float(d), float(t)))

    def cancel(self) -> None:
        self.reporter.cancel()

    def run(self) -> None:
        from .pipeline import run_pipeline

        try:
            job = run_pipeline(self.source, self.cfg, self.max_clips, self.force, self.reporter)
            self.succeeded.emit(str(job))
        except Cancelled:
            self.failed.emit("Остановлено")
        except Exception as e:
            logging.getLogger("clipforge").error(traceback.format_exc())
            self.failed.emit(str(e))


class DoctorThread(QThread):
    done = Signal(object)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        from .doctor import Check, run_checks

        try:
            self.done.emit(run_checks(self.cfg))
        except Exception as e:
            self.done.emit([Check("Проверка окружения", False, str(e))])


# ---------------------------------------------------------------- small widgets

def path_row(edit: QLineEdit, on_browse) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    btn = QPushButton("Обзор…")
    btn.clicked.connect(on_browse)
    lay.addWidget(btn)
    return w


class ColorButton(QPushButton):
    def __init__(self, rgb: str):
        super().__init__()
        self.setFixedWidth(90)
        self.clicked.connect(self._pick)
        self.set_rgb(rgb)

    def set_rgb(self, rgb: str) -> None:
        self.rgb = rgb.lstrip("#").upper()
        text = "#000" if QColor("#" + self.rgb).lightness() > 128 else "#fff"
        self.setText("#" + self.rgb)
        self.setStyleSheet(f"background:#{self.rgb}; color:{text}; font-weight:600;")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor("#" + self.rgb), self, "Цвет")
        if c.isValid():
            self.set_rgb(c.name()[1:])


def fmt_time(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.cfg = load_config(config_path if config_path.exists() else None)
        self.thread: Optional[PipelineThread] = None
        self.doctor_thread: Optional[DoctorThread] = None
        self.job: Optional[Path] = None
        self.clips: List[dict] = []

        self.setWindowTitle("clipforge — вирусные клипы из длинных видео")
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        tabs = QTabWidget()
        tabs.addTab(self._build_clips_tab(), "Клипы")
        tabs.addTab(self._build_settings_tab(), "Настройки")
        self.setCentralWidget(tabs)
        self.tabs = tabs

        self.bridge = LogBridge()
        self.bridge.message.connect(self._append_log)
        handler = QtLogHandler(self.bridge)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        self._cfg_to_ui()
        self._refresh_projects()

    # ---------------- clips tab

    def _build_clips_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        src = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Путь к видео или ссылка (YouTube и др.) — можно перетащить файл в окно")
        browse = QPushButton("Выбрать видео…")
        browse.clicked.connect(self._browse_video)
        src.addWidget(self.source_edit, 1)
        src.addWidget(browse)
        outer.addLayout(src)

        opts = QHBoxLayout()
        self.max_clips_spin = QSpinBox()
        self.max_clips_spin.setRange(1, 30)
        self.min_dur_spin = QSpinBox()
        self.min_dur_spin.setRange(5, 170)
        self.min_dur_spin.setSuffix(" с")
        self.max_dur_spin = QSpinBox()
        self.max_dur_spin.setRange(10, 180)
        self.max_dur_spin.setSuffix(" с")
        self.lang_combo = QComboBox()
        for name, code in LANGUAGES:
            self.lang_combo.addItem(name, code)
        self.captions_check = QCheckBox("Субтитры")
        for label, w in (("Клипов:", self.max_clips_spin), ("Длина от", self.min_dur_spin), ("до", self.max_dur_spin),
                         ("Язык:", self.lang_combo)):
            opts.addWidget(QLabel(label))
            opts.addWidget(w)
            opts.addSpacing(8)
        opts.addWidget(self.captions_check)
        opts.addStretch(1)
        self.start_btn = QPushButton("  Создать клипы  ")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start_new)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        opts.addWidget(self.start_btn)
        opts.addWidget(self.stop_btn)
        outer.addLayout(opts)

        prog = QHBoxLayout()
        self.stage_label = QLabel("Готов к работе")
        self.stage_label.setMinimumWidth(320)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        prog.addWidget(self.stage_label)
        prog.addWidget(self.progress, 1)
        outer.addLayout(prog)

        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)

        # left: project + clip list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        proj = QHBoxLayout()
        proj.addWidget(QLabel("Проект:"))
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._project_selected)
        proj.addWidget(self.project_combo, 1)
        refresh = QPushButton()
        refresh.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh.setToolTip("Обновить список проектов")
        refresh.clicked.connect(self._refresh_projects)
        proj.addWidget(refresh)
        ll.addLayout(proj)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Оценка", "Длина", "Название"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for col in range(3):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._clip_selected)
        ll.addWidget(self.table, 1)

        btns = QHBoxLayout()
        self.open_folder_btn = QPushButton("Открыть папку")
        self.open_folder_btn.clicked.connect(self._open_folder)
        self.rerender_btn = QPushButton("Перерендерить")
        self.rerender_btn.setToolTip("Пересобрать клипы с текущими настройками субтитров и кадрирования")
        self.rerender_btn.clicked.connect(lambda: self._rerun({"render"}))
        self.refind_btn = QPushButton("Найти моменты заново")
        self.refind_btn.setToolTip("Снова спросить LLM (например, после смены модели или длины клипов)")
        self.refind_btn.clicked.connect(lambda: self._rerun({"highlights"}))
        for b in (self.open_folder_btn, self.rerender_btn, self.refind_btn):
            btns.addWidget(b)
        ll.addLayout(btns)
        hsplit.addWidget(left)

        # right: player + details
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget()
        self.video.setMinimumSize(270, 420)
        self.video.setStyleSheet("background:#000;")
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        rl.addWidget(self.video, 3)

        ctl = QHBoxLayout()
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self._toggle_play)
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.sliderMoved.connect(self.player.setPosition)
        self.time_label = QLabel("0:00 / 0:00")
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.pos_slider, 1)
        ctl.addWidget(self.time_label)
        rl.addLayout(ctl)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda d: self.pos_slider.setRange(0, d))
        self.player.playbackStateChanged.connect(self._on_playback_state)

        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        rl.addWidget(self.details, 2)
        hsplit.addWidget(right)
        hsplit.setSizes([620, 560])

        vsplit.addWidget(hsplit)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setPlaceholderText("Здесь будет журнал работы")
        vsplit.addWidget(self.log_view)
        vsplit.setSizes([600, 150])
        outer.addWidget(vsplit, 1)
        return page

    # ---------------- settings tab

    def _build_settings_tab(self) -> QWidget:
        content = QWidget()
        lay = QVBoxLayout(content)

        # LLM
        g = QGroupBox("Нейросеть для поиска моментов (LLM)")
        f = QFormLayout(g)
        self.llm_local = QRadioButton("Локальная модель (llama.cpp запускается автоматически)")
        self.llm_remote = QRadioButton("Внешний сервер (OpenAI-совместимый API: Ollama, LM Studio, облако)")
        grp = QButtonGroup(self)
        grp.addButton(self.llm_local)
        grp.addButton(self.llm_remote)
        self.llm_local.toggled.connect(self._update_llm_mode)
        f.addRow(self.llm_local)
        self.server_exe_edit = QLineEdit()
        self.server_exe_edit.setPlaceholderText(r"C:\llama\llama-server.exe")
        f.addRow("llama-server.exe:", path_row(self.server_exe_edit, lambda: self._browse_into(
            self.server_exe_edit, "llama-server", "Программы (*.exe);;Все файлы (*)")))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(r"C:\models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
        f.addRow("Модель (.gguf):", path_row(self.model_path_edit, lambda: self._browse_into(
            self.model_path_edit, "Модель", "GGUF (*.gguf);;Все файлы (*)")))
        self.n_cpu_moe_spin = QSpinBox()
        self.n_cpu_moe_spin.setRange(0, 128)
        self.n_cpu_moe_spin.setToolTip("Сколько MoE-слоёв держать в оперативной памяти. 48 — все (для 8 ГБ VRAM).\n"
                                       "Если видеопамять свободна, уменьшайте — будет быстрее. 0 — для обычных моделей.")
        f.addRow("Слоёв экспертов в ОЗУ (--n-cpu-moe):", self.n_cpu_moe_spin)
        self.ctx_combo = QComboBox()
        for v in (8192, 16384, 24576, 32768):
            self.ctx_combo.addItem(str(v), v)
        f.addRow("Контекст (токенов):", self.ctx_combo)
        self.server_cmd_edit = QLineEdit()
        self.server_cmd_edit.setPlaceholderText("необязательно: своя команда запуска, заменяет поля выше")
        f.addRow("Своя команда:", self.server_cmd_edit)
        f.addRow(self.llm_remote)
        self.base_url_edit = QLineEdit()
        f.addRow("Адрес API:", self.base_url_edit)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        f.addRow("API ключ:", self.api_key_edit)
        self.model_name_edit = QLineEdit()
        f.addRow("Имя модели:", self.model_name_edit)
        self.rerank_check = QCheckBox("Второй проход: сравнить лучших кандидатов между собой")
        f.addRow(self.rerank_check)
        lay.addWidget(g)

        # Whisper
        g = QGroupBox("Распознавание речи (Whisper)")
        f = QFormLayout(g)
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["large-v3-turbo", "large-v3", "medium", "small"])
        self.whisper_model_combo.setEditable(True)
        f.addRow("Модель:", self.whisper_model_combo)
        self.whisper_device_combo = QComboBox()
        self.whisper_device_combo.addItems(["cuda", "cpu"])
        f.addRow("Устройство:", self.whisper_device_combo)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8_float16", "float16", "int8"])
        f.addRow("Точность:", self.compute_combo)
        lay.addWidget(g)

        # Captions
        g = QGroupBox("Субтитры")
        f = QFormLayout(g)
        self.font_combo = QFontComboBox()
        f.addRow("Шрифт:", self.font_combo)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(30, 200)
        f.addRow("Размер:", self.font_size_spin)
        self.max_words_spin = QSpinBox()
        self.max_words_spin.setRange(1, 8)
        f.addRow("Слов на экране:", self.max_words_spin)
        self.uppercase_check = QCheckBox("ЗАГЛАВНЫМИ БУКВАМИ")
        f.addRow(self.uppercase_check)
        self.color_btn = ColorButton("FFFFFF")
        f.addRow("Цвет текста:", self.color_btn)
        self.hl_color_btn = ColorButton("FFE600")
        f.addRow("Цвет текущего слова:", self.hl_color_btn)
        self.position_spin = QSpinBox()
        self.position_spin.setRange(40, 95)
        self.position_spin.setSuffix(" % высоты")
        f.addRow("Положение:", self.position_spin)
        lay.addWidget(g)

        # Reframe + render
        g = QGroupBox("Кадр и рендер")
        f = QFormLayout(g)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("1080 × 1920", (1080, 1920))
        self.resolution_combo.addItem("720 × 1280", (720, 1280))
        f.addRow("Разрешение:", self.resolution_combo)
        self.two_shot_check = QCheckBox("Если двое помещаются в кадр — показывать обоих")
        f.addRow(self.two_shot_check)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setToolTip("Максимальная скорость панорамы, ширин кадра в секунду")
        f.addRow("Скорость камеры:", self.speed_spin)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItem("NVIDIA NVENC H.264 (быстро)", "h264_nvenc")
        self.encoder_combo.addItem("NVIDIA NVENC H.265", "hevc_nvenc")
        self.encoder_combo.addItem("Процессор x264 (медленно)", "libx264")
        f.addRow("Кодировщик:", self.encoder_combo)
        self.cq_spin = QSpinBox()
        self.cq_spin.setRange(14, 35)
        self.cq_spin.setToolTip("Меньше — выше качество и больше файл. 19-23 — хороший диапазон.")
        f.addRow("Качество (CQ):", self.cq_spin)
        self.workdir_edit = QLineEdit()
        f.addRow("Папка проектов:", path_row(self.workdir_edit, self._browse_workdir))
        lay.addWidget(g)

        btns = QHBoxLayout()
        save = QPushButton("Сохранить настройки")
        save.setObjectName("primary")
        save.clicked.connect(self._save_settings)
        self.doctor_btn = QPushButton("Проверить окружение")
        self.doctor_btn.clicked.connect(self._run_doctor)
        btns.addWidget(save)
        btns.addWidget(self.doctor_btn)
        btns.addStretch(1)
        lay.addLayout(btns)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    # ---------------- config <-> UI

    def _cfg_to_ui(self) -> None:
        c = self.cfg
        h, l, cap, rf, rd = c.highlights, c.llm, c.captions, c.reframe, c.render
        self.max_clips_spin.setValue(h.max_clips)
        self.min_dur_spin.setValue(int(h.min_duration))
        self.max_dur_spin.setValue(int(h.max_duration))
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(c.whisper.language)))
        self.captions_check.setChecked(cap.enabled)

        local = bool(l.server_exe or l.server_cmd) or l.base_url.startswith(("http://127.0.0.1", "http://localhost"))
        self.llm_local.setChecked(local)
        self.llm_remote.setChecked(not local)
        self.server_exe_edit.setText(l.server_exe)
        self.model_path_edit.setText(l.model_path)
        self.n_cpu_moe_spin.setValue(l.n_cpu_moe)
        idx = self.ctx_combo.findData(l.ctx_size)
        if idx < 0:
            self.ctx_combo.addItem(str(l.ctx_size), l.ctx_size)
            idx = self.ctx_combo.count() - 1
        self.ctx_combo.setCurrentIndex(idx)
        self.server_cmd_edit.setText(l.server_cmd)
        self.base_url_edit.setText(l.base_url)
        self.api_key_edit.setText(l.api_key)
        self.model_name_edit.setText(l.model)
        self.rerank_check.setChecked(h.rerank)
        self._update_llm_mode()

        self.whisper_model_combo.setCurrentText(c.whisper.model)
        self.whisper_device_combo.setCurrentText(c.whisper.device)
        self.compute_combo.setCurrentText(c.whisper.compute_type)

        self.font_combo.setCurrentFont(QFont(cap.font))
        self.font_size_spin.setValue(cap.font_size)
        self.max_words_spin.setValue(cap.max_words)
        self.uppercase_check.setChecked(cap.uppercase)
        self.color_btn.set_rgb(cap.color)
        self.hl_color_btn.set_rgb(cap.highlight_color)
        self.position_spin.setValue(int(round(cap.position * 100)))

        idx = self.resolution_combo.findData((rf.output_width, rf.output_height))
        self.resolution_combo.setCurrentIndex(max(0, idx))
        self.two_shot_check.setChecked(rf.two_shot)
        self.speed_spin.setValue(rf.max_speed)
        self.encoder_combo.setCurrentIndex(max(0, self.encoder_combo.findData(rd.encoder)))
        self.cq_spin.setValue(rd.cq)
        self.workdir_edit.setText(c.workdir)

    def _ui_to_cfg(self) -> Config:
        c = copy.deepcopy(self.cfg)
        h, l, cap, rf, rd = c.highlights, c.llm, c.captions, c.reframe, c.render
        h.max_clips = self.max_clips_spin.value()
        h.min_duration = float(min(self.min_dur_spin.value(), self.max_dur_spin.value() - 5))
        h.max_duration = float(self.max_dur_spin.value())
        h.rerank = self.rerank_check.isChecked()
        c.whisper.language = self.lang_combo.currentData()
        c.whisper.model = self.whisper_model_combo.currentText().strip()
        c.whisper.device = self.whisper_device_combo.currentText()
        c.whisper.compute_type = self.compute_combo.currentText()
        if c.whisper.device == "cpu" and "float16" in c.whisper.compute_type:
            c.whisper.compute_type = "int8"

        if self.llm_local.isChecked():
            l.server_exe = self.server_exe_edit.text().strip().strip('"')
            l.model_path = self.model_path_edit.text().strip().strip('"')
            l.server_cmd = self.server_cmd_edit.text().strip()
            if not l.base_url.startswith(("http://127.0.0.1", "http://localhost")):
                l.base_url = "http://127.0.0.1:8080/v1"
                l.api_key = ""
        else:
            l.server_exe = ""
            l.server_cmd = ""
            l.base_url = self.base_url_edit.text().strip()
            l.api_key = self.api_key_edit.text().strip()
            l.model = self.model_name_edit.text().strip() or "local"
        l.n_cpu_moe = self.n_cpu_moe_spin.value()
        l.ctx_size = int(self.ctx_combo.currentData())

        cap.enabled = self.captions_check.isChecked()
        cap.font = self.font_combo.currentFont().family()
        cap.font_size = self.font_size_spin.value()
        cap.max_words = self.max_words_spin.value()
        cap.uppercase = self.uppercase_check.isChecked()
        cap.color = self.color_btn.rgb
        cap.highlight_color = self.hl_color_btn.rgb
        cap.position = self.position_spin.value() / 100

        rf.output_width, rf.output_height = self.resolution_combo.currentData()
        rf.two_shot = self.two_shot_check.isChecked()
        rf.max_speed = self.speed_spin.value()
        rd.encoder = self.encoder_combo.currentData()
        rd.cq = self.cq_spin.value()
        c.workdir = self.workdir_edit.text().strip() or "work"
        return c

    def _update_llm_mode(self) -> None:
        local = self.llm_local.isChecked()
        for w in (self.server_exe_edit, self.model_path_edit, self.n_cpu_moe_spin, self.ctx_combo, self.server_cmd_edit):
            w.setEnabled(local)
        for w in (self.base_url_edit, self.api_key_edit, self.model_name_edit):
            w.setEnabled(not local)

    def _save_settings(self, quiet: bool = False) -> None:
        self.cfg = self._ui_to_cfg()
        try:
            save_config(self.cfg, self.config_path)
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить {self.config_path}:\n{e}")
            return
        if not quiet:
            self.statusBar().showMessage(f"Настройки сохранены в {self.config_path}", 4000)
            self._refresh_projects()

    # ---------------- running

    def _busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def _start_new(self) -> None:
        source = self.source_edit.text().strip().strip('"')
        if not source:
            QMessageBox.information(self, "Нет видео", "Выберите видеофайл или вставьте ссылку.")
            return
        if not source.startswith(("http://", "https://")) and not Path(source).is_file():
            QMessageBox.warning(self, "Файл не найден", source)
            return
        self._launch(source, set())

    def _rerun(self, force: set) -> None:
        if not self.job:
            return
        try:
            video = json.loads((self.job / "source.json").read_text(encoding="utf-8"))["video"]
        except (OSError, ValueError, KeyError):
            QMessageBox.warning(self, "Нет исходника", "Для этого проекта не сохранён путь к исходному видео.")
            return
        if not Path(video).is_file():
            QMessageBox.warning(self, "Исходник не найден", f"Файл переместили или удалили:\n{video}")
            return
        self._launch(video, force)

    def _launch(self, source: str, force: set) -> None:
        if self._busy():
            return
        self._save_settings(quiet=True)
        self.player.stop()
        self.player.setSource(QUrl())  # release the file: Windows can't overwrite a file that is open
        self.log_view.clear()
        self.progress.setValue(0)
        self.thread = PipelineThread(source, copy.deepcopy(self.cfg), self.cfg.highlights.max_clips, force)
        self.thread.progress.connect(self._on_progress)
        self.thread.succeeded.connect(self._on_success)
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(lambda: self._set_running(False))
        self._set_running(True)
        self.stage_label.setText("Подготовка…")
        self.thread.start()

    def _stop(self) -> None:
        if self._busy():
            self.thread.cancel()
            self.stage_label.setText("Останавливаю после текущего шага…")
            self.stop_btn.setEnabled(False)

    def _set_running(self, running: bool) -> None:
        for w in (self.start_btn, self.rerender_btn, self.refind_btn, self.project_combo, self.doctor_btn):
            w.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _on_progress(self, stage: str, done: float, total: float) -> None:
        lo, hi = STAGE_RANGE.get(stage, (0, 100))
        frac = done / total if total else 0.0
        self.progress.setValue(int(lo + (hi - lo) * min(1.0, frac)))
        name = STAGE_NAMES.get(stage, stage)
        if stage == "transcribe" and total:
            name += f": {fmt_time(int(done * 1000))} из {fmt_time(int(total * 1000))}"
        elif stage in ("highlights", "render") and total:
            name += f": {int(done) + 1} из {int(total)}"
        self.stage_label.setText(name)

    def _on_success(self, job: str) -> None:
        self.progress.setValue(100)
        self.stage_label.setText("Готово")
        self._refresh_projects(select=Path(job))
        if self.table.rowCount():
            self.table.selectRow(0)

    def _on_failed(self, message: str) -> None:
        self.stage_label.setText(message.splitlines()[0][:120] if message else "Ошибка")
        if message != "Остановлено":
            QMessageBox.critical(self, "Ошибка", message[:2000] + "\n\nПодробности — в журнале внизу окна.")
        self._refresh_projects(select=self.job)

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # ---------------- projects & clips

    def _workdir(self) -> Path:
        return Path(self.cfg.workdir)

    def _refresh_projects(self, select: Optional[Path] = None) -> None:
        current = select or self.job
        root = self._workdir()
        jobs = []
        if root.is_dir():
            jobs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
                    and ((d / "clips.json").exists() or (d / "transcript.json").exists())]
        jobs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for d in jobs:
            self.project_combo.addItem(d.name, str(d))
        idx = self.project_combo.findData(str(current)) if current else -1
        self.project_combo.setCurrentIndex(idx if idx >= 0 else (0 if jobs else -1))
        self.project_combo.blockSignals(False)
        self._project_selected()

    def _project_selected(self) -> None:
        data = self.project_combo.currentData()
        self.job = Path(data) if data else None
        self.clips = []
        if self.job and (self.job / "clips.json").exists():
            try:
                self.clips = json.loads((self.job / "clips.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                log.warning("Не удалось прочитать clips.json: %s", e)
        self._fill_table()
        has_job = self.job is not None and not self._busy()
        self.open_folder_btn.setEnabled(self.job is not None)
        self.rerender_btn.setEnabled(has_job and bool(self.clips))
        self.refind_btn.setEnabled(has_job)

    def _clip_path(self, clip: dict) -> Optional[Path]:
        if not self.job or not clip.get("file"):
            return None
        p = self.job / clip["file"]
        return p if p.exists() else None

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self.clips))
        for row, c in enumerate(self.clips):
            dur = c["end"] - c["start"]
            items = [QTableWidgetItem(str(row + 1)), QTableWidgetItem(f"{c['score']:.1f}"),
                     QTableWidgetItem(f"{dur:.0f} с"), QTableWidgetItem(c.get("title") or "(без названия)")]
            rendered = self._clip_path(c) is not None
            for col, it in enumerate(items):
                if col < 3:
                    it.setTextAlignment(Qt.AlignCenter)
                if not rendered:
                    it.setForeground(QColor("#777"))
                    it.setToolTip("Клип ещё не отрендерен")
                self.table.setItem(row, col, it)
        self.details.clear()

    def _clip_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        c = self.clips[rows[0].row()]
        path = self._clip_path(c)
        if path:
            self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self.player.play()
        else:
            self.player.setSource(QUrl())
        self.details.setHtml(self._clip_html(c, path))

    @staticmethod
    def _clip_html(c: dict, path: Optional[Path]) -> str:
        from html import escape

        labels = {"hook_score": "Хук", "payoff_score": "Развязка", "standalone_score": "Понятность",
                  "emotion_score": "Эмоции", "viral_score": "Виральность"}
        scores = " · ".join(f"{labels.get(k, k)} <b>{v}</b>" for k, v in c.get("scores", {}).items())
        why = c.get("rerank_reason") or c.get("reason") or ""
        start = f"{int(c['start'] // 60)}:{c['start'] % 60:04.1f}"
        return (
            f"<h3 style='margin:0'>{escape(c.get('title', ''))}</h3>"
            f"<p style='color:#9aa'>Оценка <b>{c['score']:.1f}</b> · с {start} · {c['end'] - c['start']:.0f} с"
            f"{'' if path else ' · <i>не отрендерен</i>'}</p>"
            f"<p>{scores}</p>"
            f"<p><b>Хук:</b> {escape(c.get('hook', ''))}</p>"
            f"<p><b>Почему:</b> {escape(why)}</p>"
            f"<p style='color:#bbb'>{escape(c.get('text', ''))}</p>"
        )

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_position(self, pos: int) -> None:
        if not self.pos_slider.isSliderDown():
            self.pos_slider.setValue(pos)
        self.time_label.setText(f"{fmt_time(pos)} / {fmt_time(self.player.duration())}")

    def _on_playback_state(self, state) -> None:
        icon = QStyle.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState else QStyle.SP_MediaPlay
        self.play_btn.setIcon(self.style().standardIcon(icon))

    def _open_folder(self) -> None:
        if self.job:
            folder = self.job / "clips" if (self.job / "clips").is_dir() else self.job
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    # ---------------- dialogs & helpers

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", VIDEO_FILTER)
        if path:
            self.source_edit.setText(path)

    def _browse_into(self, edit: QLineEdit, title: str, flt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, edit.text(), flt)
        if path:
            edit.setText(path)

    def _browse_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка проектов", self.workdir_edit.text())
        if path:
            self.workdir_edit.setText(path)

    def _run_doctor(self) -> None:
        self.cfg = self._ui_to_cfg()
        self.doctor_btn.setEnabled(False)
        self.doctor_btn.setText("Проверяю…")
        self.doctor_thread = DoctorThread(copy.deepcopy(self.cfg))
        self.doctor_thread.done.connect(self._show_doctor)
        self.doctor_thread.start()

    def _show_doctor(self, checks) -> None:
        from html import escape

        self.doctor_btn.setEnabled(not self._busy())
        self.doctor_btn.setText("Проверить окружение")
        rows = []
        for c in checks:
            mark = "<span style='color:#4caf50'>✔</span>" if c.ok else "<span style='color:#f44336'>✖</span>"
            hint = f"<br><span style='color:#aaa'>{escape(c.hint)}</span>" if c.hint else ""
            rows.append(f"<p>{mark} {escape(c.name)}{hint}</p>")
        ok = all(c.ok for c in checks)
        dlg = QDialog(self)
        dlg.setWindowTitle("Проверка окружения")
        dlg.resize(620, 420)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setHtml(("<h3>Всё готово к работе</h3>" if ok else "<h3>Нужно исправить:</h3>") + "".join(rows))
        lay.addWidget(view)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    # ---------------- window events

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if urls:
            u = urls[0]
            self.source_edit.setText(u.toLocalFile() if u.isLocalFile() else u.toString())
            self.tabs.setCurrentIndex(0)

    def closeEvent(self, e) -> None:
        if self._busy():
            if QMessageBox.question(self, "Идёт обработка", "Остановить обработку и выйти?") != QMessageBox.Yes:
                e.ignore()
                return
            self.thread.cancel()
            self.thread.wait(30000)
        self.player.stop()
        e.accept()


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text, accent = QColor(30, 31, 34), QColor(43, 45, 48), QColor(223, 225, 229), QColor(0, 145, 234)
    p.setColor(QPalette.Window, alt)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.ToolTipBase, base)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor(56, 58, 62))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Link, accent)
    p.setColor(QPalette.PlaceholderText, QColor(130, 133, 140))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(110, 112, 118))
    app.setPalette(p)
    app.setStyleSheet("""
        QPushButton { padding: 6px 12px; border-radius: 4px; }
        QPushButton#primary { background: #0091ea; color: white; font-weight: 600; border: none; }
        QPushButton#primary:hover { background: #10a1fa; }
        QPushButton#primary:disabled { background: #3a4a55; color: #889; }
        QGroupBox { font-weight: 600; border: 1px solid #45474c; border-radius: 6px; margin-top: 12px; padding-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QProgressBar { border: 1px solid #45474c; border-radius: 4px; text-align: center; height: 18px; }
        QProgressBar::chunk { background: #0091ea; border-radius: 3px; }
        QTableWidget { gridline-color: #3a3c40; }
    """)


def run_gui(config_path: Optional[Path] = None) -> int:
    if sys.stdout is None or sys.stderr is None:
        # pythonw.exe has no console; libraries that print (download progress bars) would crash.
        stream = open("clipforge-gui.log", "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("clipforge")
    apply_dark_theme(app)
    win = MainWindow(Path(config_path or "config.yaml"))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
