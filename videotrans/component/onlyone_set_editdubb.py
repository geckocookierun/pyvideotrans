import json
import random
import re
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal, QUrl, QSettings
from PySide6.QtGui import QIcon, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox, QMenu, QInputDialog, QSplitter, QStackedLayout, QCheckBox,
    QTextEdit
)
from pydub import AudioSegment

from videotrans import tts
from videotrans.component._danspmixin import DanspMixin

from videotrans.configure.config import ROOT_DIR, tr, settings, logger,app_cfg,REDUBB_QUEUE_FILE,REDUBB_STATUS_FILE

# 需要重新配音的队列
from videotrans.util._srt_parse import ms_to_time_string
from videotrans.util.help_misc import vail_file, atomic_write_json, open_url

# 存储待配音数据,以文件名为key
_REDUBB_QUEUE={}
# 存储文件名映射行index
_FILENAME_ROW_MAP={}

# 本地内置TTS渠道，只启动一个配音任务
class ReDubb(QThread):
    uito = Signal(str)

    def __init__(self, *, parent=None, tts_dict=None, language=None):
        super().__init__(parent=parent)
        self.tts_dict = tts_dict
        self.language = language
        atomic_write_json([tts_dict],REDUBB_QUEUE_FILE)

    def _checking_task(self):
        """检测重新配音是否完成，完成发送信号，未完成重新写入文件"""
        while 1:
            if Path(REDUBB_STATUS_FILE).exists():
                return
            time.sleep(0.5)
            # _noend_dict={}
            _shound_pop=[]
            for filename,it in _REDUBB_QUEUE.items():
                if Path(REDUBB_STATUS_FILE).exists():
                    return
                if vail_file(filename):
                    _shound_pop.append(filename)
                    self.uito.emit(f"ok:{_FILENAME_ROW_MAP[filename]}")

            # 移除已生成的
            for filename in _shound_pop:
                if vail_file(filename):
                    _REDUBB_QUEUE.pop(filename)

    def run(self):
        try:
            threading.Thread(target=self._checking_task).start()
            tts.run(
                queue_tts=[self.tts_dict],
                language=self.language,
                tts_type=self.tts_dict['tts_type'],
                is_redubb=True,
                is_cuda=app_cfg.onlyone_is_cuda
            )
        except Exception as e:
            from videotrans.configure.excepts import get_msg_from_except
            except_msg = get_msg_from_except(e)
            msg = f'error:{except_msg}:\n' + traceback.format_exc()
            self.uito.emit(msg)

# 非本地，直接一个重新配音起一个配音任务
class ReDubbNolocal(QThread):
    uito = Signal(str)

    def __init__(self, *, parent=None, idx=None, tts_dict=None, language=None):
        super().__init__(parent=parent)
        self.tts_dict = tts_dict
        self.language = language
        self.idx = idx

    def run(self):
        try:
            tts.run(
                queue_tts=[self.tts_dict],
                language=self.language,
                tts_type=self.tts_dict['tts_type'],
                is_cuda=app_cfg.onlyone_is_cuda
            )
            self.uito.emit(f"ok:{self.idx}")
        except Exception as e:
            from videotrans.configure.excepts import get_msg_from_except
            except_msg = get_msg_from_except(e)
            msg = f'error:{except_msg}:\n' + traceback.format_exc()
            self.uito.emit(msg)


PASSAGE_GAP_MS = 2500
PASSAGE_CPS_FALLBACK = 18.0
PASSAGE_CHAR_HEADROOM = 0.92
BATCH_REDUB_DELAY_MS = 2000


def expand_passage_spans(times, seeds):
    """Grow each seed across neighboring cues until a pause of PASSAGE_GAP_MS."""
    n = len(times)
    ranges = []
    for seed in seeds:
        if seed < 0 or seed >= n:
            continue
        lo = hi = seed
        while lo > 0 and times[lo][0] - times[lo - 1][1] < PASSAGE_GAP_MS:
            lo -= 1
        while hi + 1 < n and times[hi + 1][0] - times[hi][1] < PASSAGE_GAP_MS:
            hi += 1
        ranges.append((lo, hi))
    ranges.sort()
    merged = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def split_passage_times(start_ms, end_ms, texts):
    """Split a fixed window across lines in proportion to their length."""
    weights = [max(len((text or '').strip()), 1) for text in texts]
    total = sum(weights)
    span = max(int(end_ms) - int(start_ms), len(weights))
    times = []
    cursor = int(start_ms)
    for i, weight in enumerate(weights):
        if i == len(weights) - 1:
            nxt = int(end_ms)
        else:
            used = sum(weights[:i + 1])
            nxt = int(start_ms) + int(round(span * used / total))
            nxt = min(max(nxt, cursor + 1), int(end_ms) - (len(weights) - i - 1))
        times.append((cursor, nxt))
        cursor = nxt
    return times


def passage_target_chars(dubbed_lines, tts_total_s, window_s):
    """Estimate how many characters fit the spoken window for this voice."""
    char_total = sum(len(text or '') for text in dubbed_lines)
    cps = (char_total / tts_total_s) if tts_total_s > 0.5 else PASSAGE_CPS_FALLBACK
    return max(40, int(window_s * cps * PASSAGE_CHAR_HEADROOM))


def clean_passage(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.I | re.S)
    text = re.sub(r'</?TRANSLATE_TEXT>', '', text, flags=re.I)
    lines = []
    for line in text.splitlines():
        line = line.strip().strip('"\'“”«»')
        line = re.sub(r'^\d+[\.\)、]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line).strip()
        if line:
            lines.append(line)
    return lines


def _split_passage_line(text):
    for pattern in (r'(?<=[\.!?…])\s+', r'(?<=[;；])\s+', r'(?<=[,，])\s+'):
        bits = [bit.strip() for bit in re.split(pattern, text, maxsplit=1) if bit.strip()]
        if len(bits) == 2 and min(len(bit) for bit in bits) >= 8:
            return bits
    return None


def fit_passage_lines(lines, count):
    """Return exactly `count` narration lines, or None."""
    lines = [line.strip() for line in lines if line and line.strip()]
    if count <= 0 or not lines:
        return None
    if len(lines) > count:
        return lines[:count - 1] + [' '.join(lines[count - 1:])]
    while len(lines) < count:
        index = max(range(len(lines)), key=lambda i: len(lines[i]))
        parts = _split_passage_line(lines[index])
        if not parts:
            return None
        lines[index:index + 1] = parts
    return lines


def _format_numbered(lines):
    return "\n".join(f"{i}. {text}" for i, text in enumerate(lines, 1))


def passage_prompt(dubbed_lines, source_lines, window_s, target_chars):
    n = len(dubbed_lines)
    low = max(40, int(target_chars * 0.88))
    source = _format_numbered([text for text in (source_lines or []) if text])
    source_block = ""
    if source.strip():
        source_block = (
            "\nSource lines carry the meaning. Prefer them when a dubbed line is "
            "split mid-word or mistranslated.\n"
            f"{source}\n"
        )
    return (
        "Retell this whole subtitle passage as third-person narration.\n"
        "Write brand-new complete sentences, as a narrator who just watched the scene.\n"
        "Do NOT shorten by deleting words from the dubbed lines. "
        "Do NOT only swap pronouns and keep the old sentence frames.\n"
        "Change verbs and phrasing. Every sentence needs a clear subject and verb.\n"
        "Keep every event, name, number, and relationship.\n"
        "Forbidden: choppy fragments, missing subjects, truncated clauses.\n"
        f"Write in the same language as the dubbed lines. "
        f"Write exactly {n} sentences, one sentence per line.\n"
        f"The spoken window is {window_s:.1f} seconds. "
        f"Keep the whole passage between {low} and {target_chars} characters, including spaces.\n"
        "No labels, no timestamps, no character counts, no explanation.\n"
        "\n"
        "Dubbed lines:\n"
        f"{_format_numbered(dubbed_lines)}\n"
        f"{source_block}\n"
        "# OUTPUT FORMAT\n"
        "Output ONLY the narration inside <TRANSLATE_TEXT> and </TRANSLATE_TEXT>.\n"
        "\n"
        "<INPUT>\n"
        "{batch_input}\n"
        "</INPUT>"
    )


def passage_retry_prompt(lines, target_chars, count):
    chars = sum(len(line) for line in lines)
    low = max(40, int(target_chars * 0.88))
    return (
        "The draft below is still too long for the spoken window. "
        "Retell the WHOLE passage again as third-person narration.\n"
        "Write brand-new complete sentences with subject and verb. "
        "Do not delete words until a sentence breaks. "
        "Do not only swap pronouns.\n"
        f"Current draft is {chars} characters. "
        f"Write exactly {count} sentences, one per line, in the same language.\n"
        f"Keep the whole passage between {low} and {target_chars} characters.\n"
        "Keep every event, name, and number.\n"
        "\n"
        f"{_format_numbered(lines)}\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "Output ONLY the narration inside <TRANSLATE_TEXT> and </TRANSLATE_TEXT>.\n"
        "\n"
        "<INPUT>\n"
        "{batch_input}\n"
        "</INPUT>"
    )


class PassageRewrite(QThread):
    """Retell each neighboring span as new narration, without touching the table."""
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, *, parent=None, spans=None, translate_type=0):
        super().__init__(parent=parent)
        self.spans = spans or []
        self.translate_type = translate_type

    def _ask(self, model, prompt, text):
        model.prompt = prompt
        raw = model._item_task(text)
        if not raw:
            return []
        return clean_passage(raw)

    def _rewrite_span(self, model, span):
        dubbed = span['dubbed']
        count = len(dubbed)
        lines = fit_passage_lines(self._ask(
            model,
            passage_prompt(dubbed, span['source'], span['window_s'], span['target_chars']),
            "\n".join(dubbed),
        ), count)
        if not lines:
            return None
        if sum(len(line) for line in lines) > span['target_chars'] * 1.15:
            shorter = fit_passage_lines(self._ask(
                model,
                passage_retry_prompt(lines, span['target_chars'], count),
                "\n".join(lines),
            ), count)
            if shorter:
                lines = shorter
        return lines

    def run(self):
        from videotrans.translator._openaicompat import OpenAICampat
        from videotrans.translator._runner import get_model_transobj
        try:
            model = get_model_transobj(translate_type=self.translate_type)
        except Exception as e:
            self.failed.emit(str(e))
            return
        if not isinstance(model, OpenAICampat):
            self.failed.emit('channel')
            return
        if getattr(model, 'ainame', '') == 'openrouter':
            model.extra_body = {"reasoning": {"enabled": False}}
        model.temperature = 0.4
        results = []
        for span in self.spans:
            try:
                lines = self._rewrite_span(model, span)
            except Exception as e:
                logger.warning(f'Passage rewrite failed: {e}')
                self.failed.emit(str(e))
                return
            if not lines:
                self.failed.emit('empty')
                return
            results.append({**span, 'new_lines': lines})
        self.finished_ok.emit(results)


class EditDubbingResultDialog(QDialog,DanspMixin):
    def __init__(
            self,
            novoice_mp4: str = None,
            language=None,
            cache_folder: str = None,
            parent=None
    ):
        super().__init__()
        self.parent = parent
        self.language = language
        self.cache_folder = cache_folder
        self.novoice_mp4 = novoice_mp4
        self._video_end_ms = -1
        self._audio_playing = False
        self.queue_tts = []
        self._rewrite_thread = None
        self._redub_queue = []
        self._batch_redub = False

        # 任务
        self.task=None
        queue_tts_file = Path(f'{cache_folder}/queue_tts.json')
        if queue_tts_file.exists():
            try:
                self.queue_tts = json.loads(queue_tts_file.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f'Failed to load queue_tts.json: {e}')

        self.setWindowTitle(tr("Proofreading and dubbing - Re-dubbing"))
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/videotrans/styles/icon.ico"))

        self.setMinimumWidth(int(parent.screen_size[0]*0.95))
        self.setMinimumHeight(int(parent.screen_size[1]*0.9))
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowMaximizeButtonHint)

        self.count_down = int(float(settings.get('countdown_sec', 1)))

        main_layout = QVBoxLayout(self)

        # Top Bar
        hstop = QHBoxLayout()
        self.prompt_label = QLabel(tr("You can check the voiceover here, or modify the text and re-encode the voiceover. Please stop the countdown before proceeding"))
        self.prompt_label.setStyleSheet('color:#999999')
        self.prompt_label.setWordWrap(True)
        hstop.addWidget(self.prompt_label)

        self.stop_button = QPushButton(f"{tr('Click here to stop the countdown')}({self.count_down})")
        self.stop_button.setStyleSheet("color:#ffff00")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setMinimumSize(QSize(300, 35))
        self.stop_button.clicked.connect(self.stop_countdown)
        hstop.addWidget(self.stop_button)
        main_layout.addLayout(hstop)

        # 操作提示
        htips = QHBoxLayout()
        
        prompt_label2 = QLabel(tr("Right-click: Menu | << >> : Adjust time")+"\n"+tr('Shortened and Exceeded mean'))
        prompt_label2.setWordWrap(True)
        prompt_label2.setStyleSheet("color:#999999")
        htips.addWidget(prompt_label2)
        self.video_hint = QLabel(tr("Previewing a dubbed line will sync-play the video segment"))
        self.video_hint.setStyleSheet("color:#ffcc00; font-size:12px; background-color:transparent;")
        self.video_hint.setWordWrap(True)
        htips.addWidget(self.video_hint)
        main_layout.addLayout(htips)
        
        # ===================== Splitter: video (top) + table (bottom) =====================
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(6)

        # --- Top area: video display ---
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #1a1a1a;")
        self.video_widget.setMinimumHeight(150)

        

        self.video_status = QLabel("")
        self.video_status.setStyleSheet("color:#aaaaaa; font-size:12px;")


        self._stack = QStackedLayout()
        self._stack.addWidget(self.video_widget)

        videostatus_layout=QHBoxLayout()
        videostatus_layout.setContentsMargins(0, 2, 0, 2)
        self.stop_play_btn=QPushButton()
        self.stop_play_btn.setToolTip(tr('Click to pause playback'))
        self.stop_play_btn.setText(tr('Click to pause playback'))
        self.stop_play_btn.clicked.connect(self._pause_play)
        self.stop_play_btn.setCursor(Qt.PointingHandCursor)
        self.stop_play_btn.hide()
        self.btn_minus = QPushButton("-")
        self.btn_minus.setToolTip(tr('Decrease table font size'))
        self.btn_minus.setFixedWidth(30)
        self.btn_plus = QPushButton("+")
        self.btn_plus.setToolTip(tr('Increase table font size'))
        self.btn_plus.setFixedWidth(30)
        self.btn_plus.clicked.connect(lambda: self.change_table_font_size(2))
        self.btn_minus.clicked.connect(lambda: self.change_table_font_size(-2))
        videostatus_layout.addStretch()
        videostatus_layout.addWidget(self.btn_minus)
        videostatus_layout.addWidget(self.btn_plus)
        videostatus_layout.addWidget(self.stop_play_btn)
        videostatus_layout.addWidget(self.video_status)
        videostatus_layout.addStretch()



        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(self._stack)
        top_layout.addLayout(videostatus_layout)
        self.splitter.addWidget(top_container)

        # --- Bottom area: table + buttons ---
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Loading
        self.loading_widget = QWidget()
        load_layout = QVBoxLayout(self.loading_widget)
        self.loading_label = QLabel(tr('Loading...'), self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        load_layout.addWidget(self.loading_label)
        bottom_layout.addWidget(self.loading_widget)

        # Table Widget (初始隐藏)
        self.table = QTableWidget()
        self.table.setVisible(False)
        sets = QSettings("pyvideotrans", "settings")
        fontsize = int(sets.value("danshipin_table_fontsize", 0))
        if fontsize>0:
            default_font = self.table.font()
            default_font.setPointSize(fontsize)  # 设置为 16px
            self.table.setFont(default_font)
            self.table.horizontalHeader().setFont(default_font)
            self.table.verticalHeader().setFont(default_font)
        bottom_layout.addWidget(self.table, 1)

        # Bottom Bar
        self.save_button = QPushButton(tr("continnextstep"))
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.setMinimumSize(QSize(400, 35))
        self.save_button.clicked.connect(self.save_and_close)
        
        cancel_button = QPushButton(tr("Terminate this mission"))
        cancel_button.setCursor(Qt.PointingHandCursor)
        cancel_button.setMaximumSize(QSize(200, 30))
        cancel_button.setStyleSheet("background-color:transparent;color:#ffff00")
        cancel_button.clicked.connect(self.cancel_and_close)
        
        help_button = QPushButton(tr("Detailed instructions"))
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setMaximumSize(QSize(200, 30))
        help_button.setStyleSheet("background-color:transparent")
        help_button.clicked.connect(self.help_doc)
        
        
        hpassage = QHBoxLayout()
        self.btn_select_exceeded = QPushButton(tr("Select exceeded lines"))
        self.btn_select_exceeded.setCursor(Qt.PointingHandCursor)
        self.btn_select_exceeded.setToolTip(tr("Tick every line whose TTS audio is longer than the subtitle slot"))
        self.btn_select_exceeded.clicked.connect(self._select_exceeded_lines)
        self.btn_rewrite_passage = QPushButton(tr("Rewrite this passage"))
        self.btn_rewrite_passage.setCursor(Qt.PointingHandCursor)
        self.btn_rewrite_passage.setToolTip(tr("Rewrite the selected line together with neighboring lines, show the new narration, then re-dub it. The passage start and end times stay the same."))
        self.btn_rewrite_passage.clicked.connect(self._rewrite_passage)
        hpassage.addWidget(self.btn_select_exceeded)
        hpassage.addWidget(self.btn_rewrite_passage)
        hpassage.addStretch()

        hautorate=QHBoxLayout()
        
        self.voice_autorate = QCheckBox()
        self.voice_autorate.setObjectName("voice_autorate")
        self.voice_autorate.setChecked(app_cfg.onlyone_voice_autorate)
        self.voice_autorate.setText(tr("Dubbing acceler"))
        
        self.video_autorate = QCheckBox()
        self.video_autorate.setObjectName("voice_autorate")
        self.video_autorate.setChecked(app_cfg.onlyone_video_autorate)
        self.video_autorate.setText(tr("Slow video"))
        
        self.remove_silent_mid = QCheckBox()
        self.remove_silent_mid.setObjectName("remove_silent_mid")
        self.remove_silent_mid.setChecked(app_cfg.onlyone_remove_silent_mid)
        self.remove_silent_mid.setText(tr("Del inline mute?"))
        
        self.align_sub_audio = QCheckBox()
        self.align_sub_audio.setObjectName("align_sub_audio")
        self.align_sub_audio.setChecked(app_cfg.onlyone_align_sub_audio)
        self.align_sub_audio.setText(tr("Align subtitles and audio"))

        self.remove_silent_mid.setVisible(not app_cfg.onlyone_voice_autorate and not app_cfg.onlyone_video_autorate)
        self.align_sub_audio.setVisible(not app_cfg.onlyone_voice_autorate and not app_cfg.onlyone_video_autorate)
        
        
        self.voice_autorate.toggled.connect(self.check_voice_autorate)
        self.video_autorate.toggled.connect(self.check_video_autorate)
        
        tipslabe=QLabel(tr('You can select again whether to enable voice acceleration video slow'))
        tipslabe.setStyleSheet("""color:#999999""")
        hautorate.addWidget(self.voice_autorate)
        hautorate.addWidget(self.video_autorate)
        hautorate.addWidget(self.remove_silent_mid)
        hautorate.addWidget(self.align_sub_audio)
        hautorate.addWidget(tipslabe)       
        hautorate.addStretch()

        bottom_layout.addLayout(hpassage)
        bottom_layout.addLayout(hautorate)
        
        bottom_btn_layout = QHBoxLayout()
        bottom_btn_layout.addStretch()
        bottom_btn_layout.addWidget(self.save_button)
        bottom_btn_layout.addWidget(cancel_button)
        bottom_btn_layout.addWidget(help_button)
        bottom_btn_layout.addStretch()
        bottom_layout.addLayout(bottom_btn_layout)

        self.splitter.addWidget(bottom_container)
        self.splitter.setSizes([int(parent.height * 0.22), int(parent.height * 0.68)])
        main_layout.addWidget(self.splitter, 1)

        # 延迟加载表格
        QTimer.singleShot(200, self.load_table)



    def check_voice_autorate(self, state):
        if state:
            self.remove_silent_mid.setVisible(False)
            self.align_sub_audio.setVisible(False)
        elif not self.video_autorate.isChecked():
            self.remove_silent_mid.setVisible(True)
            self.align_sub_audio.setVisible(True)

    def check_video_autorate(self, state):
        if state:
            self.remove_silent_mid.setVisible(False)
            self.align_sub_audio.setVisible(False)
        elif not self.voice_autorate.isChecked():
            self.remove_silent_mid.setVisible(True)
            self.align_sub_audio.setVisible(True)

    @staticmethod
    def help_doc():
        open_url('https://pyvideotrans.com/danshipin')

    def load_table(self):
        """极致性能加载表格"""
        if not self.isVisible():
            return
        try:
            if Path(REDUBB_QUEUE_FILE).exists():
                Path(REDUBB_QUEUE_FILE).unlink(missing_ok=True)
            if Path(REDUBB_STATUS_FILE).exists():
                Path(REDUBB_STATUS_FILE).unlink(missing_ok=True)
        except OSError:
            pass

        try:
            # 1. 创建 QTableWidget - 6列
            # 列：Line | Play | Start | End | Status | Text
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels([
                tr("Line"),
                tr("Time Axis"),
                '\u23F5',
                '\u270D'+tr("Start Time")+'/s',
                '\u270D'+tr("End Time")+'/s',
                tr("Dubbed Status"),
                '\u270D'+tr("Subtitle Text")
            ])
            
            # 2.
            self.table.setShowGrid(True)
            self.table.setAlternatingRowColors(False)
            self.table.setWordWrap(True)
            self.table.setMouseTracking(False)
            self.table.setFocusPolicy(Qt.StrongFocus)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
            
            # 垂直表头
            v_header = self.table.verticalHeader()
            v_header.setVisible(False)
            v_header.setSectionResizeMode(QHeaderView.ResizeToContents)
            v_header.setMinimumSectionSize(24)
            
            # 水平表头
            h_header = self.table.horizontalHeader()
            h_header.setSectionResizeMode(0, QHeaderView.Fixed)  # Line
            h_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Play
            h_header.setSectionResizeMode(2, QHeaderView.Fixed)  # Play
            h_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Start
            h_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # End
            h_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Status
            h_header.setSectionResizeMode(6, QHeaderView.Stretch)  # Text
            
            # 设置列宽
            self.table.setColumnWidth(0, 78)   # Line + checkbox
            self.table.setColumnWidth(1, 100)   # Timex
            self.table.setColumnWidth(2, 30)   # Play
            self.table.setColumnWidth(3, 100)   # Start
            self.table.setColumnWidth(4, 100)   # End
            self.table.setColumnWidth(5, 180)  # Status
            

            self.table.setStyleSheet(self.table_style_css)
            
            # 3. 预计算数据
            self._precompute_data()
            
            # 4. 设置行数
            total_rows = len(self.queue_tts)
            self.table.setRowCount(total_rows)
            
            # 5. 分批填充数据
            self._batch_fill(0, min(total_rows, 200))
            
            # 6. 显示表格
            self.loading_widget.setVisible(False)
            self.table.setVisible(True)
            
            # 7. 延迟加载剩余数据
            if total_rows > 200:
                QTimer.singleShot(0, lambda: self._load_remaining(200))
            
            # 8. 连接信号
            self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
            self.table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.table.customContextMenuRequested.connect(self._show_context_menu)
            # 拦截表格按键事件（Ctrl+方向键调时间）
            self.table.installEventFilter(self)
            
            # 9. 启动倒计时
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_countdown)
            self.timer.start(1000)
            self._play_with_video(None,0,5)
            if self.parent:
                self.raise_()                
                self.activateWindow()
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.loading_label.setText(f"Error: {e}")

    def _precompute_data(self):
        """预计算所有显示数据"""
        for item in self.queue_tts:
            duration = (item['end_time'] - item['start_time']) / 1000.0
            dubbing = float(item.get('dubbing_s', 0.0))
            diff = round(dubbing - duration, 3)
            
            if dubbing <= 0.0:
                msg = tr('No audio')
            elif diff > 0:
                msg = f'[{dubbing}s]{tr("Exceeded")} {diff}s'
            elif diff < 0:
                msg = f'[{dubbing}s]{tr("Shortened")} {abs(diff)}s'
            else:
                msg = f'{dubbing}s'
            
            item['_duration'] = duration
            item['_msg'] = msg

    def _text_change(self, item):
        _column=item.column()
        if _column in [3,4]:
            row = item.row()
            new_text = item.text().strip()
            offset=(float(new_text)*1000)-self.queue_tts[row]['start_time' if _column==3 else 'end_time']
            self._adjust_time(row,'start' if _column==3 else 'end',offset)


    def _batch_fill(self, start_row, end_row):
        """批量填充表格数据"""
        for row in range(start_row, end_row):
            data = self.queue_tts[row]
            
            # 0: Line
            line_item = QTableWidgetItem(str(data['line']))
            line_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            line_item.setCheckState(Qt.Unchecked)
            line_item.setData(Qt.UserRole, row)
            line_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, line_item)

            line_item = QTableWidgetItem(f'{data["startraw"]}->{data["endraw"]} ({(data["end_time"]-data["start_time"])/1000.0}s)')
            line_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            line_item.setData(Qt.UserRole, row)
            line_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, line_item)

            # 1: Play button
            btn = QPushButton("\u23F5")
            btn.setObjectName("playBtn")
            btn.setCursor(Qt.PointingHandCursor)
            s = data['start_time']
            e = data['end_time']
            btn.clicked.connect(lambda checked=False, _s=s, _e=e, _r=row: self._listen(_r))
            self.table.setCellWidget(row, 2, btn)
            
            # 2: Start
            start_item = QTableWidgetItem(f"{data['start_time']/1000.0}")
            start_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            start_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, start_item)
            
            # 3: End
            end_item = QTableWidgetItem(f"{data['end_time']/1000.0}")
            end_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            end_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, end_item)
            
            # 4: Status
            msg_item = QTableWidgetItem(data['_msg'])
            msg_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            msg_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            dubbing = float(data.get('dubbing_s', 0.0))
            diff=dubbing - float( data['_duration'])
            if dubbing <= 0:
                msg_item.setForeground(QColor("#ff4d4d"))
            elif  diff>0:
                msg_item.setForeground(QColor("#ff6600"))
            elif  diff<0:
                msg_item.setForeground(QColor("#ffffff"))
            else:
                msg_item.setForeground(QColor("#66ff66"))
            self.table.setItem(row, 5, msg_item)
            
            # 5: Text (可编辑)
            text_item = QTableWidgetItem(data['text'])
            text_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            text_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 6, text_item)
        self.table.itemChanged.connect(self._text_change)

    def _load_remaining(self, start_row):
        """延迟加载剩余行"""
        total = len(self.queue_tts)
        batch_size = 500  # 更大批次
        
        end_row = min(start_row + batch_size, total)
        self._batch_fill(start_row, end_row)
        
        if end_row < total:
            QTimer.singleShot(0, lambda: self._load_remaining(end_row))

    def eventFilter(self, obj, event):
        """拦截表格的 Ctrl+方向键事件，调整 Start/End 时间"""
        if obj is self.table and event.type() == event.Type.KeyPress:
            if event.modifiers() & Qt.ControlModifier:
                key = event.key()
                if key in (Qt.Key_Left, Qt.Key_Right):
                    row = self.table.currentRow()
                    col = self.table.currentColumn()
                    if row >= 0 and col in (3, 4):
                        offset = -100 if key == Qt.Key_Left else 100
                        mode = 'start' if col == 3 else 'end'
                        self._adjust_time(row, mode, offset)
                        return True  # 拦截事件，阻止表格默认行为
        return super().eventFilter(obj, event)

    def _on_cell_double_clicked(self, row, col):
        if col == 6:  # Text 列 - 编辑
            self.table.editItem(self.table.item(row, col))

    def _show_context_menu(self, pos):
        """显示右键菜单"""
        item = self.table.itemAt(pos)
        if not item:
            return
        
        row = item.row()
        col = item.column()
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2a2a2a;
                color: #ccc;
                border: 1px solid #444;
            }
            QMenu::item:selected {
                background-color: #0066cc;
            }
        """)
        
        if col == 3:  # Start 列
            menu.addAction(tr("Start -0.1s"), lambda: self._adjust_time(row, 'start', -100))
            menu.addAction(tr("Start +0.1s"), lambda: self._adjust_time(row, 'start', 100))
            menu.addSeparator()
            menu.addAction(tr("Custom adjust..."), lambda: self._custom_adjust(row, 'start'))
            
        elif col == 4:  # End 列
            menu.addAction(tr("End -0.1s"), lambda: self._adjust_time(row, 'end', -100))
            menu.addAction(tr("End +0.1s"), lambda: self._adjust_time(row, 'end', 100))
            menu.addSeparator()
            menu.addAction(tr("Custom adjust..."), lambda: self._custom_adjust(row, 'end'))
            
        elif col == 5:  # Status 列
            menu.addAction(tr("Trial dubbing"), lambda: self._listen(row))
            menu.addAction(tr("Re-dubbed"), lambda: self._redub(row))
            
        elif col == 6:  # Text 列
            menu.addAction(tr("Trial dubbing"), lambda: self._listen(row))
            menu.addAction(tr("Re-dubbed"), lambda: self._redub(row))
            menu.addSeparator()
            menu.addAction(tr("Clear text"), lambda: self._clear_text(row))
        
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _custom_adjust(self, row, mode):
        """自定义时间调整"""
        ms, ok = QInputDialog.getInt(
            self, tr("Adjust time"), 
            tr("Enter milliseconds (positive or negative):"),
            0, -10000, 10000, 100
        )
        if ok:
            self._adjust_time(row, mode, ms)

    def _clear_text(self, row):
        """清空文本"""
        text_item = self.table.item(row, 6)
        if text_item:
            text_item.setText("")
            self.queue_tts[row]['text'] = ""

    def _adjust_time(self, row, mode, offset_ms):
        """调整时间"""
        self.stop_countdown()
        
        if not (0 <= row < len(self.queue_tts)):
            return

        item = self.queue_tts[row]
        prev_item = self.queue_tts[row - 1] if row > 0 else None
        next_item = self.queue_tts[row + 1] if row < len(self.queue_tts) - 1 else None

        buffer = 0

        if mode == 'start':
            new_start = item['start_time'] + offset_ms

            limit_min = 0
            if prev_item:
                limit_min = prev_item['end_time'] + buffer

            if new_start < limit_min:
                self._show_error(tr("Cannot overlap with previous subtitle"))
                return

            limit_max = item['end_time'] - buffer
            if new_start > limit_max:
                self._show_error(tr("Start time cannot exceed End time"))
                return

            item['start_time'] = new_start
            item['start_time_source'] = new_start
            item['startraw'] = self._ms_to_fmt(new_start)

        elif mode == 'end':
            new_end = item['end_time'] + offset_ms

            limit_min = item['start_time'] + buffer
            if new_end < limit_min:
                self._show_error(tr("End time cannot be less than Start time"))
                return

            limit_max = float('inf')
            if next_item:
                limit_max = next_item['start_time'] - buffer

            if new_end > limit_max:
                self._show_error(tr("Cannot overlap with next subtitle"))
                return

            item['end_time'] = new_end
            item['end_time_source'] = new_end
            item['endraw'] = self._ms_to_fmt(new_end)

        # 更新显示
        self._refresh_row(row)

    @staticmethod
    def _ms_to_fmt(ms):
        """毫秒转时间格式"""
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

    def _refresh_row(self, row,msg=None):
        """刷新指定行"""
        item = self.queue_tts[row]
        
        # 更新Start
        start_item = self.table.item(row, 3)
        if start_item:
            start_item.setText(f"{item['start_time']/1000.0}")
        
        # 更新End
        end_item = self.table.item(row, 4)
        if end_item:
            end_item.setText(f"{item['end_time']/1000.0}")
        
        # 重新计算状态
        duration = (item['end_time'] - item['start_time']) / 1000.0
        dubbing = float(item.get('dubbing_s', 0.0))
        diff = round(dubbing - duration, 3)
        if not msg:
            if dubbing <= 0.0:
                msg = tr('No audio')
            elif diff > 0:
                msg = f'[{dubbing}s]{tr("Exceeded")}{diff}s'
            elif diff < 0:
                msg = f'[{dubbing}s]{tr("Shortened")}{abs(diff)}s'
            else:
                msg = f'{dubbing}s'
        
        msg_item = self.table.item(row, 5)
        if msg_item:
            msg_item.setText(str(msg))
            if dubbing <= 0:
                msg_item.setForeground(QColor("#ff4d4d"))
            elif diff>0:
                msg_item.setForeground(QColor("#ff6600"))
            elif diff<0:
                msg_item.setForeground(QColor("#ffffff"))
            else:
                msg_item.setForeground(QColor("#66ff66"))

    # ===================== Video + Audio playback =====================
    def _ensure_players(self):
        if hasattr(self, '_players_created'):
            return
        self._players_created = True
        self.video_player = QMediaPlayer()
        self.video_player.setVideoOutput(self.video_widget)
        self.video_player.positionChanged.connect(self._on_video_position_changed)
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.mediaStatusChanged.connect(self._on_media_status_changed)


    def _on_video_position_changed(self, position):
        """Stop video when it reaches the subtitle end time."""
        if 0 < self._video_end_ms <= position:
            try:
                self.video_player.pause()
            except Exception:
                pass
            self._video_end_ms = -1
            self.video_status.setText(tr("Playback stopped"))
            self.stop_play_btn.hide()

    def _pause_play(self):
        self.video_player.pause()
        self.audio_player.stop()
        self.audio_player.setSource(QUrl())
        self.stop_play_btn.hide()

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            # 停止播放
            self.audio_player.stop()
            # 清空媒体源，释放文件句柄
            self.audio_player.setSource(QUrl())
            self.stop_play_btn.hide()

    def _play_with_video(self, audio_path, video_start_ms, video_end_ms):
        """Play audio to completion while video plays only the segment."""
        self._ensure_players()
        self.stop_play_btn.show()
        self._pending_start = int(video_start_ms)
        self._pending_end = int(video_end_ms)

        video_needs_load = False
        play_audio=True
        try:
            if self.novoice_mp4 and Path(self.novoice_mp4).exists():
                if not self.video_player.source().toString():
                    self.video_player.setSource(QUrl.fromLocalFile(self.novoice_mp4))
                    video_needs_load = True
            else:
                self.video_status.setText(tr('No silent video frames generated yet'))
            if audio_path and Path(audio_path).exists():
                self.audio_player.setSource(QUrl.fromLocalFile(audio_path))
            else:
                play_audio=False
        except Exception as e:
            self.video_status.setText(f"Load failed: {e}")
            return

        # Stop any previous playback to reset state
        try:
            self.video_player.stop()
            self.audio_player.stop()
        except Exception:
            pass

        if video_needs_load:
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Failed to disconnect")
            self._players_pending = 1
            try:
                self.video_player.mediaStatusChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self.video_player.mediaStatusChanged.connect(self._on_media_ready)
            return

        self._do_play(video_start_ms, video_end_ms,play_audio)

    def _on_media_ready(self, status):
        from PySide6.QtMultimedia import QMediaPlayer
        if status in (QMediaPlayer.MediaStatus.BufferedMedia, QMediaPlayer.MediaStatus.LoadedMedia):
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Failed to disconnect")
            try:
                self.video_player.mediaStatusChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._do_play(self._pending_start, self._pending_end)

    def _do_play(self, video_start_ms, video_end_ms,play_audio=True):
        self._video_end_ms = int(video_end_ms)
        self._audio_playing = True

        self.video_player.setPosition(int(video_start_ms))
        if play_audio:
            self.audio_player.setPosition(0)
        self.video_player.play()
        if play_audio:
            self.audio_player.play()
        self._stack.setCurrentIndex(0)
        if play_audio:
            self.video_status.setText(
            f"\u23F5 {ms_to_time_string(ms=int(video_start_ms))} → {ms_to_time_string(ms=int(video_end_ms))}"
            )
        if not play_audio:
            self.video_player.pause()


    def _stop_playback(self):
        self._video_end_ms = -1
        self._audio_playing = False
        if not hasattr(self, '_players_created'):
            return
        try:
            self.video_player.stop()
            self.audio_player.stop()
        except Exception as e:
            logger.exception(e, exc_info=True)

    def _release_media(self):
        if not hasattr(self, '_players_created'):
            return
        self._stop_playback()
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Failed to disconnect")
        for player in [self.video_player, self.audio_player]:
            for sig in [player.positionChanged, player.mediaStatusChanged]:
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
        try:
            self.video_player.setSource(QUrl())
        except Exception:
            pass
        try:
            self.audio_player.setSource(QUrl())
        except Exception:
            pass
        import gc
        gc.collect()

    def _listen(self, row):
        """试听 — 同时播放对应视频片段"""
        self.stop_countdown()
        item = self.queue_tts[row]
        filename = item['filename']
        
        if not vail_file(filename):
            QMessageBox.information(self, tr("The audio file does not exist"), tr("The audio file does not exist"))
            return
        
        self._play_with_video(
            audio_path=filename,
            video_start_ms=int(item['start_time']),
            video_end_ms=int(item['end_time']),
        )

    def _redub(self, row):
        """重配音"""
        self.stop_countdown()
        
        # 删除旧文件
        try:
            Path(self.queue_tts[row]['filename']).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(e)
        
        # 重置时长
        self.queue_tts[row]['dubbing_s'] = 0.0
        self._refresh_row(row,msg=f'{tr("Re-dubbed")}...')
        
        # 获取当前文本
        text_item = self.table.item(row, 6)
        current_text = text_item.text() if text_item else self.queue_tts[row]['text']
        
        # 准备TTS参数
        tts_dict = self.queue_tts[row].copy()
        tts_dict['text'] = current_text
        if int(tts_dict['tts_type']) not in tts.LOCAL_BUILTIN:
            self.task = ReDubbNolocal(parent=self, idx=row,tts_dict=tts_dict, language=self.language)
            self.task.uito.connect(self._on_redub_finished, type=Qt.ConnectionType.QueuedConnection)
            self.task.start()
        else:
            _REDUBB_QUEUE[tts_dict['filename']]=tts_dict
            _FILENAME_ROW_MAP[tts_dict['filename']]=row
            try:
                atomic_write_json([it for it in _REDUBB_QUEUE.values()],REDUBB_QUEUE_FILE)
            except (OSError,PermissionError):
                _retry=0
                while 1:
                    # 可能遇到权限失败或读写冲突
                    if _retry>5:
                        raise
                    try:
                        Path(REDUBB_QUEUE_FILE).write_text(json.dumps([it for it in _REDUBB_QUEUE.values()]),encoding='utf-8')
                    except (OSError,PermissionError):
                        _retry+=1
                        time.sleep(round(random.random()*3.5,1))
                    else:
                        break
            # 启动后台线程
            if not self.task:
                self.task = ReDubb(parent=self, tts_dict=tts_dict, language=self.language)
                self.task.uito.connect(self._on_redub_finished, type=Qt.ConnectionType.QueuedConnection)
                self.task.start()


    def _on_redub_finished(self, msg):
        """重配音完成回调"""
        if msg.startswith("ok:"):
            try:
                idx = int(msg[3:])
            except (ValueError,TypeError,IndexError):
                return
            if idx < 0 or idx >= len(self.queue_tts):
                return
            item = self.queue_tts[idx]
            
            # 读取新文件时长
            try:
                if Path(item['filename']).exists():
                    item['dubbing_s'] = len(AudioSegment.from_file(item['filename'])) / 1000.0
                else:
                    item['dubbing_s'] = 0.0
            except Exception:
                item['dubbing_s'] = 0.0
            
            # 更新状态显示
            self._refresh_row(idx)

            if self._batch_redub:
                # Edge-TTS often returns NoAudioReceived when requests are back-to-back.
                delay_ms = BATCH_REDUB_DELAY_MS if self._redub_queue else 0
                QTimer.singleShot(delay_ms, self._redub_next)
                return
            
            # 播放新音频 + video segment
            self._play_with_video(
                audio_path=item['filename'],
                video_start_ms=item['start_time'],
                video_end_ms=item['end_time'],
            )
        else:
            # 崩溃停止任务
            self._redub_queue = []
            self._batch_redub = False
            self._set_passage_busy(False)
            Path(REDUBB_STATUS_FILE).write_text('end')
            QMessageBox.information(self, 'Error', msg)

    def _line_overrun(self, row):
        item = self.queue_tts[row]
        slot_s = (item['end_time'] - item['start_time']) / 1000.0
        tts_s = float(item.get('dubbing_s') or 0)
        return tts_s > slot_s and tts_s > 0, slot_s, tts_s

    def _checked_rows(self):
        rows = []
        for row in range(self.table.rowCount()):
            line_item = self.table.item(row, 0)
            if line_item and line_item.checkState() == Qt.Checked:
                rows.append(row)
        return rows

    def _set_passage_busy(self, busy):
        self.btn_select_exceeded.setEnabled(not busy)
        self.btn_rewrite_passage.setEnabled(not busy)
        if not busy:
            self.btn_rewrite_passage.setText(tr("Rewrite this passage"))

    def _select_exceeded_lines(self):
        self.stop_countdown()
        if not getattr(self, 'table', None):
            return
        checked = 0
        self.table.blockSignals(True)
        try:
            for row in range(self.table.rowCount()):
                line_item = self.table.item(row, 0)
                if not line_item:
                    continue
                overrun, _, _ = self._line_overrun(row)
                line_item.setCheckState(Qt.Checked if overrun else Qt.Unchecked)
                if overrun:
                    checked += 1
        finally:
            self.table.blockSignals(False)
        if checked == 0:
            QMessageBox.information(self, tr("Select exceeded lines"), tr("No exceeded lines"))

    def _passage_error(self, msg):
        QMessageBox.information(self, tr("Rewrite this passage"), msg)

    def _current_translate_type(self):
        from videotrans.configure.config import params
        from videotrans.translator._constants import AI_TRANS_CHANNELS, ID_NAME_DICT
        combo = getattr(self.parent, 'translate_type', None)
        if combo is None:
            self._passage_error(tr("Passage rewrite needs an AI chat channel"))
            return None
        translate_type = int(combo.currentIndex())
        if translate_type not in AI_TRANS_CHANNELS:
            self._passage_error(tr("Passage rewrite needs an AI chat channel"))
            return None
        provider = ID_NAME_DICT.get(translate_type)
        if provider and provider.key_name and not params.get(provider.key_name):
            self._passage_error(tr("Please configure the API key of the translation channel first"))
            return None
        return translate_type

    def _redub_next(self):
        if not self._redub_queue:
            self._batch_redub = False
            self._set_passage_busy(False)
            return
        self._redub(self._redub_queue.pop(0))

    def _passage_seeds(self):
        rows = self._checked_rows()
        if rows:
            return rows
        row = self.table.currentRow() if getattr(self, 'table', None) else -1
        return [row] if row >= 0 else []

    def _row_text(self, row):
        text_item = self.table.item(row, 6)
        if text_item:
            return text_item.text().strip()
        return (self.queue_tts[row].get('text') or '').strip()

    def _build_passage_spans(self, seeds):
        times = [
            (int(item['start_time']), int(item['end_time']))
            for item in self.queue_tts
        ]
        spans = []
        for lo, hi in expand_passage_spans(times, seeds):
            rows = list(range(lo, hi + 1))
            dubbed = [self._row_text(row) for row in rows]
            if not any(dubbed):
                continue
            source = [(self.queue_tts[row].get('ref_text') or '').strip() for row in rows]
            tts_total = sum(float(self.queue_tts[row].get('dubbing_s') or 0) for row in rows)
            start_ms, end_ms = times[lo][0], times[hi][1]
            window_s = max((end_ms - start_ms) / 1000.0, 0.1)
            spans.append({
                'rows': rows,
                'dubbed': dubbed,
                'source': source,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'window_s': window_s,
                'target_chars': passage_target_chars(dubbed, tts_total, window_s),
            })
        return spans

    def _rewrite_passage(self):
        self.stop_countdown()
        if not getattr(self, 'table', None):
            return
        seeds = self._passage_seeds()
        if not seeds:
            self._passage_error(tr("Select a subtitle line"))
            return
        translate_type = self._current_translate_type()
        if translate_type is None:
            return
        spans = self._build_passage_spans(seeds)
        if not spans:
            self._passage_error(tr("Select a subtitle line"))
            return
        self._set_passage_busy(True)
        self.btn_rewrite_passage.setText(tr("Rewriting passage..."))
        self._rewrite_thread = PassageRewrite(parent=self, spans=spans, translate_type=translate_type)
        self._rewrite_thread.finished_ok.connect(self._on_passage_ready)
        self._rewrite_thread.failed.connect(self._on_passage_failed)
        self._rewrite_thread.start()

    def _format_passage_preview(self, results):
        blocks = []
        for span in results:
            start = self._ms_to_fmt(span['start_ms'])
            end = self._ms_to_fmt(span['end_ms'])
            old_chars = sum(len(text or '') for text in span['dubbed'])
            new_chars = sum(len(text or '') for text in span['new_lines'])
            blocks.append(
                f"{start} --> {end}  ({span['window_s']:.1f}s)\n"
                f"{tr('Original')} ({old_chars})\n{_format_numbered(span['dubbed'])}\n\n"
                f"{tr('Rewritten')} ({new_chars})\n{_format_numbered(span['new_lines'])}"
            )
        return "\n\n----------\n\n".join(blocks)

    def _confirm_passage(self, results):
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Rewrite this passage"))
        dlg.setMinimumSize(720, 480)
        layout = QVBoxLayout(dlg)
        note = QLabel(tr("The passage start and end stay the same. Apply replaces the lines and re-dubs them."))
        note.setWordWrap(True)
        layout.addWidget(note)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(self._format_passage_preview(results))
        layout.addWidget(view)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton(tr("Cancel"))
        apply = QPushButton(tr("Apply rewrite"))
        cancel.clicked.connect(dlg.reject)
        apply.clicked.connect(dlg.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(apply)
        layout.addLayout(buttons)
        return dlg.exec() == QDialog.DialogCode.Accepted

    def _apply_passage_line(self, row, start_ms, end_ms, text):
        item = self.queue_tts[row]
        item['text'] = text
        item['start_time'] = int(start_ms)
        item['end_time'] = int(end_ms)
        item['startraw'] = self._ms_to_fmt(start_ms)
        item['endraw'] = self._ms_to_fmt(end_ms)
        if 'start_time_source' in item:
            item['start_time_source'] = int(start_ms)
        if 'end_time_source' in item:
            item['end_time_source'] = int(end_ms)
        item['_duration'] = (end_ms - start_ms) / 1000.0
        axis = self.table.item(row, 1)
        if axis:
            axis.setText(
                f"{item['startraw']}->{item['endraw']} ({item['_duration']:.3f}s)"
            )
        start_item = self.table.item(row, 3)
        if start_item:
            start_item.setText(f"{start_ms / 1000.0}")
        end_item = self.table.item(row, 4)
        if end_item:
            end_item.setText(f"{end_ms / 1000.0}")
        text_item = self.table.item(row, 6)
        if text_item:
            text_item.setText(text)
        self._refresh_row(row)

    def _on_passage_ready(self, results):
        if not results or not self._confirm_passage(results):
            self._set_passage_busy(False)
            return
        rows = []
        self.table.blockSignals(True)
        try:
            for span in results:
                times = split_passage_times(span['start_ms'], span['end_ms'], span['new_lines'])
                for row, (start_ms, end_ms), text in zip(span['rows'], times, span['new_lines']):
                    self._apply_passage_line(row, start_ms, end_ms, text)
                    rows.append(row)
        finally:
            self.table.blockSignals(False)
        self._redub_queue = rows
        self._batch_redub = True
        self._redub_next()

    def _on_passage_failed(self, msg):
        self._set_passage_busy(False)
        if msg == 'channel':
            msg = tr("Passage rewrite needs an AI chat channel")
        elif msg == 'empty':
            msg = tr("Passage rewrite failed")
        self._passage_error(msg)

    def _show_error(self, msg):
        """显示错误"""
        if hasattr(self, 'prompt_label') and self.prompt_label:
            original_text = self.prompt_label.text()
            original_style = self.prompt_label.styleSheet()
            
            self.prompt_label.setText(f"Error: {msg}")
            self.prompt_label.setStyleSheet("color:red;")
            
            def restore():
                if hasattr(self, 'prompt_label') and self.prompt_label:
                    self.prompt_label.setText(str(original_text))
                    self.prompt_label.setStyleSheet(original_style)
            QTimer.singleShot(2000, restore)

    def cancel_and_close(self):
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        Path(REDUBB_STATUS_FILE).write_text("end")
        self._release_media()
        self.reject()

    def update_countdown(self):
        self.count_down -= 1
        if hasattr(self, 'stop_button') and self.stop_button:
            self.stop_button.setText(f"{tr('Click here to stop the countdown')}({self.count_down})")

        if self.count_down <= 0:
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
            self.save_and_close()

    def stop_countdown(self):
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
            self.timer = None

        if hasattr(self, 'prompt_label') and self.prompt_label:
            try:
                self.prompt_label.deleteLater()
            except RuntimeError:
                pass
            self.prompt_label = None

        if hasattr(self, 'stop_button') and self.stop_button:
            try:
                self.stop_button.deleteLater()
            except RuntimeError:
                pass
            self.stop_button = None

    def save_and_close(self):
        self.stop_countdown()
        Path(REDUBB_STATUS_FILE).write_text("end")
        self._release_media()
        self.save_button.setDisabled(True)
        app_cfg.onlyone_video_autorate=self.video_autorate.isChecked()
        app_cfg.onlyone_voice_autorate=self.voice_autorate.isChecked()
        app_cfg.onlyone_remove_silent_mid=self.remove_silent_mid.isChecked()
        app_cfg.onlyone_align_sub_audio=self.align_sub_audio.isChecked()
        srt_str_list=[]
        for i, item in enumerate(self.queue_tts):
            start_time = self.table.item(i, 3)
            if not start_time:continue
            end_time = self.table.item(i, 4)
            start_raw=ms_to_time_string(ms=int(float(start_time.text().strip())*1000))
            end_raw=ms_to_time_string(ms=int(float(end_time.text().strip())*1000))
            # 不修改文本，以便可以单独使用 各种配音渠道支持的控制符号进行声音微调
            text_item = self.table.item(i, 6)
            text = text_item.text().strip() if text_item else item['text'].strip()
            # 删除空文本对应的音频文件
            if not text:
                Path(item['filename']).unlink(missing_ok=True)
            else:
                srt_str_list.append(f'{len(srt_str_list)+1}\n{start_raw} --> {end_raw}\n{text}')

        try:
            Path(app_cfg.onlyone_target_sub).write_text("\n\n".join(srt_str_list), encoding="utf-8")
            Path(f'{self.cache_folder}/queue_tts.json').write_text( json.dumps(self.queue_tts, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.error(f"Save subtitle failed: {e}")
            QMessageBox.critical(self, "Error", f"Save failed: {e}")
            self.save_button.setDisabled(False)
            return
        self.accept()