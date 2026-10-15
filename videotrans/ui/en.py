import platform
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import QSizePolicy, QApplication, QPlainTextEdit

from videotrans.configure import constants as ui_constants
from videotrans.configure.config import tr, settings
from videotrans.ui._setup_menus import _setup_actions_and_menus
from videotrans.ui._setup_rows import (
    _create_file_row,
    _create_asr_row,
    _create_translation_row,
    _create_tts_row,
    _create_alignment_row,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        _channel_label = 80
        _channel_com = 200

        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.splitter.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.splitter.setLineWidth(1)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.splitter.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.splitter.setMinimumWidth(600)

        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 3, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_3.setSpacing(8)

        self.verticalLayout_3.addLayout(_create_file_row(self, self.layoutWidget))
        self.verticalLayout_3.addLayout(_create_asr_row(self, self.layoutWidget))
        self.verticalLayout_3.addLayout(_create_translation_row(self, self.layoutWidget))
        self.verticalLayout_3.addLayout(_create_tts_row(self, self.layoutWidget))
        self.verticalLayout_3.addLayout(_create_alignment_row(self, self.layoutWidget))

        self.bgm_layout = QtWidgets.QHBoxLayout()
        self.bgm_layout.setObjectName("bgm_layout")

        self.is_separate = QtWidgets.QCheckBox(self.layoutWidget)
        self.is_separate.setMinimumSize(QtCore.QSize(0, 30))
        self.is_separate.setObjectName("is_separate")
        self.is_separate.setVisible(False)
        self.embed_bgm = QtWidgets.QCheckBox(self.layoutWidget)
        self.embed_bgm.setMinimumSize(QtCore.QSize(0, 30))
        self.embed_bgm.setObjectName("embed_bgm")
        self.embed_bgm.setChecked(True)
        self.embed_bgm.setVisible(False)

        self.addbackbtn = QtWidgets.QPushButton(self.layoutWidget)
        self.addbackbtn.setObjectName("addbackbtn")
        self.addbackbtn.setVisible(False)
        self.addbackbtn.setCursor(Qt.PointingHandCursor)

        self.back_audio = QtWidgets.QLineEdit(self.layoutWidget)
        self.back_audio.setObjectName("back_audio")
        self.back_audio.setVisible(False)

        self.is_loop_bgm = QtWidgets.QComboBox(self.layoutWidget)
        self.is_loop_bgm.addItems([tr('The bgm briefly lengthens'), tr('loop the BGM')])
        self.is_loop_bgm.setVisible(False)

        self.bgmvolume_label = QtWidgets.QLabel()
        self.bgmvolume_label.setText(tr("Volume BGM"))
        self.bgmvolume_label.setVisible(False)
        self.bgmvolume = QtWidgets.QLineEdit()
        self.bgmvolume.setText('0.8')
        self.bgmvolume.setMaximumWidth(60)
        self.bgmvolume.setVisible(False)
        self.bgmvolume.setToolTip(
            tr("BGM volume is a multiple of the original volume, greater than 1 increases, less than decreases"))

        self.keep_original_audio = QtWidgets.QCheckBox(self.layoutWidget)
        self.keep_original_audio.setMinimumSize(QtCore.QSize(0, 30))
        self.keep_original_audio.setObjectName("keep_original_audio")
        self.keep_original_audio.setVisible(False)

        self.original_volume_label = QtWidgets.QLabel()
        self.original_volume_label.setText(tr("Original volume"))
        self.original_volume_label.setVisible(False)
        self.original_volume = QtWidgets.QLineEdit()
        self.original_volume.setText('0.3')
        self.original_volume.setMaximumWidth(60)
        self.original_volume.setVisible(False)
        self.original_volume.setToolTip(
            tr("Original audio volume under the dubbing; less than 1 lowers it, e.g. 0.3"))

        self.set_ass = QtWidgets.QPushButton()
        self.set_ass.setText(tr('Modify hard subtitle style'))
        self.set_ass.setCursor(Qt.PointingHandCursor)
        self.set_ass.setVisible(False)

        self.cover_source_hardsub = QtWidgets.QCheckBox(self.layoutWidget)
        self.cover_source_hardsub.setMinimumSize(QtCore.QSize(0, 30))
        self.cover_source_hardsub.setObjectName("cover_source_hardsub")
        self.cover_source_hardsub.setVisible(False)

        self.cover_hardsub_pct_label = QtWidgets.QLabel()
        self.cover_hardsub_pct_label.setText(tr("Cover %"))
        self.cover_hardsub_pct_label.setVisible(False)
        self.cover_hardsub_pct = QtWidgets.QLineEdit()
        self.cover_hardsub_pct.setText('14')
        self.cover_hardsub_pct.setMaximumWidth(40)
        self.cover_hardsub_pct.setVisible(False)
        self.cover_hardsub_pct.setToolTip(
            tr("Bottom cover height as percent of frame, e.g. 11 = bottom 11% (range 5–35)"))
        # UI always shows integer percent; parser also accepts legacy 0.11 / 11%.

        self.embed_logo = QtWidgets.QCheckBox(self.layoutWidget)
        self.embed_logo.setMinimumSize(QtCore.QSize(0, 30))
        self.embed_logo.setObjectName("embed_logo")
        self.embed_logo.setVisible(False)

        self.logo_path = QtWidgets.QLineEdit()
        self.logo_path.setPlaceholderText(tr("Select a watermark image"))
        self.logo_path.setMaximumWidth(160)
        self.logo_path.setVisible(False)

        self.logobtn = QtWidgets.QPushButton()
        self.logobtn.setText(tr("Select an Image"))
        self.logobtn.setCursor(Qt.PointingHandCursor)
        self.logobtn.setVisible(False)

        self.logo_pos = QtWidgets.QComboBox()
        self.logo_pos.addItems([
            tr("Upper left"),
            tr("Upper right"),
            tr("Bottom right"),
            tr("Bottom left"),
            tr("Center"),
        ])
        self.logo_pos.setToolTip(tr("Watermark Location"))
        self.logo_pos.setVisible(False)

        self.logo_size = QtWidgets.QLineEdit()
        self.logo_size.setText(ui_constants.DEFAULT_CHANNEL_LOGO_SIZE)
        self.logo_size.setMaximumWidth(70)
        self.logo_size.setToolTip(tr("Watermark width x height"))
        self.logo_size.setVisible(False)

        _logo_file = Path(__file__).resolve().parents[2] / ui_constants.DEFAULT_CHANNEL_LOGO
        if _logo_file.is_file():
            self.logo_path.setText(_logo_file.as_posix())

        self.logo_margin = QtWidgets.QLineEdit()
        self.logo_margin.setText('10')
        self.logo_margin.setMaximumWidth(50)
        self.logo_margin.setToolTip(tr("Distance of watermark from left or right side"))
        self.logo_margin.setVisible(False)

        self.floating_channel = QtWidgets.QCheckBox(self.layoutWidget)
        self.floating_channel.setMinimumSize(QtCore.QSize(0, 30))
        self.floating_channel.setObjectName("floating_channel")
        self.floating_channel.setVisible(False)

        self.channel_name = QtWidgets.QComboBox()
        self.channel_name.addItems(list(ui_constants.CHANNEL_NAMES))
        self.channel_name.setMinimumWidth(140)
        self.channel_name.setToolTip(tr("Channel name for the floating watermark"))
        self.channel_name.setVisible(False)

        # Advanced rows (split so controls are not cramped on one line)
        self.bgm_layout.addWidget(self.is_separate)
        self.bgm_layout.addWidget(self.embed_bgm)
        self.bgm_layout.addWidget(self.keep_original_audio)
        self.bgm_layout.addWidget(self.original_volume_label)
        self.bgm_layout.addWidget(self.original_volume)
        self.bgm_layout.addWidget(self.is_loop_bgm)
        self.bgm_layout.addWidget(self.bgmvolume_label)
        self.bgm_layout.addWidget(self.bgmvolume)
        self.bgm_layout.addStretch()

        self.cover_layout = QtWidgets.QHBoxLayout()
        self.cover_layout.setObjectName("cover_layout")
        self.cover_layout.addWidget(self.addbackbtn)
        self.cover_layout.addWidget(self.back_audio)
        self.cover_layout.addWidget(self.set_ass)
        self.cover_layout.addWidget(self.cover_source_hardsub)
        self.cover_layout.addWidget(self.cover_hardsub_pct_label)
        self.cover_layout.addWidget(self.cover_hardsub_pct)
        self.cover_layout.addStretch()

        self.brand_layout = QtWidgets.QHBoxLayout()
        self.brand_layout.setObjectName("brand_layout")
        self.brand_layout.addWidget(self.embed_logo)
        self.brand_layout.addWidget(self.logo_path)
        self.brand_layout.addWidget(self.logobtn)
        self.brand_layout.addWidget(self.logo_pos)
        self.brand_layout.addWidget(self.logo_size)
        self.brand_layout.addWidget(self.logo_margin)
        self.brand_layout.addStretch()

        self.float_layout = QtWidgets.QHBoxLayout()
        self.float_layout.setObjectName("float_layout")
        self.float_layout.addWidget(self.floating_channel)
        self.float_layout.addWidget(self.channel_name)
        self.float_layout.addStretch()

        self.enable_diariz = QtWidgets.QCheckBox()
        self.enable_diariz.setToolTip(tr("Speaker classification language"))
        self.enable_diariz.setText(tr("Speaker classification"))

        self.fix_punc = QtWidgets.QComboBox()
        self.fix_punc.setToolTip(tr("Restoring punctuation marks when Chinese & English"))
        self.fix_punc.addItems([
            tr("Default punctuation"),
            tr("Restore punctuation"),
            tr("Delete punctuation")
        ])

        self.nums_diariz = QtWidgets.QComboBox()
        self.nums_diariz.setToolTip(tr("Specifying the number of speakers"))
        self.nums_diariz.addItems([tr("No limit"), "2", "3", "4", "5", "6", "7", "8", "9", "10"])

        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setObjectName("label_6")
        self.label_6.setVisible(False)

        self.voice_rate = QtWidgets.QSpinBox(self.layoutWidget)
        self.voice_rate.setMinimum(-50)
        self.voice_rate.setMaximum(50)
        self.voice_rate.setMinimumWidth(60)
        self.voice_rate.setObjectName("voice_rate")
        self.voice_rate.setVisible(False)

        self.volume_label = QtWidgets.QLabel(self.layoutWidget)
        self.volume_label.setText(tr("Volume+"))
        self.volume_label.setVisible(False)
        self.volume_rate = QtWidgets.QSpinBox(self.layoutWidget)
        self.volume_rate.setMinimum(-95)
        self.volume_rate.setToolTip(tr("Percentage of volume adjustment"))
        self.volume_rate.setMaximum(200)
        self.volume_rate.setMinimumWidth(60)
        self.volume_rate.setObjectName("volume_rate")
        self.volume_rate.setVisible(False)

        self.pitch_label = QtWidgets.QLabel(self.layoutWidget)
        self.pitch_label.setText(tr("Pitch+"))
        self.pitch_label.setVisible(False)
        self.pitch_rate = QtWidgets.QSpinBox(self.layoutWidget)
        self.pitch_rate.setVisible(False)
        self.pitch_rate.setMinimum(-100)
        self.pitch_rate.setMaximum(100)
        self.pitch_rate.setMinimumWidth(60)
        self.pitch_rate.setObjectName("pitch_rate")

        self.dubb_thread_layout = QtWidgets.QHBoxLayout()
        self.dubb_thread_layout.addWidget(self.label_6)
        self.dubb_thread_layout.addWidget(self.voice_rate)
        self.dubb_thread_layout.addWidget(self.volume_label)
        self.dubb_thread_layout.addWidget(self.volume_rate)
        self.dubb_thread_layout.addWidget(self.pitch_label)
        self.dubb_thread_layout.addWidget(self.pitch_rate)
        self.dubb_thread_layout.addStretch()

        self.dubb_thread_layout.addWidget(self.fix_punc)
        self.dubb_thread_layout.addWidget(self.remove_noise)
        self.dubb_thread_layout.addWidget(self.enable_diariz)
        self.dubb_thread_layout.addWidget(self.nums_diariz)

        self.adv_layout_outer = QtWidgets.QVBoxLayout()
        self.advcontainer = QtWidgets.QWidget()
        self.advcontainer.setLayout(self.adv_layout_outer)
        self.advcontainer.setObjectName("advContainer")
        self.advcontainer.setStyleSheet("""#advContainer {
        border: 1px solid #455364;
        padding: 0; 
    }""")

        self.adv_layout_outer.addLayout(self.dubb_thread_layout)
        self.adv_layout_outer.addLayout(self.bgm_layout)
        self.adv_layout_outer.addLayout(self.cover_layout)
        self.adv_layout_outer.addLayout(self.brand_layout)
        self.adv_layout_outer.addLayout(self.float_layout)
        self.verticalLayout_3.addWidget(self.advcontainer)
        self.advcontainer.setVisible(False)

        self.show_tips = QtWidgets.QLabel(self.layoutWidget)
        self.show_tips.setWordWrap(True)
        self.show_tips.setText(
            tr("Customize each configuration to batch video translation. When selecting a single video, you can pause to edit subtitles during processing."))
        self.show_tips.setStyleSheet(
            """background-color:transparent;border-color:transparent;color:#aaaaaa;text-align:left""")
        self.show_tips.setObjectName("show_tips")
        self.output_dir = QtWidgets.QLabel(self.layoutWidget)
        self.output_dir.setWordWrap(True)
        self.output_dir.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_dir.setStyleSheet(
            """background-color:transparent;border-color:transparent;color:#929ca7;text-align:center""")
        self.output_dir.setObjectName("output_dir")
        self.verticalLayout_3.addWidget(self.show_tips)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.enable_cuda = QtWidgets.QCheckBox(self.layoutWidget)
        self.enable_cuda.setMinimumSize(QtCore.QSize(50, 20))
        self.enable_cuda.setObjectName("enable_cuda")
        self.enable_cuda.setToolTip(tr('cudatips'))

        self.startbtn = QtWidgets.QPushButton(self.layoutWidget)
        self.startbtn.setMinimumSize(QtCore.QSize(160, 40))
        self.startbtn.setObjectName("startbtn")

        self.retrybtn = QtWidgets.QPushButton(self.layoutWidget)
        self.retrybtn.setObjectName("retrybtn")
        self.retrybtn.setVisible(False)
        self.retrybtn.setText(tr('Retry failed'))

        self.horizontalLayout_3.addStretch(1)
        self.horizontalLayout_3.addWidget(self.enable_cuda)
        self.horizontalLayout_3.addWidget(self.startbtn)
        self.horizontalLayout_3.addWidget(self.retrybtn)
        self.horizontalLayout_3.addStretch(1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addWidget(self.output_dir)

        self.scroll_area = QtWidgets.QScrollArea(self.layoutWidget)
        self.scroll_area.setStyleSheet("border-color:#32414B")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scroll_area")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()

        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scroll_area.setWidget(self.scrollAreaWidgetContents)

        self.scroll_area.setWidgetResizable(True)
        viewport = QtWidgets.QWidget(self.scroll_area)
        self.scroll_area.setWidget(viewport)
        self.processlayout = QtWidgets.QVBoxLayout(viewport)
        self.processlayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.verticalLayout_3.addWidget(self.scroll_area)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.subtitle_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.subtitle_layout.setContentsMargins(3, 0, 0, 0)
        self.subtitle_layout.setObjectName("subtitle_layout")

        self.subtitle_area = QPlainTextEdit()
        self.subtitle_area.setReadOnly(True)
        self.subtitle_area.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.subtitle_area.setObjectName("subtitle_area")
        self.subtitle_area.setPlaceholderText(
            f"\n{tr('subtitle_tips')}\n\n{tr('meitiaozimugeshi')}")

        self.subtitle_layout.addWidget(self.subtitle_area,stretch=1)


        import_srt_layout=QtWidgets.QHBoxLayout()
        self.import_subtitle = QtWidgets.QPushButton()
        self.import_subtitle.setText(tr('Import spoken SRT'))
        self.import_subtitle.setCursor(Qt.PointingHandCursor)
        self.import_subtitle_target = QtWidgets.QPushButton()
        self.import_subtitle_target.setText(tr('Import target language SRT'))
        self.import_subtitle_target.setCursor(Qt.PointingHandCursor)
        import_srt_layout.addWidget(self.import_subtitle)
        import_srt_layout.addWidget(self.import_subtitle_target)
        
        import_srt_tips=QtWidgets.QLabel(tr('This only works for translating a single video'))
        import_srt_tips.setWordWrap(True)
        import_srt_tips.setStyleSheet("font-size:12px;color:#848f9c")
        import_srt_tips.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        
        self.subtitle_layout.addWidget(import_srt_tips,stretch=0)
        self.subtitle_layout.addLayout(import_srt_layout)


        self.horizontalLayout_7.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        _setup_actions_and_menus(self, MainWindow)

        if platform.system() == 'Darwin':
            self.enable_cuda.setChecked(False)
            self.enable_cuda.hide()

        self._set_Ui_Text()

    def _set_Ui_Text(self):
        self.statusLabel = QtWidgets.QPushButton(tr("Open Documents"))
        self.statusLabel.setStyleSheet("""color:#ffff66""")
        self.statusBar.addWidget(self.statusLabel)

        self.rightbottom = QtWidgets.QPushButton(tr('juanzhu'))

        self.container = QtWidgets.QToolBar()
        self.container.addWidget(self.rightbottom)
        self.restart_btn = QtWidgets.QPushButton(tr("Restart"))
        self.container.addWidget(self.restart_btn)
        self.statusBar.addPermanentWidget(self.container)

        self.rightbottom.setStyleSheet("""color:#ffff66""")
        self.restart_btn.setStyleSheet("""color:#ffffbb""")
        self.restart_btn.setToolTip(
            tr("Click to end all tasks immediately and restart"))

        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.btn_get_video.setToolTip(
            tr("Multiple MP4 videos can be selected and automatically queued for processing"))
        self.btn_get_video.setText(tr("Select audio & video"))

        self.btn_save_dir.setToolTip(tr("Select where to save the processed output resources"))
        self.btn_save_dir.setText(tr("Save to.."))

        self.label_9.setText(tr("Translate channel"))
        self.translate_type.setToolTip(
            tr("Translation channels used in translating subtitle text"))
        self.label.setText(tr("Proxy"))
        self.label.setToolTip(
            tr("Click to view the tutorial for filling in the network proxy"))
        self.label.setCursor(Qt.PointingHandCursor)

        self.proxy.setPlaceholderText(tr("Failed to access Google services. Please set up the proxy correctly"))
        self.proxy.setToolTip(tr("Failed to access Google services. Please set up the proxy correctly"))
        self.model_name.setToolTip(tr("Whisper can select models for related channels"))
        self.listen_btn.setToolTip(tr("shuoming01"))
        self.listen_btn.setText(tr("Trial dubbing"))
        self.label_2.setText(tr("Speech language"))
        self.source_language.setToolTip(tr("The language used for the original video pronunciation"))
        self.label_3.setText(tr("Target lang"))
        self.target_language.setToolTip(tr("What language do you want to translate into"))
        self.tts_text.setText(tr("Dubbing channel"))
        self.label_4.setText(tr("Dubbing role") + " ")
        self.voice_role.setToolTip(tr("No is not dubbing"))

        self.subtitle_type.setToolTip(tr("shuoming02"))

        self.label_6.setText(tr("Dubbing speed"))
        self.voice_rate.setToolTip(tr("Overall acceleration or deceleration of voice over playback"))
        self.voice_autorate.setText(tr("Dubbing acceler"))
        self.voice_autorate.setToolTip(tr("shuoming03"))
        self.video_autorate.setText(tr("Slow video"))
        self.video_autorate.setToolTip(tr("Video Auto Slow"))

        self.remove_silent_mid.setText(tr("Del inline mute?"))
        self.remove_silent_mid.setToolTip(
            tr("Selecting this option will delete the silent intervals between subtitles"))

        self.align_sub_audio.setText(tr("Align subtitles and audio"))
        self.align_sub_audio.setToolTip(tr("If selected, it will force the subtitles and audio to align."))
        self.align_sub_audio.setChecked(True)

        self.enable_cuda.setText(tr("Enable CUDA?"))
        self.embed_bgm.setText(tr("embed_bgm"))
        self.is_separate.setText(tr("Retain original background sound"))
        self.is_separate.setToolTip(tr(
            "If selected, separate human voice and background sound, and finally output video will embed background sound"))
        self.keep_original_audio.setText(tr("Keep original audio"))
        self.keep_original_audio.setToolTip(tr(
            "Lower original speech and overlay dubbing on top. Can be used with Separate Vocal/BGM + Re-embed BGM: original vocal is ducked, then clean BGM is re-embedded."))
        self.cover_source_hardsub.setText(tr("Cover source hardsub"))
        self.cover_source_hardsub.setToolTip(tr(
            "Black out the bottom band of the source video before burning subtitles, so original burned-in hardsubs do not show under the new ones."))
        self.embed_logo.setText(tr("Embed logo"))
        self.embed_logo.setToolTip(tr(
            "Overlay a PNG/JPG logo on the final translated video (same as Tools → watermark)."))
        self.floating_channel.setText(tr("Floating channel name"))
        self.floating_channel.setToolTip(tr(
            "Slow faint channel name that drifts around the video so it barely distracts viewers."))
        self.startbtn.setText(tr("Start"))
        self.addbackbtn.setText(tr("Add background audio"))
        self.addbackbtn.setToolTip(
            tr("Add background audio for output video"))
        self.back_audio.setPlaceholderText(tr("back_audio_place"))
        self.back_audio.setToolTip(tr("back_audio_place"))

        self.menu_Key.setTitle(tr("&Setting"))
        self.menu_TTS.setTitle(tr("&TTSsetting"))
        self.menu_RECOGN.setTitle(tr("&RECOGNsetting"))
        self.menu.setTitle(tr("&Tools"))
        self.menu_H.setTitle(tr("&Help"))
        self.toolBar.setWindowTitle("toolBar")

        self.action_biaozhun.setToolTip(
            tr("Batch audio or video translation with all configuration options customizable on demand"))
        self.action_tiquzimu.setToolTip(tr("Batch recognize speech in video as srt subtitles"))
        self.fn_recogn.setToolTip(tr("stt-windows-title"))
        self.fn_peiyin.setToolTip(tr("tts-windows-title"))
        self.fn_fanyisrt.setToolTip(tr("transsrt-windows-title"))
        self.fn_peiyinrole.setToolTip(tr("multi-windows-title"))
        self.fn_vas.setToolTip(tr("hebing-windows-title"))
