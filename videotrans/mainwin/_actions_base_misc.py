import platform
import time
import tempfile
from pathlib import Path
from videotrans.configure.config import tr, app_cfg, params, settings
from videotrans.configure import constants
from videotrans.configure.constants import LISTEN_TEXT
from videotrans.util.help_misc import open_url, show_error


class WinActionBaseMiscMixin:

    def persist_ui_params(self):
        """Save current Advanced/main options so they survive restart without Start."""
        m = self.main
        data = {
            'clear_cache': m.clear_cache.isChecked(),
            'only_out_mp4': m.only_out_mp4.isChecked(),
            'fix_punc': m.replace_punc.currentIndex(),
            'tts_type': m.tts_type.currentIndex(),
            'voice_role': m.voice_role.currentText(),
            'recogn_type': m.recogn_type.currentIndex(),
            'model_name': m.model_name.currentText(),
            'remove_noise': m.remove_noise.isChecked(),
            'subtitle_type': m.subtitle_type.currentIndex(),
            'voice_autorate': m.voice_autorate.isChecked(),
            'video_autorate': m.video_autorate.isChecked(),
            'is_separate': m.is_separate.isChecked(),
            'embed_bgm': m.embed_bgm.isChecked(),
            'background_music': m.back_audio.text().strip(),
            'enable_diariz': m.enable_diariz.isChecked(),
            'recogn2pass': m.recogn2pass.isChecked(),
            'nums_diariz': m.nums_diariz.currentIndex(),
            'rephrase': m.rephrase.isChecked(),
            'is_cuda': m.enable_cuda.isChecked(),
            'output_srt': m.output_srt.currentIndex(),
            'loop_backaudio': m.is_loop_bgm.currentIndex(),
            'select_file_type': m.select_file_type.isChecked(),
            'copysrt_rawvideo': m.copysrt_rawvideo.isChecked(),
            'source_language': m.source_language.currentText(),
            'target_language': m.target_language.currentText(),
            'translate_type': m.translate_type.currentIndex(),
        }
        try:
            vr = int(m.voice_rate.value())
            data['voice_rate'] = f'+{vr}%' if vr >= 0 else f'{vr}%'
        except (TypeError, ValueError):
            data['voice_rate'] = '+0%'
        try:
            vol = int(m.volume_rate.value())
            data['volume'] = f'+{vol}%' if vol >= 0 else f'{vol}%'
        except (TypeError, ValueError):
            data['volume'] = '+0%'
        try:
            pitch = int(m.pitch_rate.value())
            data['pitch'] = f'+{pitch}Hz' if pitch >= 0 else f'{pitch}Hz'
        except (TypeError, ValueError):
            data['pitch'] = '+0Hz'
        data['keep_original_audio'] = bool(
            hasattr(m, 'keep_original_audio') and m.keep_original_audio.isChecked()
        )
        try:
            data['original_audio_volume'] = float(m.original_volume.text()) if hasattr(m, 'original_volume') else 0.3
        except (TypeError, ValueError):
            data['original_audio_volume'] = 0.3
        data['cover_source_hardsub'] = bool(
            hasattr(m, 'cover_source_hardsub') and m.cover_source_hardsub.isChecked()
        )
        try:
            from videotrans.task._stage_assemble import _normalize_cover_hardsub_pct
            raw = m.cover_hardsub_pct.text() if hasattr(m, 'cover_hardsub_pct') else 0.14
            data['cover_source_hardsub_pct'] = _normalize_cover_hardsub_pct(raw)
        except (TypeError, ValueError):
            data['cover_source_hardsub_pct'] = 0.14
        data['embed_logo'] = bool(hasattr(m, 'embed_logo') and m.embed_logo.isChecked())
        data['logo_path'] = m.logo_path.text().strip() if hasattr(m, 'logo_path') else ''
        data['logo_pos'] = m.logo_pos.currentIndex() if hasattr(m, 'logo_pos') else 0
        data['logo_size'] = (m.logo_size.text().strip() if hasattr(m, 'logo_size') else '') or '240x73'
        try:
            data['logo_margin'] = int(m.logo_margin.text().strip()) if hasattr(m, 'logo_margin') else 10
        except (TypeError, ValueError):
            data['logo_margin'] = 10
        data['floating_channel'] = bool(
            hasattr(m, 'floating_channel') and m.floating_channel.isChecked()
        )
        data['channel_name'] = (
            m.channel_name.currentText().strip() if hasattr(m, 'channel_name') else 'Gamma Arabicsub'
        ) or 'Gamma Arabicsub'
        if not data['voice_autorate'] and not data['video_autorate']:
            data['remove_silent_mid'] = m.remove_silent_mid.isChecked()
            data['align_sub_audio'] = m.align_sub_audio.isChecked()
        try:
            data['backaudio_volume'] = float(m.bgmvolume.text())
        except (TypeError, ValueError):
            data['backaudio_volume'] = 0.8
        if hasattr(m, 'narration_mode'):
            mode = m.narration_mode.currentData() or 'none'
            if mode in ('line_third', 'line_first'):
                mode = 'line'
            if mode not in ('none', 'third', 'first', 'line'):
                mode = 'none'
            data['narration_mode'] = mode
            data['third_person_narration'] = mode != 'none'
            data['narration_style'] = 'first' if mode == 'first' else 'third'
        params.getset_params(data)
        try:
            settings['aisendsrt'] = m.aisendsrt.isChecked()
            settings.save()
        except Exception:
            pass

    def check_cuda(self, state):
        res = state
        if platform.system() == 'Darwin':
            self.cfg['is_cuda'] = False
            return
        if state and app_cfg.NVIDIA_GPU_NUMS == 0:
            show_error(tr('nocuda'))
            self.main.enable_cuda.setChecked(False)
            self.main.enable_cuda.setDisabled(True)
            res = False
        self.cfg['is_cuda'] = res

    def check_voice_autorate(self, state):
        if state:
            self.main.remove_silent_mid.setVisible(False)
            self.main.align_sub_audio.setVisible(False)
        elif not self.main.video_autorate.isChecked():
            self.main.remove_silent_mid.setVisible(True)
            self.main.align_sub_audio.setVisible(True)

    def check_video_autorate(self, state):
        if state:
            self.main.remove_silent_mid.setVisible(False)
            self.main.align_sub_audio.setVisible(False)
        elif not self.main.voice_autorate.isChecked():
            self.main.remove_silent_mid.setVisible(True)
            self.main.align_sub_audio.setVisible(True)

    def cuda_isok(self):
        if not self.main.enable_cuda.isChecked() or platform.system() == 'Darwin':
            self.cfg['is_cuda'] = False
            return True

        if app_cfg.NVIDIA_GPU_NUMS == 0:
            self.cfg['is_cuda'] = False
            show_error(tr("nocuda"))
            return False
        self.cfg['is_cuda'] = True
        return True

    def listen_voice_fun(self):
        from videotrans import translator
        from videotrans.util.ListenVoice import ListenVoice
        lang = translator.get_code(show_text=self.main.target_language.currentText())
        if not lang:
            return show_error(
                tr("Please select the target language first"))
        
        text = LISTEN_TEXT.get(lang.split('-')[0])
        if not text:
            return show_error(tr("The voice is not support listen")+f' {lang=}')
        role = self.main.voice_role.currentText()
        if not role or role == 'No':
            return show_error(tr('mustberole'))

        voice_dir = tempfile.gettempdir() + '/pyvideotrans'
        if not Path(voice_dir).exists():
            Path(voice_dir).mkdir(parents=True, exist_ok=True)
        rate = int(self.main.voice_rate.value())
        if rate >= 0:
            rate = f"+{rate}%"
        else:
            rate = f"{rate}%"

        volume = int(self.main.volume_rate.value())
        volume = f'+{volume}%' if volume >= 0 else f'{volume}%'
        pitch = int(self.main.pitch_rate.value())
        pitch = f'+{pitch}Hz' if pitch >= 0 else f'{volume}Hz'
        tts_type=self.main.tts_type.currentIndex()
        voice_file = f"{voice_dir}/{tts_type}-{time.time()}.wav"
        obj = {
            "text": text,
            "rate": rate,
            "role": role,
            "filename": voice_file,
            "tts_type": tts_type,
            "language": lang,
            "volume": volume,
            "pitch": pitch,
        }
        if role == 'clone':
            show_error(
                tr("The original sound clone cannot be auditioned"))
            return
        raw_text = self.main.listen_btn.text()
        def feed(d):
            self.main.listen_btn.setDisabled(False)
            self.main.listen_btn.setText(raw_text)
            if d != "ok":
                show_error(d)

        self.main.listen_btn.setDisabled(True)
        self.main.listen_btn.setText('load...')
        wk = ListenVoice(parent=self.main, queue_tts=[obj], language=lang, tts_type=obj['tts_type'])
        wk.uito.connect(feed)
        wk.start()

    def show_listen_btn(self, role):
        voice_role = self.main.voice_role.currentText()
        from videotrans import tts
        _tip = tts.clone_tips(voice_role)
        if _tip:
            self.main.show_tips.setText(_tip)
        if role == 'No' or voice_role == 'clone':
            self.main.listen_btn.hide()
            return
        if self.main.app_mode in ['biaozhun']:
            self.main.listen_btn.show()
            self.main.listen_btn.setDisabled(False)

    def check_name(self):
        if self.main.app_mode != 'tiqu':
            for it in self.queue_mp4:
                if Path(it).suffix.lower() in constants.AUDIO_EXITS:
                    self.main.app_mode = 'tiqu'
                    break
        return True

    @staticmethod
    def open_url(title):
        open_url(title)
