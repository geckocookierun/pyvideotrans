import math
import shutil
import time
import uuid
from pathlib import Path

from videotrans.configure.config import tr, logger
from videotrans.util.help_ffmpeg import get_audio_time, runffmpeg, create_concat_txt, concat_multi_audio, \
    change_speed_rubberband
from videotrans.util.help_misc import vail_file

# Match SpeedRate / Edge-TTS convert_to_wav so amix never mixes mismatched rates.
_MIX_SAMPLE_RATE = "48000"


class AudioMixin:

    def _tmp_wav(self, suffix: str) -> str:
        return Path(
            self.cfg.cache_folder,
            f"{time.time_ns()}-{uuid.uuid4().hex[:8]}-{suffix}.wav",
        ).as_posix()

    def _pad_wav_to_ms(self, wav_path: str, target_ms: int) -> None:
        """Append silence so wav reaches at least target_ms (ffprobe units)."""
        if target_ms <= 0 or not vail_file(wav_path):
            return
        cur = get_audio_time(wav_path)
        gap = int(target_ms) - int(cur)
        if gap <= 50:
            return
        out = self._tmp_wav("apad")
        runffmpeg([
            "-y", "-i", wav_path,
            "-af", f"apad=pad_dur={gap / 1000.0}",
            "-ar", _MIX_SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_s16le",
            out,
        ], force_cpu=True)
        shutil.move(out, wav_path)
        logger.debug(f"pad audio {Path(wav_path).name}: {cur}ms -> {get_audio_time(wav_path)}ms (want {target_ms})")

    def _amix_under_dub(self, underlay_wav: str, volume: float) -> None:
        """Mix underlay under dubbing; keep dub length; force 48k on both legs."""
        if not vail_file(self.cfg.target_wav) or not vail_file(underlay_wav):
            return
        expect_ms = get_audio_time(self.cfg.target_wav)
        tmp_volume = self._tmp_wav("vol")
        tmp_out = self._tmp_wav("amix")
        self._set_volume(underlay_wav, tmp_volume, volume)
        # weights=3 1 keeps dub audible vs ducked vocal / BGM; TTS Volume UI controls overall level.
        runffmpeg([
            "-y",
            "-i", self.cfg.target_wav,
            "-i", tmp_volume,
            "-filter_complex",
            (
                f"[0:a]aresample={_MIX_SAMPLE_RATE},aformat=sample_rates={_MIX_SAMPLE_RATE}:channel_layouts=stereo[a0];"
                f"[1:a]aresample={_MIX_SAMPLE_RATE},aformat=sample_rates={_MIX_SAMPLE_RATE}:channel_layouts=stereo[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0:weights=3 1,"
                "alimiter=limit=0.95:level=0[aout]"
            ),
            "-map", "[aout]",
            "-ar", _MIX_SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_s16le",
            tmp_out,
        ], force_cpu=True)
        if not vail_file(tmp_out):
            raise RuntimeError(f"amix produced no output: {tmp_out}")
        shutil.move(tmp_out, self.cfg.target_wav)
        self._pad_wav_to_ms(self.cfg.target_wav, expect_ms)
        try:
            Path(tmp_volume).unlink(missing_ok=True)
        except OSError:
            pass

    def _back_music(self) -> None:
        if self._exit() or not self.should_dubbing or not vail_file(self.cfg.target_wav) or not vail_file( self.cfg.background_music):
            return
        try:
            self.signal(text=tr("Adding background audio"))
            vtime = get_audio_time(self.cfg.target_wav)
            atime = get_audio_time(self.cfg.background_music)
            if atime < 1:
                return
            bgm_file=self.cfg.cache_folder + f'/bgm_file.wav'
            self._set_volume(self.cfg.background_music, bgm_file, self.cfg.backaudio_volume)
            self.cfg.background_music = bgm_file
            beishu = math.ceil(vtime / atime)
            if self.cfg.loop_backaudio and beishu > 1 and vtime - 1000 > atime:
                file_list = [self.cfg.background_music for n in range(beishu + 1)]
                concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                create_concat_txt(file_list, concat_txt=concat_txt)
                concat_multi_audio(
                    concat_txt=concat_txt,
                    out=self.cfg.cache_folder + "/bgm_file_extend.wav")
                self.cfg.background_music = self.cfg.cache_folder + "/bgm_file_extend.wav"

            # BGM already volume-adjusted above; mix at unity under dub.
            self._amix_under_dub(self.cfg.background_music, 1.0)
        except Exception as e:
            logger.exception(f'添加背景音乐失败,静默跳过 {e}', exc_info=True)

    def _separate(self) -> None:
        if self._exit():
            return
        if not self.cfg.embed_bgm or not vail_file(self.cfg.instrument) or not vail_file(self.cfg.target_wav):
            return
        instrument = self._retime_original_track(self.cfg.instrument)
        try:
            self.signal(text=tr("Re-embedded background sounds"))
            vtime = get_audio_time(self.cfg.target_wav)
            atime = get_audio_time(instrument)
            if atime < 1:
                return
            beishu = math.ceil(vtime / atime)

            instrument_file = instrument
            logger.debug(f'合并背景音 {beishu=},{atime=},{vtime=}')
            if atime + 1000 < vtime:
                if int(self.cfg.loop_backaudio) == 1:
                    file_list = [instrument_file for n in range(beishu + 1)]
                    concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                    create_concat_txt(file_list, concat_txt=concat_txt)
                    concat_multi_audio(concat_txt=concat_txt,
                                             out=self.cfg.cache_folder + "/instrument-concat.wav")
                else:
                    change_speed_rubberband(instrument_file, self.cfg.cache_folder + f"/instrument-concat.wav",
                                                  vtime)
                instrument_file = self.cfg.cache_folder + f"/instrument-concat.wav"

            self._amix_under_dub(instrument_file, self.cfg.backaudio_volume)
        except Exception as e:
            logger.exception(f'重新嵌入分离的背景音失败 {e}', exc_info=True)

    def _keep_original(self) -> None:
        """Lower original speech and overlay dubbing on top.

        If vocal/BGM was separated, duck vocal.wav only so Re-embed BGM can still
        add instrument without doubling music. Otherwise duck the full original track.
        """
        if self._exit() or not getattr(self.cfg, 'keep_original_audio', False):
            return
        if not self.should_dubbing or not vail_file(self.cfg.target_wav):
            return

        # Prefer separated vocal when available (pairs cleanly with Re-embed BGM).
        underlay = None
        underlay_kind = 'original'
        if vail_file(self.cfg.vocal):
            underlay = self.cfg.vocal
            underlay_kind = 'vocal'
        elif vail_file(self.cfg.original_wav):
            underlay = self.cfg.original_wav
        else:
            underlay = Path(self.cfg.cache_folder + '/original_mix.wav').as_posix()
            try:
                runffmpeg([
                    "-y", "-i", self.cfg.name,
                    "-vn", "-ac", "2", "-ar", _MIX_SAMPLE_RATE, "-c:a", "pcm_s16le",
                    underlay,
                ])
                self.cfg.original_wav = underlay
            except Exception as e:
                logger.exception(f'导出原声失败，跳过保留原声叠加 {e}', exc_info=True)
                return
        if not vail_file(underlay):
            return
        underlay = self._retime_original_track(underlay)
        try:
            self.signal(text=tr("Mixing original audio under dubbing..."))
            vol = float(getattr(self.cfg, 'original_audio_volume', 0.3) or 0.3)
            vol = max(0.01, min(vol, 2.0))
            self._amix_under_dub(underlay, vol)
            logger.debug(f'原声叠加完成 kind={underlay_kind} volume={vol} dur={get_audio_time(self.cfg.target_wav)}ms')
        except Exception as e:
            logger.exception(f'原声叠加失败 {e}', exc_info=True)

    def _retime_original_track(self, filename):
        from videotrans.task._rate import retime_source_audio
        timeline = getattr(self, 'sync_timeline', [])
        if not timeline:
            return filename
        return retime_source_audio(filename, timeline, self.cfg.cache_folder)

    def _set_volume(self, input_file, output_file, volume=None):
        if volume is None:
            volume = self.cfg.backaudio_volume
        runffmpeg([
            "-y",
            "-i",
            input_file,
            "-ar",
            _MIX_SAMPLE_RATE,
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "-filter:a",
            f"volume={volume}",
            output_file
        ], force_cpu=True)
