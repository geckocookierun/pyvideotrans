"""Corrected ASR text must reach the file consumed by translation/export."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from videotrans.task._stage_recogn import RecognMixin
from videotrans.task._stage_translate import TranslateMixin
from videotrans.util.help_srt import get_subtitle_from_srt, get_srt_from_list


@pytest.mark.parametrize('rephrase', [True, False])
@pytest.mark.parametrize('mode', ['none', 'adaptive'])
def test_recognition_hands_corrected_text_to_translation(tmp_path, monkeypatch, rephrase, mode):
    raw = get_subtitle_from_srt(
        '1\n00:00:01,000 --> 00:00:03,000\nYou asked me no I asked her\n',
        is_file=False,
    )
    corrected = get_subtitle_from_srt(
        '1\n00:00:01,000 --> 00:00:03,000\nYou asked me? No, I asked her.\n',
        is_file=False,
    )
    source = tmp_path / 'source.srt'
    target = tmp_path / 'target.srt'
    if mode == 'adaptive':
        # An existing target must not bypass adaptive source/config validation.
        target.write_text('1\n00:00:01,000 --> 00:00:03,000\nOld translation\n')
    audio = tmp_path / 'source.wav'
    audio.write_bytes(b'audio')
    seen = []

    class Task(RecognMixin, TranslateMixin):
        precent = 0
        uuid = 'test'
        max_speakers = -1
        should_recogn = True
        should_trans = True
        should_dubbing = True

        def _exit(self):
            return False

        def signal(self, **kwargs):
            pass

        def _llmpost(self, subtitles):
            assert subtitles[0]['text'] == raw[0]['text']
            return corrected

        def _save_srt_target(self, subtitles, filename):
            Path(filename).write_text(get_srt_from_list(subtitles), encoding='utf-8')

    task = Task()
    task.cfg = SimpleNamespace(
        source_sub=str(source), target_sub=str(target), source_wav=str(audio),
        target_dir=str(tmp_path), cache_folder=str(tmp_path), remove_noise=False,
        is_separate=False, recogn_type=0, model_name='test', detect_language='en',
        is_cuda=False, subtitle_type=0, app_mode='biaozhun', fix_punc=0,
        rephrase=rephrase, narration_mode=mode, translate_type=10,
        source_language_code='en', target_language_code='vi',
    )
    monkeypatch.setattr('videotrans.task._stage_recogn.run_recogn', lambda **_: raw)

    def translate(**kwargs):
        seen.extend(kwargs['text_list'])
        return kwargs['text_list']

    monkeypatch.setattr('videotrans.task._stage_translate.run_trans', translate)
    monkeypatch.setattr('videotrans.translator._adaptive_dubbing.run_adaptive_dubbing_rewrite', translate)
    task.check_target_sub = lambda source, target: target
    task.recogn()
    task.trans()
    expected = corrected if rephrase else raw
    assert seen[0]['text'] == expected[0]['text']
    assert seen[0]['start_time'] == raw[0]['start_time']
    assert seen[0]['end_time'] == raw[0]['end_time']
    assert get_subtitle_from_srt(str(source))[0]['text'] == expected[0]['text']
