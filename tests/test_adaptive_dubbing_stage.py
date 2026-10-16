"""Exercise the TTS feedback loop without network calls or audio synthesis."""
from types import SimpleNamespace

import pytest

from videotrans.task._stage_dubbing import DubbingMixin


@pytest.mark.parametrize('adaptive,rewrite', [(False, False), (True, False), (True, True)])
def test_dubbing_writes_only_rewrites_and_final_timing(monkeypatch, adaptive, rewrite):
    saves, synthesis, fits = [], [], []
    timing = dict(start_time=100, end_time=900, startraw='00:00:00,100', endraw='00:00:00,900', time='00:00:00,100 --> 00:00:00,900')

    class Task(DubbingMixin):
        uuid = 'test'
        should_dubbing = True
        cfg = SimpleNamespace(
            narration_mode='adaptive' if adaptive else 'none', target_language_code='vi',
            cache_folder='/unused', translate_type=10, tts_type=0, is_cuda=False,
            voice_autorate=False, target_sub='/unused/target.srt',
        )

        def signal(self, **_):
            pass

        def _build_tts_queue(self):
            return [dict(line=1, text='short' if saves else 'original', **timing)]

        def _cache_tts_outputs(self):
            pass

        def _save_srt_target(self, records, filename):
            saves.append([dict(row) for row in records])

    def fit(**kwargs):
        fits.append(kwargs['allow_rewrite'])
        if rewrite and len(fits) == 1:
            return [dict(line=1, text='short', **timing)], True
        return [dict(line=1, text='short' if rewrite else 'original', **{**timing, 'end_time': 800})], False

    monkeypatch.setattr('videotrans.task._stage_dubbing.run_tts', lambda **kw: synthesis.append(kw))
    monkeypatch.setattr('videotrans.translator._adaptive_dubbing.fit_adaptive_dubbing_from_audio', fit)
    task = Task()
    task._tts()
    assert len(synthesis) == (2 if rewrite else 1)
    assert len(saves) == (2 if rewrite else 1)
    assert saves[-1][0]['end_time'] == (800 if adaptive else 900)
    assert saves[-1][0]['text'] == ('short' if rewrite else 'original')
    assert len(fits) == (len(synthesis) if adaptive else 0)
