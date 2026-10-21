from types import SimpleNamespace

import pytest

from videotrans.configure.excepts import TranslateSrtError
from videotrans.translator._language_guard import (
    incomplete_lines, require_complete_translation, translation_failure,
)


def test_full_source_sentence_blocks_translation():
    with pytest.raises(TranslateSrtError, match='dubbing was not started'):
        require_complete_translation(
            ['เพียงเพราะดื่มสุราพิษ', '只因喝了毒酒才死于华雄'], 'zh-cn', 'th')


def test_blank_cue_blocks_translation():
    assert incomplete_lines(['คำแปล', ''], 'zh-cn', 'th') == [(2, '')]


def test_short_name_and_shared_script_pass():
    require_complete_translation(['Cô ấy tên là 小明 và sống tại đây.'], 'zh', 'vi')
    require_complete_translation(['成群の恐竜が到着しました。'], 'zh', 'ja')
    assert translation_failure(['เขาอาจไปถึงระดับที่สูงกว่า'], 'zh-cn', 'th') is None


def test_untranslated_cue_stops_before_save(tmp_path, monkeypatch):
    from videotrans.task import _stage_translate
    from videotrans.task._base import BaseTask
    from videotrans.translator import _language_guard

    monkeypatch.setattr(_language_guard, 'discard_cached_translations', lambda lines: None)

    source = tmp_path / 'zh.srt'
    target = tmp_path / 'th.srt'
    source.write_text(
        '1\n00:18:31,216 --> 00:18:32,656\n只因喝了毒酒才死于华雄\n\n'
        '2\n00:18:33,136 --> 00:18:33,776\n传闻真假\n',
        encoding='utf-8')

    def translated(**kwargs):
        return [
            {'line': 1, 'time': '00:18:31,216 --> 00:18:32,656', 'text': '只因喝了毒酒才死于华雄'},
            {'line': 2, 'time': '00:18:33,136 --> 00:18:33,776', 'text': 'ข่าวลือจริงเท็จ'},
        ]

    monkeypatch.setattr(_stage_translate, 'run_trans', translated)
    task = _stage_translate.TranslateMixin()
    task.check_target_sub = BaseTask.check_target_sub
    task.cfg = SimpleNamespace(
        narration_mode='none', target_sub=str(target), source_sub=str(source),
        source_language_code='zh-cn', target_language_code='th',
        translate_type=10, is_cuda=False)
    task.uuid = 'test'
    task.precent = 0
    task.should_trans = True
    task._exit = lambda: False
    task.signal = lambda **kwargs: None
    task._save_srt_target = lambda *args, **kwargs: pytest.fail('must not save an incomplete subtitle')
    with pytest.raises(TranslateSrtError, match='#1'):
        task.trans()
    assert not target.exists()


def test_cached_untranslated_reply_is_dropped(tmp_path, monkeypatch):
    from videotrans.configure import config
    from videotrans.translator import _language_guard

    monkeypatch.setattr(config, 'TEMP_ROOT', str(tmp_path))
    cache = tmp_path / 'translate_cache'
    cache.mkdir()
    kept = cache / 'kept.txt'
    dropped = cache / 'dropped.txt'
    kept.write_text('ข่าวลือจริงเท็จ', encoding='utf-8')
    dropped.write_text('只因喝了毒酒才死于华雄', encoding='utf-8')
    (cache / 'narration').mkdir()
    scene = cache / 'narration' / 'scene.txt'
    scene.write_text('只因喝了毒酒才死于华雄', encoding='utf-8')
    _language_guard.discard_cached_translations(['只因喝了毒酒才死于华雄'])
    assert kept.is_file() and scene.is_file() and not dropped.exists()


def test_saved_untranslated_subtitle_is_refused(tmp_path, monkeypatch):
    from videotrans.task import _stage_translate
    from videotrans.translator import _language_guard

    monkeypatch.setattr(_language_guard, 'discard_cached_translations', lambda lines: None)

    source = tmp_path / 'zh.srt'
    target = tmp_path / 'th.srt'
    cue = '1\n00:19:20,512 --> 00:19:21,872\n他甚至可能达到更高境界\n'
    source.write_text(cue, encoding='utf-8')
    target.write_text(cue, encoding='utf-8')
    monkeypatch.setattr(
        _stage_translate, 'run_trans', lambda **kwargs: pytest.fail('must not dub or retranslate'))
    task = _stage_translate.TranslateMixin()
    task.cfg = SimpleNamespace(
        narration_mode='none', target_sub=str(target), source_sub=str(source),
        source_language_code='zh-cn', target_language_code='th', translate_type=10)
    task.uuid = 'test'
    task.precent = 0
    task.should_trans = True
    task._exit = lambda: False
    task.signal = lambda **kwargs: None
    with pytest.raises(TranslateSrtError, match='他甚至可能达到更高境界'):
        task.trans()
    assert '他甚至可能达到更高境界' in target.read_text(encoding='utf-8')
