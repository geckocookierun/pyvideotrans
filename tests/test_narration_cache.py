from types import SimpleNamespace

import pytest

from videotrans.translator import _narration, _openaicompat


def test_resume_after_scene_failure_reuses_completed_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(_narration, 'TEMP_ROOT', str(tmp_path))
    calls = []
    failing = True

    def translate(text):
        calls.append(text)
        if text == 'second' and failing:
            raise RuntimeError('network interrupted')
        return '<TRANSLATE_TEXT>She remembers ' + text + '.</TRANSLATE_TEXT>'

    model = SimpleNamespace(prompt='narrate {batch_input}', _item_task=translate)
    first = _narration._scene_narration(model, 'first', 1)
    with pytest.raises(RuntimeError, match='network interrupted'):
        _narration._scene_narration(model, 'second', 1)
    failing = False
    assert _narration._scene_narration(model, 'first', 1) == first
    assert _narration._scene_narration(model, 'second', 1)
    assert calls == ['first', 'second', 'second']


@pytest.mark.parametrize('field,value', [
    ('prompt', 'different target language or budget'),
    ('model_name', 'another-model'), ('api_url', 'https://different.invalid'),
    ('temperature', .7), ('extra_body', {'reasoning': {'enabled': True}}),
])
def test_changed_request_does_not_reuse_scene(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(_narration, 'TEMP_ROOT', str(tmp_path))
    calls = []
    model = SimpleNamespace(prompt='original', _item_task=lambda text: calls.append(text) or 'A reply.')
    _narration._scene_narration(model, 'source', 1)
    setattr(model, field, value)
    _narration._scene_narration(model, 'source', 1)
    assert len(calls) == 2


def test_empty_scene_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(_narration, 'TEMP_ROOT', str(tmp_path))
    model = SimpleNamespace(prompt='original', _item_task=lambda text: '')
    with pytest.raises(RuntimeError, match='empty'):
        _narration._scene_narration(model, 'source', 1)
    assert not list(tmp_path.rglob('*.txt'))


def test_sdk_retries_disabled_and_request_preserved(monkeypatch):
    constructors, requests = [], []

    def client(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **request: requests.append(request) or 'response')))

    monkeypatch.setattr(_openaicompat, 'OpenAI', client)
    model = SimpleNamespace(api_key='test', api_url='https://example.invalid', extra_body=None)
    assert _openaicompat.OpenAICampat._create_completion(model, {'model': 'test'}) == 'response'
    assert constructors[0]['max_retries'] == 0
    assert requests == [{'model': 'test', 'extra_body': None}]


@pytest.mark.parametrize('target', ['vi', 'ar', 'en'])
def test_wrong_language_retries_only_failed_scene(tmp_path, monkeypatch, target):
    monkeypatch.setattr(_narration, 'TEMP_ROOT', str(tmp_path))
    bad = '成群的埃德蒙顿龙也纷纷赶来。'
    good = 'Đàn khủng long kéo đến.' if target != 'ar' else 'وصلت الديناصورات إلى المكان.'
    replies = iter([bad, good])
    calls = []
    def translate(text):
        calls.append(text)
        return next(replies)
    model = SimpleNamespace(prompt='original', _item_task=translate)
    result = _narration._scene_narration(model, 'source', 1, 'zh-cn', target)
    assert result == [good]
    assert model.prompt == 'original'
    assert _narration._scene_narration(model, 'source', 1, 'zh-cn', target) == result
    assert len(calls) == 2


def test_invalid_old_cache_replaced_but_good_cache_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(_narration, 'TEMP_ROOT', str(tmp_path))
    bad = '成群的埃德蒙顿龙也纷纷赶来。'
    model = SimpleNamespace(prompt='original', _item_task=lambda text: bad)
    _narration._scene_narration(model, 'source', 1)  # legacy unchecked cache
    model._item_task = lambda text: 'Đàn khủng long kéo đến.'
    assert _narration._scene_narration(model, 'source', 1, 'zh-cn', 'vi') == ['Đàn khủng long kéo đến.']
    assert bad not in next(tmp_path.rglob('*.txt')).read_text()


def test_repeated_invalid_response_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(_narration, 'TEMP_ROOT', str(tmp_path))
    calls = []
    model = SimpleNamespace(prompt='original', _item_task=lambda text: calls.append(text) or '成群的埃德蒙顿龙也纷纷赶来。')
    with pytest.raises(ValueError, match='source script'):
        _narration._scene_narration(model, 'source', 1, 'zh-cn', 'vi')
    assert len(calls) == 2
    assert model.prompt == 'original'
    assert not list(tmp_path.rglob('*.txt'))


@pytest.mark.parametrize('source,target,text', [
    ('zh', 'vi', 'Cô ấy tên là 小明 và sống tại đây.'),
    ('zh', 'ja', '成群の恐竜が到着しました。'),
    ('zh', 'zh-tw', '成群的埃德蒙頓龍也紛紛趕來。'),
    ('en', 'vi', 'Đàn khủng long kéo đến.'),
])
def test_guard_allows_names_and_shared_scripts(source, target, text):
    from videotrans.translator._language_guard import validate_narration_language
    validate_narration_language([text], source, target)


def test_existing_wrong_language_srt_reaches_scene_repair(tmp_path, monkeypatch):
    from videotrans.task import _stage_translate
    target = tmp_path / 'vi.srt'
    target.write_text('1\n00:00:00,000 --> 00:00:02,000\n成群的埃德蒙顿龙也纷纷赶来。\n')
    source = tmp_path / 'zh.srt'
    source.write_text(target.read_text())
    class RepairReached(Exception):
        pass
    def repair(**kwargs):
        raise RepairReached()
    monkeypatch.setattr(_narration, 'run_third_person_narration', repair)
    task = _stage_translate.TranslateMixin()
    task.cfg = SimpleNamespace(narration_mode='third', target_sub=str(target), source_sub=str(source),
                               source_language_code='zh-cn', target_language_code='vi', translate_type=1)
    task.uuid = 'test'
    task.precent = 0
    task.should_trans = True
    task._exit = lambda: False
    task.signal = lambda **kwargs: None
    with pytest.raises(RepairReached):
        task.trans()
    assert '成群' in target.read_text()  # don't destroy existing output on failure


def test_wrong_language_blocked_before_tts(monkeypatch):
    from videotrans.task import _stage_dubbing
    task = _stage_dubbing.DubbingMixin()
    task.should_dubbing = True
    task.cfg = SimpleNamespace(narration_mode='third', source_language_code='zh-cn', target_language_code='vi')
    task._build_tts_queue = lambda: [dict(text='成群的埃德蒙顿龙也纷纷赶来。')]
    monkeypatch.setattr(_stage_dubbing, 'run_tts', lambda **kwargs: pytest.fail('must not send invalid cues'))
    with pytest.raises(ValueError, match='source script'):
        task._tts()
