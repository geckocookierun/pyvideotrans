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
