from copy import deepcopy
from types import SimpleNamespace

import pytest

from videotrans.translator import _openaicompat as module
from videotrans.util._srt_parse import get_subtitle_from_srt, ms_to_time_string


def run_correction(monkeypatch, rows, responses, chunk_size):
    monkeypatch.setattr(module, 'settings', {'llm_chunk_size': chunk_size})
    calls = []
    replies = iter(responses)
    def completion(kwargs):
        calls.append(deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])
    model = SimpleNamespace(model_name='test', temperature=.1, reasoning_effort=None,
                            api_url='https://example.invalid', ainame='test', max_tokens=1000,
                            signal=lambda **kwargs: None, _create_completion=completion)
    result = module.OpenAICampat.llm_segment(model, rows)
    return result, calls


def srt(start, count, text='original'):
    return '\n\n'.join(f'{i+1}\n{ms_to_time_string(ms=(start+i)*2000)} --> '
                      f'{ms_to_time_string(ms=(start+i)*2000+1000)}\n{text} {start+i}'
                      for i in range(count))


def test_empty_third_batch_keeps_all_184_cues(monkeypatch):
    rows = get_subtitle_from_srt(srt(0, 184), is_file=False)
    before = deepcopy(rows)
    result, calls = run_correction(monkeypatch, rows,
        [srt(0, 50, 'corrected'), srt(50, 50, 'corrected'), '<SRT></SRT>',
         srt(150, 34, 'corrected')], 50)
    assert len(result) == 184
    assert result[100:150] == before[100:150]
    assert result[99]['text'].startswith('corrected')
    assert result[150]['text'].startswith('corrected')
    assert rows == before
    assert len(calls) == 4


@pytest.mark.parametrize('bad', ['', 'Sorry, no subtitles.', srt(0, 1), srt(0, 3),
                                 srt(1, 2), srt(0, 2).replace('original 1', '')])
def test_invalid_batch_preserves_original(monkeypatch, bad):
    rows = get_subtitle_from_srt(srt(0, 2), is_file=False)
    rows[0]['spk'] = 'speaker1'
    result, calls = run_correction(monkeypatch, rows, [bad], 2)
    assert result == rows
    assert result[0] is not rows[0]
    assert len(calls) == 1


def test_valid_correction_only_changes_text(monkeypatch):
    rows = get_subtitle_from_srt(srt(0, 2), is_file=False)
    rows[0]['spk'] = 'speaker1'
    rows[0]['line'] = 9
    result, _ = run_correction(monkeypatch, rows, ['<SRT>'+srt(0, 2, 'corrected')+'</SRT>'], 2)
    expected = deepcopy(rows)
    for row in expected:
        row['text'] = row['text'].replace('original', 'corrected')
    assert result == expected
