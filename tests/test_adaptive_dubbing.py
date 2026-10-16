import json

import pytest

from videotrans.translator._adaptive_dubbing import (
    META_FILENAME,
    _build_scene_plans,
    _rewrite_scene,
    _scene_fit_bounds,
    _source_order,
    build_adaptive_scene_spans,
    fit_adaptive_dubbing_from_audio,
    missing_source_ids,
    parse_adaptive_records,
    run_adaptive_dubbing_rewrite,
    schedule_scene_records,
)


def test_scene_spans_respect_pause_and_maximum_duration():
    times = [(0, 3000), (3100, 7000), (7100, 12000), (14000, 16000)]
    assert build_adaptive_scene_spans(
        times, scene_gap_ms=900, max_scene_ms=10000
    ) == [(0, 1), (2, 2), (3, 3)]


def test_parse_records_requires_real_source_references():
    raw = "<TRANSLATE_TEXT>" + json.dumps([
        {"source_ids": ["S1", "S2"], "text": "Natural dialogue"},
        {"source_ids": ["S99"], "text": "Invalid"},
    ]) + "</TRANSLATE_TEXT>"
    records = parse_adaptive_records(raw, ["S1", "S2", "S3"])
    assert records == [{"source_ids": ["S1", "S2"], "text": "Natural dialogue"}]
    assert missing_source_ids(records, ["S1", "S2", "S3"]) == ["S3"]


def test_parse_records_rejects_unstructured_output():
    with pytest.raises(ValueError):
        parse_adaptive_records("ordinary translated text", ["S1"])


def test_parse_records_repairs_missing_commas_between_objects():
    raw = (
        '[{"source_ids":["S1"],"text":"مرحبا"}'
        '{"source_ids":["S2"],"text":"كيف حالك؟"}]'
    )
    assert parse_adaptive_records(raw, ["S1", "S2"]) == [
        {"source_ids": ["S1"], "text": "مرحبا"},
        {"source_ids": ["S2"], "text": "كيف حالك؟"},
    ]


def test_parse_records_recovers_unescaped_quote_in_dialogue():
    raw = '[{"source_ids":["S1"],"text":"قال "نعم" بسرعة"}]'
    assert parse_adaptive_records(raw, ["S1"]) == [
        {"source_ids": ["S1"], "text": 'قال "نعم" بسرعة'},
    ]


def test_rewrite_scene_retries_only_malformed_response():
    class FakeModel:
        prompt = ""
        calls = 0

        def _item_task(self, _):
            self.calls += 1
            if self.calls == 1:
                return "not JSON"
            assert "previous answer was invalid" in self.prompt
            return '[{"source_ids":["S1"],"text":"صحيح"}]'

    model = FakeModel()
    records = _rewrite_scene(
        model=model,
        scene_sources=[{"id": "S1", "text": "正确", "start_ms": 0, "end_ms": 1000}],
        global_context="S1: 正确",
        source_lang="Chinese",
        target_lang="Arabic",
        max_units=3,
        unit="words",
        window_s=1.0,
    )
    assert model.calls == 2
    assert records == [{"source_ids": ["S1"], "text": "صحيح"}]


def test_rewrite_scene_requests_only_omitted_source_cues():
    class FakeModel:
        prompt = ""
        calls = 0

        def _item_task(self, source):
            self.calls += 1
            if self.calls == 1:
                return '[{"source_ids":["S1"],"text":"الأول"}]'
            assert source == "S2: 第二"
            assert "Required source ids: S2." in self.prompt
            return '[{"source_ids":["S2"],"text":"الثاني"}]'

    model = FakeModel()
    records = _rewrite_scene(
        model=model,
        scene_sources=[
            {"id": "S1", "text": "第一", "start_ms": 0, "end_ms": 1000},
            {"id": "S2", "text": "第二", "start_ms": 1000, "end_ms": 2000},
        ],
        global_context="S1: 第一\nS2: 第二",
        source_lang="Chinese",
        target_lang="Arabic",
        max_units=8,
        unit="words",
        window_s=2.0,
    )
    assert model.calls == 2
    assert [row["source_ids"] for row in records] == [["S1"], ["S2"]]


def test_word_estimate_does_not_trigger_a_speculative_rewrite():
    class FakeModel:
        prompt = ""
        calls = 0

        def _item_task(self, _):
            self.calls += 1
            return '[{"source_ids":["S1"],"text":"one two three four"}]'

    model = FakeModel()
    records = _rewrite_scene(
        model=model,
        scene_sources=[{"id": "S1", "text": "一句", "start_ms": 0, "end_ms": 1000}],
        global_context="S1: 一句",
        source_lang="Chinese",
        target_lang="English",
        max_units=2,
        unit="words",
        window_s=1.0,
    )
    assert model.calls == 1
    assert records == [{"source_ids": ["S1"], "text": "one two three four"}]


def test_source_order_keeps_rewritten_dialogue_near_picture_order():
    records = [
        {"source_ids": ["S8"]},
        {"source_ids": ["S2", "S3"]},
        {"source_ids": ["S5"]},
    ]
    assert [row["source_ids"] for row in sorted(records, key=_source_order)] == [
        ["S2", "S3"], ["S5"], ["S8"]
    ]


def test_anchor_scheduler_stays_in_scene_and_keeps_order():
    records = [
        {"anchor_ms": 1500, "text": "a"},
        {"anchor_ms": 8000, "text": "b"},
        {"anchor_ms": 9500, "text": "c"},
    ]
    spans = schedule_scene_records(
        records,
        scene_start_ms=1000,
        scene_end_ms=10000,
        durations_ms=[2000, 2500, 1800],
    )
    assert spans[0][0] >= 1000
    assert spans[-1][1] <= 10000
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_anchor_scheduler_scales_only_timing_when_audio_overflows():
    records = [{"anchor_ms": 0}, {"anchor_ms": 1000}]
    spans = schedule_scene_records(
        records,
        scene_start_ms=0,
        scene_end_ms=1000,
        durations_ms=[900, 900],
    )
    assert spans == [(0, 500), (500, 1000)]


def test_scene_fit_bounds_borrow_only_half_of_neighboring_silence():
    meta = {"scenes": [
        {"start_ms": 0, "end_ms": 4000},
        {"start_ms": 5360, "end_ms": 11840},
        {"start_ms": 12960, "end_ms": 16000},
    ]}
    assert _scene_fit_bounds(meta, 1) == (4680, 12400)


def test_scene_planning_uses_silence_in_first_pass_budget():
    source = [
        {"text": "一", "start_time": 0, "end_time": 4000},
        {"text": "二", "start_time": 5360, "end_time": 11840},
        {"text": "三", "start_time": 12960, "end_time": 16000},
    ]
    times = [(row["start_time"], row["end_time"]) for row in source]
    plans = _build_scene_plans(
        source,
        times=times,
        spans=[(0, 0), (1, 1), (2, 2)],
        sec_per_unit=1.0,
        safety=1.0,
    )
    assert (plans[1]["fit_start_ms"], plans[1]["fit_end_ms"]) == (4680, 12400)
    assert plans[1]["max_units"] == 7


def _write_fit_meta(tmp_path, end_ms=5000):
    meta = {
        "target_lang": "English",
        "source_lang": "Chinese",
        "sec_per_unit": 0.25,
        "unit": "words",
        "scenes": [{
            "start_ms": 0,
            "end_ms": end_ms,
            "max_units": 12,
            "source": [
                {"id": "S1", "text": "一", "start_ms": 0, "end_ms": 2000},
                {"id": "S2", "text": "二", "start_ms": 2000, "end_ms": end_ms},
            ],
            "global_context": "S1: 一\nS2: 二",
            "records": [
                {"line": 1, "source_ids": ["S1"], "text": "first line", "anchor_ms": 1000},
                {"line": 2, "source_ids": ["S2"], "text": "second line", "anchor_ms": 3500},
            ],
        }],
    }
    (tmp_path / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")


def test_real_tts_measurement_retimes_inside_scene(tmp_path, monkeypatch):
    _write_fit_meta(tmp_path)
    files = [tmp_path / "1.wav", tmp_path / "2.wav"]
    for path in files:
        path.touch()
    monkeypatch.setattr("videotrans.util.help_ffmpeg.get_audio_time", lambda _: 1200)
    output, needs_tts = fit_adaptive_dubbing_from_audio(
        queue_tts=[
            {"line": 1, "filename": str(files[0])},
            {"line": 2, "filename": str(files[1])},
        ],
        cache_folder=str(tmp_path),
        translate_type=10,
        allow_rewrite=False,
    )
    assert needs_tts is False
    assert output[0]["start_time"] >= 0
    assert output[-1]["end_time"] <= 5000


def test_unresolved_real_tts_overflow_is_not_silently_truncated(tmp_path, monkeypatch):
    _write_fit_meta(tmp_path, end_ms=1000)
    files = [tmp_path / "1.wav", tmp_path / "2.wav"]
    for path in files:
        path.touch()
    monkeypatch.setattr("videotrans.util.help_ffmpeg.get_audio_time", lambda _: 900)
    with pytest.raises(RuntimeError, match="could not fit"):
        fit_adaptive_dubbing_from_audio(
            queue_tts=[
                {"line": 1, "filename": str(files[0])},
                {"line": 2, "filename": str(files[1])},
            ],
            cache_folder=str(tmp_path),
            translate_type=10,
            allow_rewrite=False,
        )


def test_allowed_soft_speed_is_used_before_requesting_rewrite(tmp_path, monkeypatch):
    _write_fit_meta(tmp_path, end_ms=1000)
    files = [tmp_path / "1.wav", tmp_path / "2.wav"]
    for path in files:
        path.touch()
    monkeypatch.setattr("videotrans.util.help_ffmpeg.get_audio_time", lambda _: 525)
    output, needs_tts = fit_adaptive_dubbing_from_audio(
        queue_tts=[
            {"line": 1, "filename": str(files[0])},
            {"line": 2, "filename": str(files[1])},
        ],
        cache_folder=str(tmp_path),
        translate_type=10,
        allow_rewrite=False,
        allow_soft_speed=True,
    )
    assert needs_tts is False
    assert output[-1]["end_time"] <= 1000
    meta = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["scenes"][0]["fit"]["status"] == "fitted_soft_speed"


def test_neighbor_silence_fits_real_reported_6990ms_scene(tmp_path, monkeypatch):
    meta = {
        "target_lang": "Arabic",
        "source_lang": "Chinese",
        "sec_per_unit": 0.48,
        "unit": "words",
        "translate_type": 10,
        "scenes": [
            {"start_ms": 226784, "end_ms": 250272, "records": []},
            {
                "start_ms": 251632,
                "end_ms": 258112,
                "max_units": 10,
                "source": [{"id": "S56", "text": "原文", "start_ms": 251632, "end_ms": 258112}],
                "global_context": "S56: 原文",
                "records": [{
                    "line": 1,
                    "source_ids": ["S56"],
                    "text": "نص عربي",
                    "anchor_ms": 254872,
                }],
            },
            {"start_ms": 259232, "end_ms": 274912, "records": []},
        ],
    }
    (tmp_path / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    audio = tmp_path / "scene.wav"
    audio.touch()
    monkeypatch.setattr("videotrans.util.help_ffmpeg.get_audio_time", lambda _: 6990)

    output, needs_tts = fit_adaptive_dubbing_from_audio(
        queue_tts=[{"line": 1, "filename": str(audio)}],
        cache_folder=str(tmp_path),
        translate_type=10,
        allow_rewrite=False,
    )

    assert needs_tts is False
    assert len(output) == 1
    fitted_meta = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
    assert fitted_meta["scenes"][1]["fit"]["available_ms"] == 7720
    assert fitted_meta["scenes"][1]["fit"]["status"] == "fitted"


def test_adaptive_rewrite_end_to_end_writes_anchor_metadata(tmp_path, monkeypatch):
    class FakeModel:
        prompt = ""
        calls = 0

        def _item_task(self, _):
            FakeModel.calls += 1
            required = self.prompt.split("Required source ids: ", 1)[1].split(".\n", 1)[0]
            ids = [value.strip() for value in required.split(",")]
            return json.dumps([
                {"source_ids": [source_id], "text": f"Arabic {source_id}"}
                for source_id in ids
            ])

    monkeypatch.setattr(
        "videotrans.translator._adaptive_dubbing._make_model", lambda *_: FakeModel()
    )
    monkeypatch.setattr(
        "videotrans.translator._adaptive_dubbing.get_source_target_code",
        lambda **_: ("Chinese", "Arabic"),
    )
    monkeypatch.setattr(
        "videotrans.translator._adaptive_dubbing.probe_target_sec_per_unit",
        lambda **_: {
            "sec_per_unit": 0.3,
            "unit": "words",
            "voice": "test",
            "rate": "+0%",
        },
    )
    source = [
        {"text": "你好", "start_time": 0, "end_time": 2000},
        {"text": "再见", "start_time": 2100, "end_time": 5000},
    ]
    output = run_adaptive_dubbing_rewrite(
        text_list=source,
        translate_type=10,
        source_code="zh-cn",
        target_code="ar",
        cache_folder=str(tmp_path),
    )
    assert [row["text"] for row in output] == ["Arabic S1", "Arabic S2"]
    assert output[-1]["end_time"] <= 5000
    meta = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["scenes"][0]["records"][0]["source_ids"] == ["S1"]
    assert meta["complete"] is True
    assert FakeModel.calls == 1

    # An identical retry restores completed scenes without another LLM call.
    run_adaptive_dubbing_rewrite(
        text_list=source,
        translate_type=10,
        source_code="zh-cn",
        target_code="ar",
        cache_folder=str(tmp_path),
    )
    assert FakeModel.calls == 1


def test_repeated_over_budget_tts_tightens_cap_and_supplies_feedback(tmp_path, monkeypatch):
    _write_fit_meta(tmp_path, end_ms=8269)
    path = tmp_path / META_FILENAME
    meta = json.loads(path.read_text())
    scene = meta['scenes'][0]
    scene['max_units'] = 21
    scene['records'][0]['text'] = ' '.join(['word'] * 14)
    scene['records'][1]['text'] = ' '.join(['word'] * 14)
    path.write_text(json.dumps(meta))
    audio = tmp_path / 'scene.wav'
    audio.touch()
    monkeypatch.setattr('videotrans.util.help_ffmpeg.get_audio_time', lambda _: 4585)

    class Model:
        prompt = ''

        def _item_task(self, _):
            assert '9.17 seconds' in self.prompt
            assert '8.27 seconds' in self.prompt
            assert 'at most 17 words' in self.prompt
            assert 'PREVIOUS DIALOGUE (too long)' in self.prompt
            assert 'word word' in self.prompt
            return '[{"source_ids":["S1","S2"],"text":"short dialogue"}]'

    monkeypatch.setattr('videotrans.translator._adaptive_dubbing._make_model', lambda *_: Model())
    output, retry = fit_adaptive_dubbing_from_audio(
        queue_tts=[{'line': line, 'filename': str(audio)} for line in (1, 2)],
        cache_folder=str(tmp_path), translate_type=10,
    )
    assert retry is True
    assert output[0]['text'] == 'short dialogue'
    assert json.loads(path.read_text())['scenes'][0]['max_units'] == 17


def test_exhausted_rewrites_persist_actual_measurements(tmp_path, monkeypatch):
    _write_fit_meta(tmp_path, end_ms=8269)
    audio = tmp_path / 'scene.wav'
    audio.touch()
    monkeypatch.setattr('videotrans.util.help_ffmpeg.get_audio_time', lambda _: 4585)
    with pytest.raises(RuntimeError, match=r'needs 10\.9% speed-up'):
        fit_adaptive_dubbing_from_audio(
            queue_tts=[{'line': line, 'filename': str(audio)} for line in (1, 2)],
            cache_folder=str(tmp_path), translate_type=10, allow_rewrite=False,
        )
    fit = json.loads((tmp_path / META_FILENAME).read_text())['scenes'][0]['fit']
    assert fit['status'] == 'overflow'
    assert fit['actual_tts_ms'] == 9170
    assert fit['available_ms'] == 8269



def _single_scene_kwargs(model, target_lang='Vietnamese'):
    return dict(
        model=model,
        scene_sources=[{'id': 'S1', 'text': '你要老师替你道歉吗？', 'start_ms': 0, 'end_ms': 3000}],
        global_context='A teacher speaks to a student.',
        source_lang='Chinese', target_lang=target_lang,
        max_units=20, unit='words', window_s=3,
    )


def test_valid_translation_needs_one_call_without_review():
    class Model:
        prompt = ''
        calls = 0

        def _item_task(self, _):
            self.calls += 1
            assert 'Translate the source dialogue from Chinese into Vietnamese' in self.prompt
            assert 'A teacher speaks to a student.' in self.prompt
            assert 'grammatical subject, action, and recipient' in self.prompt
            # Source appears once: no repeated INPUT block or review request.
            assert self.prompt.count('你要老师替你道歉吗？') == 1
            return '[{"source_ids":["S1"],"text":"Em muốn thầy xin lỗi thay em sao?"}]'

    model = Model()
    records = _rewrite_scene(**_single_scene_kwargs(model))
    assert model.calls == 1
    assert records[0]['text'] == 'Em muốn thầy xin lỗi thay em sao?'


def test_untranslated_han_retries_before_acceptance():
    class Model:
        prompt = ''
        calls = 0

        def _item_task(self, _):
            self.calls += 1
            if self.calls == 1:
                return '[{"source_ids":["S1"],"text":"老师替你道歉？"}]'
            assert 'Untranslated Han-script text remains' in self.prompt
            return '[{"source_ids":["S1"],"text":"Em muốn thầy xin lỗi thay em sao?"}]'

    model = Model()
    records = _rewrite_scene(**_single_scene_kwargs(model))
    assert model.calls == 2
    assert records[0]['text'] == 'Em muốn thầy xin lỗi thay em sao?'


def test_wrong_script_is_not_accepted_as_shortest_complete_result():
    class Model:
        prompt = ''

        def _item_task(self, _):
            return '[{"source_ids":["S1"],"text":"老师替你道歉？"}]'

    with pytest.raises(RuntimeError, match='invalid output'):
        _rewrite_scene(**_single_scene_kwargs(Model()))


@pytest.mark.parametrize('language', ['Chinese', 'Traditional Chinese', 'Japanese', 'zh-cn', 'ja'])
def test_han_remains_valid_for_han_script_targets(language):
    class Model:
        prompt = ''

        def _item_task(self, _):
            return '[{"source_ids":["S1"],"text":"先生"}]'

    assert _rewrite_scene(**_single_scene_kwargs(Model(), language))[0]['text'] == '先生'


def test_dialogue_context_excludes_distant_and_current_cues():
    from videotrans.translator._adaptive_dubbing import _scene_dialogue_context
    context = '\n'.join(f'S{i}: text {i}' for i in range(1, 41))
    result = _scene_dialogue_context(context, [{'id': 'S20'}, {'id': 'S21'}])
    assert 'S14: text 14' in result
    assert 'S19: text 19' in result
    assert 'S22: text 22' in result
    assert 'S25: text 25' in result
    assert 'S13:' not in result
    assert 'S20:' not in result
    assert 'S21:' not in result
    assert 'S26:' not in result
    assert result.index('PRECEDING DIALOGUE') < result.index('FOLLOWING DIALOGUE')


@pytest.fixture
def adaptive_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class Model:
        model_name = 'model-a'
        prompt = ''
        fail_on = None

        def __init__(self):
            self.calls = []

        def _item_task(self, _):
            required = self.prompt.split('Required source ids: ', 1)[1].split('.\n', 1)[0]
            ids = [value.strip() for value in required.split(',')]
            self.calls.append(ids)
            if self.fail_on in ids:
                raise RuntimeError('provider unavailable')
            return json.dumps([{'source_ids': [sid], 'text': f'translated {sid}'} for sid in ids])

    model = Model()
    probes = []

    def probe(**kwargs):
        probes.append(kwargs)
        return {'sec_per_unit': 0.3, 'unit': 'words', 'voice': kwargs['voice_role'], 'rate': kwargs['voice_rate']}

    monkeypatch.setattr('videotrans.translator._adaptive_dubbing._make_model', lambda *_: model)
    monkeypatch.setattr('videotrans.translator._adaptive_dubbing.get_source_target_code', lambda **_: ('Chinese', 'English'))
    monkeypatch.setattr('videotrans.translator._adaptive_dubbing.probe_target_sec_per_unit', probe)
    kwargs = dict(
        text_list=[
            {'text': '你好', 'start_time': 0, 'end_time': 2000},
            {'text': '再见', 'start_time': 4000, 'end_time': 6000},
        ],
        translate_type=10, source_code='zh-cn', target_code='en',
        voice_role='voice-a', voice_rate='+0%', cache_folder=str(tmp_path),
    )
    return SimpleNamespace(model=model, probes=probes, kwargs=kwargs, path=tmp_path / META_FILENAME)


def test_complete_checkpoint_skips_llm_and_pace_probe_and_keeps_fitted_times(adaptive_pipeline):
    p = adaptive_pipeline
    run_adaptive_dubbing_rewrite(**p.kwargs)
    meta = json.loads(p.path.read_text())
    meta['scenes'][0]['records'][0].update(start_ms=123, end_ms=876)
    p.path.write_text(json.dumps(meta))
    output = run_adaptive_dubbing_rewrite(**p.kwargs)
    assert len(p.model.calls) == 2
    assert len(p.probes) == 1
    assert (output[0]['start_time'], output[0]['end_time']) == (123, 876)


@pytest.mark.parametrize('changed', ['source', 'voice', 'rate', 'model', 'version', 'safety'])
def test_checkpoint_invalidates_for_translation_inputs(adaptive_pipeline, monkeypatch, changed):
    from videotrans.configure.config import settings
    p = adaptive_pipeline
    run_adaptive_dubbing_rewrite(**p.kwargs)
    if changed == 'source':
        p.kwargs['text_list'][0]['text'] = '您好'
    elif changed == 'voice':
        p.kwargs['voice_role'] = 'voice-b'
    elif changed == 'rate':
        p.kwargs['voice_rate'] = '+20%'
    elif changed == 'model':
        p.model.model_name = 'model-b'
    elif changed == 'version':
        meta = json.loads(p.path.read_text())
        meta['version'] -= 1
        p.path.write_text(json.dumps(meta))
    else:
        monkeypatch.setattr(settings, 'adaptive_safety_pct', 70)
    run_adaptive_dubbing_rewrite(**p.kwargs)
    assert len(p.model.calls) == 4
    assert len(p.probes) == 2


def test_partial_checkpoint_retries_only_unfinished_scenes(adaptive_pipeline):
    p = adaptive_pipeline
    p.model.fail_on = 'S2'
    with pytest.raises(RuntimeError, match='provider unavailable'):
        run_adaptive_dubbing_rewrite(**p.kwargs)
    assert json.loads(p.path.read_text())['complete'] is False
    p.model.fail_on = None
    run_adaptive_dubbing_rewrite(**p.kwargs)
    assert p.model.calls == [['S1'], ['S2'], ['S2']]
    assert json.loads(p.path.read_text())['complete'] is True


def test_matching_cache_preserves_user_edited_target(adaptive_pipeline, tmp_path):
    p = adaptive_pipeline
    run_adaptive_dubbing_rewrite(**p.kwargs)
    target = tmp_path / 'target.srt'
    target.write_text('1\n00:00:00,123 --> 00:00:00,876\nUser edited translation\n')
    output = run_adaptive_dubbing_rewrite(target_sub=str(target), **p.kwargs)
    assert output[0]['text'] == 'User edited translation'
    assert len(p.model.calls) == 2
    assert len(p.probes) == 1


@pytest.mark.parametrize('broken', ['{broken', 'null', '[]', '{"scenes": null}'])
def test_invalid_checkpoint_is_rebuilt(adaptive_pipeline, broken):
    p = adaptive_pipeline
    p.path.write_text(broken)
    output = run_adaptive_dubbing_rewrite(**p.kwargs)
    assert len(output) == 2
    assert json.loads(p.path.read_text())['complete'] is True


def test_scene_plans_store_only_bounded_context():
    source = [{'text': f'cue {i}', 'start_time': i * 1000, 'end_time': i * 1000 + 500} for i in range(1000)]
    times = [(row['start_time'], row['end_time']) for row in source]
    plans = _build_scene_plans(source, times=times, spans=[(500, 501)], sec_per_unit=0.3, safety=0.85)
    context = plans[0]['global_context']
    assert len(context.splitlines()) == 12
    assert 'S495: cue 494' in context
    assert 'S506: cue 505' in context
    assert 'S1:' not in context


@pytest.mark.parametrize('row', [
    {'source_ids': 42, 'text': 'invalid'},
    {'source_ids': {'S1': True}, 'text': 'invalid'},
    {'source_ids': ['S1'], 'text': {'translation': 'invalid'}},
    {'source_ids': ['S1'], 'text': 123},
])
def test_invalid_record_types_are_rejected_for_retry(row):
    with pytest.raises(ValueError, match='no valid dialogue records'):
        parse_adaptive_records(json.dumps([row]), ['S1'])


def test_bounded_context_produces_the_same_first_pass_prompt():
    from videotrans.translator._adaptive_dubbing import _build_scene_context, adaptive_rewrite_prompt
    source = [{'text': f'line {i}'} for i in range(100)]
    full_context = '\n'.join(f'S{i+1}: {row["text"]}' for i, row in enumerate(source))
    kwargs = dict(
        scene_sources=[{'id': f'S{i+1}', 'text': source[i]['text']} for i in (40, 41)],
        source_lang='English', target_lang='Vietnamese', max_units=20, unit='words', window_s=5,
    )
    assert adaptive_rewrite_prompt(global_context=full_context, **kwargs) == adaptive_rewrite_prompt(
        global_context=_build_scene_context(source, 40, 41), **kwargs,
    )
