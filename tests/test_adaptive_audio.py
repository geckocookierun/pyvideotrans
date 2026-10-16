import json
import math
import struct
import wave
from pathlib import Path

import pytest

from videotrans.configure.config import settings
from videotrans.task._adaptive_audio import measured_audio_ms, render_adaptive_audio
from videotrans.translator._adaptive_dubbing import _audio_slots


def _tone(path, milliseconds, sample_rate=16000):
    frames = round(milliseconds * sample_rate / 1000)
    # A distinct end tone makes accidental tail removal visible in the signal.
    samples = [int(10000 * math.sin(2 * math.pi * (880 if i > frames * .9 else 440) * i / sample_rate))
               for i in range(frames)]
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
        audio.writeframes(struct.pack(f'<{frames}h', *samples))


def _queue(path, start, end):
    return dict(line=1, filename=str(path), start_time=start, end_time=end, text='test', role='vi-VN-NamMinhNeural')


def test_reported_15370ms_scene_fits_14240ms_without_trimming(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'adaptive_namminh_speed_pct', 10)
    source = tmp_path / 'source.wav'
    _tone(source, 15370)
    target = tmp_path / 'final.wav'
    queue = [_queue(source, 1000, 15240)]
    render_adaptive_audio(queue_tts=queue, target_audio=str(target), cache_folder=str(tmp_path),
                          video_duration_ms=16000, allow_speed=True)
    report = json.loads((tmp_path / 'adaptive_audio/fit_report.json').read_text())
    clip = report['clips'][0]
    assert report['status'] == 'verified'
    assert 1 < clip['speed'] <= 1.10
    assert clip['actual_ms'] <= 14240
    assert measured_audio_ms(source) == 15370  # Raw TTS/cache remains untouched.
    with wave.open(str(target), 'rb') as audio:
        assert audio.getnframes() == 16000 * 48
        assert not any(audio.readframes(1000 * 48))  # Start anchor preserved.
    with wave.open(clip['filename'], 'rb') as audio:
        audio.setpos(max(0, audio.getnframes() - 4800))
        tail = struct.unpack('<' + 'h' * (4800 * 2), audio.readframes(4800))
        # Check the final 880Hz marker, beyond the original 14.24s cutoff.
        left = tail[::2]
        crossings = sum(a < 0 <= b for a, b in zip(left, left[1:]))
        assert 75 < crossings < 100


def test_insufficient_speed_limit_does_not_publish_or_modify_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'adaptive_namminh_speed_pct', 5)
    source = tmp_path / 'source.wav'
    _tone(source, 1537)
    target = tmp_path / 'final.wav'
    target.write_bytes(b'previous result')
    with pytest.raises(RuntimeError, match='above the allowed'):
        render_adaptive_audio(queue_tts=[_queue(source, 0, 1424)], target_audio=str(target),
                              cache_folder=str(tmp_path), video_duration_ms=1500, allow_speed=True)
    assert target.read_bytes() == b'previous result'
    assert measured_audio_ms(source) == 1537


def test_encoder_failure_is_not_ignored(tmp_path, monkeypatch):
    source = tmp_path / 'source.wav'
    _tone(source, 100)
    def fail(*args, **kwargs):
        raise RuntimeError('encoder failed')
    monkeypatch.setattr('videotrans.task._adaptive_audio.runffmpeg', fail)
    target = tmp_path / 'final.wav'
    with pytest.raises(RuntimeError, match='encoder failed'):
        render_adaptive_audio(queue_tts=[_queue(source, 10, 110)], target_audio=str(target),
                              cache_folder=str(tmp_path), video_duration_ms=120, allow_speed=False)
    assert not target.exists()


def test_integer_slots_respect_limit_for_every_clip():
    durations = [501, 548, 702]
    slots = _audio_slots(durations, 1670, 1.05)
    assert slots is not None
    assert sum(slots) == 1670
    assert all(original / slot <= 1.05 for original, slot in zip(durations, slots))
    assert _audio_slots([501, 501], 1000, 1.0) is None


def test_no_speed_mode_preserves_gaps_and_pcm_duration(tmp_path):
    source = tmp_path / 'source.wav'
    _tone(source, 200)
    queue = [_queue(source, 100, 300), {**_queue(source, 700, 900), 'line': 2}]
    target = tmp_path / 'final.wav'
    render_adaptive_audio(queue_tts=queue, target_audio=str(target), cache_folder=str(tmp_path),
                          video_duration_ms=1000, allow_speed=False)
    with wave.open(str(target), 'rb') as audio:
        assert audio.getnframes() == 48000
        audio.setpos(300 * 48)
        assert not any(audio.readframes(400 * 48))


def test_filter_that_cannot_fit_never_trims_or_exceeds_limit(tmp_path, monkeypatch):
    from videotrans.task._adaptive_audio import _prepare_clip
    source = tmp_path / 'source.wav'
    _tone(source, 1100)
    commands = []
    def ineffective_encoder(args):
        commands.append(args)
        _tone(Path(args[-1]), 1100, sample_rate=48000)
    monkeypatch.setattr('videotrans.task._adaptive_audio.runffmpeg', ineffective_encoder)
    with pytest.raises(RuntimeError, match='Audio was not trimmed'):
        _prepare_clip(str(source), tmp_path / 'processed.wav', 48000, 1.10)
    assert len(commands) == 1
    assert '-t' not in commands[0]
    assert 'atrim' not in ' '.join(commands[0])


def test_alignment_routes_adaptive_through_verified_renderer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from videotrans.task._stage_align import AlignMixin
    from videotrans.tts import EDGE_TTS
    seen = []
    class Task(AlignMixin):
        precent = 0
        uuid = 'test'
        should_dubbing = True
        ignore_align = False
        video_time = 1000
        queue_tts = [dict(line=1, text='test', start_time=100, end_time=900)]
        cfg = SimpleNamespace(
            app_mode='biaozhun', narration_mode='adaptive', voice_autorate=True,
            video_autorate=False, novoice_mp4='', target_wav=str(tmp_path / 'final.wav'),
            cache_folder=str(tmp_path), target_sub=str(tmp_path / 'target.srt'),
            fix_punc=0, tts_type=EDGE_TTS, volume='+0%',
        )
        def _exit(self):
            return False
        def signal(self, **kwargs):
            pass
        def _save_srt_target(self, queue, filename):
            pass
    def render(**kwargs):
        seen.append(kwargs)
        return kwargs['queue_tts']
    monkeypatch.setattr('videotrans.task._adaptive_audio.render_adaptive_audio', render)
    Task().align()
    assert len(seen) == 1
    assert seen[0]['allow_speed'] is True
    assert seen[0]['video_duration_ms'] == 1000


@pytest.mark.parametrize('role,key', [
    ('ar-SA-HamedNeural', 'adaptive_hamed_speed_pct'),
    ('Hamed(Male/SA)', 'adaptive_hamed_speed_pct'),
    ('vi-VN-NamMinhNeural', 'adaptive_namminh_speed_pct'),
    ('NamMinh(Male/VN)', 'adaptive_namminh_speed_pct'),
])
def test_selected_voice_limits(role, key, monkeypatch):
    from videotrans.task._adaptive_audio import adaptive_speed_limit
    monkeypatch.setattr(settings, key, 20)
    assert adaptive_speed_limit(True, role) == 1.2
    assert adaptive_speed_limit(False, role) == 1


def test_other_voice_can_exceed_both_old_limits(tmp_path, monkeypatch):
    from videotrans.task._adaptive_audio import adaptive_speed_limit
    monkeypatch.setattr(settings, 'max_audio_speed_rate', 1.05)
    monkeypatch.setattr(settings, 'adaptive_soft_speed_pct', 5, raising=False)
    source = tmp_path / 'source.wav'
    _tone(source, 3000)
    item = {**_queue(source, 0, 1000), 'role': 'en-US-GuyNeural'}
    assert math.isinf(adaptive_speed_limit(True, item['role']))
    assert adaptive_speed_limit(False, item['role']) == 1
    render_adaptive_audio(queue_tts=[item], target_audio=str(tmp_path / 'final.wav'),
                          cache_folder=str(tmp_path), video_duration_ms=1000, allow_speed=True)
    report = json.loads((tmp_path / 'adaptive_audio/fit_report.json').read_text())
    assert report['clips'][0]['speed'] >= 3
    assert report['clips'][0]['actual_ms'] <= 1000
    assert report['clips'][0]['max_speed'] is None


def test_mixed_voices_share_time_without_exceeding_restricted_voice():
    slots = _audio_slots([1200, 3000], 2000, [1.2, math.inf])
    assert sum(slots) == 2000
    assert 1200 / slots[0] <= 1.2
    assert 3000 / slots[1] > 1.2
    assert _audio_slots([1200, 3000], 1000, [1.2, math.inf]) is None
