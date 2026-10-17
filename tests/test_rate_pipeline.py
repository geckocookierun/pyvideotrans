import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from videotrans.configure.config import settings
from videotrans.task._rate import SpeedRate, audio_ms, map_time, retime_source_audio, speed_limit
from videotrans.util import tools


def tone(path, duration, hz=440):
    t = np.arange(round(duration * 16000 / 1000)) / 16000
    # A distinct final tone verifies that the last spoken portion survives.
    freq = np.where(t > duration / 1000 * .9, hz * 2, hz)
    sf.write(path, .25 * np.sin(2 * np.pi * freq * t), 16000)
    return str(path)


def video(path, seconds=4, fps='25'):
    tools.runffmpeg(['-y', '-f', 'lavfi', '-i', f'testsrc2=size=160x90:rate={fps}',
                     '-t', str(seconds), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)])
    return str(path)


def queue(path, start=1000, end=2000, line=1):
    return dict(line=line, filename=str(path), text='test', start_time=start, end_time=end)


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(settings, 'max_audio_speed_rate', 1.25)
    monkeypatch.setattr(tools, 'set_process', lambda **kwargs: None)


def run(tmp_path, cues, *, audio=True, slow=True, fps='25', seconds=4):
    src = video(tmp_path / 'video.mp4', seconds, fps)
    obj = SpeedRate(queue_tts=cues, should_audiorate=audio, should_videorate=slow,
                    novoice_mp4=src, target_audio=str(tmp_path/'dub.wav'), cache_folder=str(tmp_path))
    result = obj.run()
    return obj, result


@pytest.mark.parametrize('audio,slow,expected', [(True, True, 1.25), (False, True, 1.0)])
def test_overflow_extends_video_after_audio_limit(tmp_path, audio, slow, expected):
    source = tone(tmp_path/'tts.wav', 1800)
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    obj, result = run(tmp_path, [queue(source), queue(tone(tmp_path/'next.wav',100),2000,2100,2)], audio=audio, slow=slow)
    report = json.loads((obj.folder/'fit_report.json').read_text())
    clip = report['clips'][0]
    assert clip['speed'] == expected
    assert clip['actual_audio_ms'] <= clip['slot_end']-clip['start_time']
    assert result[0]['start_time'] == 1000
    assert tools.get_video_duration(obj.novoice_mp4) > 4000
    assert abs(audio_ms(obj.target_audio)-tools.get_video_duration(obj.novoice_mp4)) <= 1
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    assert obj.timeline[-1]['source_end'] == 4000  # tail retained
    assert 1900 <= obj.timeline[-1]['end']-obj.timeline[-1]['start'] < 1940
    samples, sr = sf.read(clip['processed'])
    tail = samples[-int(sr*.08):,0]
    crossings = np.sum((tail[:-1] < 0) & (tail[1:] >= 0))
    assert 60 < crossings < 80  # 880Hz tail marker survives


def test_audio_only_fits_without_changing_video(tmp_path):
    source = tone(tmp_path/'tts.wav', 1100)
    obj, result = run(tmp_path, [queue(source)], slow=False)
    assert tools.get_video_duration(obj.novoice_mp4) == 4000
    assert not obj.timeline
    assert result[0]['start_time'] == 1000
    assert 2000 < result[0]['end_time'] <= 2200
    assert audio_ms(obj.target_audio) == 4000


def test_audio_limit_failure_does_not_publish(tmp_path):
    source = tone(tmp_path/'tts.wav', 1800)
    target = tmp_path/'dub.wav'
    target.write_bytes(b'previous output')
    with pytest.raises(RuntimeError, match='Enable video slowdown'):
        run(tmp_path, [queue(source)], slow=False, seconds=2)
    assert target.read_bytes() == b'previous output'


def test_short_audio_preserves_cue_gaps_and_video(tmp_path):
    source = tone(tmp_path/'tts.wav', 200)
    obj, result = run(tmp_path, [queue(source, 500, 1000), queue(source, 2000, 2500, 2)])
    assert not obj.timeline
    assert [r['start_time'] for r in result] == [500,2000]
    samples,sr = sf.read(obj.target_audio)
    assert not np.any(samples[1000*48:2000*48])
    assert len(samples)/sr == 4


def test_fractional_fps_multiple_cues_has_no_accumulating_drift(tmp_path):
    source = tone(tmp_path/'tts.wav', 700)
    cues = [queue(source, 100+i*700, 600+i*700, i+1) for i in range(5)]
    obj, result = run(tmp_path, cues, fps='30000/1001')
    assert abs(audio_ms(obj.target_audio)-tools.get_video_duration(obj.novoice_mp4)) <= 2
    for a,b in zip(result,result[1:]):
        assert a['end_time'] <= b['start_time']
    for item,seg in zip(result,[s for s in obj.timeline if 'cue' in s]):
        assert abs(item['start_time']-seg['start']) <= 1
        assert item['end_time'] <= math.ceil(seg['end'])


def test_original_track_uses_same_piecewise_map(tmp_path):
    source = tone(tmp_path/'tts.wav', 1800)
    obj,_ = run(tmp_path,[queue(source),queue(tone(tmp_path/'next.wav',100),2000,2100,2)])
    sr=16000
    samples=np.zeros(4*sr)
    # Event after the elongated scene must move by that scene's added time.
    t=np.arange(int(.2*sr))/sr
    samples[3*sr:3*sr+len(t)]=.3*np.sin(2*np.pi*600*t)
    raw=tmp_path/'original.wav'
    sf.write(raw,samples,sr)
    aligned=retime_source_audio(str(raw),obj.timeline,str(tmp_path))
    result,sr=sf.read(aligned)
    onset=np.flatnonzero(np.abs(result[:,0])>.05)[0]*1000/sr
    assert abs(onset-map_time(obj.timeline,3000)) < 30
    assert abs(audio_ms(aligned)-audio_ms(obj.target_audio)) <= 1


@pytest.mark.parametrize('start,end',[(1000,1000),(-1,500),(3500,5000)])
def test_invalid_source_timing_is_rejected(tmp_path,start,end):
    source=tone(tmp_path/'tts.wav',100)
    with pytest.raises(RuntimeError,match='Invalid or overlapping'):
        run(tmp_path,[queue(source,start,end)])


def test_missing_audio_is_an_error(tmp_path):
    with pytest.raises(RuntimeError,match='Missing TTS audio'):
        run(tmp_path,[queue(tmp_path/'missing.wav')])


def test_legacy_50x_default_migrates(monkeypatch):
    monkeypatch.setattr(settings,'max_audio_speed_rate',50)
    assert speed_limit()==1.25


def test_encoder_failure_propagates(tmp_path,monkeypatch):
    from videotrans.task import _rate
    source=tone(tmp_path/'tts.wav',100)
    def fail(*args):
        raise RuntimeError('encoder failed')
    monkeypatch.setattr(_rate,'encode_audio',fail)
    with pytest.raises(RuntimeError,match='encoder failed'):
        run(tmp_path,[queue(source)])


def test_reported_scene_36620ms_24910ms(tmp_path):
    source=tone(tmp_path/'tts.wav',36620)
    obj,result=run(tmp_path,[queue(source,0,24910)],seconds=24.92)
    assert 29200 <= tools.get_video_duration(obj.novoice_mp4) <= 29500
    report=json.loads((obj.folder/'fit_report.json').read_text())
    assert report['clips'][0]['speed']==1.25
    assert result[0]['end_time'] <= tools.get_video_duration(obj.novoice_mp4)


def test_video_motion_is_slowed_not_just_frozen_at_end(tmp_path):
    from PIL import Image
    src=tmp_path/'video.mp4'
    tools.runffmpeg(['-y','-f','lavfi','-i',
        "color=c=red:size=160x90:rate=25:duration=4,drawbox=x=0:y=0:w=iw:h=ih:color=blue:t=fill:enable='gte(t,1.5)'",
        '-c:v','libx264',str(src)])
    source=tone(tmp_path/'tts.wav',1800)
    obj=SpeedRate(queue_tts=[queue(source),queue(tone(tmp_path/'next.wav',100),2000,2100,2)],should_audiorate=True,should_videorate=True,
        novoice_mp4=str(src),target_audio=str(tmp_path/'dub.wav'),cache_folder=str(tmp_path))
    obj.run()
    for second,expected in [(1.6,'red'),(1.9,'blue')]:
        frame=tmp_path/f'{second}.png'
        tools.runffmpeg(['-y','-ss',str(second),'-i',str(src),'-frames:v','1',str(frame)])
        r,g,b=Image.open(frame).convert('RGB').getpixel((80,45))
        assert (r>b) == (expected=='red')


def test_encoder_failure_after_video_processing_preserves_previous_outputs(tmp_path,monkeypatch):
    from videotrans.task import _rate
    src=video(tmp_path/'video.mp4')
    before=Path(src).read_bytes()
    target=tmp_path/'dub.wav'
    target.write_bytes(b'previous dub')
    source=tone(tmp_path/'tts.wav',1800)
    def fail(*args):
        raise RuntimeError('final audio failed')
    monkeypatch.setattr(_rate,'write_timeline_audio',fail)
    obj=SpeedRate(queue_tts=[queue(source),queue(tone(tmp_path/'next.wav',100),2000,2100,2)],should_audiorate=True,should_videorate=True,
                  novoice_mp4=src,target_audio=str(target),cache_folder=str(tmp_path))
    with pytest.raises(RuntimeError,match='final audio failed'):
        obj.run()
    assert Path(src).read_bytes()==before
    assert target.read_bytes()==b'previous dub'


def test_concat_uses_exact_frame_durations_across_many_rounded_segments(tmp_path):
    from videotrans.task._rate import concat_video_segments, video_frame_count
    piece = tmp_path / "one frame's video.mp4"
    tools.runffmpeg(['-y', '-f', 'lavfi', '-i', 'color=size=160x90:rate=30',
                    '-frames:v', '1', '-c:v', 'libx264', '-video_track_timescale', '1000', str(piece)])
    assert tools.get_video_duration(str(piece)) == 33
    segments = [dict(filename=str(piece), duration=1000 / 30, video_frames=1) for _ in range(139)]
    output = tmp_path / 'joined.mp4'
    measured = concat_video_segments(segments, output)
    assert abs(measured - 139000 / 30) <= 2
    assert video_frame_count(output) == 139
    # Without explicit frame durations, concat would produce 139 * 33 = 4587ms.
    assert measured > 4620


def test_concat_rejects_frame_loss_even_when_duration_matches(tmp_path, monkeypatch):
    from videotrans.task import _rate
    monkeypatch.setattr(tools, 'runffmpeg', lambda *args, **kwargs: None)
    monkeypatch.setattr(_rate, 'video_frame_count', lambda _: 29)
    monkeypatch.setattr(tools, 'get_video_duration', lambda _: 1000)
    with pytest.raises(RuntimeError, match='lost frames'):
        _rate.concat_video_segments([dict(filename=str(tmp_path/'piece.mp4'), duration=1000,
                                         video_frames=30)], tmp_path/'merged.mp4')


def test_tiny_cue_uses_gap_before_next_speech(tmp_path):
    source=tone(tmp_path/'tts.wav',2010)
    next_audio=tone(tmp_path/'next.wav',500)
    obj,result=run(tmp_path,[queue(source,1000,1160),queue(next_audio,20680,21680,2)],seconds=23)
    assert not obj.timeline
    assert tools.get_video_duration(obj.novoice_mp4)==23000
    assert result[1]['start_time']==20680
    report=json.loads((obj.folder/'fit_report.json').read_text())
    assert report['clips'][0]['speed']==1
    assert 3000 <= result[0]['end_time'] <= 3050


def test_tiny_cue_without_gap_can_slow_beyond_ten_times(tmp_path):
    source=tone(tmp_path/'tts.wav',2400)
    next_audio=tone(tmp_path/'next.wav',100)
    obj,result=run(tmp_path,[queue(source,1000,1160),queue(next_audio,1160,1400,2)],seconds=2)
    assert obj.timeline
    segment=next(s for s in obj.timeline if s.get('cue')==0)
    assert segment['target_ms']/160 > 10
    assert result[0]['end_time'] <= result[1]['start_time']
    assert abs(audio_ms(obj.target_audio)-tools.get_video_duration(obj.novoice_mp4))<=2
