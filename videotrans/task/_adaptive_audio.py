"""Render a fixed adaptive timeline from measured audio, without truncation."""
import json
import math
import wave
from pathlib import Path

from videotrans.configure.config import settings
from videotrans.util.help_ffmpeg import runffmpeg

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2


def adaptive_speed_limit(enabled: bool, role: str = '') -> float:
    """Only the two selected voices have an additional acceleration ceiling."""
    if not enabled:
        return 1.0
    name = str(role).split('(', 1)[0].strip().lower()
    keys = {
        'hamed': 'adaptive_hamed_speed_pct',
        'ar-sa-hamedneural': 'adaptive_hamed_speed_pct',
        'namminh': 'adaptive_namminh_speed_pct',
        'vi-vn-namminhneural': 'adaptive_namminh_speed_pct',
    }
    key = keys.get(name)
    if key is None:
        return math.inf
    percentage = float(settings.get(key, 20))
    if not math.isfinite(percentage) or percentage < 0:
        raise ValueError(f'Invalid acceleration limit: {key}')
    return 1 + percentage / 100


def measured_audio_ms(filename: str) -> int:
    """Round UP decoded sample duration, never substitute a text estimate."""
    import soundfile as sf
    info = sf.info(filename)
    if info.frames <= 0 or info.samplerate <= 0:
        raise RuntimeError(f'Empty or invalid TTS audio: {filename}')
    return math.ceil(info.frames * 1000 / info.samplerate)


def _tempo_filter(factor: float) -> str:
    if factor == 1:
        return 'anull'
    filters = []
    while factor > 2:
        filters.append('atempo=2')
        factor /= 2
    filters.append(f'atempo={factor:.12f}')
    return ','.join(filters)


def _prepare_clip(source: str, output: Path, slot_frames: int, max_speed: float) -> dict:
    """Always process the original file; retries never compound speed changes."""
    source_ms = measured_audio_ms(source)
    slot_ms = slot_frames * 1000 / SAMPLE_RATE
    factor = max(1.0, source_ms / slot_ms)
    if factor > max_speed + 1e-9:
        raise RuntimeError(
            f'TTS needs {(factor - 1) * 100:.2f}% speed-up, '
            f'above the allowed {(max_speed - 1) * 100:.2f}% '
            f'({source_ms}ms/{slot_ms:.2f}ms).'
        )
    for _ in range(3):
        output.unlink(missing_ok=True)
        runffmpeg([
            '-y', '-i', source, '-vn', '-af', _tempo_filter(factor),
            '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS), '-c:a', 'pcm_s16le', str(output),
        ])
        with wave.open(str(output), 'rb') as audio:
            frames = audio.getnframes()
            if frames <= 0:
                raise RuntimeError('Speed processing produced empty audio')
        if frames <= slot_frames:
            return dict(source_ms=source_ms, frames=frames, speed=factor,
                        actual_ms=frames * 1000 / SAMPLE_RATE, slot_ms=slot_ms)
        if factor >= max_speed:
            break
        # atempo is approximate. Use measured output to correct its error,
        # with a small margin, while never exceeding the user's speed limit.
        factor = min(max_speed, factor * frames / slot_frames * 1.002)
    raise RuntimeError(
        f'TTS still exceeds its slot after speed processing '
        f'({frames * 1000 / SAMPLE_RATE:.2f}ms/{slot_ms:.2f}ms). '
        f'Allowed speed-up: {(max_speed - 1) * 100:.2f}%. Audio was not trimmed.'
    )


def _write_silence(output, frames: int) -> None:
    while frames > 0:
        count = min(frames, SAMPLE_RATE)
        output.writeframesraw(b'\0' * count * CHANNELS * SAMPLE_WIDTH)
        frames -= count


def render_adaptive_audio(*, queue_tts, target_audio: str, cache_folder: str,
                          video_duration_ms: int, allow_speed: bool):
    """Publish audio only after every clip and the final PCM length fit exactly."""
    if not queue_tts:
        raise RuntimeError('Adaptive audio queue is empty')
    folder = Path(cache_folder) / 'adaptive_audio'
    folder.mkdir(parents=True, exist_ok=True)
    clips = []
    previous_end = 0
    final_frames = round(video_duration_ms * SAMPLE_RATE / 1000)
    for index, item in enumerate(queue_tts):
        start = round(item['start_time'] * SAMPLE_RATE / 1000)
        end = round(item['end_time'] * SAMPLE_RATE / 1000)
        if start < previous_end or end <= start or (final_frames > 0 and end > final_frames):
            raise RuntimeError(f'Invalid adaptive timeline at subtitle {item["line"]}')
        max_speed = adaptive_speed_limit(allow_speed, item.get('role', ''))
        output = folder / f'{index}.wav'
        try:
            result = _prepare_clip(item['filename'], output, end - start, max_speed)
        except Exception as exc:
            raise RuntimeError(f'Adaptive audio failed at subtitle {item["line"]}: {exc}') from exc
        clips.append(dict(line=item['line'], role=item.get('role', ''),
                          max_speed=max_speed if math.isfinite(max_speed) else None,
                          start_frame=start, end_frame=end,
                          filename=str(output), **result))
        previous_end = end
    if final_frames <= 0:
        final_frames = previous_end

    Path(target_audio).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(target_audio).with_suffix('.adaptive.tmp.wav')
    with wave.open(str(temporary), 'wb') as output:
        output.setparams((CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        cursor = 0
        for clip in clips:
            _write_silence(output, clip['start_frame'] - cursor)
            with wave.open(clip['filename'], 'rb') as audio:
                while chunk := audio.readframes(SAMPLE_RATE):
                    output.writeframesraw(chunk)
            cursor = clip['start_frame'] + clip['frames']
        _write_silence(output, final_frames - cursor)
    with wave.open(str(temporary), 'rb') as output:
        if output.getnframes() != final_frames:
            raise RuntimeError('Adaptive audio final duration verification failed')
    Path(target_audio).parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target_audio)
    (folder / 'fit_report.json').write_text(json.dumps(
        dict(status='verified', sample_rate=SAMPLE_RATE, frames=final_frames, clips=clips),
        ensure_ascii=False, indent=2,
    ), encoding='utf-8')
    return queue_tts
