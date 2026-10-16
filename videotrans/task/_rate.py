"""Measured cue alignment shared by basic translation and third-person narration.

Accelerate speech first, up to the shared ceiling; extend only the corresponding
video interval for any remainder. Source gaps and the tail remain on the timeline.
"""
import copy
import json
import math
import time
import wave
from pathlib import Path

import soundfile as sf

from videotrans.configure.config import settings
from videotrans.util import tools

SAMPLE_RATE = 48000


def tempo_filter(factor):
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Audio tempo must be positive and finite')
    filters = []
    while factor > 2:
        filters.append('atempo=2')
        factor /= 2
    while factor < .5:
        filters.append('atempo=0.5')
        factor /= .5
    filters.append(f'atempo={factor:.12f}' if factor != 1 else 'anull')
    return ','.join(filters)


def audio_ms(filename):
    info = sf.info(filename)
    if info.frames <= 0 or info.samplerate <= 0:
        raise RuntimeError(f'Empty audio: {filename}')
    return math.ceil(info.frames * 1000 / info.samplerate)


def speed_limit():
    value = float(settings.get('max_audio_speed_rate', 1.25))
    # Migrate the former permissive default (50x), including saved config files.
    if value == 50:
        return 1.25
    if not math.isfinite(value) or not 1 <= value <= 2:
        raise ValueError('Maximum audio speed must be between 1 and 2 (1.25 = +25%)')
    return value


def encode_audio(source, output, factor):
    output = Path(output)
    output.unlink(missing_ok=True)
    tools.runffmpeg(['-y', '-i', str(source), '-vn', '-af', tempo_filter(factor),
                     '-ar', str(SAMPLE_RATE), '-ac', '2', '-c:a', 'pcm_s16le', str(output)])
    with wave.open(str(output), 'rb') as audio:
        if audio.getparams()[:3] != (2, 2, SAMPLE_RATE) or audio.getnframes() <= 0:
            raise RuntimeError('Invalid processed PCM audio')
        return audio.getnframes()


def _silence(output, frames):
    while frames > 0:
        n = min(frames, SAMPLE_RATE)
        output.writeframesraw(b'\0' * n * 4)
        frames -= n


def write_timeline_audio(clips, output, final_ms):
    """Copy every processed sample; never trim speech to force a fit."""
    final_frames = math.ceil(final_ms * SAMPLE_RATE / 1000)
    with wave.open(str(output), 'wb') as out:
        out.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        cursor = 0
        for clip in clips:
            start = round(clip['start_time'] * SAMPLE_RATE / 1000)
            end = round(clip['slot_end'] * SAMPLE_RATE / 1000)
            if start < cursor:
                raise RuntimeError('Overlapping aligned speech')
            _silence(out, start - cursor)
            with wave.open(str(clip['processed']), 'rb') as audio:
                frames = audio.getnframes()
                if start + frames > end:
                    raise RuntimeError(f'Subtitle {clip["line"]} still exceeds its video interval')
                while chunk := audio.readframes(SAMPLE_RATE):
                    out.writeframesraw(chunk)
            cursor = start + frames
        if cursor > final_frames:
            raise RuntimeError('Dub audio exceeds the final video duration')
        _silence(out, final_frames - cursor)
    with wave.open(str(output), 'rb') as check:
        if check.getnframes() != final_frames:
            raise RuntimeError('Final audio length verification failed')


def video_frame_count(filename):
    data = json.loads(tools.runffprobe([
        '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames',
        '-of', 'json', str(filename),
    ]))
    return int(data['streams'][0]['nb_frames'])


def concat_video_segments(segments, output):
    """Use the frame timeline, not rounded MP4 durations, for concat offsets."""
    output = Path(output)
    listing = output.with_suffix('.ffconcat')
    lines = ['ffconcat version 1.0']
    cumulative_ms = 0.0
    previous_us = 0
    for segment in segments:
        filename = Path(segment['filename']).resolve().as_posix()
        if '\n' in filename or '\r' in filename:
            raise ValueError('Unsupported newline in video path')
        filename = filename.replace("'", "'\\''")
        cumulative_ms += segment['duration']
        end_us = round(cumulative_ms * 1000)
        # ffconcat stores microseconds: round cumulative boundaries rather than
        # each interval so even thousands of fractional-frame clips cannot drift.
        lines.extend([f"file '{filename}'", f"duration {(end_us - previous_us) / 1000000:.6f}"])
        previous_us = end_us
    listing.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tools.runffmpeg(['-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
                     '-c', 'copy', str(output)], force_cpu=True)
    expected_frames = sum(segment['video_frames'] for segment in segments)
    actual_frames = video_frame_count(output)
    if actual_frames != expected_frames:
        raise RuntimeError(f'Concatenated video lost frames: {actual_frames}/{expected_frames}')
    expected_ms = sum(segment['duration'] for segment in segments)
    actual_ms = tools.get_video_duration(str(output))
    if abs(actual_ms - expected_ms) > 2:
        raise RuntimeError(f'Concatenated video duration mismatch: {actual_ms}/{expected_ms}ms')
    return actual_ms


def map_time(timeline, value):
    """Map an original timestamp through the same piecewise video transformation."""
    for seg in timeline:
        if value <= seg['source_end']:
            ratio = (value - seg['source_start']) / (seg['source_end'] - seg['source_start'])
            return seg['start'] + max(0, ratio) * (seg['end'] - seg['start'])
    return timeline[-1]['end'] if timeline else value


class SpeedRate:
    def __init__(self, *, queue_tts=None, should_videorate=False, should_audiorate=False,
                 uuid=None, novoice_mp4=None, raw_total_time=0, target_audio=None,
                 cache_folder=None, remove_silent_mid=False, align_sub_audio=True):
        self.queue_tts = copy.deepcopy(queue_tts or [])
        self.should_videorate = bool(should_videorate)
        self.should_audiorate = bool(should_audiorate)
        self.uuid = uuid
        self.novoice_mp4 = novoice_mp4
        self.raw_total_time = raw_total_time or 0
        self.target_audio = target_audio
        self.remove_silent_mid = remove_silent_mid
        self.align_sub_audio = align_sub_audio
        self.folder = Path(cache_folder or Path(target_audio).parent) / f'sync-{time.time_ns()}'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.timeline = []

    def _prepare_audio(self, item, index, slot_ms):
        filename = item.get('filename')
        if not filename or not Path(filename).is_file():
            raise RuntimeError(f'Missing TTS audio for subtitle {item.get("line", index + 1)}')
        original = audio_ms(filename)
        ceiling = speed_limit() if self.should_audiorate else 1.0
        factor = min(ceiling, max(1.0, original / slot_ms))
        output = self.folder / f'audio-{index}.wav'
        for attempt in range(4):
            frames = encode_audio(filename, output, factor)
            actual = frames * 1000 / SAMPLE_RATE
            if actual <= slot_ms or factor >= ceiling or attempt == 3:
                break
            factor = ceiling if attempt == 2 else min(ceiling, factor * actual / slot_ms * 1.002)
        if actual > slot_ms and not self.should_videorate:
            raise RuntimeError(
                f'Subtitle {item.get("line", index + 1)} needs {actual / 1000:.3f}s '
                f'after {factor:.3f}x audio speed, but its interval is {slot_ms / 1000:.3f}s. '
                'Enable video slowdown or adjust the TTS speaking rate. Audio was not trimmed.'
            )
        return dict(processed=str(output), frames=frames, speed=factor,
                    source_audio_ms=original, actual_audio_ms=actual)

    def _video_piece(self, segment, index, fps):
        source_ms = segment['source_end'] - segment['source_start']
        target_ms = segment['target_ms']
        count = max(1, math.ceil(target_ms * fps / 1000 - 1e-8))
        duration = count * 1000 / fps
        # A cue boundary can fall inside the final frame (e.g. a 10ms tail
        # in 25fps video). Decode the containing frame instead of seeking past
        # its timestamp and producing an empty segment.
        decode_start = math.floor(segment['source_start'] * fps / 1000 + 1e-8) / fps
        decode_end = math.ceil(segment['source_end'] * fps / 1000 - 1e-8) / fps
        factor = duration / ((decode_end - decode_start) * 1000)
        output = self.folder / f'video-{index}.mp4'
        # CFR gives every segment an exact number of frames; tpad only covers
        # the fractional last frame created by quantizing the requested duration.
        filters = f'setpts={factor:.12f}*(PTS-STARTPTS),fps={fps:.12f},tpad=stop_mode=clone:stop_duration=1'
        tools.runffmpeg(['-y', '-ss', f'{decode_start:.9f}',
                         '-t', f'{decode_end-decode_start:.9f}', '-i', self.novoice_mp4,
                         '-an', '-vf', filters, '-frames:v', str(count),
                         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                         '-pix_fmt', 'yuv420p', '-fps_mode', 'cfr', str(output)], force_cpu=True)
        measured = tools.get_video_duration(str(output))
        if abs(measured - duration) > 2:
            raise RuntimeError(f'Video segment {index + 1} duration mismatch: {measured}/{duration}ms')
        if video_frame_count(output) != count:
            raise RuntimeError(f'Video segment {index + 1} frame count mismatch')
        segment['video_frames'] = count
        segment['duration'] = duration
        segment['filename'] = str(output)
        return output

    def run(self):
        if not self.queue_tts:
            return []
        if not self.should_audiorate and not self.should_videorate:
            return self._no_rate()
        if self.novoice_mp4:
            self.raw_total_time = tools.get_video_duration(self.novoice_mp4)
        elif self.should_videorate:
            raise RuntimeError('Video slowdown requires an input video')
        source_end = self.raw_total_time or max(it['end_time'] for it in self.queue_tts)
        cursor = 0
        clips = []
        for i, item in enumerate(self.queue_tts):
            start, end = int(item['start_time']), int(item['end_time'])
            if start < cursor or end <= start or end > source_end:
                raise RuntimeError(f'Invalid or overlapping source interval at subtitle {item.get("line", i + 1)}')
            if start > cursor:
                self.timeline.append(dict(source_start=cursor, source_end=start, target_ms=start-cursor))
            tools.set_process(text=f'Align audio {i + 1}/{len(self.queue_tts)}', uuid=self.uuid)
            try:
                result = self._prepare_audio(item, i, end - start)
            except Exception as exc:
                raise RuntimeError(f'Audio alignment failed at subtitle {item.get("line", i + 1)}: {exc}') from exc
            target = max(end - start, math.ceil(result['actual_audio_ms']))
            # A physical slowdown ceiling prevents accidentally expanding corrupt
            # timestamps into hours. Never silently reduce the required duration.
            if target / (end-start) > float(settings.get('max_video_pts_rate', 10)):
                raise RuntimeError(f'Subtitle {item.get("line", i + 1)} exceeds the video slowdown limit')
            segment = dict(source_start=start, source_end=end, target_ms=target, cue=i)
            self.timeline.append(segment)
            clips.append({**item, **result})
            cursor = end
        if cursor < source_end:
            self.timeline.append(dict(source_start=cursor, source_end=source_end, target_ms=source_end-cursor))
        changed = any(s['target_ms'] > s['source_end']-s['source_start'] for s in self.timeline)
        merged = None
        if changed:
            fps = float(tools.get_video_info(self.novoice_mp4, video_fps=True))
            if not math.isfinite(fps) or fps <= 0:
                raise RuntimeError('Invalid source video frame rate')
            for i, seg in enumerate(self.timeline):
                tools.set_process(text=f'Align video {i + 1}/{len(self.timeline)}', uuid=self.uuid)
                self._video_piece(seg, i, fps)
            merged = self.folder / 'merged.mp4'
            merged_duration = concat_video_segments(self.timeline, merged)
        cursor = 0.0
        for seg in self.timeline:
            seg['start'] = cursor
            cursor += seg.get('duration', seg['target_ms'])
            seg['end'] = cursor
            if 'cue' in seg:
                clip = clips[seg['cue']]
                clip['start_time'] = round(seg['start'])
                clip['slot_end'] = round(seg['end'])
                # Millisecond cue rounding must not create a one-sample overflow.
                if clip['start_time'] + clip['actual_audio_ms'] > clip['slot_end']:
                    clip['start_time'] = math.floor(seg['start'])
                clip['end_time'] = math.ceil(clip['start_time'] + clip['actual_audio_ms'])
        if merged:
            cursor = max(cursor, merged_duration)
        self.raw_total_time = math.ceil(cursor)
        target = Path(self.target_audio)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'{target.stem}-{time.time_ns()}.tmp.wav')
        write_timeline_audio(clips, temporary, self.raw_total_time)
        # No media is replaced until both complete outputs have been verified.
        if merged:
            import shutil
            backup = self.folder / 'original-video.mp4'
            shutil.move(self.novoice_mp4, str(backup))
            try:
                shutil.move(str(merged), self.novoice_mp4)
                temporary.replace(target)
            except BaseException:
                shutil.move(str(backup), self.novoice_mp4)
                raise
            backup.unlink()
        else:
            temporary.replace(target)
        (self.folder / 'fit_report.json').write_text(json.dumps(dict(
            status='verified', duration_ms=self.raw_total_time, timeline=self.timeline, clips=clips,
        ), ensure_ascii=False, indent=2), encoding='utf-8')
        for item, clip in zip(self.queue_tts, clips):
            item.update(start_time=clip['start_time'], end_time=clip['end_time'])
        if not changed:
            self.timeline = []  # Source tracks require no transformation.
        return self.queue_tts

    def _no_rate(self):
        """Preserve the original no-adjustment behavior; it may shift long cues."""
        clips = []
        cursor = 0
        for i, item in enumerate(self.queue_tts):
            original_start = item['start_time']
            start = cursor if self.remove_silent_mid else max(cursor, original_start)
            path = self.folder / f'audio-{i}.wav'
            frames = encode_audio(item['filename'], path, 1)
            end = start + math.ceil(frames * 1000 / SAMPLE_RATE)
            clips.append(dict(line=item.get('line', i+1), start_time=start, slot_end=end, processed=str(path)))
            if self.align_sub_audio:
                item.update(start_time=start, end_time=end)
            cursor = end
        target = Path(self.target_audio)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'{target.stem}-{time.time_ns()}.tmp.wav')
        write_timeline_audio(clips, temporary, max(cursor, self.raw_total_time))
        temporary.replace(target)
        return self.queue_tts


class TtsSpeedRate(SpeedRate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_videorate = False


def retime_source_audio(source, timeline, cache_folder):
    """Transform original vocal/BGM with the video map, leaving source files intact."""
    if not timeline:
        return source
    folder = Path(cache_folder) / f'underlay-{time.time_ns()}'
    folder.mkdir(parents=True, exist_ok=True)
    # Normalize and pad missing source tail with silence, never stretch the entire
    # track globally: events must remain attached to their own video intervals.
    normalized = folder / 'source.wav'
    tools.runffmpeg(['-y', '-i', str(source), '-af',
                     f'apad=whole_dur={timeline[-1]["source_end"] / 1000:.9f}',
                     '-ar', str(SAMPLE_RATE), '-ac', '2', '-c:a', 'pcm_s16le', str(normalized)])
    clips = []
    for i, seg in enumerate(timeline):
        raw = folder / f'raw-{i}.wav'
        processed = folder / f'processed-{i}.wav'
        source_ms = seg['source_end'] - seg['source_start']
        start, end = round(seg['start']), round(seg['end'])
        target_ms = end - start
        tools.runffmpeg(['-y', '-ss', f'{seg["source_start"] / 1000:.9f}',
                         '-i', str(normalized), '-t', f'{source_ms / 1000:.9f}',
                         '-ar', str(SAMPLE_RATE), '-ac', '2', '-c:a', 'pcm_s16le', str(raw)])
        factor = audio_ms(raw) / target_ms
        for _ in range(4):
            frames = encode_audio(raw, processed, factor)
            if frames <= target_ms * SAMPLE_RATE / 1000:
                break
            factor *= frames / (target_ms * SAMPLE_RATE / 1000) * 1.001
        else:
            raise RuntimeError(f'Original audio could not match video segment {i+1}')
        clips.append(dict(line=i+1, start_time=start, slot_end=end, processed=str(processed)))
    output = folder / 'aligned.wav'
    write_timeline_audio(clips, output, math.ceil(timeline[-1]['end']))
    return str(output)
