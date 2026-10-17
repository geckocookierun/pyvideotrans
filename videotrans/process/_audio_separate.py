# -*- coding: utf-8 -*-
"""Bounded-memory source separation with resumable, sample-exact chunks."""
import hashlib
import json
import time
import tempfile
import traceback
from pathlib import Path

from videotrans.configure.config import ROOT_DIR, TEMP_ROOT, logger, settings
from videotrans.process._audio_utils import _write_log


def _separate_chunks(input_file, vocal_file, instr_file, factory, model_id,
                     logs_file=None, chunk_seconds=60, context_seconds=2):
    import numpy as np
    import soundfile as sf

    source = Path(input_file)
    info = sf.info(source)
    if not info.frames or info.channels != 2:
        raise ValueError('Separation requires non-empty stereo audio')
    # Content identity survives recreation of task directories on restart.
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    digest.update(json.dumps([model_id, chunk_seconds, context_seconds, 1]).encode())
    cache = Path(TEMP_ROOT, 'separation_cache', digest.hexdigest())
    cache.mkdir(parents=True, exist_ok=True)
    chunk = round(chunk_seconds * info.samplerate)
    context = round(context_seconds * info.samplerate)
    total = (info.frames + chunk - 1) // chunk
    separator = None
    output_paths = [Path(vocal_file), Path(instr_file)]
    partials = [p.with_name(p.name + '.partial.wav') for p in output_paths]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sf.SoundFile(source) as audio, sf.SoundFile(
                partials[0], 'w', samplerate=info.samplerate, channels=2, subtype='PCM_16') as vocal, sf.SoundFile(
                partials[1], 'w', samplerate=info.samplerate, channels=2, subtype='PCM_16') as music:
            for index, start in enumerate(range(0, info.frames, chunk)):
                end = min(start + chunk, info.frames)
                files = [cache / f'{index:06d}-{kind}.wav' for kind in ('vocal', 'music')]
                def valid(path):
                    try:
                        meta = sf.info(path)
                        return (meta.frames, meta.samplerate, meta.channels) == (end-start, info.samplerate, 2)
                    except (RuntimeError, OSError):
                        return False
                _write_log(logs_file, f'Separating audio: {index + 1}/{total}')
                if not all(valid(p) for p in files):
                    if separator is None:
                        separator = factory()
                    left, right = max(0, start-context), min(info.frames, end+context)
                    audio.seek(left)
                    samples = audio.read(right-left, dtype='float32', always_2d=True).T
                    # Some models round output down to complete processing frames.
                    # Supply real context or zero padding beyond EOF, then retain
                    # exactly the source interval (including its final samples).
                    samples = np.pad(samples, ((0, 0), (0, end + context - right)))
                    samples = np.ascontiguousarray(samples)
                    output = separator.process(sample_rate=info.samplerate, samples=samples)
                    if output.sample_rate != info.samplerate or len(output.stems) != 2:
                        raise RuntimeError('Unexpected separation output format')
                    order = (0, 1) if model_id[0].startswith('spleeter') else (1, 0)
                    for path, stem in zip(files, order):
                        data = np.asarray(output.stems[stem].data).T
                        data = data[start-left:end-left]
                        if data.shape != (end-start, 2) or not np.isfinite(data).all():
                            raise RuntimeError('Incomplete or invalid separated audio chunk')
                        with tempfile.NamedTemporaryFile(dir=cache, suffix='.wav', delete=False) as handle:
                            temporary = Path(handle.name)
                        try:
                            sf.write(temporary, data, info.samplerate, subtype='FLOAT')
                            temporary.replace(path)
                        finally:
                            temporary.unlink(missing_ok=True)
                    del output, samples, data
                for path, destination in zip(files, (vocal, music)):
                    data, _ = sf.read(path, dtype='float32', always_2d=True)
                    destination.write(data)
                _write_log(logs_file, f'Separated audio: {index + 1}/{total}')
        for partial, destination in zip(partials, output_paths):
            if sf.info(partial).frames != info.frames:
                raise RuntimeError('Separated audio length mismatch')
        for partial, destination in zip(partials, output_paths):
            partial.replace(destination)
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)


def vocal_bgm(*, input_file, vocal_file, instr_file, logs_file=None,
              uvr_models='UVR-MDX-NET-Inst_HQ_4', **kw):
    import sherpa_onnx
    started = time.monotonic()
    try:
        if uvr_models.startswith('spleeter'):
            files = [Path(ROOT_DIR, 'models/onnx', name) for name in
                     ('vocals.fp16.onnx', 'accompaniment.fp16.onnx')]
            model = dict(spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
                vocals=str(files[0]), accompaniment=str(files[1])))
        else:
            files = [Path(ROOT_DIR, 'models/onnx', uvr_models + '.onnx')]
            model = dict(uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(files[0])))
        identity = [uvr_models, [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]]
        def factory():
            config = sherpa_onnx.OfflineSourceSeparationConfig(
                model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                    **model, num_threads=int(settings.get('noise_separate_nums', 4)),
                    debug=False, provider='cpu'))
            if not config.validate():
                raise ValueError('Invalid source separation model configuration')
            return sherpa_onnx.OfflineSourceSeparation(config)
        _write_log(logs_file, 'Preparing chunked audio separation')
        _separate_chunks(input_file, vocal_file, instr_file, factory, identity, logs_file)
        logger.info('Separated audio in %.1fs', time.monotonic()-started)
        return True, None
    except Exception as error:
        logger.exception('Audio separation failed')
        return False, f'{error}\n{traceback.format_exc()}'


def vocal_bgm_spleeter(*, input_file, vocal_file, instr_file, logs_file=None, **kw):
    return vocal_bgm(input_file=input_file, vocal_file=vocal_file, instr_file=instr_file,
                     logs_file=logs_file, uvr_models='spleeter', **kw)
