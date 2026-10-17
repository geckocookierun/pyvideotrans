from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from videotrans.process import _audio_separate as mod


def test_chunk_resume_exact_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'TEMP_ROOT', str(tmp_path))
    rate = 1000
    samples = np.random.default_rng(3).uniform(-.5, .5, (2501, 2)).astype('float32')
    source = tmp_path/'source.wav'
    sf.write(source, samples, rate, subtype='FLOAT')
    calls = []
    fail = True
    class Separator:
        def process(self, sample_rate, samples):
            calls.append(samples.shape[1])
            if len(calls) == 2 and fail:
                raise RuntimeError('worker failed')
            return SimpleNamespace(sample_rate=sample_rate, stems=[
                SimpleNamespace(data=samples), SimpleNamespace(data=-samples)])
    args = (source, tmp_path/'v.wav', tmp_path/'i.wav', Separator, ['spleeter'])
    with pytest.raises(RuntimeError):
        mod._separate_chunks(*args, chunk_seconds=1, context_seconds=.1)
    assert not (tmp_path/'v.wav').exists()
    fail = False
    mod._separate_chunks(*args, chunk_seconds=1, context_seconds=.1)
    # First completed chunk is reused, rather than inferred twice.
    assert calls == [1100, 1200, 1200, 701]
    for name, expected in [('v.wav', samples), ('i.wav', -samples)]:
        actual, sr = sf.read(tmp_path/name)
        assert sr == rate and actual.shape == samples.shape
        np.testing.assert_allclose(actual, expected, atol=1/32768)
    mod._separate_chunks(*args, chunk_seconds=1, context_seconds=.1)
    assert len(calls) == 4


def test_short_model_output_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'TEMP_ROOT', str(tmp_path))
    source = tmp_path/'in.wav'
    sf.write(source, np.zeros((2000,2)), 1000)
    class Bad:
        def process(self, sample_rate, samples):
            return SimpleNamespace(sample_rate=sample_rate, stems=[
                SimpleNamespace(data=samples[:,:1]),SimpleNamespace(data=samples[:,:1])])
    with pytest.raises(RuntimeError, match='Incomplete'):
        mod._separate_chunks(source,tmp_path/'v.wav',tmp_path/'i.wav',Bad,['spleeter'])
    assert not (tmp_path/'v.wav').exists()


def test_model_rounding_cannot_drop_source_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'TEMP_ROOT', str(tmp_path))
    samples=np.full((2501,2),.25,dtype='float32')
    source=tmp_path/'in.wav';sf.write(source,samples,1000,subtype='FLOAT')
    class Rounded:
        def process(self,sample_rate,samples):
            data=samples[:,:samples.shape[1]//64*64]
            return SimpleNamespace(sample_rate=sample_rate,stems=[SimpleNamespace(data=data)]*2)
    mod._separate_chunks(source,tmp_path/'v.wav',tmp_path/'i.wav',Rounded,['spleeter'],
                         chunk_seconds=1,context_seconds=.1)
    data,sr=sf.read(tmp_path/'v.wav')
    np.testing.assert_array_equal(data,samples)
