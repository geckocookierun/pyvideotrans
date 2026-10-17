from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from videotrans.task import _stage_dubbing as m
from videotrans.translator._base import BaseTrans
from videotrans.task.taskcfg import SrtItem


@pytest.mark.parametrize('text',['...', '…', '؟', ' — ♪ ', '\u200f…'])
def test_nonverbal_cue_has_exact_silence(tmp_path,text):
    row=dict(text=text,start_time=3926224,end_time=3927312,filename=str(tmp_path/'silence.wav'),ref_wav='unused')
    assert m.prepare_silent_cues([row])==[]
    data,rate=sf.read(row['filename'])
    assert len(data)==17408 and rate==16000
    assert not np.any(data)
    assert row['text']==text and row['start_time']==3926224
    assert 'ref_wav' not in row


@pytest.mark.parametrize('text',['了','أنت','xin chào','١٢٣','123','あ','… نعم'])
def test_letters_and_numbers_still_require_tts(tmp_path,text):
    row=dict(text=text,filename=str(tmp_path/'audio.wav'))
    assert m.prepare_silent_cues([row])==[row]
    assert not (tmp_path/'audio.wav').exists()


def test_pipeline_keeps_silent_cue_and_propagates_speech_error(tmp_path,monkeypatch):
    rows=[dict(text='...',start_time=1000,end_time=2000,filename=str(tmp_path/'silence.wav')),
          dict(text='أنت',start_time=3000,end_time=4000,filename=str(tmp_path/'speech.wav'))]
    task=m.DubbingMixin();task.should_dubbing=True;task.uuid='test'
    task.cfg=SimpleNamespace(target_language_code='ar',tts_type=0,is_cuda=False,target_sub='unused')
    monkeypatch.setattr(task,'_build_tts_queue',lambda:rows)
    def fail(**kwargs):
        assert kwargs['queue_tts']==[rows[1]]
        assert len(task.queue_tts)==2
        raise RuntimeError('real speech failed')
    monkeypatch.setattr(m,'run_tts',fail)
    with pytest.raises(RuntimeError,match='real speech failed'):
        task._tts()


def test_prompt_changes_invalidate_translation_cache():
    model=BaseTrans(text_list=[SrtItem(text='source')])
    model.prompt='old';key=model._get_key('source')
    model.prompt='new'
    assert model._get_key('source')!=key
