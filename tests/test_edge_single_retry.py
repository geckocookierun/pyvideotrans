import asyncio
from types import SimpleNamespace

import edge_tts
import pytest
from edge_tts.exceptions import NoAudioReceived
from pydub import AudioSegment
from videotrans.task._base import BaseTask
from videotrans.configure.excepts import DubbingSrtError


@pytest.mark.parametrize('success',[True,False])
def test_retry_discards_partial_audio_and_preserves_output(tmp_path,monkeypatch,success):
    attempts=[]
    class Communicate:
        def __init__(self,**kwargs):attempts.append(kwargs)
        async def stream(self):
            if not success or len(attempts)<2:
                yield dict(type='audio',data=b'partial')
                raise NoAudioReceived('no audio')
            yield dict(type='audio',data=b'complete')
    async def sleep(_):pass
    monkeypatch.setattr(edge_tts,'Communicate',Communicate)
    monkeypatch.setattr(asyncio,'sleep',sleep)
    monkeypatch.setattr(AudioSegment,'from_file',lambda buffer,**kw: b'valid' if buffer.read()==b'complete' else b'')
    target=tmp_path/'out.mp3';target.write_bytes(b'existing')
    task=SimpleNamespace(proxy_str='http://localhost:9999',_exit=lambda:False,signal=lambda **kw:None)
    args=dict(text='Xin chào',voice='vi-VN-NamMinhNeural',rate='+0%',volume='+0%',pitch='+0Hz')
    if success:
        asyncio.run(BaseTask._edgetts_single(task,str(target),args))
        assert target.read_bytes()==b'complete'
        assert len(attempts)==2 and attempts[-1]['proxy'] is None
    else:
        with pytest.raises(DubbingSrtError,match='after 3 attempts'):
            asyncio.run(BaseTask._edgetts_single(task,str(target),args))
        assert target.read_bytes()==b'existing'
        assert len(attempts)==3


def test_punctuation_only_rejected_before_network(tmp_path):
    task=SimpleNamespace()
    with pytest.raises(DubbingSrtError,match='no letters'):
        asyncio.run(BaseTask._edgetts_single(task,str(tmp_path/'out.mp3'),dict(text='...')))
