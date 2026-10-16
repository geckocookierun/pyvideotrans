from pathlib import Path
from types import SimpleNamespace

import pytest

from videotrans.configure._app_params import AppParams
from videotrans.configure._app_settings import AppSettings
from videotrans.task._stage_align import AlignMixin
from videotrans.translator import _narration


@pytest.mark.parametrize('old,new',[('adaptive','none'),('first','none'),('line','none'),('none','none'),('third','third')])
def test_saved_modes_migrate_to_two_supported_modes(old,new):
    params=object.__new__(AppParams)
    params._apply_dict({'narration_mode':old,'narration_style':'first','third_person_narration':True})
    assert params.narration_mode==new
    assert params.narration_style=='third'
    assert params.third_person_narration==(new=='third')


def test_settings_migrate_old_limit_and_remove_voice_specific_keys():
    settings=object.__new__(AppSettings)
    settings._apply_dict({'max_audio_speed_rate':50,'adaptive_hamed_speed_pct':20})
    assert settings.get('max_audio_speed_rate')==1.25
    assert not hasattr(settings,'adaptive_hamed_speed_pct')


@pytest.mark.parametrize('mode',['none','third'])
def test_modes_share_alignment_and_save_new_timestamps(tmp_path,monkeypatch,mode):
    from videotrans.task import _rate
    seen=[]
    timeline=[dict(source_start=0,source_end=1000,start=0,end=2000)]
    class Rate:
        def __init__(self,**kwargs):
            seen.append(kwargs)
            self.timeline=timeline
        def run(self):
            return [dict(line=1,text='hello',start_time=0,end_time=1900)]
    monkeypatch.setattr(_rate,'SpeedRate',Rate)
    class Task(AlignMixin):
        precent=0
        uuid='test'
        should_dubbing=True
        ignore_align=False
        is_audio_trans=True
        video_time=1000
        queue_tts=[dict(line=1,text='hello',start_time=0,end_time=1000)]
        cfg=SimpleNamespace(app_mode='biaozhun',narration_mode=mode,novoice_mp4='',
            voice_autorate=True,video_autorate=False,align_sub_audio=True,
            remove_silent_mid=False,target_wav=str(tmp_path/'out.wav'),cache_folder=str(tmp_path),
            fix_punc=0,target_sub=str(tmp_path/'out.srt'),tts_type=0,volume='+0%')
        def _exit(self): return False
        def signal(self,**kwargs): pass
        def _save_srt_target(self,rows,filename): self.saved=rows
    task=Task();task.align()
    assert len(seen)==1
    assert seen[0]['should_audiorate'] is True
    assert task.sync_timeline==timeline
    assert task.saved[0]['endraw']=='00:00:01,900'


def test_third_person_preserves_over_budget_content_without_extra_calls(monkeypatch):
    from videotrans.translator import _openaicompat, _runner
    calls=[]
    text='She refuses the marriage. Her father asks her to reconsider. She still refuses.'
    class Model:
        ainame='test'
        is_test=True
        def _item_task(self,value):
            calls.append(value)
            assert 'third-person' in self.prompt
            assert '2–4 seconds' in self.prompt
            assert self.prompt.replace('{batch_input}', value).count(value) == 1
            return text
    monkeypatch.setattr(_openaicompat,'OpenAICampat',Model)
    monkeypatch.setattr(_runner,'get_model_transobj',lambda **kwargs:Model())
    monkeypatch.setattr(_narration,'get_source_target_code',lambda **kwargs:('Chinese','English'))
    monkeypatch.setattr(_narration,'probe_target_sec_per_unit',lambda **kwargs:dict(sec_per_unit=.5,unit='words'))
    channel=next(iter(_narration.AI_TRANS_CHANNELS))
    result=_narration.run_third_person_narration(
        text_list=[dict(text='她拒绝这门婚事。',start_time=0,end_time=1000)],translate_type=channel,
        source_code='zh',target_code='en')
    assert len(calls)==1
    assert ' '.join(r['text'] for r in result)==text
    assert all(r['end_time']>r['start_time'] for r in result)
    assert all(a['end_time']<=b['start_time'] for a,b in zip(result,result[1:]))


def test_original_audio_retiming_failure_is_not_hidden(tmp_path,monkeypatch):
    from videotrans.task._stage_audio import AudioMixin
    from videotrans.task import _rate
    def fail(*args): raise RuntimeError('cannot retime original')
    monkeypatch.setattr(_rate,'retime_source_audio',fail)
    task=AudioMixin()
    task.sync_timeline=[dict(source_start=0,source_end=1000,start=0,end=1500)]
    task.cfg=SimpleNamespace(cache_folder=str(tmp_path))
    with pytest.raises(RuntimeError,match='cannot retime original'):
        task._retime_original_track('source.wav')
