from types import SimpleNamespace
from pathlib import Path

import pytest
from PIL import Image
from videotrans.task import _stage_assemble as mod
from videotrans.util import tools


@pytest.mark.parametrize('hardware_failure',[False,True])
def test_both_overlays_use_one_encode_and_keep_duration(tmp_path,monkeypatch,hardware_failure):
    src=tmp_path/'source.mp4';logo=tmp_path/'logo.png'
    Image.new('RGB',(20,20),'red').save(logo)
    tools.runffmpeg(['-y','-f','lavfi','-i','color=c=blue:size=160x90:rate=25',
                    '-t','2','-c:v','libx264',str(src)])
    task=mod.AssembleMixin()
    task.cfg=SimpleNamespace(cache_folder=str(tmp_path),embed_logo=True,logo_path=str(logo),
        floating_channel=True,channel_name='Test channel',logo_size='20x20',logo_pos=0,logo_margin=3)
    task.video_codec_num=264;task.uuid='test';task.hasend=False
    task.signal=lambda **kwargs:None
    monkeypatch.setattr(mod.platform,'system',lambda:'Darwin' if hardware_failure else 'Linux')
    monkeypatch.setattr(mod.settings,'force_lib',False)
    monkeypatch.setattr(task,'_get_hard_cfg',lambda **kw:([], '', 'h264_videotoolbox', ['-q:v','67']))
    task._exit=lambda:False
    real=mod.runffmpeg;calls=[]
    def run(cmd,**kw):
        calls.append(cmd)
        if hardware_failure and len(calls)==1:
            raise RuntimeError('hardware encoder unavailable')
        return real(cmd,**kw)
    monkeypatch.setattr(mod,'runffmpeg',run)
    task._apply_branding(str(src))
    assert len(calls)==(2 if hardware_failure else 1)
    graph=calls[0][calls[0].index('-filter_complex')+1]
    assert 'overlay=' in graph and 'drawtext=' in graph
    assert abs(tools.get_video_duration(str(src))-2000)<=1
    assert not list(tmp_path.glob('*.progress'))
