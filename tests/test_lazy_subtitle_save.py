import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from videotrans.util._subtitle_edit import edited_cue
from videotrans.util._srt_parse import srt_str_to_listdict


@pytest.mark.parametrize('name,startcol,endcol,textcol',[('recogn',2,3,5),('recogn2',2,3,5),('role',5,6,8),('editdubb',3,4,6)])
def test_save_preserves_all_unrendered_rows(tmp_path,name,startcol,endcol,textcol):
    # Execute the actual save method with table/media stubs; no Qt event loop.
    path=Path(__file__).parents[1]/f'videotrans/component/onlyone_set_{name}.py'
    tree=ast.parse(path.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='save_and_close')
    source=tmp_path/'sub.srt'
    rows=[dict(line=i+1,start_time=i*2000,end_time=i*2000+1000,text=f'cue {i}',spk='',filename=str(tmp_path/f'{i}.wav')) for i in range(1479)]
    def cell(row,col):
        if row>=700:return None
        values={startcol:str(row*2),endcol:str(row*2+1),textcol:f'edited {row}'}
        return SimpleNamespace(text=lambda:values.get(col,''))
    noop=lambda *args,**kwargs:None
    state=SimpleNamespace(onlyone_target_sub=str(source),line_roles={})
    env=dict(edited_cue=edited_cue,Path=Path,app_cfg=state,REDUBB_STATUS_FILE=str(tmp_path/'status'),
             json=__import__('json'),logger=SimpleNamespace(exception=noop,error=noop))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
    check=SimpleNamespace(isChecked=lambda:True)
    obj=SimpleNamespace(stop_countdown=noop,_release_media=noop,accept=noop,save_button=SimpleNamespace(setDisabled=noop),
        table=SimpleNamespace(item=cell),display_data=rows,queue_tts=rows,source_sub=str(source),target_sub=str(source),
        speakers={},cache_folder=str(tmp_path),video_autorate=check,voice_autorate=check,remove_silent_mid=check,align_sub_audio=check)
    env['save_and_close'](obj)
    saved=srt_str_to_listdict(source.read_text())
    assert len(saved)==1479
    assert saved[699]['text']=='edited 699'
    assert saved[700]['text']=='cue 700'
    assert saved[-1]['end_time']==rows[-1]['end_time']
