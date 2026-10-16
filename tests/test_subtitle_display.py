import copy
from types import SimpleNamespace

import pytest

from videotrans.util._subtitle_display import split_display_cues
from videotrans.util.help_srt import get_subtitle_from_srt


@pytest.mark.parametrize('text',[
    'Cha đừng làm vậy. Con muốn ở bên chăm sóc cha thêm một thời gian, xin cha hãy suy nghĩ lại.',
    'أبي أرجوك لا تفعل ذلك. أريد البقاء إلى جانبك لبعض الوقت، امنحني فرصة للتفكير.',
    '父亲请不要这样做。我想留在你的身边照顾你，请给我一点时间认真考虑这件事情。',
])
def test_split_preserves_content_bounds_and_input(text):
    rows=[dict(line=1,start_time=1234,end_time=11234,text=text)]
    original=copy.deepcopy(rows)
    result=split_display_cues(rows)
    assert len(result)>1
    assert rows==original
    assert ''.join(''.join(r['text'].split()) for r in result)==''.join(text.split())
    assert result[0]['start_time']==1234
    assert result[-1]['end_time']==11234
    assert all(a['end_time']==b['start_time'] for a,b in zip(result,result[1:]))
    assert all(r['end_time']-r['start_time'] >= 899 for r in result)


def test_short_duration_does_not_create_flashing_cues():
    text='A long sentence that would otherwise create several tiny flashing subtitles.'
    result=split_display_cues([dict(line=1,start_time=1000,end_time=1500,text=text)])
    assert len(result)==1
    assert result[0]['text']==text


def test_does_not_split_unbroken_word_or_remove_original_gap():
    rows=[dict(line=1,start_time=1000,end_time=2000,text='longunbrokenword'),
          dict(line=2,start_time=3000,end_time=4000,text='Next sentence.')]
    result=split_display_cues(rows)
    assert [r['text'] for r in result]==['longunbrokenword','Next sentence.']
    assert result[1]['start_time']==3000


@pytest.mark.parametrize('bilingual',[False,True])
def test_render_only_leaves_tts_subtitle_file_unchanged(tmp_path,monkeypatch,bilingual):
    from videotrans.task._stage_subtitle import SubtitleMixin
    from videotrans.configure.config import settings
    monkeypatch.setattr(settings,'other_len',24)
    target=tmp_path/'target.srt'
    text='One complete sentence. Another complete sentence. The final sentence remains intact.'
    target.write_text(f'1\n00:00:01,000 --> 00:00:11,000\n{text}\n\n2\n00:00:13,000 --> 00:00:14,000\nNext cue.\n')
    source=tmp_path/'source.srt'
    source.write_text('1\n00:00:01,000 --> 00:00:11,000\nFIRST SOURCE\n\n2\n00:00:13,000 --> 00:00:14,000\nSECOND SOURCE\n')
    original=target.read_bytes()
    task=SubtitleMixin()
    task.cfg=SimpleNamespace(target_sub=str(target),source_sub=str(source),cache_folder=str(tmp_path),
        target_dir=str(tmp_path),target_language_code='en',source_language_code='zh',target_language='English',
        narration_mode='none',subtitle_type=4 if bilingual else 2,output_srt=1)
    filename,_=task._process_subtitles()
    result=get_subtitle_from_srt(str(tmp_path/filename))
    assert target.read_bytes()==original
    assert len(result)>2
    assert result[0]['start_time']==1000
    assert result[-2]['end_time']==11000
    assert result[-1]['start_time']==13000
    if bilingual:
        assert all('FIRST SOURCE' in row['text'] for row in result[:-1])
        assert 'SECOND SOURCE' in result[-1]['text']


@pytest.mark.parametrize('text',[
    'Con muốn ở bên chăm sóc cha thêm một thời gian nữa để báo đáp công ơn của cha',
    'أريد البقاء إلى جانبك لبعض الوقت لكي أتمكن من الاعتناء بك',
    '我想留在你的身边照顾你请给我一点时间认真考虑这件事情',
    'The amount is 1,234.56 and the meeting starts at 12:30 on example.com today',
])
def test_long_text_without_pause_is_not_split_even_with_plenty_of_time(text):
    result=split_display_cues([dict(line=1,start_time=0,end_time=20000,text=text)])
    assert len(result)==1
    assert result[0]['text']==text


def test_long_clauses_split_only_at_punctuation_not_character_limit():
    first='Con muốn ở bên chăm sóc cha thêm một thời gian nữa,'
    second='xin cha hãy cho con suy nghĩ thật kỹ về chuyện này.'
    result=split_display_cues([dict(line=1,start_time=1000,end_time=11000,
                                   text=first+' '+second)])
    assert [row['text'] for row in result]==[first,second]
    assert result[0]['end_time']==result[1]['start_time']


def test_short_sentence_splits_at_punctuation():
    text='Vâng, con hiểu rồi.'
    result=split_display_cues([dict(line=1,start_time=0,end_time=5000,text=text)])
    assert [row['text'] for row in result]==['Vâng,', 'con hiểu rồi.']


def test_every_pause_splits_even_when_fragments_are_short_lived():
    text='Vâng, con hiểu; cha yên tâm! Được chứ?'
    result=split_display_cues([dict(line=1,start_time=1000,end_time=2000,text=text)])
    assert [r['text'] for r in result]==['Vâng,','con hiểu;','cha yên tâm!','Được chứ?']
    assert result[0]['start_time']==1000
    assert result[-1]['end_time']==2000
    assert all(a['end_time']==b['start_time'] for a,b in zip(result,result[1:]))
    assert all(r['end_time']>r['start_time'] for r in result)
