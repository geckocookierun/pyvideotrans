import copy
from pathlib import Path
from types import SimpleNamespace

from videotrans.process._source_cues import restore_transcript_punctuation, segment_aligned_source, repair_source_fragments
from videotrans.util.help_srt import get_subtitle_from_srt,get_srt_from_list


def aligned(text,step=.4,gap=.1):
    return [dict(word=c,start=i*(step+gap),end=i*(step+gap)+step) for i,c in enumerate(text)]


def test_long_source_never_splits_compound_words_at_time_limit():
    words=aligned('她纯真善良他们提前铲除障碍宁儿有六品灵根。')
    original=copy.deepcopy(words)
    result=segment_aligned_source([dict(words=words)],'zh')
    assert len(result)==1
    assert result[0]['text']=='她纯真善良他们提前铲除障碍宁儿有六品灵根。'
    assert result[0]['end_time']>6000
    assert words==original


def test_restores_punctuation_stripped_by_aligner_before_splitting():
    words=aligned('她纯真善良他清除障碍')
    restored=restore_transcript_punctuation(words,'她纯真善良，他清除障碍。')
    result=segment_aligned_source([dict(words=restored)],'zh')
    assert [r['text'] for r in result]==['她纯真善良，','他清除障碍。']
    assert result[0]['end_time']==round(words[4]['end']*1000)
    assert result[1]['start_time']==round(words[5]['start']*1000)


def test_alignment_mismatch_preserves_entire_transcript():
    words=aligned('不同')
    result=restore_transcript_punctuation(words,'完整原文，不应丢字。')
    assert result==[dict(word='完整原文，不应丢字。',start=words[0]['start'],end=words[-1]['end'])]


def test_decimal_is_not_a_boundary():
    words=[dict(word='3.',start=0,end=.3),dict(word='14',start=.3,end=.6),dict(word='元。',start=.6,end=1)]
    result=segment_aligned_source([dict(words=words)],'zh')
    assert [r['text'] for r in result]==['3.14元。']


def source():
    return get_subtitle_from_srt('1\n00:00:01,000 --> 00:00:04,000\n提前铲除障\n\n2\n00:00:04,100 --> 00:00:05,000\n碍。\n\n3\n00:00:07,000 --> 00:00:08,000\n下一句。\n',is_file=False)


def test_repair_cached_fragments_preserves_outer_times_text_and_input():
    rows=source();original=copy.deepcopy(rows)
    fixed=repair_source_fragments(rows,'zh')
    assert len(fixed)==2
    assert fixed[0]['text']=='提前铲除障碍。'
    assert (fixed[0]['start_time'],fixed[0]['end_time'])==(1000,5000)
    assert fixed[1]['start_time']==7000
    assert [r['text'] for r in rows]==[r['text'] for r in original]
    assert len(repair_source_fragments(fixed,'zh'))==len(fixed)


def test_different_speakers_do_not_merge():
    rows=source();rows[0]['spk']='sp1';rows[1]['spk']='sp2'
    assert len(repair_source_fragments(rows,'zh'))==3


def test_changed_source_backs_up_old_translation_and_saves_new_boundaries(tmp_path):
    from videotrans.task._stage_recogn import RecognMixin
    from videotrans.recognition._constants import QWENASR
    class Task(RecognMixin):
        should_trans=True
        def _save_srt_target(self,rows,filename):
            Path(filename).write_text(get_srt_from_list(rows))
    task=Task();task.source_srt_list=source()
    src=tmp_path/'zh.srt';dst=tmp_path/'vi.srt'
    src.write_text(get_srt_from_list(task.source_srt_list));dst.write_text('old edited translation')
    task.cfg=SimpleNamespace(recogn_type=QWENASR,detect_language='zh',target_dir=str(tmp_path),
                             source_sub=str(src),target_sub=str(dst))
    task._repair_source_cues()
    assert len(get_subtitle_from_srt(str(src)))==2
    assert not dst.exists()
    assert next(tmp_path.glob('vi.srt.cue-backup-*')).read_text()=='old edited translation'
    assert len(get_subtitle_from_srt(str(next(tmp_path.glob('zh.srt.cue-backup-*')))))==3
