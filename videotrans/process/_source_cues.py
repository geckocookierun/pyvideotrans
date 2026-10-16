"""Punctuation-aware source cues, using aligned audio timestamps without text loss."""
import copy
import re
import unicodedata

from videotrans.task.taskcfg import SrtItem
from videotrans.util._srt_parse import ms_to_time_string

_PAUSE = re.compile(r'[.!?,;:。！？，；：、،؛؟…][”’"\')）」』]*$')
_NO_SPACE = {'zh', 'zh-cn', 'zh-tw', 'ja', 'yue', 'ko', 'th', 'km'}


def _item(text, start, end, line, spk=''):
    a, b = ms_to_time_string(ms=start), ms_to_time_string(ms=end)
    return SrtItem(text=text, start_time=start, end_time=end, line=line,
                   startraw=a, endraw=b, time=f'{a} --> {b}', spk=spk)


def restore_transcript_punctuation(words, transcript):
    """Recover punctuation removed by an aligner only when all letters match.

    On mismatch preserve the complete transcript in its whole aligned interval;
    never guess character timestamps or discard unmatched recognized words.
    """
    if not words:
        return []
    transcript = unicodedata.normalize('NFC', transcript).strip()
    def lexical(char):
        return unicodedata.category(char)[0] in 'LNM'
    positions = [i for i, char in enumerate(transcript) if lexical(char)]
    expected = ''.join(transcript[i] for i in positions)
    cleaned = [''.join(c for c in unicodedata.normalize('NFC', w['word']) if lexical(c)) for w in words]
    if not expected or ''.join(cleaned) != expected:
        return [dict(word=transcript, start=words[0]['start'], end=words[-1]['end'])]
    output = []
    used = cursor = 0
    for word, letters in zip(words, cleaned):
        if not letters:
            continue
        used += len(letters)
        end = positions[used] if used < len(positions) else len(transcript)
        output.append(dict(word, word=transcript[cursor:end]))
        cursor = end
    return output


def segment_aligned_source(texts, language='zh'):
    """Split at real punctuation or long silence, never on a duration cap."""
    separator = '' if language.lower() in _NO_SPACE else ' '
    output, chunk = [], []
    def flush():
        if not chunk:
            return
        text = separator.join(w['word'].strip() for w in chunk).strip()
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        output.append(_item(text, round(chunk[0]['start']*1000),
                            round(chunk[-1]['end']*1000), len(output)+1))
    for segment in texts:
        words = segment.get('words') or []
        if not words and segment.get('text', '').strip():
            words = [dict(word=segment['text'], start=segment['start'], end=segment['end'])]
        for word in words:
            if not word.get('word', '').strip():
                continue
            split = False
            if chunk:
                previous = chunk[-1]['word'].strip()
                following = word['word'].strip()
                pause = bool(_PAUSE.search(previous))
                if re.search(r'\d[.,:，]$', previous) and following[0].isdigit():
                    pause = False
                # Punctuation tokens belong to the preceding phrase.
                is_punctuation = all(unicodedata.category(c)[0] == 'P' for c in following)
                split = (pause or word['start']-chunk[-1]['end'] >= 1.0) and not is_punctuation
            if split:
                flush()
                chunk = []
            chunk.append(word)
    flush()
    return output


def repair_source_fragments(rows, language='zh', max_gap_ms=250):
    """Join only nearby unfinished cues, respecting explicit speaker changes.

    This conservative migration does not invent internal word timings for old
    SRTs. New recognition uses the actual aligned word boundaries above.
    """
    separator = '' if language.lower() in _NO_SPACE else ' '
    result = []
    for row in rows:
        item = copy.deepcopy(row)
        if result:
            prev = result[-1]
            gap = item['start_time'] - prev['end_time']
            previous_speaker = prev.get('spk', '')
            speaker = item.get('spk', '')
            same_speaker = previous_speaker == speaker
            if 0 <= gap <= max_gap_ms and same_speaker and not _PAUSE.search(prev['text'].strip()):
                prev['text'] = prev['text'].rstrip() + separator + item['text'].lstrip()
                prev['end_time'] = item['end_time']
                prev['endraw'] = ms_to_time_string(ms=item['end_time'])
                prev['time'] = f"{prev['startraw']} --> {prev['endraw']}"
                continue
        item['line'] = len(result)+1
        result.append(item)
    return result
