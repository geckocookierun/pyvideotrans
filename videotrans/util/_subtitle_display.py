"""Split presentation cues without modifying translation, TTS or alignment input."""
import re

from videotrans.util.help_srt import ms_to_time_string

# A display boundary must be punctuation, never just a word/character limit.
_PAUSE = re.compile(r"[,.!?;:，。！？；：、،؛؟…]+[”’\"'）)」』]*")
_CJK = re.compile(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]')


def _chunks(text):
    boundaries = []
    for match in _PAUSE.finditer(text):
        end = match.end()
        if end == len(text):
            continue
        # Do not interpret decimal/thousands separators or clock times as pauses.
        if (match.start() > 0 and text[match.start()-1].isdigit()
                and text[end].isdigit()):
            continue
        # A period inside an address/initialism is not a sentence boundary.
        if (match.group().startswith('.') and not text[end].isspace()
                and not _CJK.match(text[end])):
            continue
        boundaries.append(end)
    chunks = []
    start = 0
    for end in boundaries:
        if text[start:end].strip():
            chunks.append(text[start:end])
        start = end
    chunks.append(text[start:])
    return chunks


def split_display_cues(rows):
    """Use character-weighted timing inside each already-aligned cue.

    This is display timing, not forced word alignment. Original rows are never
    changed; source_index keeps bilingual source lookup attached to the old cue.
    """
    output = []
    for source_index, row in enumerate(rows):
        start, end = int(row['start_time']), int(row['end_time'])
        text = row['text'].strip()
        parts = _chunks(text) or [text]
        # Only collapse an impossible sub-millisecond split; ordinary short
        # phrases remain separate even when they appear briefly.
        if end - start < len(parts):
            parts = [text]
        weights = [max(1, len(p.strip())) for p in parts]
        total = sum(weights)
        used = 0
        cursor = start
        for index, (part, weight) in enumerate(zip(parts, weights)):
            used += weight
            finish = start + round((end-start) * used / total)
            finish = max(cursor + 1, min(finish, end - (len(parts) - index - 1)))
            item = dict(line=len(output)+1, text=part.strip(), start_time=cursor,
                        end_time=finish, source_index=source_index)
            item['startraw'] = ms_to_time_string(ms=cursor)
            item['endraw'] = ms_to_time_string(ms=finish)
            item['time'] = f"{item['startraw']} --> {item['endraw']}"
            output.append(item)
            cursor = finish
    return output
