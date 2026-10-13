# -*- coding: utf-8 -*-
"""Third-person narration rewrite for Translation V/A.

Groups source cues into scenes, retells each scene in the target language,
then splits new timestamps inside the original scene window.

Length budget is derived from:
1. Scene window timing (from source cues)
2. Target speaking pace measured once per job via a short Edge-TTS probe
   on the dubbing voice (sec per word/character) — no fixed 1.7 ratio.
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from copy import copy
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from videotrans.configure.config import logger, settings, tr
from videotrans.configure.constants import CJK_LANG
from videotrans.translator._constants import AI_TRANS_CHANNELS
from videotrans.translator._lang_utils import get_source_target_code
from videotrans.util._srt_parse import ms_to_time_string

CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7f]')
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？!?…\.])\s*')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,，;；、])\s*')

# Short natural sentences used ONLY as TTS timing probes (not rate tables).
_CALIB_SAMPLES = {
    'vi': 'Người kể chuyện nói rằng cô gái quyết không đồng ý cuộc hôn nhân đó dù cha hết lời khuyên bảo và nhắc lại ân nghĩa năm xưa.',
    'en': 'The narrator says the young woman firmly refused the marriage even though her father kept urging her to accept it for the sake of an old favor.',
    'zh': '旁白说这个女孩坚决不同意这门婚事，尽管父亲一再劝说她接受，并且提起多年前受人恩惠的事情。',
    'zh-cn': '旁白说这个女孩坚决不同意这门婚事，尽管父亲一再劝说她接受，并且提起多年前受人恩惠的事情。',
    'zh-tw': '旁白說這個女孩堅決不同意這門婚事，儘管父親一再勸說她接受，並且提起多年前受人恩惠的事情。',
    'ja': '語り手は、父親が何度も説得し昔の恩を持ち出しでも、少女が結婚をきっぱりと拒んだと語っている。',
    'ko': '해설자는 아버지가 거듭 설득하고 옛 은혜를 들어도 소녀가 그 결혼을 단호히 거절했다고 말한다.',
    'th': 'ผู้บรรยายกล่าวว่าหญิงสาวปฏิเสธการแต่งงานอย่างเด็ดขาดแม้บิดาจะเกลี้ยกล่อมและยกบุญคุณเก่าขึ้นมาพูด',
    'fr': 'Le narrateur dit que la jeune femme a fermement refusé ce mariage malgré les insistances de son père et le rappel d une vieille dette de gratitude.',
    'de': 'Der Erzähler sagt, dass die junge Frau die Ehe trotz der Bitten ihres Vaters und dem Verweis auf eine alte Gefälligkeit entschieden ablehnte.',
    'es': 'El narrador dice que la joven rechazó firmemente el matrimonio pese a las súplicas de su padre y el recuerdo de un antiguo favor.',
    'pt': 'O narrador diz que a jovem recusou firmemente o casamento apesar dos pedidos do pai e da lembrança de um favor antigo.',
    'id': 'Narator mengatakan bahwa gadis itu menolak pernikahan itu dengan tegas meskipun ayahnya terus membujuk dan mengingat jasa lama.',
    'ru': 'Рассказчик говорит, что девушка твёрдо отказалась от этого брака, несмотря на уговоры отца и упоминание о старом одолжении.',
}


def _lang_is_cjk(lang: str) -> bool:
    s = (lang or '').strip().lower().replace('_', '-')
    if not s:
        return False
    if s in CJK_LANG or s.split('-')[0] in CJK_LANG:
        return True
    return any(k in s for k in (
        'chinese', 'japanese', 'korean', 'thai', 'cantonese', 'yue', 'khmer', 'lao',
    ))


def _text_is_cjk_heavy(text: str) -> bool:
    compact = re.sub(r'\s+', '', text or '')
    if not compact:
        return False
    return len(CJK_CHAR_RE.findall(compact)) >= max(1, int(len(compact) * 0.3))


def _narration_limits(lang: str = '') -> dict:
    """Cue/scene limits from Advanced Settings (cfg.json / setini).

    - max cue ms ← max_speech_duration_s
    - min cue ms ← min_speech_duration_ms2
    - scene gap ← min_silence_duration_ms
    - unit cap ← cjk_len / other_len
    """
    max_cue_ms = int(float(settings.get('max_speech_duration_s', 5)) * 1000)
    min_cue_ms = int(float(settings.get('min_speech_duration_ms2', 300)))
    scene_gap_ms = int(float(settings.get('min_silence_duration_ms', 600)))
    if _lang_is_cjk(lang):
        unit_cap = int(settings.get('cjk_len', 15))
    else:
        unit_cap = int(settings.get('other_len', 40))
    unit_cap = max(1, unit_cap)
    # Only merge tiny word-split fragments (e.g. "hy"|"sinh"), never full clauses.
    min_chunk = max(1, min(3, unit_cap // 10))
    if min_cue_ms > max_cue_ms:
        min_cue_ms = max_cue_ms
    return {
        'max_cue_ms': max(1, max_cue_ms),
        'min_cue_ms': max(1, min_cue_ms),
        'scene_gap_ms': max(1, scene_gap_ms),
        'unit_cap': unit_cap,
        'min_chunk': min_chunk,
    }


def expand_scene_spans(times: Sequence[Tuple[int, int]], scene_gap_ms: Optional[int] = None) -> List[Tuple[int, int]]:
    """Merge adjacent cues until a pause of scene_gap_ms or more."""
    if not times:
        return []
    if scene_gap_ms is None:
        scene_gap_ms = _narration_limits()['scene_gap_ms']
    scene_gap_ms = int(scene_gap_ms)
    ranges = []
    lo = hi = 0
    for i in range(1, len(times)):
        gap = times[i][0] - times[hi][1]
        if gap < scene_gap_ms:
            hi = i
        else:
            ranges.append((lo, hi))
            lo = hi = i
    ranges.append((lo, hi))
    return ranges


def count_speaking_units(text: str, lang: str = '') -> Tuple[int, str]:
    """Return (count, unit_name) for pacing: CJK→characters, else→words."""
    text = text or ''
    if not text.strip():
        return 0, 'characters' if _lang_is_cjk(lang) else 'words'

    use_cjk = _lang_is_cjk(lang) if lang else _text_is_cjk_heavy(text)
    if use_cjk:
        n = len(CJK_CHAR_RE.findall(text))
        if n <= 0:
            n = len(re.sub(r'\s+', '', text))
        return max(n, 1), 'characters'

    words = WORD_RE.findall(text)
    return (len(words) if words else 1), 'words'


def scene_timing(times: Sequence[Tuple[int, int]], lo: int, hi: int) -> Tuple[float, float]:
    """Return (window_s, spoken_s)."""
    start_ms, end_ms = times[lo][0], times[hi][1]
    window_s = max((end_ms - start_ms) / 1000.0, 0.1)
    spoken_s = sum(max(times[i][1] - times[i][0], 0) for i in range(lo, hi + 1)) / 1000.0
    spoken_s = max(min(spoken_s, window_s), 0.1)
    return window_s, spoken_s


def _lang_code_key(lang_code: str, lang_name: str = '') -> str:
    code = (lang_code or '').strip().lower().replace('_', '-')
    if code:
        return code
    name = (lang_name or '').strip().lower()
    mapping = (
        ('viet', 'vi'), ('english', 'en'), ('chinese', 'zh'), ('japanese', 'ja'),
        ('korean', 'ko'), ('thai', 'th'), ('french', 'fr'), ('german', 'de'),
        ('spanish', 'es'), ('portuguese', 'pt'), ('indonesian', 'id'), ('russian', 'ru'),
    )
    for needle, key in mapping:
        if needle in name:
            return key
    return 'en'


def calibration_sample(lang_code: str, lang_name: str = '') -> str:
    key = _lang_code_key(lang_code, lang_name)
    if key in _CALIB_SAMPLES:
        return _CALIB_SAMPLES[key]
    short = key.split('-')[0]
    return _CALIB_SAMPLES.get(short, _CALIB_SAMPLES['en'])


def resolve_edge_voice(voice_role: str, lang_code: str) -> Optional[str]:
    """Resolve an Edge neural voice for the target locale (prefer user's role)."""
    from videotrans.util.help_role import get_edge_rolelist

    code = _lang_code_key(lang_code)
    short = code.split('-')[0]
    if voice_role and voice_role not in ('No', '', ' '):
        try:
            resolved = get_edge_rolelist(voice_role, short)
            if resolved and resolved != 'No':
                return resolved
        except Exception:
            pass
    try:
        all_roles = get_edge_rolelist() or {}
        locale_roles = all_roles.get(short) or all_roles.get(code) or {}
        for name, real in locale_roles.items():
            if name != 'No' and real and real != 'No':
                return real
    except Exception as e:
        logger.warning(f'Edge voice list unavailable for calib: {e}')
    return None


def _normalize_edge_rate(voice_rate) -> str:
    """Match Edge-TTS rate format used by dubbing (+N% / -N%)."""
    if voice_rate is None or voice_rate == '':
        return '+0%'
    s = str(voice_rate).strip()
    if not s:
        return '+0%'
    if s.endswith('%'):
        if not s.startswith(('+', '-')):
            try:
                n = float(s[:-1])
                return f'+{int(n)}%' if n >= 0 else f'{int(n)}%'
            except ValueError:
                return '+0%'
        return s
    try:
        n = float(s)
        return f'+{int(n)}%' if n >= 0 else f'{int(n)}%'
    except ValueError:
        return '+0%'


def probe_target_sec_per_unit(
        *,
        lang_code: str,
        lang_name: str = '',
        voice_role: str = '',
        voice_rate: str = '+0%',
        cache_folder: str = '',
) -> dict:
    """Synthesize one short sample with Edge-TTS at the configured rate and measure sec/unit.

    Raises RuntimeError if measurement fails — no silent fallback.
    """
    from edge_tts import Communicate
    from videotrans.util.help_ffmpeg import get_audio_time

    sample = calibration_sample(lang_code, lang_name)
    units, unit = count_speaking_units(sample, lang_name or lang_code)
    if units < 1:
        raise RuntimeError(tr("Could not measure dubbing pace for third-person narration"))
    voice = resolve_edge_voice(voice_role, lang_code)
    if not voice:
        raise RuntimeError(tr("Could not measure dubbing pace for third-person narration"))

    rate = _normalize_edge_rate(voice_rate)
    out_dir = Path(cache_folder) if cache_folder else Path(tempfile.mkdtemp(prefix='narr_calib_'))
    out_dir.mkdir(parents=True, exist_ok=True)
    rate_tag = rate.replace('%', 'pct').replace('+', 'p').replace('-', 'm')
    out_mp3 = out_dir / f'narration_pace_calib_{_lang_code_key(lang_code)}_{rate_tag}.mp3'

    async def _save():
        communicate = Communicate(sample, voice=voice, rate=rate, volume='+0%', pitch='+0Hz')
        await communicate.save(str(out_mp3))

    try:
        asyncio.run(_save())
        if not out_mp3.exists() or out_mp3.stat().st_size < 100:
            raise RuntimeError(tr("Could not measure dubbing pace for third-person narration"))
        # get_audio_time returns milliseconds in this project.
        duration_ms = float(get_audio_time(str(out_mp3)))
        duration = duration_ms / 1000.0
        if duration < 0.3:
            raise RuntimeError(tr("Could not measure dubbing pace for third-person narration"))
        sec_per_unit = duration / units
        logger.debug(
            f'TTS pace probe: voice={voice} rate={rate} units={units} {unit} '
            f'duration={duration:.2f}s sec/unit={sec_per_unit:.4f}'
        )
        return {
            'sec_per_unit': sec_per_unit,
            'unit': unit,
            'voice': voice,
            'rate': rate,
            'sample_units': units,
            'duration': duration,
            'sample': sample,
        }
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception(f'TTS pace probe failed: {e}')
        raise RuntimeError(tr("Could not measure dubbing pace for third-person narration")) from e


def compute_length_budget(
        source_lines: Sequence[str],
        window_s: float,
        spoken_s: float,
        source_lang: str,
        target_lang: str,
        sec_per_tgt: float,
        tgt_unit: str = None,
        pace_voice: str = '',
        pace_rate: str = '',
        *,
        one_to_one: bool = False,
) -> dict:
    """Target length from scene timing + TTS-measured target speaking pace.

    one_to_one (dialogue keep-gaps): budget from spoken_s only — using window_s
    (which includes silence between cues) over-counted words and made dub audio
    longer than the timeline (picture ahead of sound).
    """
    if not sec_per_tgt or sec_per_tgt <= 0.02:
        raise RuntimeError(tr("Could not measure dubbing pace for third-person narration"))

    source_text = "\n".join(source_lines)
    src_units, src_unit = count_speaking_units(source_text, source_lang)
    if not tgt_unit:
        tgt_unit = 'characters' if _lang_is_cjk(target_lang) else 'words'

    spoken_s = max(spoken_s, 0.1)
    window_s = max(window_s, 0.1)
    src_units = max(src_units, 1)
    sec_per_src = spoken_s / src_units
    src_rate = src_units / spoken_s

    # Slight headroom vs continuous probe (names/punctuation).
    effective_sec = sec_per_tgt * (1.12 if one_to_one else 1.10)
    # Dialogue 1:1 must fit inside actual speech slots; scene narration may fill the wall window.
    budget_s = spoken_s if one_to_one else window_s
    ideal = max(1, int(round(budget_s / effective_sec)))
    low = max(1, int(round(ideal * 0.90)))
    high = max(ideal, int(round(ideal * (1.03 if one_to_one else 1.06))))

    n_src = max(1, len([t for t in source_lines if (t or '').strip()]))
    if tgt_unit == 'words':
        by_length = max(1, int(round(ideal / 10)) + 1)
    else:
        by_length = max(1, int(round(ideal / 16)) + 1)
    max_sentences = max(n_src, by_length)
    max_sentences = min(max_sentences, max(n_src * 2, by_length))
    min_sentences = max(1, min(n_src, 2 if ideal >= 18 else 1))

    return {
        'window_s': window_s,
        'spoken_s': spoken_s,
        'budget_s': budget_s,
        'src_units': src_units,
        'src_unit': src_unit,
        'tgt_unit': tgt_unit,
        'ideal': ideal,
        'low': low,
        'high': high,
        'min_sentences': min_sentences,
        'max_sentences': max_sentences,
        'n_src': n_src,
        'src_rate': src_rate,
        'sec_per_src': sec_per_src,
        'sec_per_tgt': sec_per_tgt,
        'effective_sec_per_tgt': effective_sec,
        'method': 'tts_probe',
        'pace_voice': pace_voice or '',
        'pace_rate': pace_rate or '',
        'one_to_one': one_to_one,
    }


def split_scene_times(start_ms: int, end_ms: int, texts: Sequence[str], target_lang: str = '') -> List[Tuple[int, int]]:
    weights = []
    for text in texts:
        n, _ = count_speaking_units(text or '', target_lang)
        weights.append(max(n, 1))
    total = sum(weights)
    span = max(int(end_ms) - int(start_ms), len(weights))
    out = []
    cursor = int(start_ms)
    for i, weight in enumerate(weights):
        if i == len(weights) - 1:
            nxt = int(end_ms)
        else:
            used = sum(weights[:i + 1])
            nxt = int(start_ms) + int(round(span * used / total))
            nxt = min(max(nxt, cursor + 1), int(end_ms) - (len(weights) - i - 1))
        out.append((cursor, nxt))
        cursor = nxt
    return out


def _cue_unit_cap(lang: str, sec_per_tgt: float, max_cue_ms: Optional[int] = None) -> int:
    """Units allowed per cue: min(pace-based, cjk_len/other_len from settings)."""
    lim = _narration_limits(lang)
    max_cue_ms = lim['max_cue_ms'] if max_cue_ms is None else max(lim['min_cue_ms'], int(max_cue_ms))
    pace_units = max(1, int(round((max_cue_ms / 1000.0) / max(sec_per_tgt, 0.02))))
    return max(1, min(pace_units, lim['unit_cap']))


def _merge_orphan_chunks(chunks: List[str], lang: str, max_units: int) -> List[str]:
    """Merge tiny trailing fragments back so we don't cut compounds (hy|sinh, thế|giới)."""
    if len(chunks) < 2:
        return chunks
    use_cjk = _lang_is_cjk(lang)
    min_keep = _narration_limits(lang)['min_chunk']
    out = list(chunks)
    i = len(out) - 1
    while i > 0:
        n, _ = count_speaking_units(out[i], lang)
        if n >= min_keep:
            i -= 1
            continue
        merged = f'{out[i - 1]} {out[i]}'.strip() if not use_cjk else f'{out[i - 1]}{out[i]}'
        out[i - 1] = merged
        out.pop(i)
        i -= 1
    _ = max_units
    return out or chunks


def _split_by_units(text: str, lang: str, max_units: int) -> List[str]:
    """Last-resort unit split, then merge orphans."""
    max_units = max(1, int(max_units))
    if _lang_is_cjk(lang) or _text_is_cjk_heavy(text):
        chars = CJK_CHAR_RE.findall(text) or list(re.sub(r'\s+', '', text))
        raw = [''.join(chars[i:i + max_units]) for i in range(0, len(chars), max_units)]
    else:
        words = text.split()
        if not words:
            words = WORD_RE.findall(text) or [text]
        raw = [' '.join(words[i:i + max_units]) for i in range(0, len(words), max_units)]
    return _merge_orphan_chunks([c.strip() for c in raw if c and c.strip()], lang, max_units)


def _split_text_for_cue_cap(text: str, lang: str, max_units: int) -> List[str]:
    """Split rewrite text into short timed cues.

    Always split on sentence end AND commas/clauses — otherwise unread text
    stays on screen while earlier words are still being spoken.
    Over-long clauses then split by cjk_len/other_len (settings).
    """
    text = (text or '').strip()
    if not text:
        return ['…']
    max_units = max(1, int(max_units))
    sentences = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()] or [text]
    out: List[str] = []
    for sent in sentences:
        # Always break clauses so timestamps follow speech, not one long 1:1 block.
        # Do NOT re-merge across commas — unread clauses must get their own cue.
        clauses = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sent) if p and p.strip()] or [sent]
        for clause in clauses:
            cn, _ = count_speaking_units(clause, lang)
            if cn <= max_units:
                out.append(clause)
            else:
                out.extend(_split_by_units(clause, lang, max_units))
    return out or [text]


def build_timed_cues(
        texts: Sequence[str],
        start_ms: int,
        end_ms: int,
        target_lang: str,
        sec_per_tgt: float,
        *,
        max_cue_ms: Optional[int] = None,
) -> List[Tuple[int, int, str]]:
    """Weight-split texts into [start,end], then break any cue longer than max_cue_ms."""
    if not texts:
        return []
    lim = _narration_limits(target_lang)
    min_cue_ms = lim['min_cue_ms']
    max_cue_ms = lim['max_cue_ms'] if max_cue_ms is None else max(min_cue_ms, int(max_cue_ms))
    max_units = _cue_unit_cap(target_lang, sec_per_tgt, max_cue_ms)
    expanded: List[str] = []
    for t in texts:
        expanded.extend(_split_text_for_cue_cap(t or '', target_lang, max_units))
    spans = split_scene_times(start_ms, end_ms, expanded, target_lang)
    out: List[Tuple[int, int, str]] = []
    for (a, b), text in zip(spans, expanded):
        speak_ms = int(count_speaking_units(text, target_lang)[0] * sec_per_tgt * 1000) + 200
        if b - a <= max_cue_ms and speak_ms <= max_cue_ms:
            out.append((a, min(b, a + max(min_cue_ms, speak_ms)), text))
            continue
        pieces = _split_text_for_cue_cap(text, target_lang, max(1, max_units // 2))
        if len(pieces) == 1:
            dur = min(max_cue_ms, max(min_cue_ms, speak_ms), b - a)
            out.append((a, a + dur, text))
        else:
            sub = split_scene_times(a, min(b, a + max_cue_ms * len(pieces)), pieces, target_lang)
            for (s, e), piece in zip(sub, pieces):
                e = min(e, s + max_cue_ms, b)
                if e <= s:
                    e = min(s + min_cue_ms, b)
                out.append((s, e, piece))
    return out


def build_one_to_one_cues(
        source_times: Sequence[Tuple[int, int]],
        new_lines: Sequence[str],
        target_lang: str,
        sec_per_tgt: float,
        *,
        max_cue_ms: Optional[int] = None,
) -> List[Tuple[int, int, str]]:
    """Map rewritten lines onto original cue slots (keep gaps); split long slots/text."""
    n = min(len(source_times), len(new_lines))
    out: List[Tuple[int, int, str]] = []
    lim = _narration_limits(target_lang)
    min_cue_ms = lim['min_cue_ms']
    max_cue_ms = lim['max_cue_ms'] if max_cue_ms is None else max(min_cue_ms, int(max_cue_ms))
    max_units = _cue_unit_cap(target_lang, sec_per_tgt, max_cue_ms)
    for i in range(n):
        a, b = int(source_times[i][0]), int(source_times[i][1])
        if b <= a:
            b = a + min_cue_ms
        text = (new_lines[i] or '').strip() or '…'
        chunks = _split_text_for_cue_cap(text, target_lang, max_units)
        if len(chunks) == 1:
            speak_ms = int(count_speaking_units(chunks[0], target_lang)[0] * sec_per_tgt * 1000) + 200
            if speak_ms > max_cue_ms or count_speaking_units(chunks[0], target_lang)[0] > max_units:
                chunks = _split_text_for_cue_cap(chunks[0], target_lang, max(1, max_units // 2))
        if len(chunks) == 1:
            speak_ms = int(count_speaking_units(chunks[0], target_lang)[0] * sec_per_tgt * 1000) + 200
            dur = min(max_cue_ms, max(min_cue_ms, speak_ms), b - a)
            out.append((a, a + dur, chunks[0]))
            continue
        usable_end = min(b, a + max_cue_ms * len(chunks))
        for (s, e), chunk in zip(split_scene_times(a, usable_end, chunks, target_lang), chunks):
            e = min(e, s + max_cue_ms, b)
            if e <= s:
                e = min(s + min_cue_ms, b)
            out.append((s, e, chunk))
    return out


def clean_narration_lines(text: str) -> List[str]:
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.I | re.S)
    text = re.sub(r'</?TRANSLATE_TEXT>', '', text, flags=re.I)
    lines = []
    for line in text.splitlines():
        line = line.strip().strip('"\'“”«»')
        line = re.sub(r'^\d+[\.\)、]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line).strip()
        if line:
            lines.append(line)
    return lines


def _normalize_narration_style(style: str) -> str:
    s = (style or 'third').strip().lower()
    if s in ('first', '1st', 'character', 'pov', 'ngoi1'):
        return 'first'
    return 'third'


def _pov_instructions(style: str) -> str:
    if _normalize_narration_style(style) == 'first':
        return (
            "Retell this subtitle scene as FIRST-PERSON character narration for video dubbing.\n"
            "Speak as the MAIN CHARACTER of this scene using I/me (or the natural first-person "
            "form in the target language). Put yourself inside their body and feelings.\n"
            "Pick the character with the strongest emotional stake in this scene "
            "(usually the protagonist who is being pressured, deciding, or reacting).\n"
            "Your own spoken lines become what YOU say or decide. "
            "Other speakers become what you hear, face, or react to "
            "(e.g. 'Father kept urging me…', 'She told me…') — do NOT switch to a third-person narrator.\n"
            "Stay in one consistent first-person identity for the whole scene.\n"
        )
    return (
        "Retell this subtitle scene as third-person narration for video dubbing.\n"
        "Write as a narrator who just watched the scene.\n"
        "Change verbs and phrasing into third person. Every sentence needs a clear subject and verb.\n"
    )


def line_rewrite_prompt(
        source_lines: Sequence[str],
        source_lang: str,
        target_lang: str,
) -> str:
    """1:1 cues/timestamps, but wording rewritten like third-person narration (not literal translate)."""
    source = "\n".join(f"{i}. {text}" for i, text in enumerate(source_lines, 1))
    target = (target_lang or '').strip() or 'the target language'
    source_name = (source_lang or '').strip() or 'auto'
    n = len(source_lines)
    return (
        f"REWRITE each subtitle into {target} as THIRD-PERSON narration for video dubbing.\n"
        "Same rewriting depth as scene third-person narration: brand-new complete sentences, "
        "a narrator retelling that beat — NOT a literal / word-for-word translation of the dialogue.\n"
        "Forbidden: direct translation that keeps the original spoken line shape "
        "(e.g. do not keep 'I refuse' as 'I refuse' in the target language).\n"
        "Required: third-person narrator wording with clear subject + verb "
        "(e.g. 'She firmly refused…', 'Father kept urging her…').\n"
        "Change verbs and phrasing into third person. Every output line must sound narrated, not dubbed dialogue.\n"
        f"STRUCTURE (1:1): output EXACTLY {n} lines — one rewritten sentence per source line, same order. "
        "Do NOT merge, split, skip, or add lines. Do NOT retime — only rewrite text.\n"
        "Keep roughly the same spoken length as each source line (compress if needed).\n"
        "Preserve names, numbers, relationships, threats, decisions, and emotional stance.\n"
        f"Source language: {source_name}. Target language: {target}.\n"
        "\n"
        "Source lines:\n"
        f"{source}\n"
        "\n"
        "# OUTPUT FORMAT\n"
        f"Output ONLY the {n} rewritten narration lines inside <TRANSLATE_TEXT> and </TRANSLATE_TEXT>, "
        "one sentence per source line.\n"
        "\n"
        "<INPUT>\n"
        "{batch_input}\n"
        "</INPUT>"
    )


LINE_REWRITE_BATCH = 12


def run_line_narration_rewrite(
        *,
        text_list: List,
        translate_type: int,
        source_code: str = '',
        target_code: str = '',
        uuid=None,
) -> List:
    """1:1 cues/timestamps; rewrite each line like third-person narration (not literal translate)."""
    if int(translate_type) not in AI_TRANS_CHANNELS:
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))

    from videotrans.translator._openaicompat import OpenAICampat
    from videotrans.translator._runner import get_model_transobj

    model = get_model_transobj(translate_type=translate_type, uuid=uuid)
    if not isinstance(model, OpenAICampat):
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))
    if getattr(model, 'ainame', '') == 'openrouter':
        model.extra_body = {"reasoning": {"enabled": False}}
    # Match scene narration creativity so the model does not fall back to literal translate.
    model.temperature = 0.45

    source_lang, target_lang = get_source_target_code(
        show_source=source_code,
        show_target=target_code,
        translate_type=translate_type,
    )
    source_lang = source_lang or source_code or 'auto'
    target_lang = target_lang or target_code or 'target language'

    results = []
    n = len(text_list)
    for start in range(0, n, LINE_REWRITE_BATCH):
        batch = text_list[start:start + LINE_REWRITE_BATCH]
        source_lines = []
        for it in batch:
            t = (it.get('text') if hasattr(it, 'get') else it['text']) or ''
            source_lines.append(t.strip() or '…')
        joined = "\n".join(source_lines)
        model.prompt = line_rewrite_prompt(source_lines, source_lang, target_lang)
        try:
            raw = model._item_task(joined)
        except Exception as e:
            logger.exception(f'Line rewrite failed at {start + 1}: {e}')
            raise
        new_lines = clean_narration_lines(raw or '')
        if len(new_lines) != len(source_lines):
            logger.warning(
                f'Line rewrite count mismatch at {start + 1}: '
                f'got {len(new_lines)} want {len(source_lines)}; retrying'
            )
            model.prompt = line_rewrite_prompt(source_lines, source_lang, target_lang) + (
                f"\nREMINDER: you MUST output exactly {len(source_lines)} lines.\n"
            )
            try:
                raw2 = model._item_task(joined)
                retry = clean_narration_lines(raw2 or '')
                if len(retry) == len(source_lines):
                    new_lines = retry
            except Exception as e:
                logger.exception(f'Line rewrite retry failed: {e}')
                raise
        if len(new_lines) < len(source_lines):
            new_lines = new_lines + source_lines[len(new_lines):]
        elif len(new_lines) > len(source_lines):
            new_lines = new_lines[:len(source_lines)]

        for it, text in zip(batch, new_lines):
            # SrtItem is a dataclass whose __iter__ yields field names — never dict(it).
            if is_dataclass(it) and not isinstance(it, type):
                results.append(replace(it, text=text))
            elif isinstance(it, dict):
                results.append({**it, 'text': text})
            else:
                item = copy(it)
                item['text'] = text
                results.append(item)

    logger.debug(f'Line rewrite 1:1: {n} cues (target={target_lang})')
    return results


def _dialogue_rewrite_instructions(target: str) -> str:
    """1:1 mode: translate spoken lines, then polish — keep dialogue voice (not narrator POV)."""
    return (
        f"For each source subtitle, TRANSLATE the spoken line into {target}, then REWRITE the wording "
        "so it sounds natural for video dubbing.\n"
        "Two steps in one output line: (1) direct dialogue translation of that speaker's words, "
        "(2) polish / rewrite the phrasing — not a stiff word-for-word gloss.\n"
        "Keep dialogue voice: preserve I/you/he/she as that speaker uses them. "
        "Do NOT convert into third-person narrator style "
        "(forbidden: 'She refused…', 'Father told her…' when the source is the character speaking).\n"
        "Allowed example: '我不答应' → 'Tôi nhất quyết không đồng ý cuộc hôn nhân này'.\n"
        "Forbidden example: '我不答应' → 'Cô ấy nhất quyết từ chối cuộc hôn nhân đó'.\n"
    )


def narration_prompt(
        source_lines: Sequence[str],
        budget: dict,
        source_lang: str,
        target_lang: str,
        style: str = 'third',
        one_to_one: bool = False,
) -> str:
    source = "\n".join(f"{i}. {text}" for i, text in enumerate(source_lines, 1))
    target = (target_lang or '').strip() or 'the target language'
    source_name = (source_lang or '').strip() or 'auto'
    unit = budget['tgt_unit']
    pace = budget.get('effective_sec_per_tgt') or budget.get('sec_per_tgt') or 0
    n_src = budget.get('n_src') or max(1, len(source_lines))
    pace_note = ''
    if pace > 0:
        pace_note = (
            f"This dubbing voice needs about {pace:.3f}s per {unit.rstrip('s')} "
            f"(≈{1.0 / pace:.2f} {unit}/s including short pauses).\n"
        )
    style = _normalize_narration_style(style)
    if one_to_one:
        lead = _dialogue_rewrite_instructions(target)
        structure = (
            f"STRUCTURE (1:1): output EXACTLY {n_src} lines — "
            f"one polished dialogue line per numbered source line, same order. "
            "Do NOT merge, skip, or add extra source beats.\n"
            "Keep natural spoken wording. If one source line has two beats, join them with a comma "
            "(system splits each comma clause into its own timestamp). "
            "FORBIDDEN: two full sentences in one line without a comma. "
            "Do NOT delete names, verbs, or mid-sentence meaning to shorten.\n"
        )
        sentence_rule = (
            f"Use EXACTLY {n_src} lines. One complete sentence/clause per line. "
            "No multi-sentence lines. Do not drop words from the middle of a line.\n"
        )
        rewrite_note = (
            "Do NOT leave a raw literal gloss — after translating, rewrite for natural spoken dubbing.\n"
            "Shorten only by tighter phrasing — never by cutting out the middle of the sentence.\n"
        )
    else:
        lead = _pov_instructions(style)
        structure = (
            "Stay CLOSE to the dialogue: cover every speaker turn / plot beat from the source lines. "
            "Do not skip lines, omit replies, or summarize the scene into a vague outline.\n"
            "Do NOT translate word-for-word line by line, but each source line's meaning must still appear. "
            "Prefer one complete narration sentence per source beat. "
            "Never put two story beats into one line. "
            "Do NOT delete mid-sentence words; the system will split long timestamps.\n"
        )
        sentence_rule = (
            f"Use {budget['min_sentences']}–{budget['max_sentences']} complete sentences, "
            "one sentence per line. No multi-sentence lines.\n"
        )
        rewrite_note = (
            "Do NOT translate word-for-word; rewrite as narration with clear subject and verb.\n"
        )
    return (
        f"{lead}"
        f"Write in {target} only. Do not write in the source language.\n"
        f"{structure}"
        f"{rewrite_note}"
        "Preserve names, numbers, relationships, threats, decisions, and emotional stance. "
        "If the length limit is tight, use shorter phrases — never delete mid-sentence words or names.\n"
        f"Source language: {source_name}. Target language: {target}.\n"
        f"There are {n_src} source lines in this scene — cover all of them.\n"
        "\n"
        "# TIMING BUDGET (must obey)\n"
        f"Scene wall time: {budget['window_s']:.1f}s; "
        f"speakable budget: {budget.get('budget_s', budget['window_s']):.1f}s "
        f"(source speech ≈{budget['spoken_s']:.1f}s, {budget['src_units']} source {budget['src_unit']}).\n"
        f"{pace_note}"
        f"Length for the whole scene: about {budget['ideal']} {unit} "
        f"(range {budget['low']}–{budget['high']} {unit}; do not exceed {budget['high']}).\n"
        f"{sentence_rule}"
        "\n"
        "Source lines:\n"
        f"{source}\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "Output ONLY the result inside <TRANSLATE_TEXT> and </TRANSLATE_TEXT>.\n"
        "\n"
        "<INPUT>\n"
        "{batch_input}\n"
        "</INPUT>"
    )


def narration_shorten_prompt(
        source_lines: Sequence[str],
        draft_lines: Sequence[str],
        budget: dict,
        target_lang: str,
        current_units: int,
        style: str = 'third',
        one_to_one: bool = False,
) -> str:
    target = (target_lang or '').strip() or 'the target language'
    unit = budget['tgt_unit']
    source = "\n".join(f"{i}. {text}" for i, text in enumerate(source_lines, 1))
    draft = "\n".join(draft_lines)
    style = _normalize_narration_style(style)
    n_src = budget.get('n_src') or max(1, len(source_lines))
    if one_to_one:
        mode = (
            f"polished DIALOGUE in {target} (translate then rewrite; keep speaker I/you/he; "
            "NOT third-person narrator)"
        )
        count_rule = f"Keep EXACTLY {n_src} lines (1:1 with source).\n"
    else:
        mode = (
            "first-person character narration (I/me, stay inside the protagonist)"
            if style == 'first'
            else "third-person narration"
        )
        count_rule = "One complete sentence per line. No labels.\n"
    return (
        f"The draft below is TOO LONG for dubbing ({current_units} {unit}, "
        f"max {budget['high']} {unit} for a "
        f"{budget.get('budget_s', budget['window_s']):.1f}s speakable window).\n"
        f"Rewrite the WHOLE scene again in {target} only as {mode}.\n"
        f"Total length MUST be ≤ {budget['high']} {unit} (aim ≈ {budget['ideal']} {unit}).\n"
        "CRITICAL: stay close to the source — cover every dialogue beat. "
        "Shorten by compressing phrases, NOT by deleting plot points, replies, or names.\n"
        f"{count_rule}"
        "\n"
        "Source meaning (must all remain):\n"
        f"{source}\n"
        "\n"
        "Overlong draft:\n"
        f"{draft}\n"
        "\n"
        "# OUTPUT FORMAT\n"
        f"Output ONLY {target} text inside <TRANSLATE_TEXT> and </TRANSLATE_TEXT>.\n"
        "\n"
        "<INPUT>\n"
        "{batch_input}\n"
        "</INPUT>"
    )


def count_lines_units(lines: Sequence[str], lang: str) -> int:
    return sum(count_speaking_units(line, lang)[0] for line in lines)


def trim_lines_to_budget(lines: List[str], lang: str, high: int) -> List[str]:
    """Drop trailing sentences until total units ≤ high (keep at least one)."""
    if not lines:
        return lines
    out = list(lines)
    while len(out) > 1 and count_lines_units(out, lang) > high:
        out.pop()
    # If a single line is still over, leave it — shorten rewrite should have handled it.
    return out


def shrink_lines_to_budget(lines: List[str], lang: str, high: int) -> List[str]:
    """Keep line count; shorten longest lines until total units ≤ high (for 1:1 mode)."""
    if not lines or high < 1:
        return lines
    out = list(lines)
    # Up to enough passes to shave long lines without dropping cues.
    for _ in range(80):
        if count_lines_units(out, lang) <= high:
            break
        # Shorten the currently longest line by ~15% units.
        idx = max(range(len(out)), key=lambda i: count_speaking_units(out[i], lang)[0])
        text = out[idx] or ''
        n, unit = count_speaking_units(text, lang)
        if n <= 1:
            break
        keep = max(1, int(n * 0.85))
        if unit == 'characters' or (_lang_is_cjk(lang) or _text_is_cjk_heavy(text)):
            chars = CJK_CHAR_RE.findall(text) or list(re.sub(r'\s+', '', text))
            out[idx] = ''.join(chars[:keep])
        else:
            words = WORD_RE.findall(text) or text.split()
            out[idx] = ' '.join(words[:keep])
    return out


def _make_srt_item(line: int, start_ms: int, end_ms: int, text: str) -> dict:
    startraw = ms_to_time_string(ms=start_ms)
    endraw = ms_to_time_string(ms=end_ms)
    return {
        'line': line,
        'start_time': int(start_ms),
        'end_time': int(end_ms),
        'startraw': startraw,
        'endraw': endraw,
        'time': f'{startraw} --> {endraw}',
        'text': text,
        'spk': '',
        'filename': '',
    }


def run_third_person_narration(
        *,
        text_list: List,
        translate_type: int,
        source_code: str = '',
        target_code: str = '',
        voice_role: str = '',
        voice_rate: str = '+0%',
        cache_folder: str = '',
        uuid=None,
        narration_style: str = 'third',
        one_to_one: bool = False,
) -> List:
    """Rewrite source cues as scene-level narration with new timestamps.

    one_to_one=True: same pipeline (scene windows, TTS word budget, retime),
    but prompt = translate each spoken line then polish (dialogue voice, not narrator POV).
    """
    style = _normalize_narration_style(narration_style)
    if int(translate_type) not in AI_TRANS_CHANNELS:
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))

    from videotrans.translator._openaicompat import OpenAICampat
    from videotrans.translator._runner import get_model_transobj

    model = get_model_transobj(translate_type=translate_type, uuid=uuid)
    if not isinstance(model, OpenAICampat):
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))
    if getattr(model, 'ainame', '') == 'openrouter':
        model.extra_body = {"reasoning": {"enabled": False}}
    model.temperature = 0.4

    source_lang, target_lang = get_source_target_code(
        show_source=source_code,
        show_target=target_code,
        translate_type=translate_type,
    )
    source_lang = source_lang or source_code or 'auto'
    target_lang = target_lang or target_code or 'target language'

    # One TTS pace probe per job at the configured dubbing rate — required, no fallback.
    pace = probe_target_sec_per_unit(
        lang_code=target_code or '',
        lang_name=target_lang,
        voice_role=voice_role or '',
        voice_rate=voice_rate or '+0%',
        cache_folder=cache_folder or '',
    )
    sec_per_tgt = pace['sec_per_unit']
    pace_unit = pace['unit']
    pace_voice = pace['voice']
    pace_rate = pace['rate']
    logger.debug(
        f'Narration TTS pace: {pace_voice} rate={pace_rate} style={style} -> '
        f'{sec_per_tgt:.4f}s/{pace_unit.rstrip("s")} ({1.0 / sec_per_tgt:.2f} {pace_unit}/s)'
    )

    times = [(int(it['start_time']), int(it['end_time'])) for it in text_list]
    spans = expand_scene_spans(times)
    results = []
    line_no = 1

    for lo, hi in spans:
        source_lines = [(text_list[i].get('text') if hasattr(text_list[i], 'get') else text_list[i]['text'] or '').strip()
                        for i in range(lo, hi + 1)]
        source_lines = [t for t in source_lines if t]
        if not source_lines:
            continue
        start_ms, end_ms = times[lo][0], times[hi][1]
        window_s, spoken_s = scene_timing(times, lo, hi)
        budget = compute_length_budget(
            source_lines, window_s, spoken_s, source_lang, target_lang,
            sec_per_tgt=sec_per_tgt,
            tgt_unit=pace_unit,
            pace_voice=pace_voice,
            pace_rate=pace_rate,
            one_to_one=one_to_one,
        )
        if one_to_one:
            n_src = budget['n_src']
            budget['min_sentences'] = n_src
            budget['max_sentences'] = n_src
        logger.debug(
            f'Narration budget scene {lo + 1}-{hi + 1}: window={window_s:.1f}s '
            f'spoken={spoken_s:.1f}s budget_s={budget.get("budget_s", window_s):.1f}s '
            f'rate={pace_rate} style={style} one_to_one={one_to_one} -> ideal={budget["ideal"]} '
            f'{budget["tgt_unit"]} [{budget["low"]}-{budget["high"]}]'
        )
        joined = "\n".join(source_lines)
        model.prompt = narration_prompt(
            source_lines, budget, source_lang, target_lang, style=style, one_to_one=one_to_one
        )
        try:
            raw = model._item_task(joined)
        except Exception as e:
            logger.exception(f'Narration failed for lines {lo + 1}-{hi + 1}: {e}')
            raise
        new_lines = clean_narration_lines(raw or '')
        if not new_lines:
            raise RuntimeError(
                tr("Third-person narration returned empty text for a scene")
            )

        # Enforce length without gutting dialogue: up to 2 compress rewrites, trim last.
        got = count_lines_units(new_lines, target_lang)
        for attempt in range(2):
            if got <= budget['high']:
                break
            logger.warning(
                f'Scene {lo + 1}-{hi + 1} over budget ({got}>{budget["high"]} '
                f'{budget["tgt_unit"]}); compress rewrite #{attempt + 1}'
            )
            model.prompt = narration_shorten_prompt(
                source_lines, new_lines, budget, target_lang, got,
                style=style, one_to_one=one_to_one,
            )
            try:
                raw2 = model._item_task(joined)
                shortened = clean_narration_lines(raw2 or '')
                if shortened:
                    new_lines = shortened
            except Exception as e:
                logger.exception(f'Narration shorten failed for scene {lo + 1}-{hi + 1}: {e}')
                raise
            got = count_lines_units(new_lines, target_lang)
        if got > budget['high']:
            before = got
            if one_to_one:
                # Never hard-truncate words (that deletes mid/end meaning).
                # Cue splitter + TTS rate will absorb modest over-budget text.
                logger.warning(
                    f'Scene {lo + 1}-{hi + 1} still over after compress '
                    f'({before}>{budget["high"]} {budget["tgt_unit"]}); '
                    f'keeping full wording (no truncate)'
                )
            else:
                new_lines = trim_lines_to_budget(new_lines, target_lang, budget['high'])
                logger.warning(
                    f'Scene {lo + 1}-{hi + 1} still over after compress '
                    f'({before} {budget["tgt_unit"]}); trimmed to '
                    f'{count_lines_units(new_lines, target_lang)}'
                )

        if one_to_one:
            n_src = budget['n_src']
            if len(new_lines) < n_src:
                new_lines = new_lines + source_lines[len(new_lines):n_src]
            elif len(new_lines) > n_src:
                new_lines = new_lines[:n_src]
            # Keep original cue gaps; size/split cues via Advanced Settings limits.
            src_times_ne = []
            for i in range(lo, hi + 1):
                t = (text_list[i].get('text') if hasattr(text_list[i], 'get') else text_list[i]['text'] or '').strip()
                if t:
                    src_times_ne.append(times[i])
            timed = build_one_to_one_cues(
                src_times_ne, new_lines, target_lang, sec_per_tgt
            )
        else:
            timed = build_timed_cues(
                new_lines, start_ms, end_ms, target_lang, sec_per_tgt
            )

        for a, b, text in timed:
            results.append(_make_srt_item(line_no, a, b, text))
            line_no += 1

    logger.debug(
        f'Narration style={style} one_to_one={one_to_one}: '
        f'{len(text_list)} source cues -> {len(results)} narration cues '
        f'(target={target_lang})'
    )
    return results
