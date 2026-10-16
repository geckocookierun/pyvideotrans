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
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from videotrans.configure.config import TEMP_ROOT, logger, settings, tr
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
    # MSA (Modern Standard Arabic) — used only to measure Edge-TTS speaking pace.
    'ar': (
        'يقول الراوي إن الفتاة رفضت هذا الزواج بشدة رغم إلحاح أبيها '
        'وتذكيره بفضل قديم عليه.'
    ),
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

    IMPORTANT: cjk_len / other_len are *characters per hardsub line* in the UI.
    count_speaking_units uses characters for CJK and *words* otherwise — so for
    Latin scripts we convert other_len chars → approx words (~5 chars/word).
    unit_cap is a soft target for the AI prompt (complete short lines). Latin text
    is NOT hard-chopped by word count (that created incomplete “câu cụt” cues).
    """
    max_cue_ms = int(float(settings.get('max_speech_duration_s', 5)) * 1000)
    min_cue_ms = int(float(settings.get('min_speech_duration_ms2', 300)))
    scene_gap_ms = int(float(settings.get('min_silence_duration_ms', 600)))
    if _lang_is_cjk(lang):
        unit_cap = max(1, int(settings.get('cjk_len', 15)))
    else:
        other_chars = max(1, int(settings.get('other_len', 40)))
        # ~5 Latin letters per word incl. space → words that fit one subtitle line.
        unit_cap = max(4, other_chars // 5)
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
        ('arab', 'ar'),
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


def _fallback_sec_per_unit(lang_code: str, lang_name: str, voice_rate: str) -> dict:
    """Offline pace estimate when Edge-TTS probe is unavailable (403/network)."""
    sample = calibration_sample(lang_code, lang_name)
    units, unit = count_speaking_units(sample, lang_name or lang_code)
    # Typical conversational pace; CJK counts characters, others words.
    base = 0.22 if unit == 'characters' else 0.28
    rate = _normalize_edge_rate(voice_rate)
    try:
        pct = float(str(rate).rstrip('%'))
    except ValueError:
        pct = 0.0
    # +20% rate ≈ speak in 1/1.2 of the time.
    factor = max(0.5, 1.0 + pct / 100.0)
    sec_per_unit = base / factor
    voice = resolve_edge_voice('', lang_code) or (lang_code or 'unknown')
    logger.warning(
        f'TTS pace probe unavailable; using fallback sec/{unit}={sec_per_unit:.4f} '
        f'(rate={rate}, lang={lang_code or lang_name})'
    )
    return {
        'sec_per_unit': sec_per_unit,
        'unit': unit,
        'voice': voice,
        'rate': rate,
        'sample_units': max(units, 1),
        'duration': sec_per_unit * max(units, 1),
        'sample': sample,
        'fallback': True,
    }


def probe_target_sec_per_unit(
        *,
        lang_code: str,
        lang_name: str = '',
        voice_role: str = '',
        voice_rate: str = '+0%',
        cache_folder: str = '',
) -> dict:
    """Synthesize one short sample with Edge-TTS at the configured rate and measure sec/unit.

    Retries a few times; on persistent Edge-TTS failure (e.g. 403), uses a language fallback
    so narration can continue. Dubbing itself still uses Edge-TTS with its own retries.
    """
    from edge_tts import Communicate
    from videotrans.util.help_ffmpeg import get_audio_time
    from videotrans.configure.config import ROOT_DIR, app_cfg

    sample = calibration_sample(lang_code, lang_name)
    units, unit = count_speaking_units(sample, lang_name or lang_code)
    if units < 1:
        return _fallback_sec_per_unit(lang_code, lang_name, voice_rate)
    voice = resolve_edge_voice(voice_role, lang_code)
    if not voice:
        return _fallback_sec_per_unit(lang_code, lang_name, voice_rate)

    rate = _normalize_edge_rate(voice_rate)
    out_dir = Path(cache_folder) if cache_folder else Path(tempfile.mkdtemp(prefix='narr_calib_'))
    out_dir.mkdir(parents=True, exist_ok=True)
    rate_tag = rate.replace('%', 'pct').replace('+', 'p').replace('-', 'm')
    out_mp3 = out_dir / f'narration_pace_calib_{_lang_code_key(lang_code)}_{rate_tag}.mp3'

    # Match EdgeTTS dubbing: honor proxy unless edgetts-noproxy.txt exists.
    proxy = None
    if not Path(f'{ROOT_DIR}/edgetts-noproxy.txt').exists():
        proxy = getattr(app_cfg, 'proxy', None) or None

    async def _save_once():
        kwargs = dict(voice=voice, rate=rate, volume='+0%', pitch='+0Hz')
        try:
            communicate = Communicate(sample, proxy=proxy, connect_timeout=5, **kwargs)
        except TypeError:
            communicate = Communicate(sample, **kwargs)
        await asyncio.wait_for(communicate.save(str(out_mp3)), timeout=30)

    last_err = None
    for attempt in range(3):
        try:
            if out_mp3.exists():
                out_mp3.unlink(missing_ok=True)
            asyncio.run(_save_once())
            if not out_mp3.exists() or out_mp3.stat().st_size < 100:
                raise RuntimeError('empty pace probe audio')
            duration_ms = float(get_audio_time(str(out_mp3)))
            duration = duration_ms / 1000.0
            if duration < 0.3:
                raise RuntimeError('pace probe audio too short')
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
        except Exception as e:
            last_err = e
            logger.warning(f'TTS pace probe attempt {attempt + 1}/3 failed: {e}')
            if attempt < 2:
                import time as _time
                _time.sleep(1.5 * (attempt + 1))

    logger.exception(f'TTS pace probe failed after retries: {last_err}')
    return _fallback_sec_per_unit(lang_code, lang_name, voice_rate)


def compute_length_budget(source_lines, window_s, spoken_s, source_lang, target_lang,
                          sec_per_tgt, tgt_unit=None):
    """Soft scene budget; measured audio controls the final speed adjustment."""
    if not sec_per_tgt or sec_per_tgt <= .02:
        raise RuntimeError(tr("Could not measure dubbing pace for third-person narration"))
    return {
        'window_s': max(.1, window_s),
        'ideal': max(1, round(max(.1, window_s) / (sec_per_tgt * 1.10))),
        'tgt_unit': tgt_unit or ('characters' if _lang_is_cjk(target_lang) else 'words'),
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

    Prefer sentence/comma boundaries so each cue stays a complete spoken phrase.
    Hard mid-phrase word chops (Latin) created incomplete “câu cụt” lines — the
    AI prompt must emit short complete lines instead. CJK may still fall back to
    character-length splits when a clause has no punctuation.
    """
    text = (text or '').strip()
    if not text:
        return ['…']
    max_units = max(1, int(max_units))
    sentences = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()] or [text]
    out: List[str] = []
    cjk = _lang_is_cjk(lang) or _text_is_cjk_heavy(text)
    for sent in sentences:
        # Break clauses so timestamps follow speech, not one long 1:1 block.
        clauses = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sent) if p and p.strip()] or [sent]
        for clause in clauses:
            cn, _ = count_speaking_units(clause, lang)
            if cn <= max_units or not cjk:
                # Latin: keep complete clause even if over soft unit_cap.
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


LINE_REWRITE_BATCH = 12


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


def narration_prompt(budget, source_lang, target_lang):
    return (
        f"Retell this scene from {source_lang} as third-person narration in {target_lang} only. "
        "Keep every source beat in its original order. Preserve subjects, actions, recipients, "
        "names, relationships, negation and replies. Do not invent events or omit meaning. "
        "Use short complete spoken phrases, about 2–4 seconds each, one phrase per line. "
        "Split long sentences at natural clause boundaries; do not merge neighboring story beats "
        "into a paragraph. Never split a name or leave a dangling fragment. "
        f"Aim for about {budget['ideal']} {budget['tgt_unit']} over this "
        f"{budget['window_s']:.2f}-second scene. This is a guide, not permission to delete content. "
        "Use concise natural wording; timing will be adjusted after speech synthesis. "
        "Output only the narration, one short cue per line, inside <TRANSLATE_TEXT>.\n"
        + "\n<INPUT>\n{batch_input}\n</INPUT>"
    )


def _scene_narration(model, source_text, translate_type):
    """Checkpoint validated scene text independently of per-run media caches."""
    identity = dict(version=1, channel=translate_type, prompt=model.prompt, source=source_text)
    for name in ('api_url', 'model_name', 'temperature', 'max_tokens',
                 'reasoning_effort', 'extra_body'):
        identity[name] = getattr(model, name, None)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = Path(TEMP_ROOT, 'translate_cache', 'narration', key + '.txt')
    use_cache = not getattr(model, 'is_test', False)
    if use_cache and cache.is_file():
        lines = clean_narration_lines(cache.read_text(encoding='utf-8'))
        if lines:
            return lines
    lines = clean_narration_lines(model._item_task(source_text) or '')
    if not lines:
        raise RuntimeError(tr("Third-person narration returned empty text for a scene"))
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publication prevents interrupted writes becoming valid checkpoints.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=cache.parent, delete=False) as out:
                temporary = Path(out.name)
                out.write('\n'.join(lines))
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return lines


def run_third_person_narration(*, text_list, translate_type, source_code='', target_code='',
                               voice_role='', voice_rate='+0%', cache_folder='', uuid=None):
    """One rewrite per scene; shared media alignment handles timing without text loss."""
    from videotrans.translator._openaicompat import OpenAICampat
    from videotrans.translator._runner import get_model_transobj
    if int(translate_type) not in AI_TRANS_CHANNELS:
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))
    model = get_model_transobj(translate_type=translate_type, uuid=uuid)
    if not isinstance(model, OpenAICampat):
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))
    if getattr(model, 'ainame', '') == 'openrouter':
        model.extra_body = {"reasoning": {"enabled": False}}
    model.temperature = 0.4
    source_lang, target_lang = get_source_target_code(
        show_source=source_code, show_target=target_code, translate_type=translate_type)
    source_lang = source_lang or source_code or 'auto'
    target_lang = target_lang or target_code
    pace = probe_target_sec_per_unit(lang_code=target_code, lang_name=target_lang,
                                     voice_role=voice_role, voice_rate=voice_rate,
                                     cache_folder=cache_folder)
    times = [(int(it['start_time']), int(it['end_time'])) for it in text_list]
    results = []
    for lo, hi in expand_scene_spans(times):
        source_lines = [text_list[i]['text'].strip() for i in range(lo, hi + 1)
                        if text_list[i]['text'].strip()]
        if not source_lines:
            continue
        window_s, spoken_s = scene_timing(times, lo, hi)
        budget = compute_length_budget(source_lines, window_s, spoken_s, source_lang, target_lang,
                                       sec_per_tgt=pace['sec_per_unit'], tgt_unit=pace['unit'])
        model.prompt = narration_prompt(budget, source_lang, target_lang)
        new_lines = _scene_narration(model, '\n'.join(source_lines), translate_type)
        # Keep every returned word even when the estimated budget is exceeded.
        for a, b, text in build_timed_cues(new_lines, times[lo][0], times[hi][1],
                                         target_lang, pace['sec_per_unit']):
            results.append(_make_srt_item(len(results) + 1, a, b, text))
    return results
