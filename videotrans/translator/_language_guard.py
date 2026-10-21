"""Conservative script checks, not semantic language identification."""
from pathlib import Path

import regex

from videotrans.configure.excepts import TranslateSrtError


def _scripts(code):
    code = (code or '').lower().replace('_', '-').split('-')[0]
    groups = {
        'Han': 'zh yue', 'Han Hiragana Katakana': 'ja', 'Hangul Han': 'ko',
        'Arabic': 'ar fa ur ps', 'Cyrillic': 'ru uk bg mk be kk ky',
        'Devanagari': 'hi mr ne', 'Thai': 'th', 'Hebrew': 'he',
        'Greek': 'el', 'Bengali': 'bn', 'Tamil': 'ta', 'Telugu': 'te',
        'Latin': 'en vi fr de es pt id ms it nl pl cs sk sl hr ro hu tr az sv da no nb fi et lv lt sw af ca eu gl sq',
    }
    return next((set(script.split()) for script, codes in groups.items()
                 if code in codes.split()), set())


def _foreign_pattern(source_code, target_code):
    """Source-only script pattern. None when the pair cannot be checked."""
    source, target = _scripts(source_code), _scripts(target_code)
    foreign = source - target
    if not target or not foreign:
        return None
    return regex.compile('[' + ''.join(r'\p{Script=' + s + '}' for s in foreign) + ']')


def _still_source_language(text, pattern):
    letters = sum(c.isalpha() for c in text)
    count = len(pattern.findall(text))
    return count >= 8 and count > letters * .6


def incomplete_lines(lines, source_code, target_code):
    """1-based cues that are blank or still predominantly in the source script.

    A short name such as 小明 is allowed. Languages that share a script are skipped.
    """
    pattern = _foreign_pattern(source_code, target_code)
    bad = []
    for index, text in enumerate(lines, 1):
        body = text or ''
        if not body.strip() or (pattern is not None and _still_source_language(body, pattern)):
            bad.append((index, body))
    return bad


def discard_cached_translations(lines):
    """Drop cached translator replies that still contain an untranslated cue."""
    from videotrans.configure.config import TEMP_ROOT
    folder = Path(TEMP_ROOT) / 'translate_cache'
    if not folder.is_dir():
        return
    needles = [line.strip() for line in lines if line and len(line.strip()) >= 8]
    if not needles:
        return
    for path in folder.glob('*.txt'):
        try:
            body = path.read_text(encoding='utf-8')
        except OSError:
            continue
        if any(needle in body for needle in needles):
            path.unlink(missing_ok=True)


def translation_failure(lines, source_code, target_code):
    """Return a translation-stage error, or None when every cue is translated."""
    bad = incomplete_lines(lines, source_code, target_code)
    if not bad:
        return None
    discard_cached_translations([text for _, text in bad])
    preview = []
    for index, text in bad[:6]:
        shown = text.strip() or '(empty)'
        preview.append(f'#{index} {shown[:60]}')
    extra = len(bad) - len(preview)
    tail = f' and {extra} more' if extra else ''
    return TranslateSrtError(
        f'Subtitle translation is incomplete ({source_code} → {target_code}) '
        f'and dubbing was not started. {len(bad)} cue(s) are empty or still in '
        f'the source language: ' + '; '.join(preview) + tail +
        '. Delete the target subtitle and run again to translate those cues.'
    )


def require_complete_translation(lines, source_code, target_code):
    failure = translation_failure(lines, source_code, target_code)
    if failure:
        raise failure


def validate_narration_language(lines, source_code, target_code):
    """Reject source-script sentences when the target uses another script.

    Names and short quotations are allowed. Languages sharing a script cannot
    be distinguished by this check; unknown scripts are deliberately skipped.
    """
    pattern = _foreign_pattern(source_code, target_code)
    if pattern is None:
        return
    for index, text in enumerate(lines, 1):
        if _still_source_language(text or '', pattern):
            raise ValueError(f'Narration cue {index} is still predominantly in the source '
                             f'script ({source_code} → {target_code}); translation required')
