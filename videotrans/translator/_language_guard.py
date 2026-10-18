"""Conservative script checks, not semantic language identification."""
import regex


def validate_narration_language(lines, source_code, target_code):
    """Reject source-script sentences when the target uses another script.

    Names and short quotations are allowed. Languages sharing a script cannot
    be distinguished by this check; unknown scripts are deliberately skipped.
    """
    def scripts(code):
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

    source, target = scripts(source_code), scripts(target_code)
    foreign = source - target
    if not target or not foreign:
        return
    pattern = regex.compile('[' + ''.join(r'\p{Script=' + s + '}' for s in foreign) + ']')
    for index, text in enumerate(lines, 1):
        letters = sum(c.isalpha() for c in text)
        count = len(pattern.findall(text))
        if count >= 8 and count > letters * .6:
            raise ValueError(f'Narration cue {index} is still predominantly in the source '
                             f'script ({source_code} → {target_code}); translation required')
