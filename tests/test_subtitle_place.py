from videotrans.task._stage_subtitle import (
    _margin_v_in_cover, _pieces_for_width, _place_single_line_subtitle,
    _sequence_one_line, _single_subtitle_line,
)


def test_line_is_centered_in_the_cover_band():
    # Band flush to the bottom: height 0.14 of a 288-tall playres, font 13.
    assert _margin_v_in_cover(288, 13, 0.86, 0.14) == 14


def test_place_keeps_configured_font_and_moves_onto_the_blur(tmp_path):
    ass = tmp_path / 'end.ass'
    ass.write_text(
        '[Script Info]\nPlayResX: 384\nPlayResY: 288\nWrapStyle: 0\n\n'
        '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, '
        'OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, '
        'Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n'
        'Style: Default,Arial,13,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,1,0,0,0,'
        '100,100,0,0,1,0.5,0.5,2,10,10,10,1\n\n'
        '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
        'Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,mot cau rat dai\n',
        encoding='utf-8')
    _place_single_line_subtitle(str(ass), (0.0, 0.86, 1.0, 0.14))
    text = ass.read_text(encoding='utf-8')
    assert 'WrapStyle: 2' in text
    assert '{\\fs' not in text
    assert 'mot cau rat dai' in text
    assert ',2,10,10,14,1' in text


def test_long_cue_keeps_the_later_words():
    text = 'nói xong đã bị oán khí cuồn cuộn của Lưu Linh Linh đánh bay, người trong sân cũng không dám thở'
    assert _single_subtitle_line(text) == text


def test_wide_line_takes_turns_on_one_row():
    from videotrans.task._stage_subtitle import _text_width_ems
    pieces = _pieces_for_width('một hai ba bốn', max_ems=4)
    assert len(pieces) > 1
    assert all(_text_width_ems(piece) <= 4 for piece in pieces)
    assert ' '.join(pieces) == 'một hai ba bốn'
    line = (
        'Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,'
        'nói xong đã bị oán khí cuồn cuộn của Lưu Linh Linh đánh bay người trong sân cũng không dám thở'
    )
    rows = _sequence_one_line(line, max_ems=8).splitlines()
    assert len(rows) > 1
    assert rows[0].split(',', 3)[1] == '0:00:00.00'
    assert rows[-1].split(',', 3)[2] == '0:00:02.00'
    shown = ' '.join(row.split(',', 9)[-1] for row in rows)
    assert shown.endswith('dám thở')
    assert 'Lưu Linh Linh' in shown
    assert all('\\N' not in row and '{\\fs' not in row for row in rows)
