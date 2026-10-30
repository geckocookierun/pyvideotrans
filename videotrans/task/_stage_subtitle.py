import json
import os
import re
from pathlib import Path
from typing import Union

from videotrans import translator
from videotrans.configure.config import tr, ROOT_DIR, settings, logger
from videotrans.configure import constants
from videotrans.configure.excepts import VideoTransError
from videotrans.util.help_srt import get_subtitle_from_srt, simple_wrap, set_ass_font
from videotrans.util._subtitle_display import split_display_cues


class SubtitleMixin:

    def _process_subtitles(self) -> Union[tuple[str, str], None]:
        logger.debug(f"\n======准备要嵌入的字幕:{self.cfg.subtitle_type=}===={self.cfg.target_sub=}=")
        if not Path(self.cfg.target_sub).exists() :
            raise VideoTransError(tr("No valid subtitle file exists")+self.cfg.target_sub)

        if (getattr(self.cfg, 'narration_mode', 'none') == 'third'
                or not Path(self.cfg.source_sub).exists()
                or self.cfg.source_language_code == self.cfg.target_language_code):
            if self.cfg.subtitle_type == 3:
                self.cfg.subtitle_type = 1
            elif self.cfg.subtitle_type == 4:
                self.cfg.subtitle_type = 2

        process_end_subtitle = self.cfg.cache_folder + f'/end.srt'
        maxlen = int(
            settings.get('cjk_len', 15) if self.cfg.target_language_code[:2] in constants.CJK_LANG else
            settings.get('other_len', 60))
        # Single-language burn uses the dubbed cue times as-is, one line each.
        # Bilingual burn still splits and wraps, so that path is unchanged.
        source_rows = get_subtitle_from_srt(self.cfg.target_sub)
        one_line = self.cfg.subtitle_type not in [3, 4]
        target_sub_list = source_rows if one_line else split_display_cues(source_rows)

        srt_string = ""
        _join_flag = ''
        if self.cfg.subtitle_type in [3, 4]:
            source_sub_list = get_subtitle_from_srt(self.cfg.source_sub)
            source_length = len(source_sub_list)
            source_maxlen = int(
                settings.get('cjk_len', 15) if self.cfg.source_language_code.split('-')[0] in constants.CJK_LANG   else
                settings.get('other_len', 60))

            _join_flag = self._get_join_flag()

            for i, it in enumerate(target_sub_list):
                _text = simple_wrap(it['text'].strip(), maxlen, self.cfg.target_language_code)
                srt_string += f"{it['line']}\n{it['time']}\n"
                source_index = it['source_index']
                if source_index < source_length:
                    _text_source = simple_wrap(source_sub_list[source_index]['text'], source_maxlen,
                                                     self.cfg.source_language_code)
                    _text = f'{_text_source}\n{_join_flag}{_text}' if self.cfg.output_srt == 1 else f'{_text}\n{_join_flag}{_text_source}'
                srt_string += f"{_text}\n\n"
            srt_string = srt_string.strip()
            process_end_subtitle = f"{self.cfg.cache_folder}/shuang.srt"
            Path(process_end_subtitle).write_text(srt_string, encoding='utf-8')
            Path(self.cfg.target_dir + "/shuang.srt").write_text(
                srt_string.replace('###', '') if _join_flag == '###' else srt_string, encoding='utf-8')
        else:
            for i, it in enumerate(target_sub_list):
                tmp = _single_subtitle_line(it['text'])
                srt_string += f"{it['line']}\n{it['time']}\n{tmp}\n\n"
            with Path(process_end_subtitle).open('w', encoding='utf-8') as f:
                f.write(srt_string)

        subtitle_langcode = translator.get_subtitle_code(show_target=self.cfg.target_language)
        logger.debug(
            f'最终确定字幕嵌入类型:{self.cfg.subtitle_type} ,目标字幕语言:{subtitle_langcode}, 字幕文件:{process_end_subtitle}\n')
        if self.cfg.subtitle_type in [2, 4]:
            return os.path.basename(process_end_subtitle), subtitle_langcode

        process_end_subtitle_ass = set_ass_font(process_end_subtitle)
        if one_line:
            cover_box = None
            if getattr(self.cfg, 'cover_source_hardsub', False):
                from videotrans.task._stage_assemble import cover_hardsub_geometry
                cover_box = cover_hardsub_geometry(
                    getattr(self.cfg, 'cover_source_hardsub_pct', 0.14),
                    getattr(self.cfg, 'cover_hardsub_x', 0),
                    getattr(self.cfg, 'cover_hardsub_y', -1),
                    getattr(self.cfg, 'cover_hardsub_w', 1),
                )
            _place_single_line_subtitle(process_end_subtitle_ass, cover_box)
        basename = os.path.basename(process_end_subtitle_ass)
        return basename, subtitle_langcode

    def _get_join_flag(self):
        _join_flag = ""
        if self.cfg.subtitle_type != 3 or not Path(f'{ROOT_DIR}/videotrans/ass.json').exists():
            return _join_flag
        try:
            assjson = json.loads(Path(f'{ROOT_DIR}/videotrans/ass.json').read_text(encoding='utf-8'))
        except Exception:
            logger.warning(f'未自定义样式 ass.json ，忽略')
            return _join_flag
        else:
            for k, v in assjson.items():
                if k.startswith('Bottom_') and v != assjson.get(k[7:]):
                    _join_flag = '###'
                    break
        return _join_flag


def _single_subtitle_line(text: str) -> str:
    """One dubbed cue keeps its full text and its original timestamps.

    Breaks inside the cue become spaces. A line that does not fit the frame
    is later cut into pieces and shown one after another on that same line.
    """
    return re.sub(r'\s*\n\s*', ' ', (text or '').strip())


def _text_width_ems(text: str) -> float:
    total = 0.0
    for char in text:
        if char.isspace():
            total += 0.33
        elif ord(char) > 0x2E80:
            total += 1.0
        else:
            total += 0.52
    return total


def _pieces_for_width(text: str, max_ems: float) -> list:
    """Cut a line into pieces that each fit one row, keeping every word."""
    text = (text or '').strip()
    if not text or max_ems <= 0 or _text_width_ems(text) <= max_ems:
        return [text] if text else []

    def take(token: str) -> list:
        if _text_width_ems(token) <= max_ems:
            return [token]
        out, buf = [], ''
        for char in token:
            trial = buf + char
            if buf and _text_width_ems(trial) > max_ems:
                out.append(buf)
                buf = char
            else:
                buf = trial
        if buf:
            out.append(buf)
        return out

    pieces, current = [], ''
    for word in text.split(' '):
        if not word:
            continue
        for part in take(word):
            trial = part if not current else current + ' ' + part
            if current and _text_width_ems(trial) > max_ems:
                pieces.append(current)
                current = part
            else:
                current = trial
    if current:
        pieces.append(current)
    return pieces or [text]


def _ass_to_cs(value: str) -> int:
    hours, minutes, seconds = value.strip().split(':')
    return int(round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 100))


def _cs_to_ass(value: int) -> str:
    value = max(0, int(value))
    hours, value = divmod(value, 360000)
    minutes, value = divmod(value, 6000)
    seconds, centis = divmod(value, 100)
    return f'{hours}:{minutes:02d}:{seconds:02d}.{centis:02d}'


def _sequence_one_line(line: str, max_ems: float) -> str:
    """Replace one wide dialogue with timed pieces on the same single line."""
    fields = line.split(',', 9)
    if len(fields) < 10:
        return line
    plain = re.sub(r'\{[^}]*\}', '', fields[9]).replace('\\N', ' ').replace('\\n', ' ')
    plain = re.sub(r'\s+', ' ', plain).strip()
    pieces = _pieces_for_width(plain, max_ems)
    if len(pieces) <= 1:
        fields[9] = plain or fields[9]
        return ','.join(fields)
    start, end = _ass_to_cs(fields[1]), _ass_to_cs(fields[2])
    span = max(0, end - start)
    weights = [max(1, len(piece)) for piece in pieces]
    total = sum(weights)
    rows, cursor, used = [], start, 0
    for index, piece in enumerate(pieces):
        used += weights[index]
        nxt = end if index == len(pieces) - 1 else start + int(round(span * used / total))
        nxt = min(end, max(cursor + (0 if index == len(pieces) - 1 else 1), nxt))
        if index == len(pieces) - 1:
            nxt = end
        fields[1], fields[2], fields[9] = _cs_to_ass(cursor), _cs_to_ass(nxt), piece
        rows.append(','.join(fields))
        cursor = nxt
    return '\n'.join(rows)


def _margin_v_in_cover(play_y: int, font_size: float, y: float, h: float) -> int:
    """Alignment 2: pixels from the frame bottom to the bottom of one line.

    The line stays the configured size and is centered in the blur band.
    """
    below = play_y * (1.0 - (y + h))
    band = play_y * h
    lift = max(0.0, (band - font_size) / 2.0)
    return max(0, int(round(below + lift)))


def _place_single_line_subtitle(ass_path: str, cover_box=None) -> None:
    """Keep the configured font on one line, sitting on the blur when set.

    A sentence wider than the frame is cut into pieces and those pieces take
    turns on that same line for the cue's original time. The dubbed cue is
    not changed.
    """
    path = Path(ass_path)
    if not path.is_file():
        return
    content = path.read_text(encoding='utf-8-sig')
    play_x, play_y = 384, 288
    found_x = re.search(r'(?m)^PlayResX:\s*(\d+)', content)
    found_y = re.search(r'(?m)^PlayResY:\s*(\d+)', content)
    if found_x:
        play_x = int(found_x.group(1))
    if found_y:
        play_y = int(found_y.group(1))
    if re.search(r'(?m)^WrapStyle:', content):
        content = re.sub(r'(?m)^WrapStyle:.*$', 'WrapStyle: 2', content, count=1)
    else:
        content = content.replace('[Script Info]', '[Script Info]\nWrapStyle: 2', 1)

    def _restyle(match):
        parts = match.group(1).split(',')
        if len(parts) < 23 or parts[0].strip() != 'Default' or not cover_box:
            return match.group(0)
        x, y, w, h = cover_box
        font_size = 13.0
        raw_size = parts[2].strip()
        if raw_size.replace('.', '', 1).isdigit():
            font_size = float(raw_size)
        parts[-2] = str(_margin_v_in_cover(play_y, font_size, y, h))
        if x > 0.01 or w < 0.98:
            parts[-4] = str(max(0, int(round(play_x * x))))
            parts[-3] = str(max(0, int(round(play_x * (1.0 - x - w)))))
        return 'Style: ' + ','.join(parts)

    content = re.sub(r'(?m)^Style:\s*(.*)$', _restyle, content)
    font_size, margin_l, margin_r = 13.0, 10, 10
    style = re.search(r'(?m)^Style:\s*Default,.*$', content)
    if style:
        parts = style.group(0).split(',')
        if len(parts) >= 23 and parts[2].strip().replace('.', '', 1).isdigit():
            font_size = float(parts[2])
            margin_l, margin_r = int(float(parts[-4])), int(float(parts[-3]))
    usable = max(1.0, (play_x - margin_l - margin_r) * 0.96)
    max_ems = usable / max(font_size, 1.0)

    def _page(match):
        return _sequence_one_line(match.group(0), max_ems)

    content = re.sub(r'(?m)^Dialogue:.*$', _page, content)
    path.write_text(content, encoding='utf-8')
