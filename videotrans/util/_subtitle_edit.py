"""Read edits without dropping rows that a lazy-loaded table has not rendered."""
from videotrans.util._srt_parse import ms_to_time_string


def edited_cue(table, row, data, start_column, end_column, text_column):
    start_cell = table.item(row, start_column)
    end_cell = table.item(row, end_column)
    text_cell = table.item(row, text_column)
    start = round(float(start_cell.text().strip()) * 1000) if start_cell is not None else round(data['start_time'])
    end = round(float(end_cell.text().strip()) * 1000) if end_cell is not None else round(data['end_time'])
    if start < 0 or end < start:
        raise ValueError(f'Invalid subtitle interval at row {row + 1}')
    text = text_cell.text().strip() if text_cell is not None else data['text'].strip()
    return ms_to_time_string(ms=start), ms_to_time_string(ms=end), text
