"""FFmpeg progress estimates; no synthetic percent or wall-clock-only ETA."""
import math


def render_progress(lines, duration_ms):
    fields = {}
    for line in lines:
        key, sep, value = line.strip().partition('=')
        if sep:
            fields[key] = value
    try:
        elapsed = float(fields.get('out_time_us', fields.get('out_time_ms', 'nan'))) / 1000
        speed = float(fields.get('speed', 'nan').rstrip('x'))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(elapsed) or elapsed < 0 or duration_ms <= 0:
        return None
    percent = min(100, elapsed / duration_ms * 100)
    eta = None
    if elapsed >= 5000 and math.isfinite(speed) and speed > 0:
        eta = max(0, (duration_ms - elapsed) / (1000 * speed))
    return percent, eta


def progress_message(label, lines, duration_ms):
    result = render_progress(lines, duration_ms)
    if result is None:
        return label + ' | estimating…'
    percent, eta = result
    if percent >= 100:
        return f'{label} | 100% | finalizing file…'
    remaining = 'estimating…' if eta is None else f'~{math.ceil(eta / 60)} min remaining'
    return f'{label} | {percent:.1f}% | {remaining}'
