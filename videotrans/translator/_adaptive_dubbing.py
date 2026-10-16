# -*- coding: utf-8 -*-
"""Duration-aware dialogue adaptation for video dubbing.

Translation uses neighboring dialogue as context and stays inside bounded scenes.
Every generated line keeps source cue references so dialogue can move inside a
scene without drifting away from the picture.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from videotrans.configure.config import logger, settings, tr
from videotrans.translator._constants import AI_TRANS_CHANNELS
from videotrans.translator._lang_utils import get_source_target_code
from videotrans.translator._narration import (
    _make_srt_item,
    count_lines_units,
    count_speaking_units,
    probe_target_sec_per_unit,
)


META_FILENAME = "adaptive_dubbing.json"
ALGORITHM_VERSION = 7
SCENE_SILENCE_PAD_CAP_MS = 750
MAX_SCENE_REQUESTS = 4
CONTEXT_BEFORE_CUES = 6
CONTEXT_AFTER_CUES = 4
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def adaptive_metadata_path(cache_folder: str) -> str:
    return str(Path(cache_folder) / META_FILENAME)


def _load_metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("scenes", []), list):
            return value
        return {}
    except (OSError, ValueError):
        return {}


def _save_metadata(path: Path, meta: dict) -> None:
    """Replace a checkpoint only after the full JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_adaptive_scene_spans(
        times: Sequence[Tuple[int, int]],
        *,
        scene_gap_ms: int,
        max_scene_ms: int,
) -> List[Tuple[int, int]]:
    """Split cues on meaningful pauses while preventing oversized LLM scenes."""
    if not times:
        return []
    scene_gap_ms = max(1, int(scene_gap_ms))
    max_scene_ms = max(3000, int(max_scene_ms))
    spans: List[Tuple[int, int]] = []
    lo = 0
    for i in range(1, len(times)):
        gap = int(times[i][0]) - int(times[i - 1][1])
        would_span = int(times[i][1]) - int(times[lo][0])
        if gap >= scene_gap_ms or would_span > max_scene_ms:
            spans.append((lo, i - 1))
            lo = i
    spans.append((lo, len(times) - 1))
    return spans


def _cue_id(index: int) -> str:
    return f"S{index + 1}"


def _source_text(item) -> str:
    value = item.get("text", "") if hasattr(item, "get") else item["text"]
    return (value or "").strip()


def _build_scene_context(text_list: Sequence, lo: int, hi: int) -> str:
    """Store only the local window, including cues needed by partial retries."""
    return "\n".join(
        f"{_cue_id(i)}: {_source_text(text_list[i])}"
        for i in range(max(0, lo - CONTEXT_BEFORE_CUES), min(len(text_list), hi + CONTEXT_AFTER_CUES + 1))
        if _source_text(text_list[i])
    )


def _scene_dialogue_context(global_context: str, scene_sources: Sequence[dict]) -> str:
    """Keep adjacent turns, without repeating the requested cues as context."""
    rows = global_context.splitlines()
    numbered = []
    for row in rows:
        match = re.match(r'^S(\d+):', row)
        if match:
            numbered.append((int(match.group(1)), row))
    indices = [int(row['id'][1:]) for row in scene_sources if re.fullmatch(r'S\d+', row['id'])]
    if not numbered or not indices:
        return global_context
    lo, hi = min(indices), max(indices)
    before = [row for index, row in numbered if lo - CONTEXT_BEFORE_CUES <= index < lo]
    after = [row for index, row in numbered if hi < index <= hi + CONTEXT_AFTER_CUES]
    return 'PRECEDING DIALOGUE:\n' + '\n'.join(before) + '\nFOLLOWING DIALOGUE:\n' + '\n'.join(after)


def adaptive_rewrite_prompt(
        *,
        scene_sources: Sequence[dict],
        global_context: str,
        source_lang: str,
        target_lang: str,
        max_units: int,
        unit: str,
        window_s: float,
        missing_ids: Sequence[str] = (),
        format_error: str = "",
        tts_feedback: str = "",
) -> str:
    source_rows = "\n".join(f"{row['id']}: {row['text']}" for row in scene_sources)
    required = ", ".join(row["id"] for row in scene_sources)
    feedback = []
    if missing_ids:
        feedback.append("Return the omitted source ids: " + ", ".join(missing_ids) + ".")
    if format_error:
        feedback.append("The previous answer was invalid: " + format_error + ". Return corrected JSON.")
    if tts_feedback:
        feedback.append(tts_feedback)
    return (
        f"Translate the source dialogue from {source_lang or 'auto'} into {target_lang} for dubbing. "
        f"Every output text must be in {target_lang}, including transliterated names. "
        "Meaning and speaker roles take priority over brevity. Translate faithfully; "
        "shorten only as needed for the time budget.\n"
        "Read neighboring dialogue to resolve references. Each source cue can contain multiple speakers. "
        "Keep utterances in their original order, with a single speaking turn per output record. "
        "Never combine different speakers in one record. Preserve the grammatical subject, action, "
        "and recipient of every clause. "
        "A kinship/title noun can refer to the speaker, listener, or someone else: resolve it from the "
        "source grammar and context. Never carry pronouns across a change of speaker or replace an "
        "explicit subject with a different person. Preserve questions, replies, negation, names and "
        "relationships; do not invent a relationship to choose a pronoun.\n"
        f"Time budget: {window_s:.2f}s; at most {max_units} {unit} total. "
        "Split at natural phrase or speaker boundaries; do not omit meaning to make lines short. "
        "Reconstruct clauses across cue breaks before translating; do not translate broken words "
        "in isolation. Keep proper names consistent with the source.\n"
        "Do not drop a statement or reply to meet the word budget. The budget is a shortening "
        "target, never permission to omit source meaning.\n"
        + '\n'.join(feedback) + '\n'
        "Context only (do not translate outside the requested source cues):\n"
        + _scene_dialogue_context(global_context, scene_sources) + '\n'
        f"Required source ids: {required}.\n"
        "Cover every source id where its meaning is translated. Return only a JSON array with "
        "source_ids and text, where several source ids may share a line and one source id may have "
        "several lines. No explanation, no speaker labels.\n"
        '[{"source_ids":["S1","S2"],"text":"translated dialogue"}]\n'
        f"TRANSLATE THESE CUES INTO {target_lang}:\n{source_rows}\n"
    )


def parse_adaptive_records(raw: str, valid_ids: Sequence[str]) -> List[dict]:
    """Parse and validate structured LLM output without silently restoring source text."""
    text = (raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"</?TRANSLATE_TEXT>", "", text, flags=re.I).strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    left, right = text.find("["), text.rfind("]")
    if left < 0 or right < left:
        raise ValueError("adaptive rewrite did not return a JSON array")
    payload = text[left:right + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as original_error:
        # Smaller/cheaper chat models occasionally omit a separator even when
        # explicitly asked for JSON. Repair only unambiguous structural slips;
        # source-id validation below still prevents accepting unrelated text.
        repaired = re.sub(r"}\s*{", "},{", payload)
        repaired = re.sub(
            r"]\s*(?=[\"']?text[\"']?\s*:)",
            "],",
            repaired,
            flags=re.I,
        )
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            data = _parse_loose_adaptive_records(payload)
            if not data:
                raise original_error
    if not isinstance(data, list) or not data:
        raise ValueError("adaptive rewrite returned an empty JSON array")
    valid = set(valid_ids)
    records: List[dict] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        content = row.get("text")
        if not isinstance(content, str):
            continue
        content = content.strip()
        ids = row.get("source_ids") or []
        if isinstance(ids, str):
            ids = re.findall(r"S\d+", ids, flags=re.I)
        if not isinstance(ids, list):
            continue
        ids = [str(x).upper() for x in ids if str(x).upper() in valid]
        ids = list(dict.fromkeys(ids))
        if content and ids:
            records.append({"source_ids": ids, "text": content})
    if not records:
        raise ValueError("adaptive rewrite returned no valid dialogue records")
    return records


def _parse_loose_adaptive_records(payload: str) -> List[dict]:
    """Recover schema-shaped records from slightly malformed model JSON.

    This deliberately requires both ``source_ids`` and ``text`` in each
    candidate. It does not guess dialogue from arbitrary prose.
    """
    starts = list(re.finditer(r"[\"']?source_ids[\"']?\s*:\s*\[", payload, flags=re.I))
    rows: List[dict] = []
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(payload)
        chunk = payload[match.start():stop]
        ids_end = chunk.find("]", match.end() - match.start())
        if ids_end < 0:
            continue
        ids = re.findall(r"S\d+", chunk[:ids_end + 1], flags=re.I)
        text_match = re.search(r"[\"']?text[\"']?\s*:\s*", chunk[ids_end + 1:], flags=re.I)
        if not ids or not text_match:
            continue
        value_start = ids_end + 1 + text_match.end()
        value = chunk[value_start:]
        object_end = value.rfind("}")
        if object_end >= 0:
            value = value[:object_end]
        value = value.strip().rstrip(",").strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        value = value.replace(r'\"', '"').replace(r"\'", "'").strip()
        if value:
            rows.append({"source_ids": ids, "text": value})
    return rows


def missing_source_ids(records: Sequence[dict], required_ids: Sequence[str]) -> List[str]:
    covered = {sid for row in records for sid in row.get("source_ids", [])}
    return [sid for sid in required_ids if sid not in covered]


def _source_order(record: dict) -> int:
    values = []
    for source_id in record.get("source_ids", []):
        try:
            values.append(int(str(source_id)[1:]))
        except (TypeError, ValueError):
            continue
    return min(values) if values else 10**9


def _anchor_for_ids(source_ids: Sequence[str], source_by_id: Dict[str, dict]) -> int:
    mids = [
        (int(source_by_id[sid]["start_ms"]) + int(source_by_id[sid]["end_ms"])) // 2
        for sid in source_ids if sid in source_by_id
    ]
    return int(round(sum(mids) / len(mids))) if mids else 0


def schedule_scene_records(
        records: Sequence[dict],
        *,
        scene_start_ms: int,
        scene_end_ms: int,
        durations_ms: Sequence[int],
) -> List[Tuple[int, int]]:
    """Place dialogue near semantic anchors while staying inside scene bounds."""
    if not records:
        return []
    start = int(scene_start_ms)
    end = max(start + len(records), int(scene_end_ms))
    available = end - start
    durations = [max(1, int(x)) for x in durations_ms]
    if len(durations) < len(records):
        durations += [1] * (len(records) - len(durations))
    durations = durations[:len(records)]
    total = sum(durations)
    if total > available:
        scale = available / max(total, 1)
        durations = [max(1, int(round(x * scale))) for x in durations]
        # Rounding may leave a tiny overflow.
        while sum(durations) > available:
            idx = max(range(len(durations)), key=durations.__getitem__)
            durations[idx] -= 1

    anchors = []
    previous = start
    for row in records:
        anchor = int(row.get("anchor_ms") or previous)
        anchor = min(max(anchor, start), end)
        anchor = max(anchor, previous)
        anchors.append(anchor)
        previous = anchor

    desired = [max(start, min(end - durations[i], anchors[i] - durations[i] // 2)) for i in range(len(records))]
    starts: List[int] = []
    cursor = start
    for i, wanted in enumerate(desired):
        s = max(cursor, wanted)
        starts.append(s)
        cursor = s + durations[i]

    if cursor > end:
        cursor = end
        for i in range(len(records) - 1, -1, -1):
            starts[i] = min(starts[i], cursor - durations[i])
            cursor = starts[i]
        if starts[0] < start:
            cursor = start
            for i in range(len(records)):
                starts[i] = cursor
                cursor += durations[i]

    return [(s, min(end, s + durations[i])) for i, s in enumerate(starts)]


def _scene_fit_bounds(
        meta: dict,
        scene_index: int,
        pad_cap_ms: int = SCENE_SILENCE_PAD_CAP_MS,
) -> Tuple[int, int]:
    """Borrow a small, non-overlapping share of silence around a scene."""
    scenes = meta.get("scenes", [])
    scene = scenes[scene_index]
    if "fit_start_ms" in scene and "fit_end_ms" in scene:
        return int(scene["fit_start_ms"]), int(scene["fit_end_ms"])
    start = int(scene["start_ms"])
    end = int(scene["end_ms"])
    cap = max(0, int(pad_cap_ms))
    if scene_index > 0:
        previous_end = int(scenes[scene_index - 1]["end_ms"])
        gap = max(0, start - previous_end)
        start -= min(cap, gap // 2)
    if scene_index + 1 < len(scenes):
        next_start = int(scenes[scene_index + 1]["start_ms"])
        gap = max(0, next_start - end)
        end += min(cap, gap // 2)
    return start, end


def _build_scene_plans(
        text_list: Sequence,
        *,
        times: Sequence[Tuple[int, int]],
        spans: Sequence[Tuple[int, int]],
        sec_per_unit: float,
        safety: float,
) -> List[dict]:
    """Plan every scene and its usable silence before asking the LLM to write."""
    plans: List[dict] = []
    for lo, hi in spans:
        sources = []
        for index in range(lo, hi + 1):
            source_text = _source_text(text_list[index])
            if source_text:
                sources.append({
                    "id": _cue_id(index),
                    "text": source_text,
                    "start_ms": int(times[index][0]),
                    "end_ms": int(times[index][1]),
                })
        if not sources:
            continue
        plans.append({
            "source_lo": lo,
            "source_hi": hi,
            "start_ms": int(times[lo][0]),
            "end_ms": int(times[hi][1]),
            "source": sources,
            "global_context": _build_scene_context(text_list, lo, hi),
            "records": [],
        })

    planning_meta = {"scenes": plans}
    for scene_index, scene in enumerate(plans):
        fit_start, fit_end = _scene_fit_bounds(planning_meta, scene_index)
        scene["fit_start_ms"] = fit_start
        scene["fit_end_ms"] = fit_end
        usable_s = max(0.2, (fit_end - fit_start) / 1000.0)
        scene["max_units"] = max(
            2,
            int(usable_s / max(float(sec_per_unit), 0.02) * float(safety)),
        )
    return plans


def _make_model(translate_type: int, uuid=None):
    if int(translate_type) not in AI_TRANS_CHANNELS:
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))
    from videotrans.translator._openaicompat import OpenAICampat
    from videotrans.translator._runner import get_model_transobj

    model = get_model_transobj(translate_type=translate_type, uuid=uuid)
    if not isinstance(model, OpenAICampat):
        raise RuntimeError(tr("Third-person narration needs an AI chat channel"))
    if getattr(model, "ainame", "") == "openrouter":
        model.extra_body = {"reasoning": {"enabled": False}}
    model.temperature = 0.25
    return model


def _rewrite_scene(
        *,
        model,
        scene_sources: Sequence[dict],
        global_context: str,
        source_lang: str,
        target_lang: str,
        max_units: int,
        unit: str,
        window_s: float,
        tts_feedback: str = "",
) -> List[dict]:
    required = [row["id"] for row in scene_sources]
    source_by_id = {row["id"]: row for row in scene_sources}
    pending = list(scene_sources)
    collected: List[dict] = []
    format_error = ""
    for attempt in range(MAX_SCENE_REQUESTS):
        pending_ids = [row["id"] for row in pending]
        joined = "\n".join(f"{row['id']}: {row['text']}" for row in pending)
        if collected:
            used_units = count_lines_units([row["text"] for row in collected], target_lang)
            remaining_units = max(2, max_units - used_units)
            pending_ms = sum(max(1, int(row["end_ms"]) - int(row["start_ms"])) for row in pending)
            total_ms = sum(max(1, int(row["end_ms"]) - int(row["start_ms"])) for row in scene_sources)
            proportional_units = max(2, int(max_units * pending_ms / max(total_ms, 1)))
            request_max_units = min(remaining_units, proportional_units)
            request_window_s = max(0.2, pending_ms / 1000.0)
        else:
            request_max_units = max_units
            request_window_s = window_s
        model.prompt = adaptive_rewrite_prompt(
            scene_sources=pending,
            global_context=global_context,
            source_lang=source_lang,
            target_lang=target_lang,
            max_units=request_max_units,
            unit=unit,
            window_s=request_window_s,
            missing_ids=pending_ids if collected else (),
            format_error=format_error,
            tts_feedback=tts_feedback,
        )
        raw = model._item_task(joined)
        try:
            records = parse_adaptive_records(raw or "", pending_ids)
            if _unexpected_han(records, target_lang):
                raise ValueError(f"Untranslated Han-script text remains; output {target_lang} only")
        except ValueError as exc:
            format_error = str(exc)
            if attempt + 1 < MAX_SCENE_REQUESTS:
                logger.warning(f"Adaptive rewrite returned invalid output; retrying scene: {exc}")
                continue
            raise RuntimeError(
                "Adaptive dubbing rewrite returned invalid output after retrying the scene"
            ) from exc
        format_error = ""
        collected.extend(records)
        missing = missing_source_ids(collected, required)
        if not missing:
            # Word count is an estimate. Only measured TTS overflow warrants
            # shortening complete dialogue; avoid speculative LLM rewrites.
            return sorted(collected, key=_source_order)
        # Preserve accepted dialogue and ask only for the omitted cues. This is
        # both cheaper and more reliable than rewriting the complete scene.
        pending = [source_by_id[source_id] for source_id in missing]
        logger.warning(f"Adaptive rewrite missing source ids {missing}; requesting only missing cues")
    missing = missing_source_ids(collected, required)
    raise RuntimeError(f"Adaptive dubbing rewrite omitted source cues: {', '.join(missing)}")


def _unexpected_han(records: Sequence[dict], target_lang: str) -> bool:
    """Reject untranslated Han text for targets that do not use Han script.

    This is a script guard, not a general language or semantic classifier.
    """
    language = target_lang.strip().lower()
    if re.search(r'chinese|japanese|korean|cantonese|mandarin', language) or re.match(
        r'^(zh|ja|ko|yue)(?:[-_]|$)', language
    ):
        return False
    return any(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]', row['text'])
               for row in records)


def _rewrite_signature(
        text_list: Sequence,
        *,
        source_code: str,
        target_code: str,
        translate_type: int,
        voice_role: str,
        voice_rate: str,
        gap_ms: int,
        max_scene_ms: int,
        safety: float,
        model_config: dict,
) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "model": model_config,
        "scene_silence_pad_cap_ms": SCENE_SILENCE_PAD_CAP_MS,
        "source": [
            [_source_text(row), int(row["start_time"]), int(row["end_time"])]
            for row in text_list
        ],
        "source_code": source_code,
        "target_code": target_code,
        "translate_type": int(translate_type),
        "voice_role": voice_role,
        "voice_rate": voice_rate,
        "gap_ms": gap_ms,
        "max_scene_ms": max_scene_ms,
        "safety": round(safety, 4),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _estimated_durations(records: Sequence[dict], target_lang: str, sec_per_unit: float) -> List[int]:
    return [
        max(250, int(count_speaking_units(row["text"], target_lang)[0] * sec_per_unit * 1000) + 120)
        for row in records
    ]


def _rebuild_output(meta: dict, duration_map: Dict[Tuple[int, int], int] | None = None) -> List[dict]:
    output: List[dict] = []
    line_no = 1
    for scene_index, scene in enumerate(meta["scenes"]):
        records = scene["records"]
        estimates = _estimated_durations(records, meta["target_lang"], float(meta["sec_per_unit"]))
        if duration_map:
            estimates = [duration_map.get((scene_index, i), estimates[i]) for i in range(len(records))]
        fit_start, fit_end = _scene_fit_bounds(meta, scene_index)
        spans = schedule_scene_records(
            records,
            scene_start_ms=fit_start,
            scene_end_ms=fit_end,
            durations_ms=estimates,
        )
        for row, (a, b) in zip(records, spans):
            item = _make_srt_item(line_no, a, b, row["text"])
            output.append(item)
            row["line"] = line_no
            row["start_ms"] = a
            row["end_ms"] = b
            line_no += 1
    return output


def _set_scene_records(scene: dict, records: List[dict]) -> None:
    by_id = {row["id"]: row for row in scene["source"]}
    for record in records:
        record["anchor_ms"] = _anchor_for_ids(record["source_ids"], by_id)
    scene["records"] = records


def _cached_output(meta: dict, target_sub: str) -> List[dict] | None:
    """Restore saved timing; preserve an existing, user-editable target SRT."""
    if not meta.get("complete"):
        return None
    try:
        records = [row for scene in meta["scenes"] for row in scene["records"]]
        output = [_make_srt_item(i, row["start_ms"], row["end_ms"], row["text"])
                  for i, row in enumerate(records, 1)]
    except (KeyError, TypeError, ValueError):
        return None
    if not output:
        return None
    if target_sub and Path(target_sub).is_file():
        from videotrans.util.help_srt import get_subtitle_from_srt
        existing = get_subtitle_from_srt(target_sub)
        if existing:
            return existing
    return output


def run_adaptive_dubbing_rewrite(
        *,
        text_list: List,
        translate_type: int,
        source_code: str = "",
        target_code: str = "",
        voice_role: str = "",
        voice_rate: str = "+0%",
        cache_folder: str = "",
        target_sub: str = "",
        uuid=None,
) -> List[dict]:
    """Translate once per scene; reuse matching checkpoints before probing TTS."""
    if not any(_source_text(row) for row in text_list):
        return []
    model = _make_model(translate_type, uuid)
    gap_ms = max(1, int(settings.get("adaptive_scene_gap_ms", 900)))
    max_scene_ms = max(3000, int(float(settings.get("adaptive_scene_max_s", 30)) * 1000))
    safety = min(0.95, max(0.65, float(settings.get("adaptive_safety_pct", 85)) / 100.0))
    signature = _rewrite_signature(
        text_list, source_code=source_code, target_code=target_code,
        translate_type=translate_type, voice_role=voice_role, voice_rate=voice_rate,
        gap_ms=gap_ms, max_scene_ms=max_scene_ms, safety=safety,
        model_config={key: getattr(model, key, None) for key in (
            "model_name", "api_url", "temperature", "reasoning_effort", "extra_body",
        )},
    )
    path = Path(adaptive_metadata_path(cache_folder))
    previous = _load_metadata(path)
    if previous.get("version") != ALGORITHM_VERSION or previous.get("rewrite_signature") != signature:
        previous = {}
    cached = _cached_output(previous, target_sub)
    if cached is not None:
        return cached

    source_lang, target_lang = get_source_target_code(
        show_source=source_code, show_target=target_code, translate_type=translate_type,
    )
    source_lang = source_lang or source_code or "auto"
    target_lang = target_lang or target_code or "target language"
    pace = probe_target_sec_per_unit(
        lang_code=target_code, lang_name=target_lang, voice_role=voice_role,
        voice_rate=voice_rate, cache_folder=cache_folder,
    )
    times = [(int(row["start_time"]), int(row["end_time"])) for row in text_list]
    planned_scenes = _build_scene_plans(
        text_list, times=times,
        spans=build_adaptive_scene_spans(times, scene_gap_ms=gap_ms, max_scene_ms=max_scene_ms),
        sec_per_unit=float(pace["sec_per_unit"]), safety=safety,
    )
    cached_scenes = {
        (scene.get("source_lo"), scene.get("source_hi")): scene
        for scene in previous.get("scenes", []) if isinstance(scene, dict)
    }
    meta = {
        "version": ALGORITHM_VERSION,
        "complete": False,
        "rewrite_signature": signature,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_code": source_code,
        "target_code": target_code,
        "translate_type": int(translate_type),
        "sec_per_unit": float(pace["sec_per_unit"]),
        "unit": pace["unit"],
        "voice": pace.get("voice", ""),
        "voice_rate": pace.get("rate", voice_rate),
        "scenes": [],
    }
    for scene_no, scene in enumerate(planned_scenes, 1):
        cached = cached_scenes.get((scene["source_lo"], scene["source_hi"]))
        if cached and not missing_source_ids(cached.get("records", []), [row["id"] for row in scene["source"]]):
            meta["scenes"].append(cached)
            logger.debug(f"Adaptive scene {scene_no}/{len(planned_scenes)} restored from checkpoint")
            continue
        window_s = max(0.2, (scene["fit_end_ms"] - scene["fit_start_ms"]) / 1000.0)
        logger.debug(
            f"Adaptive scene {scene_no}/{len(planned_scenes)} "
            f"window={window_s:.2f}s max={scene['max_units']} {pace['unit']}"
        )
        records = _rewrite_scene(
            model=model, scene_sources=scene["source"], global_context=scene["global_context"],
            source_lang=source_lang, target_lang=target_lang,
            max_units=scene["max_units"], unit=pace["unit"], window_s=window_s,
        )
        _set_scene_records(scene, records)
        meta["scenes"].append(scene)
        _save_metadata(path, meta)

    output = _rebuild_output(meta)
    meta["complete"] = True
    _save_metadata(path, meta)
    logger.debug(f"Adaptive dubbing rewrite: {len(text_list)} source cues -> {len(output)} cues")
    return output


def _audio_slots(durations, available, speed_limits):
    """Allocate integer slots without exceeding any individual voice ceiling."""
    import math
    if isinstance(speed_limits, (int, float)):
        speed_limits = [speed_limits] * len(durations)
    if sum(durations) <= available:
        return list(durations)
    slots = [max(1, math.ceil(d / limit - 1e-9))
             for d, limit in zip(durations, speed_limits)]
    remaining = available - sum(slots)
    if remaining < 0:
        return None
    total = sum(durations)
    extra = remaining
    for i, duration in enumerate(durations):
        share = min(duration - slots[i], extra * duration // total)
        slots[i] += share
        remaining -= share
    for i, duration in enumerate(durations):
        share = min(duration - slots[i], remaining)
        slots[i] += share
        remaining -= share
    return slots


def fit_adaptive_dubbing_from_audio(
        *,
        queue_tts: Sequence[dict],
        cache_folder: str,
        translate_type: int,
        uuid=None,
        allow_rewrite: bool = True,
        allow_soft_speed: bool = False,
) -> Tuple[List[dict], bool]:
    """Measure real TTS, rewrite overflowing scenes, and retime accepted scenes.

    Returns (new subtitles, needs_tts_again).
    """
    from videotrans.task._adaptive_audio import measured_audio_ms, adaptive_speed_limit

    path = Path(adaptive_metadata_path(cache_folder))
    if not path.exists():
        return [], False
    meta = json.loads(path.read_text(encoding="utf-8"))
    queue_by_line = {int(row["line"]): row for row in queue_tts}
    durations: Dict[Tuple[int, int], int] = {}
    overflow: List[Tuple[int, int, int]] = []
    for scene_index, scene in enumerate(meta["scenes"]):
        total = 0
        limits = []
        actual = []
        for record_index, record in enumerate(scene["records"]):
            q = queue_by_line.get(int(record.get("line", -1)))
            if not q or not Path(q.get("filename", "")).is_file():
                raise RuntimeError(f'Missing TTS audio for subtitle {record.get("line")}')
            duration = measured_audio_ms(q["filename"])
            limits.append(adaptive_speed_limit(allow_soft_speed, q.get('role', '')))
            actual.append(duration)
            durations[(scene_index, record_index)] = duration
            total += duration
        fit_start, fit_end = _scene_fit_bounds(meta, scene_index)
        available = fit_end - fit_start
        required_speed_pct = max(0.0, (total / max(available, 1) - 1.0) * 100.0)
        scene["fit"] = {
            "available_ms": available,
            "actual_tts_ms": total,
            "required_speed_pct": round(required_speed_pct, 2),
            "status": "fits",
        }
        slots = _audio_slots(actual, available, limits)
        if slots is None:
            scene["fit"]["status"] = "overflow"
            overflow.append((scene_index, total, available))
        else:
            for i, slot in enumerate(slots):
                durations[(scene_index, i)] = slot
            if total > available:
                scene["fit"]["status"] = "soft_speed"

    if overflow and allow_rewrite:
        model = _make_model(translate_type, uuid)
        for scene_index, actual_ms, available_ms in overflow:
            scene = meta["scenes"][scene_index]
            scene["fit"]["status"] = "rewrite_pending"
            current_units = count_lines_units(
                [row["text"] for row in scene["records"]], meta["target_lang"]
            )
            ratio = available_ms / max(actual_ms, 1)
            new_max = max(2, int(current_units * ratio * 0.90))
            # A provider may ignore the previous cap. Tighten that cap too,
            # otherwise repeated over-budget answers can repeat the same request.
            previous_max = int(scene.get("max_units", current_units))
            new_max = min(new_max, max(2, int(previous_max * ratio * 0.90)))
            previous_dialogue = json.dumps(
                [{"source_ids": row["source_ids"], "text": row["text"]}
                 for row in scene["records"]], ensure_ascii=False,
            )
            tts_feedback = (
                f"The previous dialogue synthesized to {actual_ms / 1000:.2f} seconds, "
                f"but only {available_ms / 1000:.2f} seconds are available. "
                f"It used {current_units} {meta['unit']}. Rewrite it more concisely; "
                "do not repeat the same wording. Preserve all essential meaning and source ids.\n"
                f"PREVIOUS DIALOGUE (too long):\n{previous_dialogue}\n"
            )
            logger.warning(
                f"Adaptive scene {scene_index + 1} TTS overflow: {actual_ms}ms > "
                f"{available_ms}ms; rewrite max {current_units}->{new_max} {meta['unit']}"
            )
            records = _rewrite_scene(
                model=model,
                scene_sources=scene["source"],
                global_context=scene["global_context"],
                source_lang=meta["source_lang"],
                target_lang=meta["target_lang"],
                max_units=new_max,
                unit=meta["unit"],
                window_s=available_ms / 1000.0,
                tts_feedback=tts_feedback,
            )
            _set_scene_records(scene, records)
            scene["max_units"] = new_max
            scene["fit"]["status"] = "rewritten_for_tts"
        output = _rebuild_output(meta)
        _save_metadata(path, meta)
        return output, True

    if overflow:
        _save_metadata(path, meta)
        details = ", ".join(
            f"scene {scene_index + 1}: {actual_ms / 1000:.2f}s/{available_ms / 1000:.2f}s "
            f"(needs {meta['scenes'][scene_index]['fit']['required_speed_pct']:.1f}% speed-up)"
            for scene_index, actual_ms, available_ms in overflow
        )
        raise RuntimeError(
            "Adaptive dubbing could not fit the following scenes after all rewrite attempts: "
            f"{details}. Lower the safety percentage, allow more rewrite attempts, or choose a faster voice. "
            "Enable audio auto-rate to allow speed-up. Hamed and NamMinh have separate "
            "adaptive acceleration limits; other voices have no configured ceiling."
        )
    output = _rebuild_output(meta, durations)
    for scene in meta["scenes"]:
        status = scene.get("fit", {}).get("status")
        if status == "fits":
            scene["fit"]["status"] = "fitted"
        elif status == "soft_speed":
            scene["fit"]["status"] = "fitted_soft_speed"
    _save_metadata(path, meta)
    return output, False
