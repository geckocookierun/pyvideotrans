# -*- coding: utf-8 -*-
"""Duration-aware dialogue adaptation for video dubbing.

The source transcript is used as global context, but rewriting happens inside
bounded scenes.  Every generated line keeps references to source cue ids so the
new dialogue can move inside a scene without drifting away from the picture.
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
    _narration_limits,
    count_lines_units,
    count_speaking_units,
    probe_target_sec_per_unit,
)


META_FILENAME = "adaptive_dubbing.json"
ALGORITHM_VERSION = 4
SCENE_SILENCE_PAD_CAP_MS = 750
MAX_SCENE_REQUESTS = 4
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def adaptive_metadata_path(cache_folder: str) -> str:
    return str(Path(cache_folder) / META_FILENAME)


def adaptive_metadata_complete(cache_folder: str) -> bool:
    path = Path(adaptive_metadata_path(cache_folder))
    if not path.exists():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return bool(meta.get("complete")) and meta.get("version") == ALGORITHM_VERSION
    except (OSError, ValueError, TypeError):
        return False


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


def _build_global_context(text_list: Sequence, lo: int, hi: int, limit: int = 30000) -> str:
    """Use the whole transcript when it fits; otherwise keep a wide local window."""
    rows = [f"{_cue_id(i)}: {_source_text(item)}" for i, item in enumerate(text_list) if _source_text(item)]
    full = "\n".join(rows)
    if len(full) <= limit:
        return full

    # Long videos can exceed a model context window. Keep the opening context
    # (names/setting) plus a large neighborhood around the current scene.
    local_lo = max(0, lo - 60)
    local_hi = min(len(text_list) - 1, hi + 60)
    opening = "\n".join(rows[:25])
    local = "\n".join(
        f"{_cue_id(i)}: {_source_text(text_list[i])}"
        for i in range(local_lo, local_hi + 1)
        if _source_text(text_list[i])
    )
    prefix = f"{opening}\n...\n"
    return prefix + local[:max(0, limit - len(prefix))]


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
        budget_error: str = "",
        tts_feedback: str = "",
) -> str:
    line_unit_cap = int(_narration_limits(target_lang).get("unit_cap", 8))
    source_rows = "\n".join(
        f"{row['id']} [{row['start_ms']}..{row['end_ms']} ms]: {row['text']}"
        for row in scene_sources
    )
    required = ", ".join(row["id"] for row in scene_sources)
    missing_note = ""
    if missing_ids:
        missing_note = (
            "\nYour previous answer failed to cover these source ids: "
            f"{', '.join(missing_ids)}. Return a corrected complete answer.\n"
        )
    format_note = ""
    if format_error:
        format_note = (
            "\nYour previous answer was not valid JSON. Return the complete scene again. "
            "Check every comma, quote, bracket and brace carefully.\n"
        )
    budget_note = ""
    if budget_error:
        budget_note = (
            f"\n{budget_error} Return the complete scene again with no more than "
            f"{max_units} {unit}. Shorten wording, not meaning.\n"
        )
    return (
        "Adapt subtitles into a compact dubbing script. This is translation plus rewriting, "
        "not line-by-line subtitle translation.\n"
        f"Translate from {source_lang or 'auto'} into {target_lang}. Output {target_lang} only.\n"
        "You MAY merge, split, reorder wording inside this scene, and change the number of lines. "
        "Keep the event order, character intent, names, relationships, numbers, questions, replies, "
        "decisions and emotional stance. Do not add events.\n"
        "Write natural spoken dialogue for dubbing. Keep character point of view and pronouns; "
        "do not turn dialogue into narrator prose.\n"
        "Resolve speaker, addressee, and who does what to whom from the surrounding transcript "
        "before translating. A subtitle cue can contain multiple speakers; cue boundaries are "
        "not speaker boundaries. Kinship terms and titles may be self-reference, direct address, "
        "or third-person reference: determine their role from grammar and context, not a fixed mapping. "
        "Keep separate turns distinct, including a short reply at the end of a cue. "
        "When shortening, preserve the agent, recipient, relationships, negation, and question intent. "
        "Before returning, compare each translated action and pronoun against the source; "
        "correct any reversed roles or accidental self-directed actions. Do not invent identities "
        "when context is insufficient.\n"
        f"The scene lasts {window_s:.2f} seconds. The complete output must be at most "
        f"{max_units} {unit}. Use fewer words when possible without losing essential meaning.\n"
        f"Each output line must be a complete short spoken phrase of about at most "
        f"{line_unit_cap} {unit}; split at a natural breath boundary if needed.\n"
        "Every source id must occur in source_ids exactly where its meaning is represented. "
        "Several source ids may map to one rewritten line; one source id may map to several lines.\n"
        f"Required source ids: {required}.\n"
        f"{missing_note}{format_note}{budget_note}\n{tts_feedback}\n"
        "GLOBAL TRANSCRIPT CONTEXT (reference only; rewrite only the current scene):\n"
        f"{global_context}\n\n"
        "CURRENT SCENE:\n"
        f"{source_rows}\n\n"
        "Return only a valid JSON array inside <TRANSLATE_TEXT> tags. No markdown. "
        "Put a comma between every object and escape any double quote inside dialogue text. Schema:\n"
        '[{"source_ids":["S1","S2"],"text":"rewritten target-language dialogue"}]\n'
        "\n<INPUT>\n{batch_input}\n</INPUT>"
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
        content = str(row.get("text") or "").strip()
        ids = row.get("source_ids") or []
        if isinstance(ids, str):
            ids = re.findall(r"S\d+", ids, flags=re.I)
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
            "global_context": _build_global_context(text_list, lo, hi),
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
    budget_error = ""
    best_complete: List[dict] = []
    best_units = 10**9
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
            budget_error=budget_error,
            tts_feedback=tts_feedback,
        )
        raw = model._item_task(joined)
        try:
            records = parse_adaptive_records(raw or "", pending_ids)
        except (ValueError, json.JSONDecodeError) as exc:
            format_error = str(exc)
            if attempt + 1 < MAX_SCENE_REQUESTS:
                logger.warning(f"Adaptive rewrite returned malformed JSON; retrying scene: {exc}")
                continue
            raise RuntimeError(
                "Adaptive dubbing rewrite returned invalid JSON after retrying the scene"
            ) from exc
        format_error = ""
        budget_error = ""
        collected.extend(records)
        missing = missing_source_ids(collected, required)
        if not missing:
            candidate = sorted(collected, key=_source_order)
            actual_units = count_lines_units([row["text"] for row in candidate], target_lang)
            if actual_units < best_units:
                best_complete = candidate
                best_units = actual_units
            if actual_units <= max_units:
                # Stable sort protects picture sync if a model returns valid ids out of order.
                return candidate
            budget_error = (
                f"Your previous complete answer used {actual_units} {unit}, exceeding the limit."
            )
            logger.warning(
                f"Adaptive rewrite exceeded budget {actual_units}>{max_units} {unit}; "
                "retrying complete scene"
            )
            # A too-long complete answer must be rewritten as a whole; appending
            # fragments cannot reduce it.
            pending = list(scene_sources)
            collected = []
            continue
        # Preserve accepted dialogue and ask only for the omitted cues. This is
        # both cheaper and more reliable than rewriting the complete scene.
        pending = [source_by_id[source_id] for source_id in missing]
        logger.warning(f"Adaptive rewrite missing source ids {missing}; requesting only missing cues")
    missing = missing_source_ids(collected, required)
    if best_complete:
        # Real synthesized duration remains authoritative. Returning the
        # shortest complete candidate lets the TTS feedback loop make the
        # final decision when a provider repeatedly ignores the word limit.
        logger.warning(
            f"Adaptive rewrite still exceeded text budget after {MAX_SCENE_REQUESTS} requests; "
            f"using shortest complete result ({best_units}/{max_units} {unit})"
        )
        return best_complete
    if missing:
        raise RuntimeError(f"Adaptive dubbing rewrite omitted source cues: {', '.join(missing)}")
    return sorted(collected, key=_source_order)


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
) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
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
        for record_index, (row, (a, b)) in enumerate(zip(records, spans)):
            item = _make_srt_item(line_no, a, b, row["text"])
            output.append(item)
            row["line"] = line_no
            row["start_ms"] = a
            row["end_ms"] = b
            row["record_index"] = record_index
            line_no += 1
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
        uuid=None,
) -> List[dict]:
    """Create a duration-aware script and persist source anchors for TTS fitting."""
    model = _make_model(translate_type, uuid)
    source_lang, target_lang = get_source_target_code(
        show_source=source_code,
        show_target=target_code,
        translate_type=translate_type,
    )
    source_lang = source_lang or source_code or "auto"
    target_lang = target_lang or target_code or "target language"
    pace = probe_target_sec_per_unit(
        lang_code=target_code,
        lang_name=target_lang,
        voice_role=voice_role,
        voice_rate=voice_rate,
        cache_folder=cache_folder,
    )
    sec_per_unit = float(pace["sec_per_unit"])
    times = [(int(it["start_time"]), int(it["end_time"])) for it in text_list]
    gap_ms = int(settings.get("adaptive_scene_gap_ms", max(900, int(settings.get("min_silence_duration_ms", 600)))))
    max_scene_ms = int(float(settings.get("adaptive_scene_max_s", 30)) * 1000)
    safety = min(0.95, max(0.65, float(settings.get("adaptive_safety_pct", 85)) / 100.0))
    spans = build_adaptive_scene_spans(
        times,
        scene_gap_ms=gap_ms,
        max_scene_ms=max_scene_ms,
    )
    planned_scenes = _build_scene_plans(
        text_list,
        times=times,
        spans=spans,
        sec_per_unit=sec_per_unit,
        safety=safety,
    )
    path = Path(adaptive_metadata_path(cache_folder))
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = _rewrite_signature(
        text_list,
        source_code=source_code,
        target_code=target_code,
        translate_type=translate_type,
        voice_role=voice_role,
        voice_rate=voice_rate,
        gap_ms=gap_ms,
        max_scene_ms=max_scene_ms,
        safety=safety,
    )
    cached_scenes = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("rewrite_signature") == signature:
                cached_scenes = {
                    (int(scene["source_lo"]), int(scene["source_hi"])): scene
                    for scene in previous.get("scenes", [])
                }
        except (OSError, ValueError, TypeError, KeyError):
            cached_scenes = {}
    meta = {
        "version": ALGORITHM_VERSION,
        "complete": False,
        "rewrite_signature": signature,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_code": source_code,
        "target_code": target_code,
        "translate_type": int(translate_type),
        "sec_per_unit": sec_per_unit,
        "unit": pace["unit"],
        "voice": pace.get("voice", ""),
        "voice_rate": pace.get("rate", voice_rate),
        "scenes": [],
    }

    for scene_no, plan in enumerate(planned_scenes, 1):
        lo = int(plan["source_lo"])
        hi = int(plan["source_hi"])
        scene_sources = plan["source"]
        fit_start = int(plan["fit_start_ms"])
        fit_end = int(plan["fit_end_ms"])
        window_s = max(0.2, (fit_end - fit_start) / 1000.0)
        max_units = int(plan["max_units"])
        context = plan["global_context"]
        cached = cached_scenes.get((lo, hi))
        if cached and not missing_source_ids(cached.get("records", []), [row["id"] for row in scene_sources]):
            meta["scenes"].append(cached)
            logger.debug(f"Adaptive scene {scene_no}/{len(planned_scenes)} restored from checkpoint")
            continue
        logger.debug(
            f"Adaptive scene {scene_no}/{len(planned_scenes)} cues={lo + 1}-{hi + 1} "
            f"window={window_s:.2f}s max={max_units} {pace['unit']}"
        )
        records = _rewrite_scene(
            model=model,
            scene_sources=scene_sources,
            global_context=context,
            source_lang=source_lang,
            target_lang=target_lang,
            max_units=max_units,
            unit=pace["unit"],
            window_s=window_s,
        )
        by_id = {row["id"]: row for row in scene_sources}
        for record in records:
            record["anchor_ms"] = _anchor_for_ids(record["source_ids"], by_id)
        plan["records"] = records
        meta["scenes"].append(plan)
        # Keep completed API scenes so the task Retry action can resume after a
        # later scene fails instead of paying to rewrite the whole video again.
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    output = _rebuild_output(meta)
    meta["complete"] = True
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(f"Adaptive dubbing rewrite: {len(text_list)} source cues -> {len(output)} cues")
    return output


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
    from videotrans.util.help_ffmpeg import get_audio_time

    path = Path(adaptive_metadata_path(cache_folder))
    if not path.exists():
        return [], False
    meta = json.loads(path.read_text(encoding="utf-8"))
    queue_by_line = {int(row["line"]): row for row in queue_tts}
    durations: Dict[Tuple[int, int], int] = {}
    overflow: List[Tuple[int, int, int]] = []
    soft_speed_pct = max(0.0, float(settings.get("adaptive_soft_speed_pct", 5))) if allow_soft_speed else 0.0
    # Allow 1% for codec/container duration rounding. Additional tolerance is
    # safe only when the normal audio-speed alignment stage is enabled.
    tolerance = 1.01 + soft_speed_pct / 100.0

    for scene_index, scene in enumerate(meta["scenes"]):
        total = 0
        for record_index, record in enumerate(scene["records"]):
            q = queue_by_line.get(int(record.get("line", -1)))
            duration = 0
            if q and Path(q.get("filename", "")).exists():
                duration = int(get_audio_time(q["filename"]))
            if duration <= 0:
                duration = _estimated_durations(
                    [record], meta["target_lang"], float(meta["sec_per_unit"])
                )[0]
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
        if total > int(available * tolerance):
            scene["fit"]["status"] = "overflow"
            overflow.append((scene_index, total, available))
        elif total > available and soft_speed_pct > 0:
            scene["fit"]["status"] = "soft_speed"
            logger.debug(
                f"Adaptive scene {scene_index + 1} uses {required_speed_pct:.2f}% final speed-up "
                f"within configured {soft_speed_pct:.2f}%"
            )

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
            by_id = {row["id"]: row for row in scene["source"]}
            for record in records:
                record["anchor_ms"] = _anchor_for_ids(record["source_ids"], by_id)
            scene["records"] = records
            scene["max_units"] = new_max
            scene["fit"]["status"] = "rewritten_for_tts"
        output = _rebuild_output(meta)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return output, True

    if overflow:
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        details = ", ".join(
            f"scene {scene_index + 1}: {actual_ms / 1000:.2f}s/{available_ms / 1000:.2f}s "
            f"(needs {meta['scenes'][scene_index]['fit']['required_speed_pct']:.1f}% speed-up)"
            for scene_index, actual_ms, available_ms in overflow
        )
        raise RuntimeError(
            "Adaptive dubbing could not fit the following scenes after all rewrite attempts: "
            f"{details}. Lower the safety percentage, allow more rewrite attempts, or choose a faster voice. "
            "To allow audio speed-up, enable audio auto-rate and set the adaptive soft speed "
            "percentage at least as high as the required speed-up."
        )
    output = _rebuild_output(meta, durations)
    for scene in meta["scenes"]:
        status = scene.get("fit", {}).get("status")
        if status == "fits":
            scene["fit"]["status"] = "fitted"
        elif status == "soft_speed":
            scene["fit"]["status"] = "fitted_soft_speed"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, False
