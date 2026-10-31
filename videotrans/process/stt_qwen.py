# 语音识别，新进程执行
# 返回元组
# 失败：第一个值为False，则为失败，第二个值存储失败原因
# 成功，第一个值存在需要的返回值，不需要时返回True，第二个值为None
import json, traceback
import re
from pathlib import Path
from typing import List


from videotrans.configure.config import logger


def qwen3asr_fun(
        cut_audio_list=None,
        logs_file=None,
        local_dir=None,
        local_dir_align=None,
        model_name=None,
        detect_language=None,
        force_align=False,  # 是否需要对齐时间戳:只有明确指定属于这些语言中的某个 ["zh","en","ja",'ko','yue','fr','es','it','de','pt','ru'] 才支持
        hotword=None,
        **kw
):
    import torch
    from videotrans.process._stt_utils import _write_log
    from videotrans.process._source_cues import restore_transcript_punctuation, segment_aligned_source
    from transformers import AutoProcessor, AutoModelForMultimodalLM,AutoModelForTokenClassification

    try:
        batch_size=2
        srts = [item for item in json.loads(Path(cut_audio_list).read_text(encoding='utf-8'))]
        if not force_align:
            processor = AutoProcessor.from_pretrained(local_dir)
            model = AutoModelForMultimodalLM.from_pretrained(local_dir, device_map=kw.get('device_name', 'auto'))
            msg = f'Load {model_name} running on {model.device}'
            _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
            logger.debug(f'QwenASR:{local_dir}，{msg}，{detect_language=}')

            # 不返还时间戳数据
            srts_chunk = [srts[i:i + batch_size] for i in range(0, len(srts), batch_size)]
            for i, it_list in enumerate(srts_chunk):
                audio = [it['filename'] for it in it_list]
                inputs = processor.apply_transcription_request(
                    audio, language=[detect_language for it in audio],
                ).to(model.device, model.dtype)

                output_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                transcriptions = processor.decode(generated_ids, return_format="transcription_only")
                for i, text in enumerate(transcriptions):
                    it_list[i]['text'] = text
                
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": "\n".join([it['text'] for it in it_list])}))

            return srts, None
            
        

        # 需要返回时间戳。识别和强制对齐分开加载，避免两个模型同时占满 MPS。
        texts = [{
            "start": 0,
            "end": 0,
            "text": "",
            "words": []
        }]
        language = None
        device_name = kw.get('device_name', 'auto')

        def _free_mps():
            import gc
            gc.collect()
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        asr_processor = AutoProcessor.from_pretrained(local_dir)
        asr_model = AutoModelForMultimodalLM.from_pretrained(local_dir, device_map=device_name)
        msg = f'Load {model_name} running on {asr_model.device}'
        _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
        logger.debug(f'QwenASR:{local_dir}，{msg}，{detect_language=}, {force_align=}')

        transcripts = []
        for i, it in enumerate(srts):
            inputs = asr_processor.apply_transcription_request(audio=it['filename'], language=detect_language)
            inputs = inputs.to(asr_model.device, asr_model.dtype)
            with torch.inference_mode():
                output_ids = asr_model.generate(**inputs, max_new_tokens=1024)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]
            transcript = parsed["transcription"]
            language = parsed["language"] or "English"
            transcripts.append((transcript, language))
            del inputs, output_ids, generated_ids, parsed
            _write_log(logs_file, json.dumps({"type": "subtitle", "text": transcript[:90]+"..."}))
        del asr_model, asr_processor
        _free_mps()

        aligner_processor = AutoProcessor.from_pretrained(local_dir_align)
        aligner_model = AutoModelForTokenClassification.from_pretrained(
            local_dir_align, dtype='auto', device_map=device_name)
        _write_log(logs_file, json.dumps({"type": "logs", "text": "ASR unloaded, aligning timestamps"}))

        for i, it in enumerate(srts):
            transcript, language = transcripts[i]
            if not (transcript or '').strip():
                continue
            aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
                audio=it['filename'],
                transcript=transcript,
                language=language,
            )
            aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)
            with torch.inference_mode():
                outputs = aligner_model(**aligner_inputs)
            timestamps = aligner_processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=aligner_model.config.timestamp_token_id,
            )[0]
            del aligner_inputs, outputs

            offset = it['start_time'] / 1000.0
            if texts[0]['words'] == [] and timestamps:
                texts[0]['start'] = float(timestamps[0]['start_time']) + offset
            aligned_words = []
            for item in timestamps:
                aligned_words.append({
                    "word": item['text'],
                    "start": float(item['start_time']) + offset,
                    "end": float(item['end_time']) + offset
                })
            texts[0]['words'].extend(restore_transcript_punctuation(aligned_words, transcript))
            _free_mps()
        if texts[0]['words']:
            texts[0]['end'] = texts[0]['words'][-1]['end']
        del aligner_model, aligner_processor
        _free_mps()

        srts = segment_aligned_source(texts, "zh" if language in ["Chinese", "Cantonese", "Japanese", "Korean"] else 'en')
        return srts, None
    except BaseException as e:
        msg = traceback.format_exc()
        return False, f'{e}:{msg}'
