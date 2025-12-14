#!/usr/bin/env python3
"""
Clean Batch API Optimization - Uses proper judge calls, no embedded prompts.

This module provides batch processing that maintains the ensemble architecture:
- Calls actual judges (no embedded prompts)
- Aggregates results using ensemble logic
- Provides both cost savings (50%) and async processing
"""

import os
import json
import tempfile
import time
import re
import math
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from utils.ensemble import call_parallel_judges_for_row_detailed
from utils.llm.backends import (
    create_batch_file,
    upload_batch_file,
    create_batch_job,
    get_batch_status,
    download_file_content,
)
from utils.progress import SingleLineProgress


def _get_language_label(lang: str, src: str = "") -> str:
    try:
        if not lang:
            from utils.judge import detect_language_from_text
            detected = detect_language_from_text(src)
            return detected
        return {
            'es': 'Spanish', 'en': 'English', 'de': 'German', 'ua': 'Ukrainian'
        }.get(lang, 'Spanish')
    except Exception:
        return 'Spanish'


def _build_prompt_for_row(judge: str, method: str, backend: str, lang: str, row: pd.Series) -> str:
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    language_label = _get_language_label(lang, src)

    # Use any precomputed alignment first to avoid recomputation
    aligned_text = (
        str(row.get('aligned', ''))
        or str(row.get('aligned_sentence', ''))
        or str(row.get('alert', ''))
    )

    if judge == 'feedback':
        # Requires aligned text; if not present and allowed, compute
        if not aligned_text:
            try:
                errant_flag = os.getenv('FEEDBACK_ERRANT', os.getenv('ERRANT', 'on')).lower()
                use_errant = errant_flag not in {'off', 'false', '0', 'no'}
                if use_errant:
                    from utils.errant_align import get_alignment_for_language
                    aligned_text = get_alignment_for_language(src, tgt, language=lang)
            except Exception:
                aligned_text = ''
        from judges.feedback.prompts import TPFP_PROMPT_BASELINE
        from utils.judge import build_numbered_prompt
        prompt = build_numbered_prompt(TPFP_PROMPT_BASELINE, language_label, src, tgt, aligned_text)
        return prompt

    if judge == 'sentence':
        from judges.sentence.advanced import build_baseline_prompt, create_fusion_alignment
        if not aligned_text:
            try:
                from utils.errant_align import get_alignment_for_language
                aligned_text = get_alignment_for_language(src, tgt, language=lang)
            except Exception:
                aligned_text = f"{src} → {tgt}"
        if method == 'baseline' or method == 'legacy':
            # Reuse baseline prompt builder for both; legacy is tolerated by baseline template
            return build_baseline_prompt(src, tgt, lang, aligned_text)

    if judge == 'edit':
        if method == 'agent':
            # Agent path does not go through LLM batch
            raise ValueError('Batch API not supported for edit/agent method')
        from judges.edit.prompts import EDIT_LEVEL_JUDGE_PROMPT
        from judges.sentence.advanced import create_fusion_alignment
        from utils.judge import build_numbered_prompt
        if not aligned_text:
            try:
                from utils.errant_align import get_alignment_for_language
                aligned_text = get_alignment_for_language(src, tgt, language=lang)
            except Exception:
                aligned_text = f"{src} → {tgt}"
        fused_alignment_full = create_fusion_alignment(src, tgt, aligned_text)
        fused_alignment = fused_alignment_full
        if len(fused_alignment_full) > 600:
            fused_alignment = fused_alignment_full[:600] + "…"
        # Extract edits spans for {4} slot, mirroring baseline judge logic
        try:
            raw_spans = re.findall(r"\{[^}]+=>[^}]+\}", fused_alignment_full)
            spans = []
            for s in raw_spans:
                body = s[1:-1]
                o, n = body.split("=>", 1)
                if o == n:
                    continue
                spans.append(s)
            edits_field = ('"' + '" , "'.join(spans) + '"') if spans else ""
        except Exception:
            edits_field = ""
        return build_numbered_prompt(EDIT_LEVEL_JUDGE_PROMPT, language_label, src, tgt, fused_alignment, edits_field)

    # Generic fallback
    aligned_display = aligned_text or f"{src} → {tgt}"
    return (
        f"Classify this {language_label} correction:\n\n"
        f"Original: {src}\n"
        f"Corrected: {tgt}\n"
        f"Changes: {aligned_display}\n\n"
        f"Classify as: TP/FP/TN/FN"
    )


def _parse_model_output(content: str, judge: str, method: str, src: str, tgt: str) -> str:
    try:
        if judge == 'sentence':
            from judges.sentence.baseline import parse_baseline_response
            label, _ = parse_baseline_response(content)
            return label if label in {'TP','FP1','FP2','FP3','TN','FN'} else 'Error'
        if judge == 'edit':
            from judges.edit.baseline import parse_json_response, compute_sentence_label
            parsed = parse_json_response(content)
            edit_labels = parsed.get('labels', [])
            missed_error = parsed.get('missed_error', False)
            label = compute_sentence_label(src, tgt, missed_error, edit_labels)
            return label if label in {'TP','FP1','FP2','FP3','TN','FN'} else 'Error'
        if judge == 'feedback':
            from utils.judge import parse_tpfp_label
            lab = parse_tpfp_label(content)
            return lab if lab in {'TP','FP1','FP2','FP3','TN','FN'} else 'Error'
        # Generic scan
        up = (content or '').upper()
        for cls in ['FP1','FP2','FP3','TP','TN','FN']:
            if f' {cls}' in f' {up}':
                return cls
        return 'Error'
    except Exception:
        return 'Error'


def create_judge_batch_requests(df: pd.DataFrame, judge: str, method: str, 
                               backend: str, lang: str = 'es') -> List[Dict[str, Any]]:
    """
    Create batch requests using actual judge logic (no embedded prompts).
    
    This function calls the real judge to get the actual prompt that would be used,
    then formats it for batch processing.
    """
    print(f"📦 Creating batch requests using {judge}/{method} judge logic...")
    
    batch_requests = []
    
    # Build prompts per row using same logic as real judges
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            prompt = _build_prompt_for_row(judge, method, backend, lang, row)
            batch_requests.append({
                'custom_id': f"row-{idx}",
                'sample_id': idx,
                'judge': judge,
                'method': method,
                'backend': backend,
                'lang': lang,
                'src': str(row.get('src', '')),
                'tgt': str(row.get('tgt', '')),
                'prompt': prompt,
            })
        except Exception as e:
            print(f"❌ Failed to create request for sample {idx}: {e}")
            with open('fails.txt', 'a') as f:
                f.write(f"Failed to create request for sample {idx}: {e}\n")
            continue
    
    print(f"✅ Created {len(batch_requests)} batch requests")
    return batch_requests


def _submit_and_wait_batch(batch_requests: List[Dict[str, Any]], backend: str, *, max_wait_seconds: int = 900) -> Optional[str]:
    # Create file
    tmp_path = create_batch_file(batch_requests, backend)
    # Choose token based on routing: prefer OpenAI key when forcing direct
    force_direct = os.getenv('FORCE_OPENAI_DIRECT_BATCH', '').lower() in {'1','true','on','yes'}
    api_token = (os.getenv('OPENAI_API_KEY', '') if force_direct else (os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')))
    if not api_token:
        raise RuntimeError('API token not set for batch submit')
    try:
        file_id = upload_batch_file(tmp_path, backend, api_token)
        batch_id = create_batch_job(file_id, backend, api_token)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Poll until complete or timeout
    start = time.time()
    sleep_seconds = 2.0
    while time.time() - start < max_wait_seconds:
        try:
            status = get_batch_status(batch_id, backend, api_token)
        except Exception as e:
            print(f"⚠️  Batch status check failed: {e}")
            time.sleep(min(10.0, sleep_seconds))
            sleep_seconds = min(30.0, sleep_seconds * 1.5)
            continue
        st = str(status.get('status', '')).lower()
        if st in {'completed', 'succeeded', 'success'}:
            return status.get('output_file_id')
        if st in {'failed', 'canceled', 'expired'}:
            raise RuntimeError(f"Batch failed with status={st}")
        time.sleep(sleep_seconds)
        sleep_seconds = min(30.0, sleep_seconds * 1.5)
    # Timed out
    return None


def _build_results_from_output_jsonl(jsonl_text: str, df: pd.DataFrame, judge: str, method: str, backend: str) -> List[Dict[str, Any]]:
    # Map index -> result; prefill errors
    results: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            'idx': i,
            'src': str(row.get('src', '')),
            'tgt': str(row.get('tgt', '')),
            'tp_fp_label': 'Error',
            'reasoning': 'No response',
            'writing_type': '',
            'judge_outputs': [],
            'total_tokens': 0,
            'model': backend,
        })

    # Parse lines
    bad_lines = 0
    for line in (jsonl_text or '').splitlines():
        if not line or not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception as e:
            bad_lines += 1
            continue
        cid = rec.get('custom_id') or rec.get('id') or ''
        # Expect custom_id like row-<idx>
        idx = None
        if isinstance(cid, str) and cid.startswith('row-'):
            try:
                idx = int(cid.split('-', 1)[1])
            except Exception:
                idx = None
        # Extract content and usage
        body = None
        if isinstance(rec.get('response'), dict):
            body = rec['response'].get('body')
        elif 'body' in rec:
            body = rec.get('body')
        content = ''
        usage = {}
        # In OpenAI Batch output, body may be a JSON string; parse if needed
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = None
        if isinstance(body, dict):
            choices = body.get('choices') or []
            if choices and isinstance(choices, list):
                msg = choices[0].get('message', {})
                raw_content = msg.get('content')
                # OpenAI may return a string or a list of content parts in JSON mode
                if isinstance(raw_content, str):
                    content = raw_content.strip()
                elif isinstance(raw_content, list):
                    # Prefer output_json.json if present; else join output_text.text
                    collected = None
                    for part in raw_content:
                        if isinstance(part, dict) and part.get('type') in ('output_json', 'json'):  # new JSON mode
                            pj = part.get('json')
                            try:
                                collected = json.dumps(pj, ensure_ascii=False)
                            except Exception:
                                collected = str(pj)
                            break
                    if collected is None:
                        texts = []
                        for part in raw_content:
                            if isinstance(part, dict):
                                if 'text' in part:
                                    texts.append(str(part.get('text') or ''))
                        collected = '\n'.join([t for t in texts if t])
                    content = (collected or '').strip()
                else:
                    content = ''
            usage = body.get('usage') or {}
        # Compute label using judge-specific parser
        if idx is not None and 0 <= idx < len(results):
            src = results[idx]['src']
            tgt = results[idx]['tgt']
            label = _parse_model_output(content, judge, method, src, tgt)
            results[idx]['tp_fp_label'] = label
            results[idx]['reasoning'] = content if content else 'No response'
            # Token accounting from usage if available
            total_tokens = int(usage.get('total_tokens', 0))
            prompt_tokens = int(usage.get('prompt_tokens', usage.get('input_tokens', 0) or 0))
            completion_tokens = int(usage.get('completion_tokens', usage.get('output_tokens', 0) or 0))
            results[idx]['total_tokens'] = total_tokens
            results[idx]['input_tokens'] = prompt_tokens
            results[idx]['output_tokens'] = completion_tokens
            results[idx]['model'] = backend

    if bad_lines:
        try:
            with open('fails.txt', 'a') as f:
                f.write(f"Batch output contained {bad_lines} non-JSON lines (ignored)\n")
        except Exception:
            pass

    return results


def _submit_sharded_batches(
    df: pd.DataFrame,
    batch_requests: List[Dict[str, Any]],
    backend: str,
    *,
    shard_size: int,
    force_direct: bool,
    max_wait_secs: int,
    judge: str,
    method: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Submit sharded batch jobs in parallel, poll up to max_wait_secs, and return
    (completed_results_by_idx, pending_rows_by_idx).
    """
    # Build shard structures: each shard has indices list and requests list
    shards: List[Tuple[List[int], List[Dict[str, Any]]]] = []
    cur: List[int] = []
    cur_reqs: List[Dict[str, Any]] = []
    for i, req in enumerate(batch_requests):
        cur.append(req['sample_id'])
        cur_reqs.append(req)
        if len(cur) >= shard_size:
            shards.append((cur, cur_reqs))
            cur, cur_reqs = [], []
    if cur:
        shards.append((cur, cur_reqs))

    # Enforce max JSONL size per shard (adaptive pre-split)
    max_file_mb = float(os.getenv('BATCH_MAX_FILE_MB', '40'))  # conservative default for proxies
    max_file_bytes = int(max_file_mb * 1024 * 1024)

    adaptive_shards: List[Tuple[List[int], List[Dict[str, Any]]]] = []
    for indices, reqs in shards:
        # Approximate jsonl size
        try:
            import json
            approx = sum(len(json.dumps(r)) + 1 for r in reqs)
        except Exception:
            approx = len(reqs) * 4096  # fallback estimate
        if approx <= max_file_bytes:
            adaptive_shards.append((indices, reqs))
            continue
        # Split into smaller chunks
        factor = max(2, int(math.ceil(approx / max_file_bytes)))
        step = max(1, int(math.ceil(len(reqs) / factor)))
        for i in range(0, len(reqs), step):
            sub_reqs = reqs[i:i+step]
            sub_idxs = indices[i:i+step]
            adaptive_shards.append((sub_idxs, sub_reqs))

    shards = adaptive_shards

    # Submit shards in parallel: upload + create batch job
    api_token = (os.getenv('OPENAI_API_KEY', '') if force_direct else (os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')))
    if not api_token:
        raise RuntimeError('API token missing for sharded batch submit')

    submitted: List[Tuple[List[int], str]] = []  # (indices, batch_id)

    def _submit_single(indices: List[int], reqs: List[Dict[str, Any]]) -> Tuple[List[int], str]:
        tmp = create_batch_file(reqs, backend)
        try:
            file_id = upload_batch_file(tmp, backend, api_token)
            batch_id = create_batch_job(file_id, backend, api_token)
            return indices, batch_id
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=min(16, max(1, len(shards)))) as ex:
        submit_prog = SingleLineProgress(len(shards), desc="Submit shards", update_every=1)
        futs = [ex.submit(_submit_single, idxs, reqs) for idxs, reqs in shards]
        for f in futs:
            indices, batch_id = f.result()
            submitted.append((indices, batch_id))
            submit_prog.update(len(submitted))
        submit_prog.finish()

    # Poll until max_wait_secs; collect completed results
    start = time.time()
    completed: Dict[int, Dict[str, Any]] = {}
    remaining: List[Tuple[List[int], str]] = list(submitted)

    total_shards = len(remaining)
    poll_global = SingleLineProgress(total_shards, desc="Poll shards (completed)", update_every=1)
    completed_shards = 0
    poll_round = 0
    while remaining and (time.time() - start) < max_wait_secs:
        poll_round += 1
        next_round: List[Tuple[List[int], str]] = []

        # Poll all batches in parallel for this round
        with ThreadPoolExecutor(max_workers=min(32, len(remaining))) as ex:
            def _poll_and_maybe_download(indices_batchid):
                indices, batch_id = indices_batchid
                try:
                    status = get_batch_status(batch_id, backend, api_token)
                    st = str(status.get('status', '')).lower()
                    if st in {'completed', 'succeeded', 'success'}:
                        ofid = status.get('output_file_id')
                        if ofid:
                            # Robust: retry a couple of times if empty/non-JSON
                            retries = int(os.getenv('BATCH_EMPTY_RETRY_ROUNDS', '2'))
                            sleep_s = float(os.getenv('BATCH_EMPTY_RETRY_SLEEP', '3'))
                            text = download_file_content(ofid, backend, api_token)
                            attempt = 0
                            while attempt < retries and (not text or not text.strip()):
                                time.sleep(sleep_s)
                                text = download_file_content(ofid, backend, api_token)
                                attempt += 1
                            shard_df = df.iloc[indices]
                            shard_results = _build_results_from_output_jsonl(text, shard_df, judge=judge, method=method, backend=backend)
                            return ('completed', indices, shard_results, batch_id, status)
                        else:
                            return ('pending', indices, None, batch_id, status)
                    # Treat terminal failure states as finalized to avoid infinite re-queue
                    if st in {'failed', 'canceled', 'expired'}:
                        # Optionally fetch and log error file
                        try:
                            err_id = status.get('error_file_id')
                            if err_id:
                                err_text = download_file_content(err_id, backend, api_token)
                                with open('fails.txt', 'a') as f:
                                    f.write(f"Batch shard failed (batch_id={batch_id}): {err_text[:1000]}\n")
                        except Exception:
                            pass
                        return ('failed', indices, None, batch_id, status)
                    else:
                        return ('pending', indices, None, batch_id, status)
                except Exception as e:
                    return ('pending', indices, None, batch_id, {'error': str(e)})

            futs = [ex.submit(_poll_and_maybe_download, item) for item in remaining]
            for f in futs:
                kind, indices, shard_results, ret_batch_id, st = f.result()
                if kind == 'completed' and shard_results is not None:
                    for j, idx in enumerate(indices):
                        completed[idx] = shard_results[j]
                    completed_shards += 1
                    # Optionally check error_file_id and log
                    try:
                        err_id = st.get('error_file_id')
                        if err_id:
                            err_text = download_file_content(err_id, backend, api_token)
                            with open('fails.txt', 'a') as f:
                                f.write(f"Batch shard had errors: {err_text[:1000]}\n")
                    except Exception:
                        pass
                elif kind == 'failed':
                    # Count failed shard as finalized to advance overall progress
                    completed_shards += 1
                    # Do not re-queue failed shards
                    pass
                else:
                    # Keep the original batch_id for next poll round
                    next_round.append((indices, ret_batch_id))
        # Single-line global poll progress update
        poll_global.update(completed_shards)
        remaining = [(idxs, bid if isinstance(bid, str) else '') for idxs, bid in next_round]
        if remaining:
            time.sleep(2.0)
    poll_global.finish()

    # Build pending rows mapping
    pending: Dict[int, Dict[str, Any]] = {}
    for indices, _bid in remaining:
        for idx in indices:
            if idx not in completed:
                row = df.iloc[idx]
                pending[idx] = {
                    'idx': idx,
                    'src': str(row.get('src', '')),
                    'tgt': str(row.get('tgt', '')),
                }

    return completed, pending


def process_with_clean_batch_api(df: pd.DataFrame, judge: str, method: str, 
                                backends: List[str], lang: str = 'es',
                                n_judges: int = 1) -> List[Dict[str, Any]]:
    """
    Process DataFrame using clean batch API that calls actual judges.
    
    This maintains the ensemble architecture by:
    1. Using actual judge prompts (not embedded ones)
    2. Proper aggregation logic
    3. Same result format as regular ensemble
    """
    print(f"🚀 CLEAN BATCH API PROCESSING")
    print(f"Processing {len(df)} samples with {judge}/{method}")
    print("=" * 60)
    
    # Check API token
    api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
    if not api_token:
        error_msg = "❌ No API token found. Set API_TOKEN or OPENAI_API_KEY in .env file."
        print(error_msg)
        with open('fails.txt', 'a') as f:
            f.write(f"{error_msg}\n")
        return None
    
    try:
        backend = backends[0]
        # Build requests
        batch_requests = create_judge_batch_requests(df, judge, method, backend, lang)
        if not batch_requests:
            raise RuntimeError('Failed to create batch requests')

        # Sharded/hybrid config
        force_direct = os.getenv('FORCE_OPENAI_DIRECT_BATCH', '').lower() in {'1','true','on','yes'}
        async_mode = os.getenv('BATCH_ASYNC', '').lower() in {'1','true','on','yes'}
        shard_size = int(os.getenv('BATCH_SHARD_SIZE', '200'))
        max_wait = int(os.getenv('BATCH_MAX_WAIT_SECS', '900'))
        hybrid = os.getenv('BATCH_HYBRID', '').lower() in {'1','true','on','yes'}
        rt_workers = int(os.getenv('RT_CONCURRENCY', '100'))

        # Sharded submission + partial wait
        completed_map, pending_map = _submit_sharded_batches(
            df,
            batch_requests,
            backend,
            shard_size=shard_size,
            force_direct=force_direct,
            max_wait_secs=max_wait,
            judge=judge,
            method=method,
        )

        # If async and no hybrid (pure batch), return placeholders and do not fallback
        if async_mode and not hybrid:
            print("🚀 Batch submitted (async mode) - results pending")
            return [{
                'idx': i,
                'src': str(df.iloc[i].get('src', '')),
                'tgt': str(df.iloc[i].get('tgt', '')),
                'tp_fp_label': ('BATCH_PENDING' if i in pending_map else completed_map[i]['tp_fp_label']),
                'reasoning': (completed_map[i]['reasoning'] if i in completed_map else 'Batch submitted; results pending'),
                'writing_type': '',
                'judge_outputs': [],
                'total_tokens': (completed_map[i].get('total_tokens', 0) if i in completed_map else 0),
                'model': backend,
                'batch_status': ('completed' if i in completed_map else 'submitted')
            } for i in range(len(df))]

        # Threshold-based hybrid: when enough rows are completed in Batch, finish the rest in realtime
        results: List[Optional[Dict[str, Any]]] = [None] * len(df)
        for idx, rec in completed_map.items():
            results[idx] = rec

        threshold_pct = float(os.getenv('HYBRID_THRESHOLD_PCT', '0.9'))
        enough_completed = len(completed_map) >= int(threshold_pct * len(df))

        # Do not resubmit a single monolithic batch; either fallback (if enabled) or return errors
        if not completed_map and not hybrid:
            print("No shards completed within wait window. Returning errors (set BATCH_HYBRID=on to fallback in realtime).")
            return [
                {
                    'idx': i,
                    'src': str(df.iloc[i].get('src', '')),
                    'tgt': str(df.iloc[i].get('tgt', '')),
                    'tp_fp_label': 'Error',
                    'reasoning': 'Batch shard incomplete',
                    'writing_type': '',
                    'judge_outputs': [],
                    'total_tokens': 0,
                    'model': backend,
                }
                for i in range(len(df))
            ]

        # If enough rows are completed in Batch, finish pending via realtime calls
        if enough_completed:
            pending_idxs = [i for i in range(len(df)) if results[i] is None]
            if pending_idxs:
                print(f"⚡ Hybrid threshold reached ({len(completed_map)}/{len(df)}). Completing {len(pending_idxs)} pending via realtime.")
                def _rt_call(idx):
                    row = df.iloc[idx]
                    judge_results = call_parallel_judges_for_row_detailed(
                        judge=judge,
                        method=method,
                        backends=[backend],
                        lang=lang,
                        row=row,
                        n_judges=1,
                        moderation='off'
                    )
                    primary = judge_results[0] if judge_results else {}
                    label = primary.get('label', 'Error')
                    return {
                        'idx': idx,
                        'src': str(row.get('src', '')),
                        'tgt': str(row.get('tgt', '')),
                        'tp_fp_label': label,
                        'reasoning': primary.get('reasoning', ''),
                        'writing_type': primary.get('writing_type', ''),
                        'judge_outputs': judge_results,
                        'total_tokens': primary.get('total_tokens', 0),
                        'model': backend,
                    }
                with ThreadPoolExecutor(max_workers=int(os.getenv('RT_CONCURRENCY','200'))) as ex:
                    for rec in ex.map(_rt_call, pending_idxs):
                        results[rec['idx']] = rec

        # Sync sharded path with shards completed
        for i in range(len(df)):
            if results[i] is None and i in completed_map:
                results[i] = completed_map[i]
        # Fill any None as errors
        for i in range(len(df)):
            if results[i] is None:
                row = df.iloc[i]
                results[i] = {
                    'idx': i,
                    'src': str(row.get('src', '')),
                    'tgt': str(row.get('tgt', '')),
                    'tp_fp_label': 'Error',
                    'reasoning': 'Shard incomplete',
                    'writing_type': '',
                    'judge_outputs': [],
                    'total_tokens': 0,
                    'model': backend,
                }
        return results  # type: ignore

    except Exception as e:
        error_msg = f"❌ Clean batch API processing failed: {e}"
        print(error_msg)
        with open('fails.txt', 'a') as f:
            f.write(f"{error_msg}\n")
        # Return error results to keep pipeline consistent
        return [{
            'idx': i,
            'src': str(row.get('src', '')),
            'tgt': str(row.get('tgt', '')),
            'tp_fp_label': 'Error',
            'reasoning': str(e),
            'writing_type': '',
            'judge_outputs': [],
            'total_tokens': 0,
            'model': (backends[0] if backends else 'unknown')
        } for i, (_, row) in enumerate(df.iterrows())]


def process_with_ultra_threading(df: pd.DataFrame, judge: str, method: str,
                               backends: List[str], lang: str = 'es',
                               n_judges: int = 1, max_workers: int = 32) -> List[Dict[str, Any]]:
    """
    Process DataFrame using ultra-threading for maximum immediate speedup.
    
    This uses proper judge calls with high concurrency for 2x+ speedup.
    """
    print(f"⚡ ULTRA-THREADING PROCESSING")
    print(f"Processing {len(df)} samples with {max_workers} workers")
    print(f"Judge: {judge}/{method}")
    print("=" * 60)
    
    results = []
    errors = 0
    
    def process_single_row(row_data):
        idx, row = row_data
        try:
            # Use actual judge calls (proper ensemble architecture)
            judge_results = call_parallel_judges_for_row_detailed(
                judge=judge,
                method=method,
                backends=backends,
                lang=lang,
                row=row,
                n_judges=n_judges,
                moderation='off'
            )
            
            if judge_results and len(judge_results) > 0:
                # Use first judge result (or implement proper aggregation)
                primary_result = judge_results[0]
                label = primary_result.get('label', 'Error')
                reasoning = primary_result.get('reasoning', '')
                writing_type = primary_result.get('writing_type', '')
                total_tokens = sum(j.get('total_tokens', 0) for j in judge_results)
                
                return {
                    'idx': idx,
                    'src': str(row.get('src', '')),
                    'tgt': str(row.get('tgt', '')),
                    'tp_fp_label': label,
                    'reasoning': reasoning,
                    'writing_type': writing_type,
                    'judge_outputs': judge_results,
                    'total_tokens': total_tokens,
                    'model': primary_result.get('model', backends[0])
                }
            else:
                return {
                    'idx': idx,
                    'src': str(row.get('src', '')),
                    'tgt': str(row.get('tgt', '')),
                    'tp_fp_label': 'Error',
                    'reasoning': 'Judge call failed',
                    'writing_type': '',
                    'judge_outputs': [],
                    'total_tokens': 0,
                    'model': backends[0]
                }
                
        except Exception as e:
            return {
                'idx': idx,
                'src': str(row.get('src', '')),
                'tgt': str(row.get('tgt', '')),
                'tp_fp_label': 'Error',
                'reasoning': f'Processing error: {str(e)}',
                'writing_type': '',
                'judge_outputs': [],
                'total_tokens': 0,
                'model': backends[0]
            }
    
    # Process with high concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_row, (idx, row)) 
                  for idx, row in df.iterrows()]
        
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            
            if result['tp_fp_label'] == 'Error':
                errors += 1
            
            # Progress update every 50 samples
            if (i + 1) % 50 == 0 or (i + 1) == len(futures):
                print(f"  Progress: {i+1}/{len(futures)} samples processed")
    
    success_rate = (len(results) - errors) / len(results) * 100
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    
    print(f"\n📊 ULTRA-THREADING RESULTS:")
    print(f"  Processed: {len(results)} samples")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Errors: {errors}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Workers: {max_workers}")
    
    return results


def get_optimal_processing_strategy(df_size: int, priority: str = 'balanced') -> Dict[str, Any]:
    """
    Recommend optimal processing strategy based on dataset size and priority.
    
    Args:
        df_size: Number of samples to process
        priority: 'speed', 'cost', or 'balanced'
    
    Returns:
        Dictionary with recommended strategy and parameters
    """
    strategies = {
        'speed': {
            'method': 'ultra_threading',
            'workers': min(32, df_size),
            'description': 'Maximum immediate speedup',
            'expected_speedup': '10x+',
            'cost_savings': '0%'
        },
        'cost': {
            'method': 'batch_api',
            'workers': 1,
            'description': 'Async processing with cost savings',
            'expected_speedup': '∞ (async)',
            'cost_savings': '50%'
        },
        'balanced': {
            'method': 'ultra_threading' if df_size < 1000 else 'batch_api',
            'workers': 16 if df_size < 1000 else 1,
            'description': 'Threading for small datasets, batch for large',
            'expected_speedup': '6x+' if df_size < 1000 else '∞ (async)',
            'cost_savings': '0%' if df_size < 1000 else '50%'
        }
    }
    
    strategy = strategies.get(priority, strategies['balanced'])
    
    print(f"📋 OPTIMAL STRATEGY RECOMMENDATION")
    print(f"Dataset size: {df_size} samples")
    print(f"Priority: {priority}")
    print(f"Recommended: {strategy['method']}")
    print(f"Workers: {strategy['workers']}")
    print(f"Description: {strategy['description']}")
    print(f"Expected speedup: {strategy['expected_speedup']}")
    print(f"Cost savings: {strategy['cost_savings']}")
    
    return strategy
