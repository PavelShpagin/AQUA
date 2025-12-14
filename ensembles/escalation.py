#!/usr/bin/env python3
"""
Dynamic Escalation Ensemble (Language-agnostic)

Two routing modes:
- deterministic (default):
  * One small classification call (baseline/edit) + deterministic router using
    simple, language-agnostic risk flags derived from the aligned span.
  * If risky/low-trust: escalate to the next expert backend.
- llm: legacy model-driven router (combined classify+route with the small model,
  falling back to two calls: small classify + small router).

This keeps costs low (1 call on non-escalated cases) and improves stability by
decoupling classification from routing.
"""

import argparse
import os
import re
from typing import Dict, Any, List, Tuple, Set
import pandas as pd

from utils.ensemble import (
    call_single_judge_for_row_detailed,
    create_result_dict,
    process_rows_parallel,
    normalize_label,
    format_opinions,
)
from utils.optimization.runner import run_optimized_process_rows
from utils.judge import call_model_with_pricing, parse_tpfp_label
from ensembles.prompts import (
    FEEDBACK_FINAL_JUDGMENT_PROMPT,
    EDIT_FINAL_JUDGMENT_PROMPT,
    ESCALATION_ROUTER_PROMPT_GENERIC,
    COMBINED_CLASSIFY_ROUTE_PROMPT,
)


def _extract_edit_spans(aligned: str) -> List[Tuple[str, str]]:
    """Extract {old=>new} pairs from an aligned string.

    Returns a list of (old, new). If none are found, returns an empty list.
    """
    if not aligned:
        return []
    try:
        pattern = re.compile(r"\{(.*?)=>(.*?)\}")
        pairs = pattern.findall(aligned)
        return [(o.strip(), n.strip()) for (o, n) in pairs]
    except Exception:
        return []


def _has_alnum(text: str) -> bool:
    for ch in text:
        if ch.isalnum():
            return True
    return False


def _compute_risk_flags(src: str, tgt: str, aligned: str, small_label: str) -> List[str]:
    """Compute simple, language-agnostic risk flags from edit spans.

    Heuristics are intentionally conservative and fast.
    """
    flags: List[str] = []
    pairs = _extract_edit_spans(aligned)
    if not pairs and src and tgt and src != tgt:
        # Fallback: treat entire pair as one span
        pairs = [(src, tgt)]

    # Track counters to infer long/complex rewrites
    changed_tokens_total = 0
    for old, new in pairs:
        old_tokens = old.split()
        new_tokens = new.split()
        changed_tokens_total += abs(len(new_tokens) - len(old_tokens)) + min(len(old_tokens), len(new_tokens))

        combined = (old + new).strip()
        if not _has_alnum(combined):
            flags.append('punctuation_only')

        # Numbers changed
        nums_old = re.findall(r"\d+", old)
        nums_new = re.findall(r"\d+", new)
        if nums_old != nums_new and (nums_old or nums_new):
            flags.append('number_change')

        # Proper-noun change (very simple heuristic: capitalized token changed)
        proper_old = [t for t in old_tokens if t[:1].isupper() and _has_alnum(t)]
        proper_new = [t for t in new_tokens if t[:1].isupper() and _has_alnum(t)]
        if proper_old != proper_new and (proper_old or proper_new):
            flags.append('proper_noun_change')

        # URL/email/code-ish fragments altered
        if re.search(r"https?://|www\.|@|[#{}()\[\]<>]", old + new):
            flags.append('structured_token_change')

    # Long rewrite risk
    if changed_tokens_total >= 6:
        flags.append('long_rewrite_risk')

    # Label-informed risk (keep minimal)
    if small_label in {'FP1', 'FP2', 'Error'}:
        flags.append('low_trust_label')

    return list(dict.fromkeys(flags))  # de-dup, preserve order


def _route_deterministic(small_label: str, flags: List[str], available_backends: List[str],
                         escalate_labels: Set[str], escalate_flags: Set[str]) -> str:
    """Return backend name to escalate to, or '' if no escalation."""
    should_escalate = (small_label in escalate_labels) or any(f in escalate_flags for f in flags)
    if not should_escalate:
        return ''
    # Prefer the first available expert after small
    if len(available_backends) >= 2:
        return available_backends[1]
    if len(available_backends) >= 3:
        return available_backends[2]
    return ''


def build_router_prompt(experts: List[str], *, small_label: str, small_reason: str, src: str, tgt: str, aligned: str) -> str:
    # experts: list like ["Expert", "Senior", "Principal"] mapped to actual backends by order
    experts_list = ", ".join(experts)
    experts_json = " | ".join(f'"{e}"' for e in experts)
    return ESCALATION_ROUTER_PROMPT_GENERIC.format(
        experts_list=experts_list,
        experts_json=experts_json,
        small_label=small_label,
        small_reason=small_reason,
        src=src,
        tgt=tgt,
        aligned=aligned
    )


def process_row(idx_row: Tuple[int, pd.Series], args) -> Dict[str, Any]:
    idx, row = idx_row
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    aligned = str(row.get('aligned', '') or row.get('aligned_sentence', '') or row.get('alert', ''))

    # Backends in preference order
    small = args.backends[0]
    mid = args.backends[1] if len(args.backends) > 1 else None
    top = args.backends[2] if len(args.backends) > 2 else None

    judge_outputs: List[Dict[str, Any]] = []

    api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')

    # 1) Small classification call (baseline/edit)
    small_out = call_single_judge_for_row_detailed(
        args.judge, args.method, small, args.lang, row, args.moderation,
        optimization=(args.optimization == 'on')
    )
    small_label = normalize_label(small_out.get('label', 'Error'))
    small_reason = small_out.get('reasoning','')
    small_writing = small_out.get('writing_type','')
    judge_outputs.append(small_out)

    escalate_to = None

    if getattr(args, 'router', 'deterministic') == 'llm':
        # Legacy: LLM-based router (combined classify+route, then fallback)
        combined_prompt = COMBINED_CLASSIFY_ROUTE_PROMPT.format(src=src, tgt=tgt, aligned=aligned)
        ok_c, content_c, _t_c, _p_c = call_model_with_pricing(combined_prompt, small, api_token=api_token, moderation=False, temperature_override=0.0)
        try:
            import json
            data_c = json.loads(content_c) if ok_c and content_c else {}
            # Prefer expert recommendation; otherwise keep deterministic decision below
            if str(data_c.get('escalate', False)).lower() in {'true','1'}:
                exp = str(data_c.get('expert','none')).lower()
                if exp in {'o3','principal'} and top:
                    escalate_to = top
                elif exp in {'gpt-4o','4o','senior'} and mid:
                    escalate_to = mid
                elif mid:
                    escalate_to = mid
        except Exception:
            pass
    else:
        # Deterministic router
        flags = _compute_risk_flags(src, tgt, aligned, small_label)
        escalate_to_choice = _route_deterministic(
            small_label,
            flags,
            args.backends,
            getattr(args, 'escalate_labels', {'FP1','FP2','Error'}),
            getattr(args, 'escalate_flags', {'meaning_change_risk','number_change','proper_noun_change','long_rewrite_risk'})
        )
        escalate_to = escalate_to_choice or None

    if not escalate_to:
        return create_result_dict(row, idx, small_label, small_reason, small_writing, judge_outputs)

    expert_out = call_single_judge_for_row_detailed(
        args.judge, args.method, escalate_to, args.lang, row, args.moderation,
        optimization=(args.optimization == 'on')
    )
    expert_label = normalize_label(expert_out.get('label','Error'))
    judge_outputs.append(expert_out)

    # Optional finalizer: only if conflict between small and expert
    if expert_label != small_label:
        labels = [normalize_label(o.get('label','Error')) for o in judge_outputs if o.get('label')]
        opinions = format_opinions(judge_outputs)
        if getattr(args, 'finalizer', 'top') == 'small':
            final_backend = args.backends[0]
        elif getattr(args, 'finalizer', 'top') == 'mid':
            final_backend = args.backends[1] if len(args.backends) > 1 else args.backends[-1]
        else:
            final_backend = args.backends[2] if len(args.backends) > 2 else (args.backends[1] if len(args.backends) > 1 else args.backends[0])
        prompt = FEEDBACK_FINAL_JUDGMENT_PROMPT.format(opinions, src, tgt)
        ok_f, content_f, _tok_f, pricing_f = call_model_with_pricing(prompt, final_backend, api_token=api_token, moderation=False, temperature_override=0.0)
        final_label = parse_tpfp_label(content_f) if ok_f else expert_label
        final_rec = {'label': final_label, 'reasoning': content_f or '', 'writing_type': '', 'model': final_backend}
        judge_outputs.append(final_rec)
        return create_result_dict(row, idx, final_label, content_f or expert_out.get('reasoning',''), expert_out.get('writing_type',''), judge_outputs)

    return create_result_dict(row, idx, expert_label, expert_out.get('reasoning',''), expert_out.get('writing_type',''), judge_outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback','edit'])
    ap.add_argument('--method', required=True)
    ap.add_argument('--backends', nargs='+', required=True, help='Escalation order: small [mid] [top]')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--finalizer', choices=['small','mid','top'], default='top')
    ap.add_argument('--moderation', choices=['on','off'], default='off')
    ap.add_argument('--optimization', default='off', choices=['on','off'])
    ap.add_argument('--router', default='deterministic', choices=['deterministic','llm'])
    ap.add_argument('--escalate_labels', default='FP1,FP2,Error', help='Comma-separated labels to force escalation')
    ap.add_argument('--escalate_flags', default='meaning_change_risk,number_change,proper_noun_change,long_rewrite_risk', help='Comma-separated risk flags to trigger escalation')
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Normalize moderation flag for downstream helper
    args.moderation = 'on' if args.moderation in {'on','true','1','yes'} else 'off'

    # Parse routing config
    if isinstance(args.escalate_labels, str):
        args.escalate_labels = {s.strip() for s in args.escalate_labels.split(',') if s.strip()}
    if isinstance(args.escalate_flags, str):
        args.escalate_flags = {s.strip() for s in args.escalate_flags.split(',') if s.strip()}

    def proc(row):
        return process_row(row, args)

    if args.optimization == 'on':
        def process_fn(r):
            return proc((r.name, r))
        results = run_optimized_process_rows(df, process_fn, desc='escalation/optimized', target_shards=None, workers_per_shard=max(400, args.workers))
    else:
        results = process_rows_parallel(df, proc, args.workers, 'Escalation routing')

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # Print distribution
    from utils.judge import print_judge_distribution
    print_judge_distribution(results, f'Escalation Ensemble ({args.judge}/{args.method})')


if __name__ == '__main__':
    main()




