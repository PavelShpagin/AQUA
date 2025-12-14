#!/usr/bin/env python3
"""
Debate-Routed Mini-Agent Ensemble (cheap → modular triage → expert → oracle)

Inspiration: iter_critic (buffered consensus), inner_debate (argumentation),
and modular triage (nonsense/meaning/quality signals).

Natural pipeline:
1) Cheap baseline judge (e.g., gpt-4.1-nano, feedback/baseline) → initial label
2) If uncertain, run modular triage (same cheap backend) to extract signals
3) Optionally add a second cheap judge (e.g., gpt-4o-mini) to probe disagreement
4) Synthesize a compact debate text from the opinions/signals
5) Expert final judge (e.g., gpt-4.1 or gpt-4o) decides after reading the debate
6) Optional oracle (e.g., o3) only if expert fails

Escalation heuristic (simple/clean): escalate unless cheap == TP.
This yields strong accuracy with minimal added cost on TP-heavy corpora.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ensemble import (
    call_judge_in_process,
    create_result_dict,
    normalize_label,
    process_rows_parallel,
)
from utils.judge import (
    parse_tpfp_label,
    add_pricing_to_result_dict,
    call_model_with_pricing,
)
from utils.modular import classify_modular
from judges.feedback import prompts as fb_prompts
from ensembles.prompts import FEEDBACK_DEBATE_PROMPT


def _call_final_feedback_debate(debate_text: str, src: str, tgt: str, backend: str) -> Dict[str, Any]:
    prompt = FEEDBACK_DEBATE_PROMPT.replace('{0}', debate_text).replace('{1}', src).replace('{2}', tgt)
    api_token = os.getenv('OPENAI_API_KEY', '') or os.getenv('API_TOKEN', '')
    ok, content, _tokens, pricing = call_model_with_pricing(
        prompt, backend, api_token=api_token, moderation=False, temperature_override=0.0
    )
    label = parse_tpfp_label(content) if ok else 'Error'
    out = {
        'label': normalize_label(label),
        'reasoning': (content or '').strip() if content else 'Error',
        'model': backend,
    }
    if ok:
        out = add_pricing_to_result_dict(out, pricing)
    return out


def _format_debate(opinions: List[Dict[str, Any]], modular_info: Dict[str, Any]) -> str:
    parts: List[str] = []
    for i, op in enumerate(opinions, 1):
        parts.append(f"Judge {i}: {op.get('label','Error')}\nReasoning: {op.get('reasoning','')}")
    if modular_info:
        ns = modular_info.get('nonsense_score')
        mc = modular_info.get('meaning_change_score')
        ql = modular_info.get('quality_score')
        parts.append(
            "Modular triage:\n"
            f"- Nonsense: {ns}\n- Meaning change: {mc}\n- Quality: {ql}"
        )
    return "\n\n".join(parts)


def is_uncertain(label: str) -> bool:
    lab = normalize_label(label)
    return lab in {'FP1', 'FP2', 'FP3', 'TN', 'FN', 'Error'}


def process_single_row(row_data: Tuple[int, pd.Series], args) -> Dict[str, Any]:
    idx, row = row_data
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))

    judge_outputs: List[Dict[str, Any]] = []

    # 1) Cheap baseline pass
    cheap = call_judge_in_process(
        judge=args.judge,
        method=args.method,
        src=src,
        tgt=tgt,
        backend=args.cheap_backend,
        lang=args.lang,
        moderation=(args.moderation == 'on'),
        api_token=os.getenv('OPENAI_API_KEY', '') or os.getenv('API_TOKEN', ''),
        aligned_text=str(row.get('aligned', '')) or str(row.get('aligned_sentence', '')) or str(row.get('alert', '')),
    )
    cheap_label = normalize_label(cheap.get('tp_fp_label', 'Error'))
    judge_outputs.append({
        'label': cheap_label,
        'reasoning': cheap.get('reasoning', ''),
        'model': args.cheap_backend,
        'input_tokens': cheap.get('input_tokens', 0),
        'output_tokens': cheap.get('output_tokens', 0),
        'reasoning_tokens': cheap.get('reasoning_tokens', 0),
        'cached_tokens': cheap.get('cached_tokens', 0),
        'total_cost_usd': cheap.get('total_cost_usd', 0.0),
    })

    if not is_uncertain(cheap_label):
        return create_result_dict(row, idx, cheap_label, cheap.get('reasoning', ''), cheap.get('writing_type',''), judge_outputs)

    # 2) Modular triage on cheap backend
    modular_info = {}
    try:
        m_label, m_reason, dbg = classify_modular(
            src, tgt,
            backend=args.cheap_backend,
            api_token=os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', ''),
            language_label={'en':'English','de':'German','ua':'Ukrainian','es':'Spanish'}.get(args.lang, 'English'),
            prompts=fb_prompts,
            demo=False,
            moderation=False,
            return_debug=True,
        )
        modular_info = dbg or {}
        judge_outputs.append({
            'label': normalize_label(m_label),
            'reasoning': m_reason,
            'model': f"modular@{args.cheap_backend}",
        })
    except Exception as e:
        judge_outputs.append({'label':'Error','reasoning':f"Modular failed: {e}", 'model': f"modular@{args.cheap_backend}"})

    # 3) Optional second cheap judge
    if args.cheap2_backend:
        cheap2 = call_judge_in_process(
            judge=args.judge,
            method=args.method,
            src=src,
            tgt=tgt,
            backend=args.cheap2_backend,
            lang=args.lang,
            moderation=(args.moderation == 'on'),
            api_token=os.getenv('OPENAI_API_KEY', '') or os.getenv('API_TOKEN', ''),
            aligned_text=str(row.get('aligned', '')) or str(row.get('aligned_sentence', '')) or str(row.get('alert', '')),
        )
        cheap2_label = normalize_label(cheap2.get('tp_fp_label', 'Error'))
        judge_outputs.append({
            'label': cheap2_label,
            'reasoning': cheap2.get('reasoning', ''),
            'model': args.cheap2_backend,
            'input_tokens': cheap2.get('input_tokens', 0),
            'output_tokens': cheap2.get('output_tokens', 0),
            'reasoning_tokens': cheap2.get('reasoning_tokens', 0),
            'cached_tokens': cheap2.get('cached_tokens', 0),
            'total_cost_usd': cheap2.get('total_cost_usd', 0.0),
        })

    # Synthesize debate
    debate_text = _format_debate(judge_outputs, modular_info)

    # 4) Expert decision after debate
    expert = _call_final_feedback_debate(debate_text, src, tgt, backend=args.expert_backend)
    expert_label = normalize_label(expert.get('label','Error'))
    judge_outputs.append(expert)

    if expert_label != 'Error':
        return create_result_dict(row, idx, expert_label, expert.get('reasoning',''), '', judge_outputs)

    # 5) Optional oracle
    if args.oracle_backend:
        oracle = _call_final_feedback_debate(debate_text, src, tgt, backend=args.oracle_backend)
        oracle_label = normalize_label(oracle.get('label','Error'))
        judge_outputs.append(oracle)
        return create_result_dict(row, idx, oracle_label, oracle.get('reasoning',''), '', judge_outputs)

    return create_result_dict(row, idx, expert_label, expert.get('reasoning',''), '', judge_outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback'])
    ap.add_argument('--method', required=True, choices=['baseline', 'legacy'])
    ap.add_argument('--cheap_backend', required=True)
    ap.add_argument('--cheap2_backend', default='')
    ap.add_argument('--expert_backend', required=True)
    ap.add_argument('--oracle_backend', default='')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--moderation', choices=['on','off'], default='off')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    def process_func(row_data):
        return process_single_row(row_data, args)
    results = process_rows_parallel(df, process_func, args.workers, desc='debate_router')

    out_df = pd.DataFrame(results)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    try:
        from utils.judge import print_judge_distribution
        print_judge_distribution(results, 'Debate-Routed Ensemble')
    except Exception:
        pass


if __name__ == '__main__':
    main()




