#!/usr/bin/env python3
"""
Iterative Consensus Escalation (iter_escalation)

Goal: Achieve ~93% 4-class accuracy at <$2/10K by combining:
- K-votes from a small model (multi-vote classification per case)
- Consensus check (2/3 majority + nonadjacency guard)
- Conditional escalation to stronger experts (gpt-4o, o3) when consensus is weak
- Final judgment by top expert over collected opinions

Consensus rule (for labels in [TP, FP3, FP2, FP1]):
- Let c* be the most frequent label across all collected votes
- Accept consensus if:
  (a) freq(c*) ≥ ceil(2/3 * total_votes)
  (b) For nonadjacent classes to c*, freq ≤ floor(1/4 * total_votes)
- Otherwise escalate to next expert tier, append its K-votes, re-check; stop at top tier

Nonadjacent classes:
- nonadj(TP) = {FP2, FP1}
- nonadj(FP3) = {FP1}
- nonadj(FP2) = {TP}
- nonadj(FP1) = {TP, FP3}

Outputs one final TP/FP3/FP2/FP1 label per row.
"""

import argparse
import os
import json
from collections import Counter
from typing import Dict, Any, List, Tuple
import pandas as pd

from utils.ensemble import (
    process_rows_parallel,
    create_result_dict,
    normalize_label,
    format_opinions,
    call_single_judge_for_row_detailed,
)
from utils.optimization.runner import run_optimized_process_rows
from utils.judge import call_model_with_pricing, parse_tpfp_label
from ensembles.prompts import (
    FEEDBACK_VOTES_PROMPT,
    FEEDBACK_FINAL_JUDGMENT_PROMPT,
    ESCALATION_ROUTER_PROMPT_GENERIC,
    ENSEMBLE_PAIR_CORRECTNESS_PROMPT,
    ENSEMBLE_REWARD_PROMPT,
)

LABELS = ['TP','FP3','FP2','FP1']
NONADJ = {
    'TP': {'FP2','FP1'},
    'FP3': {'FP1'},
    'FP2': {'TP'},
    'FP1': {'TP','FP3'},
}

def build_votes_prompt(src: str, tgt: str, aligned: str, n_votes: int) -> str:
    return FEEDBACK_VOTES_PROMPT.format(src=src, tgt=tgt, aligned=aligned, n_votes=n_votes)


def parse_votes(content: str) -> List[str]:
    try:
        data = json.loads(content or '{}')
        votes = data.get('votes', [])
        out = []
        for v in votes:
            lab = normalize_label(str(v))
            if lab in LABELS:
                out.append(lab)
        return out
    except Exception:
        return []


def has_consensus(votes: List[str]) -> Tuple[bool, str]:
    if not votes:
        return False, ''
    cnt = Counter(votes)
    total = len(votes)
    c_star, f_star = cnt.most_common(1)[0]
    need = (2*total + 2) // 3  # ceil(2/3 * total)
    guard = total // 4         # floor(1/4 * total)
    ok_major = f_star >= need
    ok_nonadj = True
    for lab, f in cnt.items():
        if lab in NONADJ.get(c_star, set()) and f > guard:
            ok_nonadj = False
            break
    return (ok_major and ok_nonadj), c_star


def call_k_votes(backend: str, src: str, tgt: str, aligned: str, n_votes: int, api_token: str) -> Tuple[List[str], Dict[str, Any]]:
    prompt = build_votes_prompt(src, tgt, aligned, n_votes)
    ok, content, _tok, pricing = call_model_with_pricing(prompt, backend, api_token=api_token, moderation=False, temperature_override=0.0)
    votes = parse_votes(content if ok else '')
    meta = {'model': backend, 'raw': content or '', 'n_votes': n_votes, 'token_usage': (_tok or {}), 'pricing': (pricing or {})}
    return votes, meta


def process_row(idx_row: Tuple[int, pd.Series], args) -> Dict[str, Any]:
    idx, row = idx_row
    src = str(row.get('src',''))
    tgt = str(row.get('tgt',''))
    aligned = str(row.get('aligned','') or row.get('aligned_sentence','') or row.get('alert',''))

    api_token = os.getenv('API_TOKEN','') or os.getenv('OPENAI_API_KEY','')

    # Backends in order: small, mid, top
    tiers = list(args.backends)
    n_votes_small = args.k_small

    votes_all: List[str] = []
    opinions: List[Dict[str, Any]] = []

    # 0) Small single-pass baseline for routing context
    small_out = call_single_judge_for_row_detailed(args.judge, args.method, tiers[0], args.lang, row, 'off')
    small_label_single = normalize_label(small_out.get('label','Error'))
    opinions.append(small_out)

    # 1) Small tier votes
    v_small, meta_small = call_k_votes(tiers[0], src, tgt, aligned, n_votes_small, api_token)
    votes_all.extend(v_small)
    opinions.append({'label': v_small[0] if v_small else 'Error', 'reasoning': meta_small.get('raw',''), 'model': tiers[0]})
    ok, c_star = has_consensus(votes_all)
    if ok:
        return create_result_dict(row, idx, c_star, meta_small.get('raw',''), '', opinions)

    # 1.5) Router gating to avoid unnecessary expert calls
    # Build expert names list
    expert_names: List[str] = []
    if len(tiers) >= 2:
        expert_names.append('Expert')
    if len(tiers) >= 3:
        expert_names.append('Senior')
    opinions_prompt = ESCALATION_ROUTER_PROMPT_GENERIC.format(
        experts_list=", ".join(expert_names or ['Expert']),
        experts_json=" | ".join(f'"{e}"' for e in (expert_names or ['Expert'])),
        small_label=small_label_single,
        small_reason=str(small_out.get('reasoning',''))[:1000],
        src=src,
        tgt=tgt,
        aligned=aligned
    )
    ok_r, content_r, _tok_r, _pricing_r = call_model_with_pricing(opinions_prompt, tiers[0], api_token=api_token, moderation=False, temperature_override=0.0)
    want_escalate = True
    bucket = 'medium'
    if ok_r and content_r:
        try:
            data_r = json.loads(content_r)
            want_escalate = str(data_r.get('escalate', True)).lower() in {'true','1'}
            bucket = str(data_r.get('confidence_bucket','medium')).lower()
        except Exception:
            want_escalate = True
            bucket = 'medium'
    # Add lightweight pair-correctness + reward gating (two single calls on small)
    pc_prompt = ENSEMBLE_PAIR_CORRECTNESS_PROMPT.format(src=src, tgt=tgt, aligned=aligned)
    ok_pc, content_pc, _tok_pc, _pricing_pc = call_model_with_pricing(pc_prompt, tiers[0], api_token=api_token, moderation=False, temperature_override=0.0)
    reward_prompt = ENSEMBLE_REWARD_PROMPT.format(src=src, tgt=tgt, aligned=aligned)
    ok_rw, content_rw, _tok_rw, _pricing_rw = call_model_with_pricing(reward_prompt, tiers[0], api_token=api_token, moderation=False, temperature_override=0.0)
    src_ok = tgt_ok = True
    reward = 0
    try:
        dpc = json.loads(content_pc or '{}'); src_ok = bool(dpc.get('source_correct', True)); tgt_ok = bool(dpc.get('target_correct', True))
    except Exception:
        src_ok = tgt_ok = True
    try:
        drw = json.loads(content_rw or '{}'); reward = int(drw.get('improvement', 0))
    except Exception:
        reward = 0

    # Heuristics: if both correct and reward in {0,1} with small votes leaning TP/FP3, avoid escalation
    cnt_small = Counter(votes_all)
    small_major, small_freq = (cnt_small.most_common(1)[0] if cnt_small else ('TP', 0))
    ambiguous_tp_fp3 = (src_ok and tgt_ok and reward in {0,1} and small_major in {'TP','FP3'})
    # Escalate on router signal, risky small label, OR ambiguous TP/FP3 to reduce TP bias
    should_escalate = (want_escalate or (bucket in {'vlow','low','medium'}) or (small_label_single in {'FP1','FP2','Error'}) or ambiguous_tp_fp3)
    if not should_escalate:
        # No escalation allowed: accept plurality from small votes; break ties by small single label
        cnt = Counter(votes_all)
        if cnt:
            most = cnt.most_common()
            if len(most) > 1 and most[0][1] == most[1][1] and small_label_single in LABELS:
                return create_result_dict(row, idx, small_label_single, meta_small.get('raw',''), '', opinions)
            return create_result_dict(row, idx, most[0][0], meta_small.get('raw',''), '', opinions)
        return create_result_dict(row, idx, small_label_single or 'TP', meta_small.get('raw',''), '', opinions)

    # 2) Escalate to mid if exists
    if len(tiers) >= 2:
        v_mid, meta_mid = call_k_votes(tiers[1], src, tgt, aligned, args.k_mid, api_token)
        votes_all.extend(v_mid)
        opinions.append({'label': v_mid[0] if v_mid else 'Error', 'reasoning': meta_mid.get('raw',''), 'model': tiers[1]})
        ok, c_star = has_consensus(votes_all)
        if ok:
            return create_result_dict(row, idx, c_star, meta_mid.get('raw',''), '', opinions)

    # 3) Escalate to top if exists
    if len(tiers) >= 3:
        v_top, meta_top = call_k_votes(tiers[2], src, tgt, aligned, args.k_top, api_token)
        votes_all.extend(v_top)
        opinions.append({'label': v_top[0] if v_top else 'Error', 'reasoning': meta_top.get('raw',''), 'model': tiers[2]})
        ok, c_star = has_consensus(votes_all)
        if ok:
            return create_result_dict(row, idx, c_star, meta_top.get('raw',''), '', opinions)

    # 4) Final judgment by top available model over opinions (iter_critic-style)
    final_backend = tiers[min(2, len(tiers)-1)]
    formatted = format_opinions(opinions)
    prompt = FEEDBACK_FINAL_JUDGMENT_PROMPT.format(formatted, src, tgt)
    ok_f, content_f, _tok_f, pricing_f = call_model_with_pricing(prompt, final_backend, api_token=api_token, moderation=False, temperature_override=0.0)
    final_label = parse_tpfp_label(content_f) if ok_f else (votes_all[0] if votes_all else 'TP')
    opinions.append({'label': final_label, 'reasoning': content_f or '', 'model': final_backend})
    return create_result_dict(row, idx, final_label, content_f or '', '', opinions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback'])
    ap.add_argument('--method', required=True)
    ap.add_argument('--backends', nargs='+', required=True, help='Order: small [mid] [top]')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--optimization', default='off', choices=['on','off'])
    ap.add_argument('--k_small', type=int, default=3)
    ap.add_argument('--k_mid', type=int, default=2)
    ap.add_argument('--k_top', type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    def proc(row):
        return process_row(row, args)

    if args.optimization == 'on':
        def process_fn(r):
            return proc((r.name, r))
        results = run_optimized_process_rows(df, process_fn, desc='iter_escalation/optimized', target_shards=None, workers_per_shard=max(400, args.workers))
    else:
        results = process_rows_parallel(df, proc, args.workers, 'Iterative consensus escalation')

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    from utils.judge import print_judge_distribution
    print_judge_distribution(results, f'Iterative Consensus Escalation ({args.judge}/{args.method})')

if __name__ == '__main__':
    main()
