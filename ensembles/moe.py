#!/usr/bin/env python3
"""
Mixture-of-Experts (MoE) Ensemble

- A small router (gpt-4o-mini) decides which expert to call: small/mid/top.
- We then call exactly ONE expert to get the final classification.
- Goal: simplify flow, reduce variance, and keep cost low while maintaining accuracy.
"""

import argparse
import os
import pandas as pd
from typing import Dict, Any, List, Tuple

from utils.ensemble import (
    call_single_judge_for_row_detailed,
    create_result_dict,
    process_rows_parallel,
)
from utils.optimization.runner import run_optimized_process_rows
from ensembles.prompts import MOE_ROUTER_PROMPT
from utils.judge import call_model_with_pricing


def build_moe_router_prompt(experts: List[str], *, src: str, tgt: str, aligned: str) -> str:
    experts_list = ", ".join(experts)
    experts_json = " | ".join(f'"{e}"' for e in experts)
    return MOE_ROUTER_PROMPT.format(
        experts_list=experts_list,
        experts_json=experts_json,
        src=src,
        tgt=tgt,
        aligned=aligned,
    )


def process_row(idx_row: Tuple[int, pd.Series], args) -> Dict[str, Any]:
    idx, row = idx_row
    src = str(row.get('src',''))
    tgt = str(row.get('tgt',''))
    aligned = str(row.get('aligned','') or row.get('aligned_sentence','') or row.get('alert',''))

    # Backends: small (router), mid, top
    small = args.backends[0]
    mid = args.backends[1] if len(args.backends) > 1 else None
    top = args.backends[2] if len(args.backends) > 2 else None

    # 1) Router decides target expert
    expert_names: List[str] = []
    if len(args.backends) >= 1:
        expert_names.append('Small')
    if len(args.backends) >= 2:
        expert_names.append('Expert')
    if len(args.backends) >= 3:
        expert_names.append('Senior')

    prompt = build_moe_router_prompt(expert_names or ['Small'], src=src, tgt=tgt, aligned=aligned)
    api_token = os.getenv('API_TOKEN','') or os.getenv('OPENAI_API_KEY','')
    ok, content, _tok, pricing = call_model_with_pricing(prompt, small, api_token=api_token, moderation=False, temperature_override=0.0)

    # Map chosen expert to backend
    target_backend = small
    if ok and content:
        try:
            import json
            data = json.loads(content)
            exp = str(data.get('expert','Small'))
            if exp.lower() in {'senior'} and top:
                target_backend = top
            elif exp.lower() in {'expert'} and mid:
                target_backend = mid
            else:
                target_backend = small
        except Exception:
            target_backend = small

    # 2) Call exactly one chosen expert
    result = call_single_judge_for_row_detailed(args.judge, args.method, target_backend, args.lang, row, args.moderation)
    label = result.get('label','Error')
    reasoning = result.get('reasoning','')
    writing_type = result.get('writing_type','')

    return create_result_dict(row, idx, label, reasoning, writing_type, [result, {'router': content or ''}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback','edit'])
    ap.add_argument('--method', required=True)
    ap.add_argument('--backends', nargs='+', required=True, help='Backends order: small [mid] [top]')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--moderation', choices=['on','off'], default='off')
    ap.add_argument('--optimization', default='off', choices=['on','off'])
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    def proc(row):
        return process_row(row, args)

    if args.optimization == 'on':
        def process_fn(r):
            return proc((r.name, r))
        results = run_optimized_process_rows(df, process_fn, desc='moe/optimized', target_shards=None, workers_per_shard=max(400, args.workers))
    else:
        results = process_rows_parallel(df, proc, args.workers, 'MoE routing')

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    from utils.judge import print_judge_distribution
    print_judge_distribution(results, f'MoE Ensemble ({args.judge}/{args.method})')

if __name__ == '__main__':
    main()


