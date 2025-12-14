#!/usr/bin/env python3
"""
Iterative Critic Ensemble with separate TP/FP* and TN/FN buffer logic.
"""

import argparse
import os
import sys
import tempfile
import pandas as pd
from typing import Dict, Any, List
from collections import Counter

# Load environment variables from .env file
try:
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ensemble import (
    call_parallel_judges_for_row_detailed, call_single_judge_for_row_detailed,
    check_consensus, format_opinions, create_result_dict, process_rows_parallel,
    judge_row_with_filter
)
from ensembles.prompts import (
    FEEDBACK_FINAL_JUDGMENT_PROMPT, 
    SENTENCE_FINAL_JUDGMENT_PROMPT, 
    EDIT_FINAL_JUDGMENT_PROMPT
)
from utils.judge import call_model_with_pricing, add_pricing_to_result_dict
from utils.optimization.runner import run_optimized_process_rows
try:
    from utils.llm.settings import get_final_judge_temperature
except Exception:
    def get_final_judge_temperature(default: float = 0.0) -> float:
        return default


def get_final_judge_prompt(judge_type: str) -> str:
    """Get the appropriate final judgment prompt for the judge type."""
    if judge_type == 'feedback':
        return FEEDBACK_FINAL_JUDGMENT_PROMPT
    elif judge_type in ['sentence', 'edit']:
        return SENTENCE_FINAL_JUDGMENT_PROMPT if judge_type == 'sentence' else EDIT_FINAL_JUDGMENT_PROMPT
    else:
        return FEEDBACK_FINAL_JUDGMENT_PROMPT  # fallback


def call_final_judge(judge: str, method: str, backend: str, lang: str, 
                    row: pd.Series, top_classes: List[str], 
                    judge_results: List[Dict], backend_offset: int) -> Dict[str, str]:
    """Call final judge with top 2 classes context."""
    
    # Build final judge prompt with top 2 classes context
    final_prompt = get_final_judge_prompt(judge)
    
    # Create opinions string with top 2 classes context
    top_classes_context = f"The top 2 most frequent labels from previous judges are: {top_classes[0]} and {top_classes[1]}.\n\nPlease make a final decision between these options.\n\n"
    opinions = top_classes_context + format_opinions(judge_results)
    
    # Get language label
    from utils.judge import get_language_label, build_numbered_prompt
    language_label = get_language_label(lang)
    
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    # Build prompt
    if judge in ['sentence', 'edit']:
        from utils.errant_align import get_alignment_for_language
        # Generate alignment (not directly used in final prompt schema)
        _ = get_alignment_for_language(src, tgt, language=lang)
        # Final judgment prompts expect: {0}=opinions, {1}=Original, {2}=Suggested
        prompt = build_numbered_prompt(final_prompt, opinions, src, tgt)
    else:
        # Feedback judge uses same order: {0}=opinions, {1}=Original, {2}=Suggested
        prompt = build_numbered_prompt(final_prompt, opinions, src, tgt)
    
    # Call model
    api_token = os.getenv('API_TOKEN', '')
    ok, content, _, pricing_info = call_model_with_pricing(
        prompt,
        backend,
        api_token=api_token,
        moderation=False,
        temperature_override=get_final_judge_temperature(0.0),
    )
    
    if not ok:
        src = str(row.get('src', ''))
        tgt = str(row.get('tgt', ''))
        fallback = 'TN' if judge in ['sentence','edit'] and src == tgt else 'TP'
        return {'label': fallback, 'reasoning': 'Final judge failed | Fallback applied', 'writing_type': ''}
    
    # Parse response (prefer 6-class for sentence/edit, 4-class for feedback, TN/FN for tnfn)
    from utils.judge import parse_tpfp_label, parse_writing_type
    if judge == 'tnfn':
        from utils.judge import parse_tnfn_label
        label = parse_tnfn_label(content)
    elif judge in ['sentence', 'edit']:
        import re as _re
        up = content.upper() if content else ''
        m = _re.search(r'"CLASSIFICATION"\s*:\s*"(TP|FP1|FP2|FP3|TN|FN)"', up)
        if m:
            label = m.group(1)
        else:
            # Fallback: scan for any of the 6 classes
            label = 'TP'
            for cand in ['FP1','FP2','FP3','TN','FN','TP']:
                if f' {cand}' in f' {up}':
                    label = cand
                    break
    else:
        label = parse_tpfp_label(content)
    
    writing_type = parse_writing_type(content)
    
    result = {
        'label': label,
        'reasoning': content,
        'writing_type': writing_type
    }
    # Add pricing information
    result = add_pricing_to_result_dict(result, pricing_info)
    return result


def process_single_row(row_data: tuple, args) -> Dict[str, Any]:
    """Process a single row with iter_critic logic using separate TP/FP* and TN/FN buffers."""
    idx, row = row_data
    
    # Optional multi-target filtering: accept first TP/TN among candidate columns
    if getattr(args, 'filter', None):
        accepted = judge_row_with_filter(
            idx,
            row,
            judge=args.judge,
            method=args.method,
            backends=args.backends,
            lang=args.lang,
            n_judges=args.n_judges,
            moderation=args.moderation,
            filter_cols=args.filter,
        )
        if accepted is not None:
            return accepted
    
    # Initialize separate buffers
    tpfp_buffer = []  # TP, FP1, FP2, FP3
    tnfn_buffer = []  # TN, FN
    all_judge_results = []  # For opinion formatting
    
    # Backends by tier
    small_backend = args.backends[0]
    mid_backend = args.backends[1] if len(args.backends) > 1 else None
    top_backend = args.backends[2] if len(args.backends) > 2 else None

    # Initial judge calls - PARALLEL for n_judges
    # In lite mode, call only the small backend for initial votes
    initial_backends = [small_backend] if getattr(args, 'lite', False) else args.backends
    judge_outputs = call_parallel_judges_for_row_detailed(
        judge=args.judge,
        method=args.method,
        backends=initial_backends,
        lang=args.lang,
        row=row,
        n_judges=args.n_judges,
        backend_offset=idx * args.n_judges,  # Stagger across rows
        moderation=args.moderation
    )
    
    # Populate initial buffers
    from utils.ensemble import normalize_label
    for output in judge_outputs:
        label = normalize_label(output.get('label', 'Error'))
        reasoning = output.get('reasoning', '')
        writing_type = output.get('writing_type', '')
        
        # Check for errors
        if label == "Error":
            return create_result_dict(row, idx, "Error", "One or more judges returned Error", writing_type, all_judge_results)
        
        # Add to appropriate buffer
        if label in ['TP', 'FP1', 'FP2', 'FP3']:
            tpfp_buffer.append(label)
        elif label in ['TN', 'FN']:
            tnfn_buffer.append(label)
        
        # Keep track of all results for opinions
        # Store the full output for pricing aggregation
        all_judge_results.append(output)
    
    # Check if either buffer has achieved consensus
    def check_buffer_consensus(buffer_labels):
        if not buffer_labels:
            return None
        return check_consensus(buffer_labels)
    
    # Check initial consensus
    tpfp_consensus = check_buffer_consensus(tpfp_buffer)
    tnfn_consensus = check_buffer_consensus(tnfn_buffer)
    
    if tpfp_consensus and len(tpfp_buffer) >= len(tnfn_buffer):
        # TP/FP* buffer has consensus and is largest/equal
        last = judge_outputs[-1] if judge_outputs else {'reasoning': '', 'writing_type': ''}
        return create_result_dict(row, idx, tpfp_consensus, f"TP/FP* consensus: {tpfp_consensus}", last.get('writing_type', ''), all_judge_results)
    elif tnfn_consensus and len(tnfn_buffer) > len(tpfp_buffer):
        # TN/FN buffer has consensus and is largest
        last = judge_outputs[-1] if judge_outputs else {'reasoning': '', 'writing_type': ''}
        return create_result_dict(row, idx, tnfn_consensus, f"TN/FN consensus: {tnfn_consensus}", last.get('writing_type', ''), all_judge_results)
    
    # If lite mode: do a single mid-tier probe, then finalize
    if getattr(args, 'lite', False):
        # Try one mid-tier probe if available
        if not (tpfp_consensus or tnfn_consensus) and mid_backend is not None:
            additional = call_single_judge_for_row_detailed(
                judge=args.judge,
                method=args.method,
                backend=mid_backend,
                lang=args.lang,
                row=row,
                moderation=args.moderation,
                opinions=format_opinions(all_judge_results)
            )
            add_label = normalize_label(additional.get('label', 'Error'))
            if add_label in ['TP','FP1','FP2','FP3']:
                tpfp_buffer.append(add_label)
            elif add_label in ['TN','FN']:
                tnfn_buffer.append(add_label)
            all_judge_results.append(additional)

            # Re-check consensus
            tpfp_consensus = check_buffer_consensus(tpfp_buffer)
            tnfn_consensus = check_buffer_consensus(tnfn_buffer)
            if tpfp_consensus and len(tpfp_buffer) >= len(tnfn_buffer):
                return create_result_dict(row, idx, tpfp_consensus, f"TP/FP* consensus after mid probe: {tpfp_consensus}", "", all_judge_results)
            if tnfn_consensus and len(tnfn_buffer) > len(tpfp_buffer):
                return create_result_dict(row, idx, tnfn_consensus, f"TN/FN consensus after mid probe: {tnfn_consensus}", "", all_judge_results)

        # Finalize: pick final backend based on conflict severity
        def has_severe_conflict(labels: List[str]) -> bool:
            s = set(labels)
            return ('TP' in s and 'FP1' in s) or ('TP' in s and 'FP2' in s) or ('FP1' in s and 'FP3' in s)

        selected_buffer = tpfp_buffer if len(tpfp_buffer) >= len(tnfn_buffer) else tnfn_buffer
        buffer_type = "TP/FP*" if selected_buffer is tpfp_buffer else "TN/FN"

        # Top-2 from selected buffer
        def top_two_from(buf: List[str], buf_type: str) -> List[str]:
            if not buf:
                return []
            cnt = Counter(buf)
            if buf_type == 'TP/FP*':
                order = {'FP1':1,'FP2':2,'FP3':3,'TP':4}
            else:
                order = {'FN':1,'TN':2}
            sorted_labels = sorted(cnt.items(), key=lambda x: (-x[1], order.get(x[0], 99)))
            return [lbl for lbl,_ in sorted_labels[:2]]

        top_two = top_two_from(selected_buffer, buffer_type)
        # Choose final backend: prefer mid; escalate to top only for severe conflicts and when available
        final_backend = mid_backend or small_backend
        if has_severe_conflict(tpfp_buffer) and top_backend is not None:
            final_backend = top_backend

        final_judge_result = call_final_judge(
            judge=args.judge,
            method=args.method,
            backend=final_backend,
            lang=args.lang,
            row=row,
            top_classes=top_two if top_two else (['TP','FP3'] if buffer_type=='TP/FP*' else ['TN','FN']),
            judge_results=all_judge_results,
            backend_offset=0
        )
        return create_result_dict(row, idx, final_judge_result['label'], final_judge_result['reasoning'], final_judge_result.get('writing_type',''), all_judge_results + [final_judge_result])

    # No consensus - run iterative process (full mode)
    iteration = 0
    backend_cycle = args.n_judges + (idx * args.n_judges)
    
    while iteration < args.max_iters:
        iteration += 1
        
        # Format opinions from previous judges (pass last n judges as context)
        recent_results = all_judge_results[-args.n_judges:] if len(all_judge_results) >= args.n_judges else all_judge_results
        opinions = format_opinions(recent_results)
        
        # Call additional judge with opinions
        backend = args.backends[backend_cycle % len(args.backends)]
        backend_cycle += 1
        
        additional = call_single_judge_for_row_detailed(
            judge=args.judge,
            method=args.method,
            backend=backend,
            lang=args.lang,
            row=row,
            moderation=args.moderation,
            opinions=opinions
        )
        additional_label = normalize_label(additional.get('label', 'Error'))
        
        # Check for errors
        if additional_label == "Error":
            # Skip this iteration and continue, do not fail the row
            continue
        
        # Add to appropriate buffer
        if additional_label in ['TP', 'FP1', 'FP2', 'FP3']:
            tpfp_buffer.append(additional_label)
        elif additional_label in ['TN', 'FN']:
            tnfn_buffer.append(additional_label)
        
        # Add to all results for opinions
        # Store the full result for pricing aggregation
        all_judge_results.append(additional)
        
        # Check for consensus again
        tpfp_consensus = check_buffer_consensus(tpfp_buffer)
        tnfn_consensus = check_buffer_consensus(tnfn_buffer)
        
        if tpfp_consensus and len(tpfp_buffer) >= len(tnfn_buffer):
            return create_result_dict(row, idx, tpfp_consensus, f"TP/FP* consensus after {iteration} iterations: {tpfp_consensus}", "", all_judge_results)
        elif tnfn_consensus and len(tnfn_buffer) > len(tpfp_buffer):
            return create_result_dict(row, idx, tnfn_consensus, f"TN/FN consensus after {iteration} iterations: {tnfn_consensus}", "", all_judge_results)
    
    # Max iterations reached - select dominant buffer and get top 2 classes
    if len(tpfp_buffer) > len(tnfn_buffer):
        # TP/FP* buffer is larger
        selected_buffer = tpfp_buffer
        buffer_type = "TP/FP*"
    elif len(tnfn_buffer) > len(tpfp_buffer):
        # TN/FN buffer is larger  
        selected_buffer = tnfn_buffer
        buffer_type = "TN/FN"
    else:
        # Equal size - prefer TP/FP* buffer
        selected_buffer = tpfp_buffer
        buffer_type = "TP/FP*"
    
    # Get top 2 classes from selected buffer using gradation order
    def get_buffer_top_two(buffer_labels, buffer_type):
        if not buffer_labels:
            return []
        
        label_counts = Counter(buffer_labels)
        
        if buffer_type == "TP/FP*":
            # Order: FP1 < FP2 < FP3 < TP (lower is higher priority)
            gradation_order = {'FP1': 1, 'FP2': 2, 'FP3': 3, 'TP': 4}
        else:  # TN/FN
            # Order: FN < TN (lower is higher priority)
            gradation_order = {'FN': 1, 'TN': 2}
        
        # Sort by count (descending) then by gradation order (ascending for priority)
        sorted_labels = sorted(label_counts.items(), 
                              key=lambda x: (-x[1], gradation_order.get(x[0], 999)))
        
        # Return top 2 labels
        return [label for label, count in sorted_labels[:2]]
    
    top_two = get_buffer_top_two(selected_buffer, buffer_type)
    
    if len(top_two) < 2:
        # Fallback: use most frequent label from selected buffer
        counter = Counter(selected_buffer)
        most_common = counter.most_common(1)[0][0]
        return create_result_dict(row, idx, most_common, f"No consensus, {buffer_type} buffer dominant, using most frequent: {most_common}", "", all_judge_results)
    
    # Final judge with top 2 classes
    final_judge_result = call_final_judge(
        judge=args.judge,
        method=args.method,
        backend=args.backends[backend_cycle % len(args.backends)],
        lang=args.lang,
        row=row,
        top_classes=top_two,
        judge_results=all_judge_results,
        backend_offset=backend_cycle
    )
    
    return create_result_dict(row, idx, final_judge_result['label'], final_judge_result['reasoning'], final_judge_result.get('writing_type', ''), all_judge_results + [final_judge_result])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback', 'tnfn', 'sentence', 'edit'])
    ap.add_argument('--method', required=True, choices=['baseline', 'advanced', 'modular', 'agent', 'tagger'])
    ap.add_argument('--backends', nargs='+', required=True, help='List of LLM backends')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--n_judges', type=int, default=2)
    ap.add_argument('--max_iters', type=int, default=3)
    ap.add_argument('--lite', action='store_true', help='Use cost-efficient lite mode: small-only initial votes, one mid probe, selective finalization')
    ap.add_argument('--input', required=True, help='Input CSV file')
    ap.add_argument('--output', required=True, help='Output CSV file')
    ap.add_argument('--filter', nargs='*', default=None, help='Ordered list of target columns to try; keep first TP/TN')
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--moderation', choices=['on', 'off'], default='off')
    ap.add_argument('--optimization', default='off', choices=['on','off'])
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    # Do not force in-process judge unless optimization is ON
    if args.optimization == 'on':
        os.environ['CURRENT_WORKERS'] = str(args.workers)
        os.environ['USE_IN_PROCESS_JUDGE'] = '1'
    
    # Process rows in parallel (ensemble-level parallelism)
    def process_func(row_data):
        return process_single_row(row_data, args)
    
    # Optimized path via shared runner if enabled
    if args.optimization == 'on':
        def process_fn(row):
            return process_func((row.name, row))
        results = run_optimized_process_rows(df, process_fn, desc='iter_critic/optimized', target_shards=None, workers_per_shard=max(400, args.workers))
    else:
        results = process_rows_parallel(df, process_func, args.workers, "Processing rows")
    
    # Save results
    output_df = pd.DataFrame(results)
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if output has a directory component
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    
    # Print distribution analysis
    from utils.judge import print_judge_distribution
    print_judge_distribution(results, f"Iterative Critic Ensemble ({args.judge}/{args.method})")


if __name__ == '__main__':
    main()
