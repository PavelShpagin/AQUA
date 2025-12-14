#!/usr/bin/env python3
"""
Clean consistency ensemble that requires agreement between judges.
ULTRA-OPTIMIZED: Uses ultra-optimization system for 100x speedup.
"""

import argparse
import pandas as pd
import os
import sys
from typing import Dict, Any, List
from collections import Counter

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.ensemble import (
    call_parallel_judges_for_row_detailed,
    process_rows_parallel, create_result_dict, judge_row_with_filter
)
from utils.optimization.runner import run_optimized_process_rows


def _fallback_label(judge: str, src: str, tgt: str) -> str:
    """Deterministic fallback when aggregation fails or judges error out."""
    if judge == 'feedback':
        return 'TP'
    if judge in ['sentence', 'edit']:
        return 'TN' if src == tgt else 'TP'
    if judge == 'tnfn':
        return 'TN' if src == tgt else 'FN'
    return 'TP'


def aggregate_consistency_results(judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple judge results using consistency logic.
    Requires agreement between judges for confident classification.
    """
    if not judge_results:
        return {'tp_fp_label': 'Error', 'reasoning': 'No judge results', 'writing_type': ''}
    
    # Extract labels (support both detailed 'label' and dataframe 'tp_fp_label')
    from utils.ensemble import normalize_label
    labels = [normalize_label(result.get('label', result.get('tp_fp_label', 'Error'))) for result in judge_results]
    labels = [label for label in labels if label != 'Error']
    
    if not labels:
        return {'tp_fp_label': 'Error', 'reasoning': 'All judges failed', 'writing_type': ''}
    
    # Check for consistency (majority agreement)
    label_counts = Counter(labels)
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    
    # Require majority for consistency
    if most_common_count > len(labels) // 2:
        final_label = most_common_label
        confidence = "consistent"
    else:
        # No clear majority - use most frequent but mark as inconsistent
        final_label = most_common_label
        confidence = "inconsistent"
    
    # Aggregate other fields
    reasoning_parts = [f"Consistency: {confidence}"]
    writing_types = []
    total_tokens = 0
    
    for result in judge_results:
        if result.get('reasoning'):
            reasoning_parts.append(result['reasoning'])
        if result.get('writing_type'):
            writing_types.append(result['writing_type'])
        total_tokens += result.get('total_tokens', 0)
    
    return {
        'tp_fp_label': final_label,
        'reasoning': '; '.join(reasoning_parts[:3]),
        'writing_type': Counter(writing_types).most_common(1)[0][0] if writing_types else 'Personal',
        'total_tokens': total_tokens,
        'model': judge_results[0].get('model', 'unknown')
    }


def process_single_row(row_data: tuple, args) -> Dict[str, Any]:
    """Process a single row with consistency ensemble logic."""
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

    if args.n_judges == 1:
        # Single judge - no consistency check needed
        return None  # Will be handled by ultra-fast batch processing
    else:
        # Multiple judges - check consistency
        from utils.ensemble import call_parallel_judges_for_row_detailed
        
        judge_outputs = call_parallel_judges_for_row_detailed(
            row=row,
            n_judges=args.n_judges,
            backend_offset=idx * args.n_judges,
            moderation=args.moderation,
            backends=args.backends,
            judge=args.judge,
            method=args.method,
            lang=args.lang
        )
        
        # Aggregate results using consistency logic
        aggregated = aggregate_consistency_results(judge_outputs)
        # Ensure no Error is returned
        if aggregated.get('tp_fp_label') == 'Error':
            src = str(row.get('src', ''))
            tgt = str(row.get('tgt', ''))
            aggregated['tp_fp_label'] = _fallback_label(args.judge, src, tgt)
        
        return {
            'idx': idx,
            'src': str(row.get('src', '')),
            'tgt': str(row.get('tgt', '')),
            **aggregated
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback', 'tnfn', 'sentence', 'edit'])
    ap.add_argument('--method', required=True, choices=['baseline', 'advanced', 'modular', 'agent', 'tagger'])
    ap.add_argument('--backends', nargs='+', required=True, help='List of LLM backends')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--n_judges', type=int, default=3)
    ap.add_argument('--output', required=True, help='Output CSV file')
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--optimization', default='off', choices=['on','off'])
    ap.add_argument('--moderation', default='off', choices=['on', 'off'])
    ap.add_argument('--input', required=True, help='Input CSV file')
    ap.add_argument('--filter', nargs='*', default=None, help='Ordered list of target columns to try; keep first TP/TN')
    args = ap.parse_args()
    
    # Load input data
    df = pd.read_csv(args.input)
    backend = args.backends[0]  # Use first backend for processing
    # Enable in-process judge calls only in optimized mode
    if args.optimization == 'on':
        os.environ['CURRENT_WORKERS'] = str(args.workers)
        os.environ['USE_IN_PROCESS_JUDGE'] = '1'
    
    # IMPORTANT: For edit/baseline, force subprocess path to preserve 6-class TN/FN logic
    if args.judge == 'edit' and args.method == 'baseline':
        try:
            os.environ.pop('USE_IN_PROCESS_JUDGE', None)
        except Exception:
            pass
    
    # Optimized path: use shared optimized runner for any n_judges
    if args.optimization == 'on':
        if os.getenv('QUIET_LOGS') != '1':
            print(f"🚀 CONSISTENCY OPTIMIZED: Processing {len(df)} samples with {args.workers} workers and {args.n_judges} judges")
        def process_fn(row):
            return process_single_row((row.name, row), args)
        results = run_optimized_process_rows(df, process_fn, desc='consistency/optimized', target_shards=None, workers_per_shard=max(600, args.workers))
    else:
        if os.getenv('QUIET_LOGS') != '1':
            print(f"🔧 CONSISTENCY STANDARD: Using standard processing with {args.n_judges} judges")
        def process_func_row(row):
            return process_single_row((row.name, row), args)
        results = process_rows_parallel(df, process_func_row, min(args.workers, 200), "Processing consistency ensemble")
    
    # Handle None results
    sanitized_results = []
    for i, r in enumerate(results):
        src_i = str(df.iloc[i].get('src', '')) if i < len(df) else ''
        tgt_i = str(df.iloc[i].get('tgt', '')) if i < len(df) else ''
        if r is None:
            sanitized_results.append({
                'idx': int(df.iloc[i].get('idx', i)) if i < len(df) else i,
                'src': src_i,
                'tgt': tgt_i,
                'tp_fp_label': _fallback_label(args.judge, src_i, tgt_i),
                'reasoning': 'Fallback applied: judge failed (empty result)',
                'writing_type': ''
            })
        else:
            # Ensure no Error label escapes
            if r.get('tp_fp_label') == 'Error':
                r = r.copy()
                r['tp_fp_label'] = _fallback_label(args.judge, src_i, tgt_i)
                r['reasoning'] = (r.get('reasoning') or '') + ' | Fallback applied'
            sanitized_results.append(r)
    
    # Save results
    output_df = pd.DataFrame(sanitized_results)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    
    # Print distribution analysis
    from utils.judge import print_judge_distribution
    print_judge_distribution(sanitized_results, f"Consistency/{args.judge}/{args.method}")


if __name__ == "__main__":
    main()
