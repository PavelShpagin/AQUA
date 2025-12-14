#!/usr/bin/env python3
"""
Weighted ensemble for GEC judge predictions.

This ensemble contains both aggregation and processing logic (merged).
"""

import argparse
import os
import time
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.ensemble import call_parallel_judges_for_row_detailed, judge_row_with_filter, print_filter_annotation_stats
from utils.progress import SingleLineProgress


def create_result_dict(row, idx: int, label: str, reasoning: str, writing_type: str, judge_outputs: List[Dict]) -> Dict[str, Any]:
    """Create result dictionary - preserve input row idx for gold join parity."""
    row_idx = row.get('idx', idx)
    try:
        row_idx = int(row_idx)
    except Exception:
        pass
    return {
        'idx': row_idx,
        'src': str(row.get('src', '')),
        'tgt': str(row.get('tgt', '')),
        'aligned_sentence': str(row.get('aligned_sentence', row.get('aligned', row.get('alert', '')))),
        'tp_fp_label': label,
        'reasoning': reasoning,
        'writing_type': writing_type,
        'judge_outputs': judge_outputs,
        'total_tokens': sum(j.get('total_tokens', 0) for j in judge_outputs),
        'model': judge_outputs[0].get('model', '') if judge_outputs else ''
    }


def aggregate_weighted(labels: List[str], weights: List[float]) -> str:
    """Aggregate labels using average-weight logic suggested by user.

    Idea:
    - Split predictions into TP/FP* vs TN/FN buckets and pick the bucket with more votes.
    - Map each class to a numeric weight and compute the average.
    - Return the first class whose threshold is <= average (preserves the priority order).
    
    Notes:
    - Supports optional 'TP-S'/'TP-W' inputs; maps plain 'TP' to strong TP.
    - Ignores unknown/error labels.
    - The 'weights' argument is unused (kept for API compatibility/minimal changes).
    """
    if not labels:
        return 'Error'

    # Normalize labels
    norm = []
    for lab in labels:
        up = str(lab).strip().upper()
        if up == 'TP':
            up = 'TP-S'  # treat plain TP as strong TP
        norm.append(up)

    # Buckets
    tpfp_classes = [x for x in norm if x in ['TP-S', 'TP-W', 'FP1', 'FP2', 'FP3']]
    tnfn_classes = [x for x in norm if x in ['TN', 'FN']]

    # Weights
    tpfp_weights = {'TP-S': 5, 'TP-W': 3, 'FP3': 1, 'FP2': -1, 'FP1': -3}
    tnfn_weights = {'TN': 1, 'FN': -1}

    # Choose dominant bucket (ties prefer TP/FP* to keep original behavior)
    if len(tpfp_classes) >= len(tnfn_classes):
        selected = tpfp_classes
        mapping = tpfp_weights
    else:
        selected = tnfn_classes
        mapping = tnfn_weights

    if not selected:
        return 'Error'

    total = len(selected)
    summed = sum(mapping.get(cls, 0) for cls in selected)
    avg = round(summed / max(1, total), 2)

    # Strict threshold logic in declared order
    for cls, threshold in mapping.items():
        if avg >= threshold:
            # Collapse TP-S/TP-W back to TP for downstream consumers
            return 'TP' if cls in {'TP-S', 'TP-W'} else cls
    return 'Error'


def process_single_row_fast(row_data, args) -> Dict[str, Any]:
    """Process a single row using proper judge calls with pricing."""
    idx, row = row_data
    
    try:
        # Use proper judge calls - no embedded prompts
        judge_outputs = call_parallel_judges_for_row_detailed(
            judge=args.judge,
            method=args.method,
            backends=args.backends,
            lang=args.lang,
            row=row,
            n_judges=args.n_judges,
            backend_offset=0,
            moderation=args.moderation
        )
        
        if not judge_outputs:
            return create_result_dict(row, idx, 'Error', 'No judge outputs', '', [])
        
        # Aggregate results
        labels = [output.get('label', 'Error') for output in judge_outputs]
        weights = [1.0] * len(labels)  # Equal weights
        
        final_label = aggregate_weighted(labels, weights)
        
        # Get reasoning from first successful judge
        reasoning = next((output.get('reasoning', '') for output in judge_outputs 
                         if output.get('reasoning')), '')
        writing_type = next((output.get('writing_type', '') for output in judge_outputs 
                           if output.get('writing_type')), '')
        
        return create_result_dict(row, idx, final_label, reasoning, writing_type, judge_outputs)
        
    except Exception as e:
        error_msg = f"Processing error for sample {idx}: {str(e)}"
        with open('fails.txt', 'a') as f:
            f.write(f"{error_msg}\n")
        return create_result_dict(row, idx, 'Error', error_msg, '', [])


def process_rows_with_dynamic_timing(df: pd.DataFrame, process_func, workers: int, description: str) -> List[Dict[str, Any]]:
    """Process rows with dynamic timing and DETERMINISTIC ordering."""
    print(f"\n{description}")
    print(f"Using {workers} workers")
    
    start_time = time.time()
    total = len(df)
    
    # Pre-allocate results array to guarantee ordering
    results: List[Dict[str, Any]] = [None] * total  # type: ignore
    
    prog = SingleLineProgress(total, desc=description, update_every=200)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Create futures with explicit index mapping for deterministic ordering
        futures = {executor.submit(process_func, (idx, row)): idx for idx, row in df.iterrows()}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            original_index = futures[future]
            results[original_index] = result
            completed += 1
            # Single-line progress update
            prog.update(completed)
    
    # Finish progress line
    prog.finish()

    # Fill any None results (should not happen)
    for i, result in enumerate(results):
        if result is None:
            results[i] = {'idx': i, 'tp_fp_label': 'Error', 'reasoning': 'Processing failed', 'success': False}
    
    return results


# Processing strategies (merged from utils/ensemble_processor)

def process_with_speed_optimization(df: pd.DataFrame, args) -> List[Dict[str, Any]]:
    """High-concurrency processing using detailed judge calls."""
    workers = min(200, len(df))
    print(f"SPEED OPTIMIZATION: Using {workers} workers")

    def process_func(row_data):
        return process_single_row(row_data, args)

    return process_rows_with_dynamic_timing(df, process_func, workers, "Processing samples")


def process_with_regular_optimization(df: pd.DataFrame, args) -> List[Dict[str, Any]]:
    """Moderate concurrency processing."""
    workers = min(100, getattr(args, 'workers', 50) * 2, len(df))
    print(f"REGULAR PROCESSING: Using {workers} workers")

    def process_func(row_data):
        return process_single_row(row_data, args)

    return process_rows_with_dynamic_timing(df, process_func, workers, "Processing samples")


def process_with_real_batch_api(df: pd.DataFrame, args) -> List[Dict[str, Any]]:
    """Execute real Batch API flow and return parsed results."""
    print("REAL BATCH API: submitting batch job and waiting for results...")
    from utils.batch_optimization import process_with_clean_batch_api
    return process_with_clean_batch_api(
        df=df,
        judge=args.judge,
        method=args.method,
        backends=args.backends,
        lang=args.lang,
        n_judges=args.n_judges,
    ) or []

def analyze_and_save_results(results: List[Dict], output_path: str, args):
    """Save results and print analysis."""
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    # Enhanced analysis with distributions
    total = len(results)
    print(f"\nWeighted Ensemble ({args.judge}/{args.method}) Analysis")
    print("=" * 60)
    print(f"Total samples: {total}")
    
    # Prediction distribution
    pred_dist = df_results['tp_fp_label'].value_counts()
    print(f"\nPrediction Distribution:")
    for label, count in pred_dist.items():
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Original vs Corrected Distribution
    if 'src' in df_results.columns and 'tgt' in df_results.columns:
        no_change_count = sum(1 for _, row in df_results.iterrows() 
                             if str(row.get('src', '')).strip() == str(row.get('tgt', '')).strip())
        change_count = total - no_change_count
        
        print(f"\nOriginal vs Corrected Distribution:")
        print(f"  No changes: {no_change_count} ({no_change_count/total*100:.1f}%)")
        print(f"  Has changes: {change_count} ({change_count/total*100:.1f}%)")
    
    # Filter annotation statistics (only when filter is used)
    print_filter_annotation_stats(df_results, getattr(args, 'filter', None))
    
    # Clean pricing calculation using pricing table - separated concern
    from utils.pricing_reporter import calculate_and_report_pricing
    backend = args.backends[0] if args.backends else 'unknown'
    use_batch_pricing = hasattr(args, 'batch') and args.batch == 'on'
    pricing_report, pricing_summary = calculate_and_report_pricing(results, backend, use_batch_pricing)


# Clean implementation - no embedded prompts, no broken batch processing


def process_single_row(row_data: tuple, args) -> Dict[str, Any]:
    """Process single row with weighted ensemble logic."""
    idx, row = row_data

    # Optional multi-target filtering handled in utils.ensemble for modularity
    filter_cols = getattr(args, 'filter', None)
    if filter_cols:
        annotated = judge_row_with_filter(
            idx,
            row,
            judge=args.judge,
            method=args.method,
            backends=args.backends,
            lang=args.lang,
            n_judges=args.n_judges,
            moderation=args.moderation,
            filter_cols=filter_cols,
            annotate=True,
        )
        if annotated is not None:
            return annotated

    # Use existing ensemble function for judge calls
    judge_outputs = call_parallel_judges_for_row_detailed(
        judge=args.judge,
        method=args.method,
        backends=args.backends,
        lang=args.lang,
        row=row,
        n_judges=args.n_judges,
        backend_offset=0,
        moderation=args.moderation,
        opinions=None,
        optimization=(args.optimization == 'on')
    )
    
    if not judge_outputs:
        return create_result_dict(row, idx, 'Error', 'No judge outputs', '', [])
    
    # Simple weighted aggregation
    row_labels = [output.get('label', 'Error') for output in judge_outputs]
    weights = [1.0] * len(judge_outputs)  # Equal weights for simplicity
    final_label = aggregate_weighted(row_labels, weights)
    
    # Use last judge's reasoning
    last = judge_outputs[-1] if judge_outputs else {'reasoning': '', 'writing_type': ''}
    return create_result_dict(row, idx, final_label, last.get('reasoning', ''), 
                              last.get('writing_type', ''), judge_outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback', 'tnfn', 'sentence', 'edit'])
    ap.add_argument('--method', required=True, choices=['baseline', 'advanced', 'modular', 'agent', 'agent_v1', 'agent_v2', 'ultra_sota', 'ultimate', 'tagger'])
    ap.add_argument('--backends', nargs='+', required=True)
    ap.add_argument('--lang', required=True)
    ap.add_argument('--n_judges', type=int, default=1)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--moderation', default='off', choices=['on', 'off'])
    ap.add_argument('--optimization', default='off', choices=['off', 'on'])
    ap.add_argument('--batch', default='off', choices=['off', 'on'])
    ap.add_argument('--filter', nargs='*', default=None)
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--tgt', default='tgt', help='Target column name (default: tgt)')
    args = ap.parse_args()

    print("WEIGHTED ENSEMBLE")
    print("=" * 60)
    print(f"Judge: {args.judge}/{args.method}")
    print(f"Backends: {args.backends}")
    print(f"Language: {args.lang}")
    print(f"Optimization: {args.optimization}")
    
    # Red Sparta compatibility check
    if os.getenv('SPARTA_ENV') or os.getenv('LLM_PROXY_PROD_HOST'):
        print("Red Sparta environment detected - using LLM Proxy routing")
    else:
        print("Local environment detected - using direct API routing")

    # Load data
    df = pd.read_csv(args.input)
    
    # Handle custom target column name
    if args.tgt != 'tgt' and args.tgt in df.columns:
        if 'tgt' in df.columns:
            print(f"Warning: Both 'tgt' and '{args.tgt}' columns exist. Using '{args.tgt}' and renaming 'tgt' to 'tgt_original'")
            df = df.rename(columns={'tgt': 'tgt_original'})
        df = df.rename(columns={args.tgt: 'tgt'})
        print(f"Using '{args.tgt}' column as target (renamed to 'tgt')")
    elif args.tgt != 'tgt':
        print(f"Warning: Specified target column '{args.tgt}' not found. Using default 'tgt' column")
    
    if args.samples and args.samples > 0 and args.samples < len(df):
        df = df.head(args.samples)
    print(f"Processing {len(df)} samples")
    
    # Start timing (monotonic for accurate elapsed time)
    start_time = time.perf_counter()
    
    # Choose processing strategy
    # Processing with different strategies - clean separation of concerns
    use_batch = hasattr(args, 'batch') and args.batch == 'on'
    use_optimization = hasattr(args, 'optimization') and args.optimization == 'on'
    
    if use_batch:
        # REAL OpenAI Batch API processing - 50% cost savings
        results = process_with_real_batch_api(df, args)
        # If batch is async and results are placeholders, skip metrics/logging
        try:
            pending_count = sum(1 for r in results if str(r.get('tp_fp_label','')) == 'BATCH_PENDING' or str(r.get('batch_status','')).lower() in {'submitted','pending'})
        except Exception:
            pending_count = 0
        if results and pending_count >= int(0.9 * len(results)):
            print("Batch submitted (async). Results pending. Skipping metrics and analysis until results are available.")
            # Save a minimal pending file for tracking
            try:
                import pandas as _pd
                _pd.DataFrame(results).to_csv(args.output, index=False)
            except Exception:
                pass
            return
        
    elif use_optimization:
        # Speed optimization with high concurrency
        results = process_with_speed_optimization(df, args)
        
    else:
        # Regular processing
        results = process_with_regular_optimization(df, args)
    
    # Clean up None results  
    clean_results = []
    for i, result in enumerate(results):
        if result is None:
            row = df.iloc[i]
            result = create_result_dict(row, i, 'Error', 'Processing failed', '', [])
        clean_results.append(result)
    
    # Calculate performance metrics
    total_time = time.perf_counter() - start_time
    samples_per_sec = len(df) / total_time if total_time > 0 else 0
    
    print(f"\nPERFORMANCE METRICS")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Processed: {len(clean_results)} samples")
    
    # Performance comparison with targets
    if len(df) >= 1000:
        # Extrapolate to 4K samples
        time_for_4k = total_time * (4000 / len(df))
        print(f"Estimated time for 4K samples: {time_for_4k:.1f}s ({time_for_4k/60:.1f} min)")
        
        if time_for_4k <= 180:  # 3 minutes
            print("Meets original 2-3 minute target for 4K samples")
        elif time_for_4k <= 120:  # 2 minutes  
            print("Exceeds original target - under 2 minutes!")
        else:
            print("Does not meet 2-3 minute target")
    
    # Save and analyze
    analyze_and_save_results(clean_results, args.output, args)
    print(f"\nResults saved to {os.path.dirname(args.output)}")


if __name__ == "__main__":
    main()