#!/usr/bin/env python3
"""
Inner Debate ensemble that creates debates between the two most dominant classes.

Algorithm:
1. Find the 2 most dominating labels using priority rules (FP1>FP2>FP3>FN>TP/TN)
2. Take the last min(m, k) judges from each dominant class
3. Create alternating debate reasoning for final judge
4. Final judge makes decision after seeing the debate

PARALLELIZED: Uses shared ensemble utilities for optimal performance.
"""

import argparse
import pandas as pd
import os
import sys
from typing import Dict, Any, List, Tuple, Optional

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

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ensemble import (
    call_parallel_judges_for_row_detailed,
    call_single_judge_for_row_detailed,
    process_rows_parallel, create_result_dict
)
from utils.llm.backends import call_model
from utils.llm.moderation import check_openai_moderation
from utils.optimization.runner import run_optimized_process_rows
from utils.judge import (
    parse_tpfp_label, parse_writing_type,
    call_model_with_pricing, add_pricing_to_result_dict
)
try:
    from utils.llm.settings import get_final_judge_temperature
except Exception:
    def get_final_judge_temperature(default: float = 0.0) -> float:
        return default
from ensembles.prompts import (
    FEEDBACK_DEBATE_PROMPT, SENTENCE_DEBATE_PROMPT, EDIT_DEBATE_PROMPT
)


def get_dominance_priority(label: str, src: str = "", tgt: str = "") -> int:
    """Get dominance priority for a label (lower number = higher dominance)."""
    priority_map = {
        'FP1': 1,  # Highest dominance
        'FP2': 2,
        'FP3': 3,
        'FN': 4,
    }
    
    if label in priority_map:
        return priority_map[label]
    elif label in ['TN', 'TP']:
        # TN/TP tie-breaking: TN dominates if src==tgt, else TP dominates
        if src == tgt:
            return 5 if label == 'TN' else 6  # TN wins if src==tgt
        else:
            return 5 if label == 'TP' else 6  # TP wins if src!=tgt
    else:
        return 999  # Unknown labels have lowest dominance


def get_two_most_dominant_classes(judge_outputs: List[Dict[str, Any]], 
                                  src: str = "", tgt: str = "") -> Tuple[List[str], List[str]]:
    """
    Get the two most dominant classes and their corresponding labels.
    Returns: (class1_labels, class2_labels) where class1 is more dominant.
    """
    if not judge_outputs:
        return [], []
    
    # Group labels by class
    class_groups = {}
    for output in judge_outputs:
        label = output.get('label', 'Error')
        if label != 'Error':
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(output)
    
    if not class_groups:
        return [], []
    
    # Sort classes by dominance (lower priority number = higher dominance)
    sorted_classes = sorted(class_groups.items(), 
                          key=lambda x: get_dominance_priority(x[0], src, tgt))
    
    # Take top 2 classes
    if len(sorted_classes) >= 2:
        class1_outputs = sorted_classes[0][1]
        class2_outputs = sorted_classes[1][1]
        return class1_outputs, class2_outputs
    elif len(sorted_classes) == 1:
        # If only one class, return it and empty list
        return sorted_classes[0][1], []
    else:
        return [], []


def create_debate_opinions(class1_outputs: List[Dict[str, Any]], 
                          class2_outputs: List[Dict[str, Any]],
                          class1_label: str,
                          class2_label: str) -> str:
    """Create alternating debate opinions for the final judge."""
    if not class1_outputs and not class2_outputs:
        return ""
    
    if not class2_outputs:
        # Only one class, no debate
        reasoning = class1_outputs[-1].get('reasoning', 'No reasoning provided')
        return f"{class1_label} Argument:\n{reasoning}"
    
    # Take last min(m, k) judges from each class
    min_count = min(len(class1_outputs), len(class2_outputs))
    selected_class1 = class1_outputs[-min_count:] if class1_outputs else []
    selected_class2 = class2_outputs[-min_count:] if class2_outputs else []
    
    # Create alternating debate
    debate_parts = []
    
    for i in range(min_count):
        # Class 1 argument
        if i < len(selected_class1):
            reasoning1 = selected_class1[i].get('reasoning', 'No reasoning provided')
            debate_parts.append(f"{class1_label} Argument:\n{reasoning1}")
        
        # Class 2 argument
        if i < len(selected_class2):
            reasoning2 = selected_class2[i].get('reasoning', 'No reasoning provided')
            debate_parts.append(f"{class2_label} Argument:\n{reasoning2}")
    
    return "\n\n".join(debate_parts)


def create_final_judge_opinions(debate_text: str, class1_label: str, class2_label: str) -> str:
    """Format debate text into opinions format for the final judge."""
    if not debate_text:
        return ""
    
    header = f"## DEBATE BETWEEN {class1_label} AND {class2_label}\n"
    return header + debate_text


def call_debate_judge_for_row(judge: str, method: str, backend: str, lang: str,
                              row: pd.Series, moderation: str, debate_text: str,
                              class1_label: str, class2_label: str) -> Dict[str, Any]:
    """Call a single judge with custom debate prompt."""
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    # Moderation check
    if moderation == 'on':
        moderation_input = f"{src}\n{tgt}"
        if check_openai_moderation(moderation_input):
            return {
                'label': 'Error',
                'reasoning': 'Content flagged by moderation',
                'writing_type': ''
            }
    
    # Select appropriate debate prompt based on judge type
    if judge == 'feedback':
        prompt = FEEDBACK_DEBATE_PROMPT.format(debate_text, src, tgt)
    elif judge == 'sentence':
        prompt = SENTENCE_DEBATE_PROMPT.format(debate_text, src, tgt)
    elif judge == 'edit':
        prompt = EDIT_DEBATE_PROMPT.format(debate_text, src, tgt)
    else:
        # Fallback to sentence prompt
        prompt = SENTENCE_DEBATE_PROMPT.format(debate_text, src, tgt)
    
    try:
        ok, content, total_tokens, pricing_info = call_model_with_pricing(
            prompt,
            backend,
            api_token=os.getenv('API_TOKEN', ''),
            moderation=False,
            temperature_override=get_final_judge_temperature(0.0),
        )
        
        if not ok:
            return {
                'label': 'Error',
                'reasoning': 'LLM call failed',
                'writing_type': ''
            }
        
        # Parse label from JSON response
        import re
        up = content.upper()
        if judge == 'feedback':
            m = re.search(r'"CLASSIFICATION"\s*:\s*"(TP|FP1|FP2|FP3)"', up)
        else:
            m = re.search(r'"CLASSIFICATION"\s*:\s*"(TP|FP1|FP2|FP3|TN|FN)"', up)
        
        label = m.group(1) if m else parse_tpfp_label(content)
        writing_type = parse_writing_type(content)
        
        result = {
            'label': label,
            'reasoning': content.strip(),
            'writing_type': writing_type
        }
        
        # Add pricing info
        return add_pricing_to_result_dict(result, pricing_info)
        
    except Exception as e:
        print(f"ERROR in call_debate_judge_for_row: {e}", file=sys.stderr)
        return {
            'label': 'Error',
            'reasoning': str(e),
            'writing_type': ''
        }


def process_single_row(row_data: tuple, args) -> Dict[str, Any]:
    """Process a single row with inner debate ensemble logic."""
    idx, row = row_data
    # Extract core fields early for fallbacks
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    # Call all judges for this row in parallel (detailed)
    judge_outputs = call_parallel_judges_for_row_detailed(
        judge=args.judge,
        method=args.method,
        backends=args.backends,
        lang=args.lang,
        row=row,
        n_judges=args.n_judges,
        backend_offset=idx * args.n_judges,  # Stagger across rows
        moderation=args.moderation
    )
    
    if not judge_outputs:
        # fallback heuristic
        fallback = 'TN' if args.judge in ['sentence','edit'] and src == tgt else 'TP'
        return create_result_dict(row, idx, fallback, 'Fallback: no judge outputs', '', [])
    
    # Check if we have enough diverse opinions for debate
    from utils.ensemble import normalize_label
    labels = [normalize_label(o.get('label', o.get('tp_fp_label', 'Error'))) for o in judge_outputs]
    labels = [l for l in labels if l != 'Error']
    unique_labels = list(set(labels))
    
    if len(unique_labels) == 0:
        # All judges failed
        # Fallback on heuristic
        fallback = 'TN' if args.judge in ['sentence','edit'] and src == tgt else 'TP'
        return create_result_dict(row, idx, fallback, 'Fallback: all judges failed', '', judge_outputs)
    elif len(unique_labels) == 1:
        # Perfect consensus - no debate needed, just return the unanimous label
        unanimous_label = unique_labels[0]
        last = judge_outputs[-1] if judge_outputs else {'reasoning': '', 'writing_type': ''}
        return create_result_dict(row, idx, unanimous_label, 
                                f"Unanimous decision ({unanimous_label}): {last.get('reasoning', '')}", 
                                last.get('writing_type', ''), judge_outputs)
    # If len(unique_labels) < 2 but not 0 or 1, continue to debate logic
    
    # Get two most dominant classes
    class1_outputs, class2_outputs = get_two_most_dominant_classes(judge_outputs, src, tgt)
    
    if not class1_outputs:
        fallback = 'TN' if args.judge in ['sentence','edit'] and src == tgt else 'TP'
        return create_result_dict(row, idx, fallback, 'Fallback: no valid classes found', '', judge_outputs)
    
    class1_label = normalize_label(class1_outputs[0].get('label', 'Error'))
    class2_label = normalize_label(class2_outputs[0].get('label', 'Unknown')) if class2_outputs else 'None'
    
    # Create debate opinions
    debate_text = create_debate_opinions(class1_outputs, class2_outputs, class1_label, class2_label)
    opinions = create_final_judge_opinions(debate_text, class1_label, class2_label)
    
    # Call final judge with custom debate prompt
    # Use the most reliable backend (first one)
    final_backend = args.backends[0] if args.backends else 'gpt-4o-mini'
    
    try:
        final_result = call_debate_judge_for_row(
            judge=args.judge,
            method=args.method,
            backend=final_backend,
            lang=args.lang,
            row=row,
            moderation=args.moderation,
            debate_text=debate_text,
            class1_label=class1_label,
            class2_label=class2_label
        )
        
        final_label = final_result.get('label', 'Error')
        final_reasoning = final_result.get('reasoning', 'Final judge failed')
        final_writing_type = final_result.get('writing_type', '')
        
        # Add debate context to reasoning
        context_reasoning = f"DEBATE DECISION:\n{final_reasoning}\n\n--- DEBATE CONTEXT ---\n{debate_text}"
        
        if final_label == 'Error':
            fallback = 'TN' if args.judge in ['sentence','edit'] and src == tgt else 'TP'
            final_label = fallback
            context_reasoning += ' | Fallback applied'
        return create_result_dict(row, idx, final_label, context_reasoning, 
                                final_writing_type, judge_outputs + [final_result])
        
    except Exception as e:
        print(f"ERROR in final judge for row {idx}: {e}", file=sys.stderr)
        # Fallback to most dominant class
        final_label = class1_label
        final_reasoning = f"Final judge failed ({e}), using most dominant class: {class1_label}"
        return create_result_dict(row, idx, final_label, final_reasoning, '', judge_outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, choices=['feedback', 'tnfn', 'sentence', 'edit'])
    ap.add_argument('--method', required=True, choices=['baseline', 'advanced', 'modular', 'agent', 'tagger'])
    ap.add_argument('--backends', nargs='+', required=True, help='List of LLM backends')
    ap.add_argument('--lang', required=True)
    ap.add_argument('--n_judges', type=int, default=3, help='Number of initial judges (min 2 for debate)')
    ap.add_argument('--input', required=True, help='Input CSV file')
    ap.add_argument('--output', required=True, help='Output CSV file')
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--moderation', default='off', choices=['on', 'off'])
    ap.add_argument('--optimization', default='off', choices=['on','off'])
    args = ap.parse_args()
    
    # Validate n_judges
    if args.n_judges < 2:
        print("WARNING: inner_debate ensemble requires at least 2 judges for meaningful debate", file=sys.stderr)
        args.n_judges = 2

    # Load input data
    df = pd.read_csv(args.input)
    
    # Create process function with args bound
    def process_func(row_data):
        return process_single_row(row_data, args)
    
    # Determine safe concurrency (no auto-optim unless optimization is ON)
    if args.optimization == 'on':
        os.environ['USE_IN_PROCESS_JUDGE'] = '1'
        os.environ['PROCESSING_MODE'] = 'bulk'
        os.environ['ULTRA_FAST_MODE'] = '1'
        os.environ['CURRENT_WORKERS'] = str(args.workers)
    
    # Optimized path via shared runner if env OPTIMIZATION is on (propagated by shell/run_judge)
    if args.optimization == 'on':
        def process_fn(row):
            return process_func((row.name, row))
        results = run_optimized_process_rows(df, process_fn, desc='inner_debate/optimized', target_shards=None, workers_per_shard=max(400, args.workers))
    else:
        safe_workers = args.workers
        results = process_rows_parallel(df, process_func, safe_workers, "Processing inner debate")
    
    # Replace any None results with structured Error rows
    sanitized_results = []
    for i, r in enumerate(results):
        if r is None:
            sanitized_results.append({
                'idx': int(df.iloc[i].get('idx', i)),
                'src': str(df.iloc[i].get('src', '')),
                'tgt': str(df.iloc[i].get('tgt', '')),
                'tp_fp_label': 'Error',
                'reasoning': 'Inner debate ensemble failed (empty result)'
            })
        else:
            sanitized_results.append(r)
    
    # Save results
    output_df = pd.DataFrame(sanitized_results)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    
    # Print distribution analysis
    from utils.judge import print_judge_distribution
    print_judge_distribution(results, f"Inner Debate Ensemble ({args.judge}/{args.method})")


if __name__ == '__main__':
    main()
