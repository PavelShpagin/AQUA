#!/usr/bin/env python3
"""
Legacy sentence judge: single-call classifier returning JSON with a label.

Minimal implementation: uses the final judgment sentence prompt to classify
each (src, tgt) into one of TP/FP1/FP2/FP3/TN/FN and writes predictions.
"""

import os
import argparse
import re
import pandas as pd
from judges.sentence.prompts import SYSTEM_M
from utils.judge import (
    print_judge_distribution,
    process_rows_with_progress,
    call_model_with_pricing,
    add_pricing_to_result_dict,
    get_language_label,
    detect_language_from_text,
)


def main():
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input CSV file (if not provided, uses gold dataset based on --lang)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--llm_backend', required=True)
    # If --lang is omitted, we'll still auto-detect per-row for prompt injection
    parser.add_argument('--lang', default='en', help='Language for gold dataset selection (prompts auto-detect if omitted)')
    # Conservative default for baseline stability
    default_workers = 50
    try:
        if os.getenv('OPTIMIZATION', 'off') == 'on':
            default_workers = 200
    except Exception:
        pass
    parser.add_argument('--workers', type=int, default=default_workers, help='Number of workers for baseline threading')
    parser.add_argument('--moderation', default='off', choices=['on', 'off'], help='Moderation (ignored for legacy)')
    parser.add_argument('--judge', default='sentence', help='Judge type (ignored)')
    parser.add_argument('--method', default='legacy', help='Method (ignored)')
    parser.add_argument('--debug', default='off', choices=['on', 'off'])
    parser.add_argument('--optimization', default='off', choices=['on','off'], help='Enable 2x speed/2x cost optimization')
    args = parser.parse_args()

    # Determine input file: use provided --input or select gold dataset based on --lang
    if args.input:
        input_file = args.input
    else:
        # Auto-select gold dataset based on language
        gold_datasets = {
            'en': 'data/eval/gold_en.csv',
            'de': 'data/eval/gold_de.csv',
            'ua': 'data/eval/gold_ua.csv'
        }
        input_file = gold_datasets.get(args.lang, 'data/eval/gold_en.csv')
        print(f"Auto-selected gold dataset: {input_file}")

    # Debug mode: run small sample with actual SYSTEM_M prompt
    if args.debug == 'on':
        from utils.debug import run_debug_test, save_debug_log

        def debug_wrapper(src, tgt, llm_backend, lang, **kwargs):
            # Map code/label to human-readable name for $provided_language
            provided_language = get_language_label(lang)

            # Full diff output - no truncation
            diff_output = f"[(0, '{src}'), (1, '{tgt}')]"

            # Format the SYSTEM_M prompt
            import string
            system_template = string.Template(SYSTEM_M)
            system_prompt = system_template.substitute(provided_language=provided_language)
            full_prompt = (
                system_prompt
                + f"\n*   **Original Text:** `{src}`"
                + f"\n*   **Suggested Text:** `{tgt}`"
                + f"\n*   **DiffMatchPatch output:** `{diff_output}`"
            )
            
            # Debug: Print to verify we're using the right prompt
            print(f"DEBUG: Using SYSTEM_M prompt, length={len(full_prompt)} chars")

            ok, content, _tokens, _pricing = call_model_with_pricing(
                full_prompt, llm_backend, api_token=os.getenv('API_TOKEN', ''), moderation=False
            )

            # Parse label using the same regex approach as below
            label = 'Error'
            reasoning = ''
            if ok and content:
                m = re.search(r'"classifications"\s*:\s*\[([^\]]*)\]', content, re.IGNORECASE)
                if m:
                    classifications = m.group(1)
                    up = classifications.upper()
                    if 'TP' in up:
                        label = 'TP'
                    elif 'TN' in up:
                        label = 'TN'
                    elif 'FN' in up:
                        label = 'FN'
                    elif 'FP' in up:
                        label = 'FP'
                if label == 'FP':
                    s = re.search(r'"fp_severity"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
                    if s:
                        severity = s.group(1).upper()
                        if 'FP1' in severity or 'CRITICAL' in severity:
                            label = 'FP1'
                        elif 'FP2' in severity or 'MEDIUM' in severity:
                            label = 'FP2'
                        elif 'FP3' in severity or 'MINOR' in severity:
                            label = 'FP3'
                r = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
                if r:
                    reasoning = r.group(1)

            return {
                'prompt': full_prompt,
                'output': content if ok else 'Error',
                'reasoning': reasoning or (content if ok else 'Error'),
                'label': label
            }

        logs = run_debug_test(debug_wrapper, 'sentence', 'legacy', [args.llm_backend], 'all')
        save_debug_log(logs, 'sentence', 'legacy')
        return

    df = pd.read_csv(input_file)

    def process_row(row):
        src = str(row.get('src', ''))
        tgt = str(row.get('tgt', ''))

        # Build comprehensive SYSTEM_M prompt with proper formatting
        # Choose human-readable language label: use forced --lang or auto-detect from src
        provided_language = get_language_label(args.lang) if args.lang else detect_language_from_text(src)

        # Full diff output - no truncation
        diff_output = f"[(0, '{src}'), (1, '{tgt}')]"

        # Format the prompt with the template variables
        import string
        system_template = string.Template(SYSTEM_M)
        system_prompt = system_template.substitute(provided_language=provided_language)

        # Append required input bullets exactly as the prompt describes
        full_prompt = (
            system_prompt
            + f"\n*   **Original Text:** `{src}`"
            + f"\n*   **Suggested Text:** `{tgt}`"
            + f"\n*   **DiffMatchPatch output:** `{diff_output}`"
        )
        ok, content, _tokens, pricing_info = call_model_with_pricing(
            full_prompt, args.llm_backend, api_token=os.getenv('API_TOKEN', ''), moderation=False
        )

        label = 'Error'
        reasoning = ''
        fp_severity = ''
        if ok and content:
            # Extract classifications array (SYSTEM_M returns an array)
            m = re.search(r'"classifications"\s*:\s*\[([^\]]*)\]', content, re.IGNORECASE)
            if m:
                # Extract the first classification from the array
                classifications = m.group(1)
                # Look for TP, TN, FP, or FN
                if 'TP' in classifications.upper():
                    label = 'TP'
                elif 'TN' in classifications.upper():
                    label = 'TN'
                elif 'FN' in classifications.upper():
                    label = 'FN'
                elif 'FP' in classifications.upper():
                    label = 'FP'  # Will be refined by severity
            
            # Extract FP severity if it's an FP
            if label == 'FP':
                s = re.search(r'"fp_severity"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
                if s:
                    severity = s.group(1).upper()
                    if 'FP1' in severity or 'CRITICAL' in severity:
                        label = 'FP1'
                    elif 'FP2' in severity or 'MEDIUM' in severity:
                        label = 'FP2'
                    elif 'FP3' in severity or 'MINOR' in severity:
                        label = 'FP3'
            
            # Try to extract explanation (reasoning)
            r = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
            if r:
                reasoning = r.group(1)

        result = {
            'idx': row.get('idx', ''),
            'src': src,
            'tgt': tgt,
            'tp_fp_label': label,
            'reasoning': reasoning if reasoning else (content if ok else 'Error'),
        }
        
        # Add pricing information to result
        if pricing_info:
            result = add_pricing_to_result_dict(result, pricing_info)
        
        return result

    rows = process_rows_with_progress(
        df, process_row, desc=f"Sentence Legacy ({args.llm_backend})",
        workers=args.workers, optimization=(args.optimization=='on')
    )
    out = pd.DataFrame(rows)

    # Print distribution analysis
    print_judge_distribution(rows, f"Sentence Legacy ({args.llm_backend})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()



