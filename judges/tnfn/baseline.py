#!/usr/bin/env python3
"""
TN/FN judge (baseline) with real LLM calls using TNFN_PROMPT.
Compatible with the consistency ensemble and `shell/run_judge.sh`.
"""

import os
import sys
import argparse
import pandas as pd

from judges.tnfn.prompts import TNFN_PROMPT
from utils.llm.backends import call_model
from utils.llm.moderation import check_openai_moderation
from utils.judge import (
    detect_language_from_text,
    get_language_label,
    build_numbered_prompt,
    parse_tnfn_label,
    process_rows_with_progress,
    call_model_with_pricing,
    add_pricing_to_result_dict,
    print_judge_distribution,
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
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--llm_backend', required=True)
    # Accept --lang to force language for prompts (if not provided, auto-detect with langid)
    parser.add_argument('--lang', default='', help='Force language for prompts (e.g., es, en, de, ua). If empty, auto-detect with langid.')
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--moderation', default='off', choices=['on', 'off'])
    parser.add_argument('--opinions', type=str, default="", help='Previous judge opinions for iter_critic')
    parser.add_argument('--final_judge', action='store_true', help='Use final judgment prompt for iter_critic')
    args = parser.parse_args()


    df = pd.read_csv(args.input)

    def process_row(row):
        src = str(row.get('src', ''))
        
        # Use forced language from --lang parameter, or auto-detect from source text
        if args.lang:
            # Force language from config/command line
            language_label = get_language_label(args.lang)
        else:
            # Auto-detect language from source text using langid
            language_label = detect_language_from_text(src)
        
        # Moderation check
        if args.moderation == 'on':
            moderation_input = src
            if check_openai_moderation(moderation_input):
                return {
                    'idx': row.get('idx', ''), 'src': src, 'tgt': row.get('tgt', ''),
                    'tp_fp_label': 'Error', 'reasoning': 'Error'
                }

        # For TNFN, we only need the source text
        prompt = build_numbered_prompt(TNFN_PROMPT, language_label, src, args.opinions)
        
        api_token = os.getenv('API_TOKEN', '')
        ok, content, total_tokens, pricing_info = call_model_with_pricing(
            prompt, args.llm_backend, api_token=api_token, moderation=False
        )
        
        if not ok:
            result = {
                'idx': row.get('idx', ''), 'src': src, 'tgt': row.get('tgt', ''),
                'tp_fp_label': 'Error', 'reasoning': 'Error'
            }
            return add_pricing_to_result_dict(result, pricing_info)
        
        label = parse_tnfn_label(content)
        result = {
            'idx': row.get('idx', ''),
            'src': src,
            'tgt': row.get('tgt', ''),
            'tp_fp_label': label,
            'reasoning': content.strip(),
            'writing_type': '',  # TNFN doesn't extract writing type
        }
        return add_pricing_to_result_dict(result, pricing_info)

    rows = process_rows_with_progress(df, process_row, desc=f"tnfn/baseline ({args.llm_backend})", workers=args.workers)
    out_df = pd.DataFrame(rows)
    
    # Print distribution analysis
    print_judge_distribution(rows, f"TNFN Baseline ({args.llm_backend})")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
