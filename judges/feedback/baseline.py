#!/usr/bin/env python3
"""
Feedback judge (baseline): classify TP/FP* using baseline prompt (no agent tools).
Uses unified LLM router and robust parsing of the classification.
"""

import os
import argparse
import sys
import pandas as pd

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from judges.feedback.prompts import TPFP_PROMPT_BASELINE, TPFP_PROMPT_BASELINE_SIMPLE
from utils.llm.backends import call_model
from utils.llm.moderation import check_openai_moderation
from utils.judge import detect_language_from_text, get_language_label, get_language_code, build_numbered_prompt, parse_tpfp_label, process_rows_with_progress, call_model_with_pricing, add_pricing_to_result_dict
# Use optimized ERRANT alignment (100x faster with caching)
try:
    from utils.errant_align import get_alignment_for_language
    # Using optimized ERRANT alignment
except ImportError:
    # Fallback to simple diff if ERRANT not available
    def get_alignment_for_language(src, tgt, lang='en', **kwargs):
        if src == tgt:
            return src
        return f"{src} → {tgt}"
    print("WARN: Using simple alignment fallback", file=sys.stderr)
from utils.processing.model_setup import setup_language_models


def _compute_aligned_and_edit(src: str, tgt: str, aligned: str) -> tuple[str, str]:
    """Return (aligned_injected, edit_token) for logging and prompt injection."""
    edit_token = ""
    left_part, right_part = "", ""
    try:
        if '{' in aligned and '=>' in aligned and '}' in aligned:
            start = aligned.find('{')
            end = aligned.find('}', start + 1)
            if start != -1 and end != -1 and '=>' in aligned[start:end+1]:
                maybe = aligned[start:end+1]
                edit_token = maybe
                inner = maybe.strip('{}')
                if '=>' in inner:
                    left_part, right_part = inner.split('=>', 1)
        if not edit_token:
            from difflib import SequenceMatcher
            src_words = src.split()
            tgt_words = tgt.split()
            sm = SequenceMatcher(a=src_words, b=tgt_words)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != 'equal':
                    left_part = ' '.join(src_words[i1:i2]).strip()
                    right_part = ' '.join(tgt_words[j1:j2]).strip()
                    edit_token = f"{{{left_part}=>{right_part}}}"
                    break
        if not edit_token:
            edit_token = "{=>}"
    except Exception:
        edit_token = "{=>}"
        left_part, right_part = "", ""

    aligned_injected = aligned
    try:
        if '{' not in aligned_injected or '=>' not in aligned_injected or '}' not in aligned_injected:
            if left_part:
                idx = src.find(left_part)
                if idx != -1:
                    aligned_injected = src[:idx] + edit_token + src[idx + len(left_part):]
                else:
                    aligned_injected = f"{src} {edit_token}"
            else:
                aligned_injected = f"{src} {edit_token}"
    except Exception:
        aligned_injected = aligned or f"{src} {edit_token}"

    return aligned_injected, edit_token


def build_prompt(language_label: str, src: str, tgt: str, aligned: str, opinions: str = "") -> str:
    aligned_injected, edit_token = _compute_aligned_and_edit(src, tgt, aligned)
    return build_numbered_prompt(TPFP_PROMPT_BASELINE, language_label, src, tgt, aligned_injected, edit_token)


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
    parser.add_argument('--input', required=False)  # Made optional for debug mode
    parser.add_argument('--output', required=False)  # Made optional for debug mode
    parser.add_argument('--llm_backend', required=True)
    # Accept --lang to force language for prompts (if not provided, auto-detect with langid)
    parser.add_argument('--lang', default='', help='Force language for prompts (e.g., es, en, de, ua). If empty, auto-detect with langid.')
    parser.add_argument('--workers', type=int, default=200)  # Optimized for high-speed processing
    parser.add_argument('--moderation', default='off', choices=['on', 'off'])
    parser.add_argument('--opinions', type=str, default='')
    parser.add_argument('--final_judge', default='false', choices=['true', 'false'])
    parser.add_argument('--debug', default='off', choices=['on', 'off'])
    parser.add_argument('--optimization', default='off', choices=['on','off'])
    args = parser.parse_args()
    
    # Debug mode - run test and exit
    if args.debug == 'on':
        from utils.debug import run_debug_test, save_debug_log
        
        def debug_wrapper(src, tgt, llm_backend, lang, **kwargs):
            # Use forced language from --lang parameter, or auto-detect from source text
            if args.lang:
                # Force language from config/command line
                language_label = get_language_label(args.lang)
            else:
                # Auto-detect language from source text using langid
                language_label = detect_language_from_text(src)
            # Generate aligned sentence for debug mode
            aligned = get_alignment_for_language(src, tgt, language=lang.lower(), merge=True)
            
            # Check if alignment failed
            if aligned is None:
                return {
                    'prompt': 'Error: Alignment generation failed',
                    'output': 'Error: Failed to generate alignment',
                    'reasoning': 'Error: Failed to generate alignment for text comparison',
                    'label': 'Error'
                }
            
            prompt = build_prompt(language_label, src, tgt, aligned)
            ok, content, tokens, pricing_info = call_model_with_pricing(
                prompt, llm_backend, 
                api_token=os.getenv('API_TOKEN', ''), 
                moderation=False
            )
            label = parse_tpfp_label(content) if ok else 'Error'
            return {
                'prompt': prompt,
                'output': content,
                'reasoning': content,
                'label': label
            }
        
        logs = run_debug_test(
            debug_wrapper,
            'feedback',
            'baseline',
            [args.llm_backend],
            'all'  # Always test both EN and UA in debug mode
        )
        save_debug_log(logs, 'feedback', 'baseline')
        return
    
    # Regular mode - require input/output
    if not args.input or not args.output:
        parser.error("--input and --output are required when not in debug mode")

    is_final_judge = args.final_judge == 'true'
    use_moderation = args.moderation == 'on'

    df = pd.read_csv(args.input)

    # ERRANT toggle: if disabled, do not load models and require aligned fields
    errant_env = os.getenv('FEEDBACK_ERRANT', os.getenv('ERRANT', 'on')).lower()
    use_errant = errant_env not in {'off', 'false', '0', 'no'}

    language_models = {}
    if use_errant:
        # Preload spaCy models once for the detected languages in the dataset
        # Detect languages from a sample of the dataset
        sample_size = min(100, len(df))
        sample_langs = set()
        for i in range(sample_size):
            src = str(df.iloc[i]['src'])
            detected_label = detect_language_from_text(src)
            sample_langs.add(get_language_code(detected_label))
        
        # Preload models for detected languages
        print(f"Preloading spaCy models for languages: {sample_langs}")
        for lang_code in sample_langs:
            nlp, annotator = setup_language_models(lang_code)
            if nlp and annotator:
                language_models[lang_code] = (nlp, annotator)
                print(f"Loaded models for {lang_code}")
            else:
                print(f"Warning: Could not load models for {lang_code}, will use fallback alignment")
    else:
        print("ERRANT disabled for feedback judge (FEEDBACK_ERRANT=off). Expecting aligned/aligned_sentence/alert in input.")

    def process_row(row):
        src = str(row['src']); tgt = str(row['tgt'])
        
        # Use forced language from --lang parameter, or auto-detect from source text
        if args.lang:
            # Force language from config/command line
            language_label = get_language_label(args.lang)
        else:
            # Auto-detect language from source text using langid
            language_label = detect_language_from_text(src)
        
        # Map language label/code to normalized code
        lang_code = get_language_code(args.lang if args.lang else language_label)
        
        # Prefer precomputed aligned column, then aligned_sentence, then alert
        aligned_sentence = row.get('aligned', '') or row.get('aligned_sentence', '')
        if not aligned_sentence and 'alert' in row:
            aligned_sentence = str(row['alert'])
        
        # Generate aligned sentence if not available
        if not aligned_sentence:
            if use_errant:
                # Use preloaded models if available
                if lang_code in language_models:
                    nlp, annotator = language_models[lang_code]
                    aligned_sentence = get_alignment_for_language(src, tgt, language=lang_code, nlp=nlp, annotator=annotator)
                else:
                    # Try to generate alignment without preloaded models
                    aligned_sentence = get_alignment_for_language(src, tgt, language=lang_code)
                # Check if alignment failed
                if aligned_sentence is None:
                    return {
                        'idx': row['idx'], 'src': src, 'tgt': tgt,
                        'tp_fp_label': 'Error', 
                        'reasoning': 'Error: Failed to generate alignment for text comparison',
                        'aligned_sentence': ''
                    }
            else:
                # Strict behavior: ERRANT disabled and no aligned provided
                return {
                    'idx': row['idx'], 'src': src, 'tgt': tgt,
                    'tp_fp_label': 'Error',
                    'reasoning': 'Error: Missing aligned/aligned_sentence/alert and ERRANT is disabled',
                    'aligned_sentence': ''
                }
        
        if use_moderation:
            flagged, _ = check_openai_moderation(src + "\n" + tgt)
            if flagged:
                return {
                    'idx': row['idx'], 'src': src, 'tgt': tgt,
                    'tp_fp_label': 'Error', 'reasoning': 'Error',
                    'aligned_sentence': aligned_sentence
                }
        
        # Toggle prompt variant via env FEEDBACK_SIMPLE_PROMPT=on
        use_simple = os.getenv('FEEDBACK_SIMPLE_PROMPT', 'off').lower() in {'on','true','1','yes'}
        prompt_template = TPFP_PROMPT_BASELINE_SIMPLE if use_simple else TPFP_PROMPT_BASELINE
        # Compute aligned injection and edit span for both prompt and logging
        aligned_injected, edit_token = _compute_aligned_and_edit(src, tgt, aligned_sentence)
        prompt = build_numbered_prompt(prompt_template, language_label, src, tgt, aligned_injected, edit_token)
        
        # Do not truncate prompts; preserve full baseline prompt for reproducibility
        # Prefer OPENAI_API_KEY for OpenAI models, else API_TOKEN for proxy
        api_token = os.getenv('OPENAI_API_KEY', '') or os.getenv('API_TOKEN', '')
        ok, content, total_tokens, pricing_info = call_model_with_pricing(
            prompt, args.llm_backend, api_token=api_token, moderation=False, temperature_override=0.0
        )
        if not ok:
            label = 'Error'
            content = 'Error'
        else:
            label = parse_tpfp_label(content)
        
        result = {
            'idx': row['idx'],
            'src': src,
            'tgt': tgt,
            'tp_fp_label': label,
            'reasoning': content.strip(),
            'aligned_sentence': aligned_injected,
            'edit_span': edit_token
        }
        
        # Add pricing information to the result
        if ok:  # Only add pricing info for successful calls
            result = add_pricing_to_result_dict(result, pricing_info)
        
        return result

    rows = process_rows_with_progress(
        df, process_row, desc=f"Judge evaluation ({args.llm_backend})",
        workers=args.workers, optimization=(args.optimization=='on')
    )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()


