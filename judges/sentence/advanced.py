#!/usr/bin/env python3
"""
judges/sentence/baseline.py - Sentence-level baseline judge with fusion alignment
===============================================================================

This judge uses fusion alignment - merging all individual edits into one large 
comprehensive edit, then classifying it using the TP/FP3/FP2/FP1/TN/FN model.
"""

import argparse
import pandas as pd
import os
import sys
import re
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.judge import call_model_with_pricing, add_pricing_to_result_dict
from utils.errant_align import create_errant_alignment_optimized as build_inline_alignment
from utils.diff_extension import diff_by_words
import langid


def create_fusion_alignment(src: str, tgt: str, pre_aligned: str = "") -> str:
    """Create fusion alignment text using real diff-match-patch word-level diff.

    If pre_aligned is provided, return it as-is (for precomputed alignments).
    """
    if pre_aligned:
        return pre_aligned
    try:
        if src == tgt:
            return f"No change: '{src}'"
        diffs = diff_by_words(src, tgt, cleanup_semantic=True)
        parts = []
        for op, text in diffs:
            segment = text.strip()
            if not segment:
                continue
            if op == 0:
                parts.append(f"= {segment}")
            elif op == 1:
                parts.append(f"+ {segment}")
            else:
                parts.append(f"- {segment}")
        joined = " " .join(parts)
        # Keep prompt compact
        if len(joined) > 500:
            joined = joined[:500] + "…"
        return f"DIFF (word-level): {joined}"
    except Exception as e:
        return f"Diff error: {str(e)}"


def build_baseline_prompt(src: str, tgt: str, lang: str, pre_aligned: str = "") -> str:
    """Build enhanced prompt for sentence/baseline with fusion alignment."""
    lang_name_map = {'en': 'English', 'de': 'German', 'ua': 'Ukrainian', 'es': 'Spanish'}
    provided_language = lang_name_map.get(lang, lang)
    
    fusion_alignment = create_fusion_alignment(src, tgt, pre_aligned)
    
    prompt = f"""You are an expert {provided_language} Grammatical Error Correction (GEC) evaluator.

Your task is to evaluate whether a suggested correction is appropriate, using FUSION ALIGNMENT analysis.

**FUSION ALIGNMENT ANALYSIS:**
{fusion_alignment}

**Classification Guidelines:**
- **TP (True Positive)**: Correction fixes genuine grammatical errors appropriately
- **FP1 (False Positive - Critical)**: Correction introduces errors or significantly worsens the text
- **FP2 (False Positive - Medium)**: Correction is unnecessary but doesn't harm meaning
- **FP3 (False Positive - Minor)**: Correction is stylistic preference, not grammatical necessity
- **TN (True Negative)**: No correction needed, texts are equivalent
- **FN (False Negative)**: Original text has errors that weren't corrected

**Original Text:** `{src}`
**Suggested Text:** `{tgt}`

Provide your analysis in this JSON format (do not truncate or omit any keys):
{{
    "fusion_analysis": "Brief analysis of the fused changes",
    "classification": "TP|FP1|FP2|FP3|TN|FN",
    "reasoning": "Detailed explanation of your classification decision"
}}"""

    return prompt


def parse_baseline_response(content: str) -> Tuple[str, str]:
    """Parse the baseline judge response."""
    if not content:
        return 'TN', 'No response'
    
    classification_match = re.search(r'"classification"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
    if classification_match:
        classification = classification_match.group(1).upper()
        valid_labels = ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']
        if classification in valid_labels:
            label = classification
        else:
            label = 'TN'
    else:
        content_upper = content.upper()
        for possible_label in ['FP1', 'FP2', 'FP3', 'TP', 'TN', 'FN']:
            if possible_label in content_upper:
                label = possible_label
                break
        else:
            label = 'TN'
    
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1)
    else:
        reasoning = content[:200].replace('\n', ' ').strip()
    
    return label, reasoning


def call_single_judge_for_row_detailed(judge: str, method: str, backend: str, lang: str, 
                                     row: pd.Series, moderation: str = "off", 
                                     opinions: str = None) -> Dict[str, Any]:
    """Call sentence/baseline judge for a single row with detailed output."""
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    # Prefer precomputed aligned column if present
    pre_aligned = str(row.get('aligned', ''))
    prompt = build_baseline_prompt(src, tgt, lang, pre_aligned)
    api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
    
    ok, content, tokens, pricing_info = call_model_with_pricing(
        prompt, backend, api_token=api_token, moderation=(moderation == "on")
    )
    
    if ok and content:
        label, reasoning = parse_baseline_response(content)
    else:
        label, reasoning = 'Error', f'API call failed: {content}'
    
    result = {
        'label': label,
        'reasoning': reasoning,
        'writing_type': 'baseline_fusion',
        'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
        'output': content[:500] + '...' if content and len(content) > 500 else content,
        'api_success': ok
    }
    
    add_pricing_to_result_dict(result, pricing_info)
    return result


def main():
    """Main function for sentence/baseline judge."""
    parser = argparse.ArgumentParser(description='Sentence-level baseline judge with fusion alignment')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--llm_backend', required=True, help='LLM backend to use')
    parser.add_argument('--lang', default='', help='Force language for prompts (e.g., es, en, de, ua). If empty, auto-detect with langid.')
    parser.add_argument('--workers', type=int, default=50, help='Number of workers')
    parser.add_argument('--moderation', default='off', choices=['on', 'off'], help='Content moderation')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    print(f"Processing {len(df)} samples with sentence/baseline judge")
    
    results = []
    
    for idx, row in df.iterrows():
        if args.lang:
            # Force language from config/command line
            detected_lang = args.lang
        else:
            # Auto-detect language from source text using langid
            detected_lang, _ = langid.classify(str(row.get('src', '')))
        
        result = call_single_judge_for_row_detailed(
            judge='sentence', method='baseline', backend=args.llm_backend,
            lang=detected_lang, row=row, moderation=args.moderation
        )
        
        result.update({
            'idx': idx,
            'src': str(row.get('src', '')),
            'tgt': str(row.get('tgt', '')),
            'tp_fp_label': result['label'],
            'lang': detected_lang
        })
        
        results.append(result)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples")
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    
    from collections import Counter
    labels = [r['tp_fp_label'] for r in results]
    distribution = Counter(labels)
    
    print(f"\nSentence/Baseline Results:")
    print(f"Total samples: {len(results)}")
    print(f"Label distribution: {dict(distribution)}")
    
    success_rate = sum(1 for r in results if r.get('api_success', False)) / len(results)
    print(f"API success rate: {success_rate:.3f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()