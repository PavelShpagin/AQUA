#!/usr/bin/env python3
"""
Common judge utilities shared across methods and judge types.

- Language mapping
- Safe numbered placeholder replacement
- Label parsers (TP/FP*, TN/FN)
- Concurrent row processing with progress bar
"""

from __future__ import annotations

import sys
import os
from typing import Callable, Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import langid
from utils.llm.backends import call_model
from utils.optimization.runner import run_optimized_process_rows
try:
    from utils.llm.settings import get_final_judge_temperature
except Exception:
    def get_final_judge_temperature(default: float = 0.0) -> float:
        return default
from utils.pricing import calculate_cost
from utils.distribution import analyze_distribution, print_distribution_summary, add_distribution_to_result_dict


LANGUAGE_LABELS: Dict[str, str] = {
    'en': 'English',
    'de': 'German', 
    'ua': 'Ukrainian',
    'uk': 'Ukrainian',  # langid uses 'uk' for Ukrainian
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    # Extended coverage (100+)
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'bs': 'Bosnian',
    'br': 'Breton',
    'my': 'Burmese',
    'ca': 'Catalan',
    'ceb': 'Cebuano',
    'zh': 'Chinese',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'co': 'Corsican',
    'cr': 'Cree',
    'cy': 'Welsh',
    'eo': 'Esperanto',
    'fa': 'Persian',
    'fr-ca': 'French (Canada)',
    'fy': 'Frisian',
    'ga': 'Irish',
    'gd': 'Scottish Gaelic',
    'gl': 'Galician',
    'gu': 'Gujarati',
    'ha': 'Hausa',
    'haw': 'Hawaiian',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hmn': 'Hmong',
    'hr': 'Croatian',
    'ht': 'Haitian Creole',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'ig': 'Igbo',
    'is': 'Icelandic',
    'ja': 'Japanese',
    'jv': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'ky': 'Kyrgyz',
    'la': 'Latin',
    'lb': 'Luxembourgish',
    'lo': 'Lao',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'ms': 'Malay',
    'ml': 'Malayalam',
    'mt': 'Maltese',
    'mi': 'Maori',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'nb': 'Norwegian Bokmål',
    'nn': 'Norwegian Nynorsk',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ps': 'Pashto',
    'qu': 'Quechua',
    'rw': 'Kinyarwanda',
    'sa': 'Sanskrit',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sm': 'Samoan',
    'sn': 'Shona',
    'so': 'Somali',
    'sr': 'Serbian',
    'st': 'Sesotho',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'te': 'Telugu',
    'tg': 'Tajik',
    'th': 'Thai',
    'ti': 'Tigrinya',
    'tk': 'Turkmen',
    'tl': 'Tagalog',
    'tn': 'Tswana',
    'to': 'Tongan',
    'tr': 'Turkish',
    'tt': 'Tatar',
    'ug': 'Uyghur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zu': 'Zulu',
}


def detect_language_from_text(text: str) -> str:
    """
    Detect language from the first sentence of text using langid.
    Returns the full language name (e.g., 'English', 'German').
    
    Note: langid.classify() always returns a language code, never None.
    We trust langid's best guess even with low confidence.
    """
    # Take first sentence for detection (up to first period, or first 100 chars)
    first_sentence = text.split('.')[0][:100].strip()
    
    # If empty text, use the full text as fallback
    if not first_sentence:
        first_sentence = text[:100].strip()
    
    # If still empty, langid will handle it and return its best guess
    detected_code, confidence = langid.classify(first_sentence)
    return LANGUAGE_LABELS.get(detected_code, detected_code.title())


def get_language_label(code: str) -> str:
    """Convert language code to human-readable language name.
    
    Args:
        code: Language code (e.g., 'es', 'en', 'de', 'ua')
        
    Returns:
        Human-readable language name (e.g., 'Spanish', 'English', 'German', 'Ukrainian')
        Falls back to title-cased code if not found.
    """
    return LANGUAGE_LABELS.get(code.lower(), code.title())


def get_language_code(identifier: str) -> str:
    """Normalize a language identifier (code or label) to a language code.
    
    - Accepts codes like 'pt', 'EN', 'ua', 'uk'
    - Accepts labels like 'Portuguese', 'English', 'Ukrainian'
    - Prefers 'uk' for Ukrainian; keeps 'ua' if explicitly provided
    - Falls back to lower-cased input
    """
    try:
        if not identifier:
            raise ValueError("Unsupported language: empty identifier")
        ident = identifier.strip()
        if not ident:
            raise ValueError("Unsupported language: empty identifier")
        # If already a known code
        low = ident.lower()
        if low in LANGUAGE_LABELS:
            # Normalize Ukrainian to 'uk' unless explicitly 'ua'
            if low == 'ua':
                return 'ua'
            if low == 'uk':
                return 'uk'
            return low
        # Try to match by label
        for code, label in LANGUAGE_LABELS.items():
            if label.lower() == ident.lower():
                if code in ('uk', 'ua'):
                    return 'uk'
                return code
        # Unsupported language
        raise ValueError(f"Unsupported language: '{identifier}'")
    except Exception:
        raise


def build_numbered_prompt(template: str, *args: str) -> str:
    """Safely replace numbered placeholders {0},{1},{2},{3} without touching JSON braces."""
    out = template
    for idx, val in enumerate(args):
        out = out.replace(f'{{{idx}}}', val)
    return out


def parse_tpfp_label(text: str) -> str:
    """Extract classification label from model output.

    Returns one of: TP, FP1, FP2, FP3, TN, FN, or Error (fallback).
    The model is expected to either embed the label plainly in text or
    inside a JSON key called "classification" (case-insensitive).
    """

    if not text:
        return 'Error'

    up = text.upper()

    # 1. JSON pattern
    import re
    # a) classification: "TP"
    m = re.search(r'"CLASSIFICATION"\s*:\s*"([A-Z0-9]+)"', up)
    if m:
        cls = m.group(1)
        if cls in {'TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN'}:
            return cls
    # b) classifications: ["TP", ...] (plural form used by some prompts)
    m = re.search(r'"CLASSIFICATIONS"\s*:\s*\[(.*?)\]', up)
    if m:
        within = m.group(1)
        # pick first token-like label inside the array (with or without quotes)
        m2 = re.search(r'"?(TP|TN|FN|FP1|FP2|FP3|FP)"?', within)
        if m2:
            label = m2.group(1)
            # If it's just "FP", check for fp_severity
            if label == 'FP':
                severity_match = re.search(r'"FP_SEVERITY"\s*:\s*"?(FP[123])', up)
                if severity_match:
                    return severity_match.group(1)
            return label
    # c) label: "TP"
    m = re.search(r'"LABEL"\s*:\s*"(TP|TN|FN|FP1|FP2|FP3)"', up)
    if m:
        return m.group(1)

    # 2. Plain-text search (prefix space avoids substrings)
    for cls in ['TP', 'TN', 'FN', 'FP1', 'FP2', 'FP3']:
        if f' {cls}' in f' {up}':
            return cls

    return 'Error'


def parse_tnfn_label(text: str) -> str:
    if not text:
        return 'FN'
    up = text.upper()
    import re
    m = re.search(r'"CLASSIFICATION"\s*:\s*"(TN|FN)"', up)
    if m:
        return m.group(1)
    return 'TN' if ' TN' in f' {up}' else 'FN'


def parse_writing_type(text: str) -> str:
    """Extract writing type from model response supporting both keys.
    Looks for either "type_of_writing" (baseline feedback JSON) or "writing_type".
    Returns empty string if not found.
    """
    if not text:
        return ''
    import re
    # Try type_of_writing first
    m = re.search(r'"type_of_writing"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback to writing_type
    m = re.search(r'"writing_type"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ''


def call_model_with_pricing(prompt: str, backend: str, api_token: str, *, 
                           moderation: bool = False, no_temperature: bool = False, temperature_override: float | None = None) -> Tuple[bool, str, int, Dict[str, Any]]:
    """
    Backward-compatible wrapper around call_model that provides pricing information.
    
    Returns:
        Tuple[bool, str, int, Dict[str, Any]]: (success, content, total_tokens, pricing_info)
        
        pricing_info contains:
        - 'token_usage': Dict with input_tokens, output_tokens, reasoning_tokens, cached_tokens, total_tokens
        - 'cost_breakdown': Dict with input_cost_usd, output_cost_usd, reasoning_cost_usd, cached_cost_usd, total_cost_usd
        - 'model': str
    """
    # Fallback: if api_token is empty, try common env vars to avoid silent failures
    if not api_token:
        import os as _os
        api_token = (
            _os.getenv('OPENAI_API_KEY', '')
            or _os.getenv('openai_api_key', '')
            or _os.getenv('openai_api_token', '')
            or _os.getenv('API_TOKEN', '')
        )
    # Choose token type carefully based on backend flavor (proxy vs direct OpenAI)
    try:
        b_lower = (backend or '').lower()
        from utils.llm.backends import is_red_sparta as _is_sparta
        if _is_sparta():
            # On Red Sparta, always prefer API_TOKEN for llm-proxy-backed routes
            import os as _os
            original_token = api_token
            api_token = _os.getenv('API_TOKEN', api_token)
            if _os.getenv('LLM_DEBUG'):
                print(f"DEBUG [call_model_with_pricing]: Red Sparta detected, backend={backend}, original_token={'set' if original_token else 'empty'}, using API_TOKEN={'set' if api_token else 'NOT SET'}")
        else:
            # Heuristic: proxy-style backends and nano variants should use API_TOKEN, not OpenAI keys
            looks_like_proxy = any(k in b_lower for k in ['oai_', 'gas_', 'aws_', 'bedrock', 'gemini', 'nano'])
            if not looks_like_proxy:
                looks_like_openai = ('gpt' in b_lower) or ('openai' in b_lower)
                if looks_like_openai:
                    # If provided token doesn't look like an OpenAI key, but OPENAI_API_KEY exists, use it
                    if not (api_token or '').startswith('sk-'):
                        import os as _os
                        openai_key = _os.getenv('OPENAI_API_KEY', '')
                        if openai_key:
                            api_token = openai_key
            # Otherwise keep api_token as-is for proxy-backed models
    except Exception:
        pass

    # Default to temperature=0 for reproducibility unless explicitly overridden
    if temperature_override is None:
        temperature_override = 0.0
    success, content, token_usage = call_model(
        prompt, backend, api_token,
        moderation=moderation,
        no_temperature=no_temperature,
        temperature_override=temperature_override,
    )
    
    # Calculate costs
    cost_breakdown = calculate_cost(
        backend,
        input_tokens=token_usage.get('input_tokens', 0),
        output_tokens=token_usage.get('output_tokens', 0),
        reasoning_tokens=token_usage.get('reasoning_tokens', 0),
        cached_tokens=token_usage.get('cached_tokens', 0)
    )
    
    pricing_info = {
        'token_usage': token_usage,
        'cost_breakdown': {
            'input_cost_usd': cost_breakdown.input_cost,
            'output_cost_usd': cost_breakdown.output_cost,
            'reasoning_cost_usd': cost_breakdown.reasoning_cost,
            'cached_cost_usd': cost_breakdown.cached_cost,
            'total_cost_usd': cost_breakdown.total_cost
        },
        'model': backend
    }
    
    return success, content, token_usage.get('total_tokens', 0), pricing_info


def call_model_legacy(prompt: str, backend: str, api_token: str, *, 
                     moderation: bool = False, no_temperature: bool = False) -> Tuple[bool, str, int]:
    """
    Legacy backward-compatible wrapper that returns the old format.
    
    For judges that haven't been updated yet, this provides the old (success, content, tokens) signature.
    """
    success, content, _, _ = call_model_with_pricing(prompt, backend, api_token, 
                                                   moderation=moderation, no_temperature=no_temperature)
    return success, content, _


def process_rows_with_progress(
    df,
    process_fn: Callable[[Any], Dict[str, Any]],
    *,
    desc: str,
    workers: int = 1,
    optimization: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run process_fn(row) across a pandas DataFrame with optional concurrency.
    Preserves input order; shows a tqdm progress bar.
    """
    # Use requested workers without global cap; baseline mode is capped in shell
    safe_workers = max(1, workers or 1)

    # Optimized path
    if optimization:
        # Enable 2x cheaper pricing only in optimized mode
        os.environ['BATCH_PRICING'] = '1'
        return run_optimized_process_rows(
            df, process_fn, desc=desc, target_shards=None, workers_per_shard=safe_workers
        )

    # Serial
    if not workers or workers <= 1:
        rows: List[Dict[str, Any]] = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc=desc, leave=True, mininterval=0.1, file=sys.stdout, disable=(os.getenv('QUIET_LOGS') == '1')):
            rows.append(process_fn(r))
        return rows

    # Concurrent
    results: List[Dict[str, Any]] = [None] * len(df)  # type: ignore
    with ThreadPoolExecutor(max_workers=safe_workers) as ex:
        futures = {ex.submit(process_fn, r): i for i, (_, r) in enumerate(df.iterrows())}
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=True, mininterval=0.1, file=sys.stdout, disable=(os.getenv('QUIET_LOGS') == '1')):
            pass
        for fut, i in futures.items():
            results[i] = fut.result()
    return results


def print_judge_distribution(results: List[Dict[str, Any]], 
                           judge_name: str = "Judge") -> None:
    """
    Print distribution analysis for judge results including pricing if available.
    
    Args:
        results: List of result dictionaries from judge processing
        judge_name: Name of the judge for display
    """
    if not results:
        print(f"\n{judge_name} Distribution: No results to analyze")
        return
    
    # Print distribution analysis
    # Auto-detect column names for distribution analysis
    pred_col = None
    gold_col = None
    
    if results:
        sample = results[0]
        # Check for various prediction column name patterns (prefer tp_fp_label)
        for col_name in ['tp_fp_label', 'label', 'pred_specialized', 'judge_label']:
            if col_name in sample:
                pred_col = col_name
                break
        
        # Check for gold column patterns  
        for col_name in ['gold_specialized', 'gold_label', 'tp_fp_label_gold']:
            if col_name in sample:
                gold_col = col_name
                break
        
        # Fallback defaults
        if pred_col is None:
            pred_col = 'pred_specialized'
        if gold_col is None:
            gold_col = 'gold_specialized'
    
    distribution = analyze_distribution(results, pred_col=pred_col, gold_col=gold_col)
    summary = print_distribution_summary(
        distribution, 
        f"{judge_name} Distribution Analysis", 
        show_confusion=False
    )
    print(summary)
    
    # Print pricing analysis if pricing/token data is available
    pricing_results = [r for r in results if ('total_cost_usd' in r) or ('input_tokens' in r)]
    if pricing_results:
        from utils.pricing import PricingTracker, extrapolate_costs
        
        tracker = PricingTracker()
        for result in pricing_results:
            model = result.get('model', 'unknown')
            tracker.add_usage(
                model,
                input_tokens=int(result.get('input_tokens', 0)),
                output_tokens=int(result.get('output_tokens', 0)),
                reasoning_tokens=int(result.get('reasoning_tokens', 0)),
                cached_tokens=int(result.get('cached_tokens', 0))
            )
        
        # Always show pricing analysis if we have token data
        summary = tracker.get_summary()
        total_cost = summary['total_cost_usd']
        
        # Calculate cache savings
        total_cached = sum(r.get('cached_tokens', 0) for r in pricing_results)
        total_input = sum(r.get('input_tokens', 0) for r in pricing_results)
        
        print(f"\n{judge_name} Pricing Analysis")
        print("=" * 50)
        print(f"Total samples processed: {len(results)}")
        print(f"Total tokens consumed: {summary['total_tokens']:,}")
        
        # Show detailed token breakdown
        total_input_tokens = sum(r.get('input_tokens', 0) for r in pricing_results)
        total_output_tokens = sum(r.get('output_tokens', 0) for r in pricing_results)
        total_reasoning_tokens = sum(r.get('reasoning_tokens', 0) for r in pricing_results)
        
        print(f"Input tokens: {total_input_tokens:,}")
        print(f"Output tokens: {total_output_tokens:,}")
        if total_reasoning_tokens > 0:
            print(f"Reasoning tokens: {total_reasoning_tokens:,}")
        
        # Show cache statistics if caching was used
        if total_cached > 0 and total_input > 0:
            cache_rate = (total_cached / total_input) * 100
            # Calculate savings from caching
            from utils.pricing import get_model_pricing
            if pricing_results and 'model' in pricing_results[0]:
                model = pricing_results[0]['model']
                pricing = get_model_pricing(model)
                if pricing:
                    hypothetical_cost = (total_cached / 1_000_000) * pricing["input"]
                    actual_cached_cost = (total_cached / 1_000_000) * pricing["cached"]
                    cache_savings = hypothetical_cost - actual_cached_cost
                    print(f"Cache hit rate: {cache_rate:.1f}% ({total_cached:,}/{total_input:,} tokens)")
                    print(f"Cache savings: ${cache_savings:.4f}")
        elif total_input > 0:
            print(f"Cache hit rate: 0.0% (no caching)")
        
        # Show detailed cost breakdown
        total_input_cost = sum(r.get('input_cost_usd', 0) for r in pricing_results)
        total_output_cost = sum(r.get('output_cost_usd', 0) for r in pricing_results)
        total_reasoning_cost = sum(r.get('reasoning_cost_usd', 0) for r in pricing_results)
        
        print(f"Input cost: ${total_input_cost:.6f}")
        print(f"Output cost: ${total_output_cost:.6f}")
        if total_reasoning_cost > 0:
            print(f"Reasoning cost: ${total_reasoning_cost:.6f}")
        print(f"Total cost: ${total_cost:.6f}")
        # Judges meta
        num_judges = 0
        try:
            # Detect number of judges by counting per-row judge_outputs if present
            sample = results[0]
            if 'judge_outputs' in sample and isinstance(sample['judge_outputs'], list):
                num_judges = len(sample['judge_outputs'])
        except Exception:
            pass
        if num_judges:
            print(f"Judges per row: {num_judges}")
        
        if len(results) > 0:
            print(f"Average cost per request: ${total_cost/len(results):.6f}")
        
        # Extrapolation
        if len(results) > 0:
            extrapolation = extrapolate_costs(total_cost, len(results), 10000)
            print(f"\n10K EXTRAPOLATION:")
            print(f"Projected cost for 10,000 requests: ${extrapolation['extrapolated_cost']:.2f}")
            
            # Cost assessment with dollar amount
            current_cost = extrapolation['extrapolated_cost']
            if current_cost == 0:
                assessment = "ZERO COST RECORDED" 
            elif current_cost < 10:
                assessment = f"LOW COST: ${current_cost:.2f} for 10K requests"
            elif current_cost < 50:
                assessment = f"MODERATE COST: ${current_cost:.2f} for 10K requests"
            elif current_cost < 200:
                assessment = f"HIGH COST: ${current_cost:.2f} for 10K requests"
            else:
                assessment = f"VERY HIGH COST: ${current_cost:.2f} for 10K requests"
            
            print(f"Cost Assessment: {assessment}")
        print("=" * 50)


def add_pricing_to_result_dict(result_dict: Dict[str, Any], pricing_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to add pricing information to a judge result dictionary.
    
    Args:
        result_dict: The judge's result dictionary (e.g., with 'idx', 'src', 'tgt', 'tp_fp_label', etc.)
        pricing_info: Pricing information from call_model_with_pricing
        
    Returns:
        Updated result dictionary with pricing columns added
    """
    result_dict = result_dict.copy()
    
    # Add token usage
    token_usage = pricing_info['token_usage']
    result_dict['input_tokens'] = token_usage.get('input_tokens', 0)
    result_dict['output_tokens'] = token_usage.get('output_tokens', 0) 
    result_dict['reasoning_tokens'] = token_usage.get('reasoning_tokens', 0)
    result_dict['cached_tokens'] = token_usage.get('cached_tokens', 0)
    result_dict['total_tokens'] = token_usage.get('total_tokens', 0)
    
    # Add cost breakdown
    cost_breakdown = pricing_info['cost_breakdown']
    result_dict['input_cost_usd'] = cost_breakdown['input_cost_usd']
    result_dict['output_cost_usd'] = cost_breakdown['output_cost_usd']
    result_dict['reasoning_cost_usd'] = cost_breakdown['reasoning_cost_usd']
    result_dict['cached_cost_usd'] = cost_breakdown['cached_cost_usd']
    result_dict['total_cost_usd'] = cost_breakdown['total_cost_usd']
    
    # Add model info
    result_dict['model'] = pricing_info['model']
    
    return result_dict


