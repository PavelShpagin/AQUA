"""Shared utilities for ensemble orchestration."""

import subprocess
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
import tempfile
import os
import sys
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from utils.judge import parse_writing_type, add_pricing_to_result_dict

# In-process optimization imports
try:
    from utils.llm.backends import call_model
    from utils.judge import detect_language_from_text, parse_tpfp_label, call_model_with_pricing
    from utils.errant_align import get_alignment_for_language
    IN_PROCESS_AVAILABLE = True
except ImportError:
    IN_PROCESS_AVAILABLE = False


# Global executor to reuse threads across rows and reduce overhead
try:
    _JUDGE_POOL_SIZE = int(os.getenv('JUDGE_POOL', '512'))
except Exception:
    _JUDGE_POOL_SIZE = 512
_GLOBAL_JUDGE_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, _JUDGE_POOL_SIZE))


def call_judge(judge: str, method: str, backend: str, lang: str, 
               input_file: str, workers: int = 50, moderation: str = "off",
               opinions: Optional[str] = None) -> pd.DataFrame:
    """Call a single judge method and return the results as DataFrame."""
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_output = tmp_file.name
    
    try:
        # Build command
        import sys as _sys
        cmd = [
            _sys.executable, '-m', f'judges.{judge}.{method}',
            '--input', input_file,
            '--output', tmp_output,
            '--llm_backend', backend,
            '--lang', lang,
            '--workers', str(workers),
            '--moderation', moderation
        ]
        
        if opinions:
            cmd.extend(['--opinions', opinions])
        
        # Execute judge (pass environment to subprocess so .env vars are available)
        env = os.environ.copy()
        # Ensure API keys are present inside subprocess
        if not env.get('API_TOKEN'):
            env['API_TOKEN'] = env.get('OPENAI_API_KEY', env.get('openai_api_key', ''))
        if not env.get('OPENAI_API_KEY') and env.get('API_TOKEN'):
            env['OPENAI_API_KEY'] = env['API_TOKEN']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        
        if result.returncode != 0:
            # Non-zero exit: try to salvage output if the judge wrote it
            try:
                if os.path.exists(tmp_output) and os.path.getsize(tmp_output) > 0:
                    return pd.read_csv(tmp_output)
            except Exception:
                pass
            # Fall back to returning an explicit Error row constructed from input
            try:
                in_df = pd.read_csv(input_file)
                # Ensure required columns exist
                out = in_df.copy()
                out['tp_fp_label'] = 'Error'
                out['reasoning'] = (result.stderr or result.stdout or 'Judge subprocess failed').strip()[:2000]
                return out
            except Exception:
                # Last resort: synthesize a minimal one-row dataframe
                return pd.DataFrame([
                    {
                        'idx': 0,
                        'src': '',
                        'tgt': '',
                        'tp_fp_label': 'Error',
                        'reasoning': (result.stderr or result.stdout or 'Judge subprocess failed').strip()[:2000],
                    }
                ])
        
        # Read results on success
        df = pd.read_csv(tmp_output)
        return df
        
    finally:
        # Cleanup
        if os.path.exists(tmp_output):
            os.unlink(tmp_output)


def get_classification_mode(judge: str) -> str:
    """Get classification mode for judge type."""
    if judge == 'feedback':
        return '4-class'
    elif judge in ['edit', 'sentence']:
        return '6-class'
    elif judge == 'tnfn':
        return 'binary'
    else:
        raise ValueError(f"Unknown judge type: {judge}")


def get_non_adjacent_labels(label: str) -> List[str]:
    """Get non-adjacent labels for consensus checking."""
    non_adjacent = {
        'TP': ['FP2', 'FP1'],
        'FP3': ['FP1'], 
        'FP2': ['TP'],
        'FP1': ['TP', 'FP3'],
        'TN': ['FN'],
        'FN': ['TN']
    }
    return non_adjacent.get(label, [])


def check_consensus(labels: List[str]) -> Optional[str]:
    """
    Check consensus using the IterCriticEnsemble algorithm from docs/general.md:
    - Most frequent class c* must have frequency >= 2/3
    - Each non-adjacent class must have frequency <= 1/4
    """
    if not labels:
        return None
    
    # Count label frequencies
    label_counts = {}
    for label in labels:
        if label != "Error":  # Skip error labels
            lab = normalize_label(label)
            if lab == 'Error':
                continue
            label_counts[lab] = label_counts.get(lab, 0) + 1
    
    if not label_counts:
        return None
    
    total_valid = sum(label_counts.values())
    if total_valid == 0:
        return None
    
    # Find most frequent class c*
    max_count = max(label_counts.values())
    most_frequent = [label for label, count in label_counts.items() if count == max_count]
    
    if len(most_frequent) > 1:
        return None  # Tie, no consensus
    
    c_star = most_frequent[0]
    c_star_freq = label_counts[c_star] / total_valid
    
    # Check 2/3 threshold for c*
    if c_star_freq < 2/3:
        return None
    
    # Check 1/4 threshold for non-adjacent labels
    non_adjacent = get_non_adjacent_labels(c_star)
    for non_adj_label in non_adjacent:
        if non_adj_label in label_counts:
            non_adj_freq = label_counts[non_adj_label] / total_valid
            if non_adj_freq > 1/4:
                return None
    
    return c_star


def is_unanimous(labels: List[str], judge: str) -> bool:
    """Check if labels reach consensus using the proper algorithm."""
    consensus = check_consensus(labels)
    return consensus is not None


def get_top_two_classes(labels: List[str]) -> List[str]:
    """Get the 2 most frequent classes for final judgment."""
    if not labels:
        return []
    
    # Count label frequencies, excluding errors
    label_counts = {}
    for label in labels:
        if label != "Error":
            lab = normalize_label(label)
            if lab == 'Error':
                continue
            label_counts[lab] = label_counts.get(lab, 0) + 1
    
    if not label_counts:
        return []
    
    # Sort by frequency, then alphabetically for ties
    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Return top 2 (or fewer if less than 2 unique labels)
    return [label for label, _ in sorted_labels[:2]]


def format_opinions(judge_results: List[Dict[str, Any]]) -> str:
    """Format previous judge results into opinions string."""
    if not judge_results:
        return ""
    
    opinions = []
    for i, result in enumerate(judge_results, 1):
        # Prefer 'label' (from detailed judge calls), fallback to 'tp_fp_label'
        label = result.get('label', result.get('tp_fp_label', 'Unknown'))
        reasoning = result.get('reasoning', 'No reasoning provided')
        opinions.append(f"Judge {i}: {label}\nReasoning: {reasoning}")
    text = "\n\n".join(opinions)
    # Programmatically add header once here; prompts only contain the {opinions} placeholder
    return f"## ADDITIONAL OPINIONS\n{text}" if text else ""


def normalize_label(label: str) -> str:
    """Normalize any variant/noisy label to strict taxonomy: TP, FP1, FP2, FP3, TN, FN.
    Unknowns -> 'Error'."""
    if not label:
        return 'Error'
    up = str(label).strip().upper()
    if up in {'TP','FP1','FP2','FP3','TN','FN'}:
        return up
    return 'Error'


def aggregate_weighted(labels: List[str], weights: Dict[str, float]) -> str:
    """Aggregate labels using weighted voting with category separation."""
    if not labels:
        return "Error"
    
    # Ignore Error labels instead of failing hard
    labels = [l for l in labels if l != "Error"]
    if not labels:
        return "Error"
    
    # Separate TP/FP* and TN/FN categories
    # Normalize any noise (e.g., TFN) to strict taxonomy
    labels = [normalize_label(x) for x in labels]
    tpfp_labels = [x for x in labels if x in ['TP', 'FP1', 'FP2', 'FP3']]
    tnfn_labels = [x for x in labels if x in ['TN', 'FN']]
    
    # Choose dominant category
    if len(tpfp_labels) >= len(tnfn_labels):
        selected_labels = tpfp_labels
        # Use TP/FP* weights (updated to match your logic)
        selected_weights = {'TP': 5, 'FP3': 1, 'FP2': -1, 'FP1': -3}
    else:
        selected_labels = tnfn_labels
        # Use TN/FN weights
        selected_weights = {'TN': 1, 'FN': -1}
    
    if not selected_labels:
        return "Error"
    
    # Calculate average weight
    total_weight = sum([selected_weights.get(label, 0) for label in selected_labels])
    avg_weight = total_weight / len(selected_labels)
    
    # Find class with weight closest to (and <=) average weight
    # Sort by weight descending to find first match
    sorted_classes = sorted(selected_weights.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, weight in sorted_classes:
        if avg_weight >= weight:
            return class_name
    
    return "Error"


def aggregate_consistency(labels: List[str], judge: str, src: str = "", tgt: str = "") -> str:
    """Aggregate labels using majority voting with custom tie-breaking order."""
    if not labels:
        return "Error"
    
    # Check for errors
    if any(label == "Error" for label in labels):
        return "Error"
    
    # Remove the 6-class outlier handling entirely - all labels compete equally
    # with the new tie-breaking order: FP1 < FP2 < FP3 < FN < TN/TP
    
    # Standard majority vote
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]
    
    # Custom tie-breaking order: FP1 < FP2 < FP3 < FN < TN/TP
    def tie_break_order(label):
        order_map = {'FP1': 1, 'FP2': 2, 'FP3': 3, 'FN': 4}
        
        if label in order_map:
            return order_map[label]
        elif label in ['TN', 'TP']:
            # Special TN/TP tie-breaking: TN if src==tgt, else TP
            if src == tgt:
                return 5 if label == 'TN' else 6  # TN wins if src==tgt
            else:
                return 5 if label == 'TP' else 6  # TP wins if src!=tgt
        else:
            return 7  # Other labels last
    
    # Sort candidates by custom order and return first
    return sorted(candidates, key=tie_break_order)[0]


def call_single_judge_for_row(judge: str, method: str, backend: str, lang: str,
                              row: pd.Series, moderation: str = "off",
                              opinions: Optional[str] = None) -> str:
    """Call a single judge for a single row and return the label."""
    # Create single-row temp file ensuring required columns exist
    required_cols = ['idx', 'src', 'tgt']
    single_row_dict = {k: row.get(k, '') for k in required_cols}
    # Preserve extra columns if present
    for k in row.index:
        if k not in single_row_dict:
            single_row_dict[k] = row.get(k)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_input:
        single_row = pd.DataFrame([single_row_dict])
        single_row.to_csv(tmp_input.name, index=False)
        tmp_input_path = tmp_input.name
    
    try:
        # Call judge
        judge_result = call_judge(
            judge=judge,
            method=method,
            backend=backend,
            lang=lang,
            input_file=tmp_input_path,
            workers=1,  # Single row, single worker
            moderation=moderation,
            opinions=opinions
        )
        
        if len(judge_result) > 0:
            from utils.ensemble import normalize_label as _norm
            return _norm(str(judge_result.iloc[0]['tp_fp_label']))
        else:
            return "Error"
            
    except Exception as e:
        import traceback
        print(f"ERROR in call_single_judge_for_row: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return "Error"
        
    finally:
        if os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)


def call_judge_in_process(judge: str, method: str, src: str, tgt: str, backend: str, 
                         lang: str = 'es', moderation: bool = False, api_token: str = '',
                         aligned_text: str = '') -> Dict[str, Any]:
    """
    PRODUCTION-READY in-process judge calling - 100x faster than subprocess.
    """
    if not IN_PROCESS_AVAILABLE:
        return {'tp_fp_label': 'Error', 'reasoning': 'In-process mode not available'}
    
    try:
        # Language detection (avoid expensive detection when lang is provided)
        language_label = detect_language_from_text(src) if not lang else {
            'es': 'Spanish', 'en': 'English', 'de': 'German', 'ua': 'Ukrainian'
        }.get(lang, 'Spanish')
        
        # Universal prompt builder (preserve original judge prompts)
        if judge == 'feedback':
            # Baseline feedback does not use alignment; skip ERRANT to avoid spaCy loads per thread
            if method == 'baseline':
                from judges.feedback.baseline import build_prompt as _build_legacy_feedback_prompt
                prompt = _build_legacy_feedback_prompt(language_label, src, tgt)
            else:
                # Baseline feedback requires alignment for the prompt
                errant_flag = os.getenv('FEEDBACK_ERRANT', os.getenv('ERRANT', 'on')).lower()
                use_errant = errant_flag not in {'off', 'false', '0', 'no'}
                aligned = aligned_text
                if not aligned:
                    if use_errant:
                        aligned = get_alignment_for_language(src, tgt, language=lang)
                    else:
                        # Strict behavior when ERRANT is disabled: require aligned/aligned_sentence/alert
                        return {
                            'tp_fp_label': 'Error',
                            'reasoning': 'Error: aligned/aligned_sentence/alert not provided (ERRANT disabled)'
                        }
                # Use baseline's builder to inject Aligned and Edit properly (5 placeholders)
                try:
                    from judges.feedback.advanced import build_prompt as _build_feedback_prompt
                    prompt = _build_feedback_prompt(language_label, src, tgt, aligned)
                except Exception:
                    from judges.feedback.prompts import TPFP_PROMPT_BASELINE
                    from utils.judge import build_numbered_prompt
                    # Fallback to 4-slot prompt (may leave {4} literal if present)
                    prompt = build_numbered_prompt(TPFP_PROMPT_BASELINE, language_label, src, tgt, aligned)
            
        elif judge == 'sentence':
            from judges.sentence.advanced import build_baseline_prompt, create_fusion_alignment, parse_baseline_response
            # Treat 'baseline' as the main enhanced baseline path
            if method in ('advanced', 'baseline'):
                pre_aligned = aligned_text or get_alignment_for_language(src, tgt, language=lang)
                if not pre_aligned:
                    pre_aligned = f"{src} → {tgt}"
                prompt = build_baseline_prompt(src, tgt, lang, pre_aligned)
            elif method == 'legacy':
                # Build older 4-class prompt (TP/FP1-3/TN/FN not supported). Reuse baseline prompt but drop TN/FN lines.
                pre_aligned = aligned_text or create_fusion_alignment(src, tgt)
                if not pre_aligned:
                    pre_aligned = f"{src} → {tgt}"
                prompt = build_baseline_prompt(src, tgt, lang, pre_aligned)

        elif judge == 'edit':
            if method == 'agent' or method == 'agent_v1':
                # SOTA Edit Agent - call directly without LLM prompt building
                if method == 'agent_v1':
                    from judges.edit._legacy.agent_v1 import call_single_judge_for_row_detailed as call_agent_judge
                else:
                    from judges.edit._legacy.agent import call_single_judge_for_row_detailed as call_agent_judge
                
                # Create a row-like object for the agent
                class RowWrapper:
                    def __init__(self, src, tgt):
                        self.src = src
                        self.tgt = tgt
                    def get(self, key, default=''):
                        return getattr(self, key, default)
                
                row = RowWrapper(src, tgt)
                
                # Call the agent directly
                result = call_agent_judge(
                    judge='edit',
                    method='agent', 
                    backend=backend,
                    lang=lang,
                    row=row,
                    moderation=moderation,
                    optimization=True  # Always use optimization in ensemble
                )
                
                # Return in expected format
                return {
                    'tp_fp_label': result.get('label', 'Error'),
                    'reasoning': result.get('reasoning', ''),
                    'writing_type': result.get('writing_type', 'agent'),
                    'confidence': result.get('confidence', 0.5),
                    'tools_used': result.get('tools_used', ''),
                    'cost_estimate': result.get('cost_estimate', 0.0),
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_cost_usd': result.get('cost_estimate', 0.0)
                }
            elif method == 'agent_v2':
                # Agent v2 routing identical to agent (separate module for experimentation)
                from judges.edit._legacy.agent_v2 import call_single_judge_for_row_detailed as call_agent_judge
                class RowWrapper:
                    def __init__(self, src, tgt):
                        self.src = src
                        self.tgt = tgt
                    def get(self, key, default=''):
                        return getattr(self, key, default)
                row = RowWrapper(src, tgt)
                result = call_agent_judge(
                    judge='edit',
                    method='agent_v2',
                    backend=backend,
                    lang=lang,
                    row=row,
                    moderation=moderation,
                    optimization=True
                )
                return {
                    'tp_fp_label': result.get('label', 'Error'),
                    'reasoning': result.get('reasoning', ''),
                    'writing_type': result.get('writing_type', 'agent_v2'),
                    'confidence': result.get('confidence', 0.5),
                    'tools_used': result.get('tools_used', ''),
                    'cost_estimate': result.get('cost_estimate', 0.0),
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_cost_usd': result.get('cost_estimate', 0.0)
                }
            else:
                # Edit baseline-compatible: per-edit JSON -> sentence 6-class (TP/FP1/FP2/FP3/TN/FN)
                from judges.edit.prompts import EDIT_LEVEL_JUDGE_PROMPT
                from judges.edit.advanced import parse_json_response, compute_sentence_label
                from judges.sentence.advanced import create_fusion_alignment
                from utils.judge import build_numbered_prompt
                raw_align = aligned_text or get_alignment_for_language(src, tgt, language=lang)
                if not raw_align:
                    raw_align = f"{src} → {tgt}"
                # Use compact fusion alignment to keep prompt size safe
                fused_alignment = create_fusion_alignment(src, tgt, raw_align)
                if len(fused_alignment) > 600:
                    fused_alignment = fused_alignment[:600] + "…"
                # Extract explicit edit spans for the {4} slot to avoid leaving a literal {4} in the prompt
                try:
                    import re as _re
                    raw_spans = _re.findall(r"\{[^}]+=>[^}]+\}", fused_alignment)
                    spans = []
                    for s in raw_spans:
                        body = s[1:-1]
                        o, n = body.split("=>", 1)
                        if (o.strip()=='' and n.strip()==''):
                            continue
                        if o == n:
                            continue
                        spans.append(s)
                    edits_field = ('"' + '" , "'.join(spans) + '"') if spans else ""
                except Exception:
                    edits_field = ""
                prompt = build_numbered_prompt(EDIT_LEVEL_JUDGE_PROMPT, language_label, src, tgt, fused_alignment, edits_field)

        else:
            # Generic fallback
            prompt = f"""Classify this {language_label} correction:

Original: {src}
Corrected: {tgt}
Changes: {aligned}

Classify as: TP/FP/TN/FN"""
        
        # Direct LLM call with optimized backends (temperature=0 for reproducibility)
        # Deterministic call (temperature=0). Keep signature compatible.
        ok, content, tokens, pricing_info = call_model_with_pricing(
            prompt, backend, api_token=api_token, moderation=moderation, temperature_override=0.0
        )
        
        # Disable cross-backend model fallbacks by default (env-gated)
        ALLOW_FALLBACKS = os.getenv('ENABLE_BACKEND_FALLBACK', '0').lower() in {'1','true','on'}
        if ALLOW_FALLBACKS and not ok and 'gemini' in backend.lower():
            for fallback in ['gemini-2.0-flash-lite', 'gas_gemini20_flash_lite']:
                if fallback != backend:
                    ok, content, tokens, pricing_info = call_model_with_pricing(
                        prompt, fallback, api_token=api_token, moderation=moderation, temperature_override=0.0
                    )
                    if ok:
                        break
        
        elif ALLOW_FALLBACKS and not ok and ('gpt' in backend.lower() or '4.1' in backend):
            for fallback in ['gpt-4o', 'gpt-4o-mini']:
                if fallback != backend:
                    ok, content, tokens, pricing_info = call_model_with_pricing(
                        prompt, fallback, api_token=api_token, moderation=moderation, temperature_override=0.0
                    )
                    if ok:
                        break
        
        # Parse result (mirror judge scripts)
        model_json_for_reason = None
        if ok and content:
            if judge == 'sentence':
                if method == 'baseline':
                    label, _ = parse_baseline_response(content)
                elif method == 'legacy':
                    # Legacy parser: accept 4-class or 6-class outputs
                    import re, json
                    label = 'Error'
                    try:
                        data = json.loads(content)
                        label = str(data.get('classification', '')).upper()
                    except Exception:
                        pass
                    if label not in {'TP','FP1','FP2','FP3','TN','FN'}:
                        m = re.search(r'"?CLASSIFICATION"?\s*[:=]\s*"?(TP|FP1|FP2|FP3|TN|FN)"?', content, re.I)
                        if m:
                            label = m.group(1).upper()
                    if label not in {'TP','FP1','FP2','FP3','TN','FN'}:
                        up = content.upper()
                        for cls in ['FP1','FP2','FP3','TP','TN','FN']:
                            if f' {cls}' in f' {up}':
                                label = cls; break
                    if label not in {'TP','FP1','FP2','FP3','TN','FN'}:
                        label = 'Error'
                # Ensure parsed_label is set for sentence path
                parsed_label = label
            elif judge == 'edit':
                parsed = parse_json_response(content)
                # Clean junk edit keys like {=>}, {}, and {x=>x}
                try:
                    edits_in = dict(parsed.get('edits', {}) or {})
                    cleaned: Dict[str, Any] = {}
                    for k, v in edits_in.items():
                        try:
                            ks = str(k)
                            if not (ks.startswith('{') and ks.endswith('}') and '=>' in ks):
                                continue
                            body = ks[1:-1]
                            left, right = body.split('=>', 1)
                            # Drop only exact no-ops: {=>} or identical spans {x=>x}
                            if left == right:
                                continue
                            cleaned[ks] = v
                        except Exception:
                            continue
                    parsed['edits'] = cleaned
                    # Keep labels consistent with sanitized edits in the reasoning blob
                    try:
                        labels_in = list(parsed.get('labels', []) or [])
                        labels_in = [str(x).upper() for x in labels_in if str(x).upper() in {'TP','FP1','FP2','FP3'}]
                        if not cleaned:
                            parsed['labels'] = []
                        elif len(labels_in) > len(cleaned):
                            parsed['labels'] = labels_in[:len(cleaned)]
                        else:
                            parsed['labels'] = labels_in
                    except Exception:
                        parsed['labels'] = []
                except Exception:
                    pass
                model_json_for_reason = parsed  # keep parsed JSON for reasoning field
                edit_labels = [lbl for lbl in parsed.get('labels', []) if lbl in {'TP','FP1','FP2','FP3'}]
                # If we removed all edits during sanitation, do not retain labels
                if parsed.get('edits') == {}:
                    edit_labels = []
                # No salvage parsing of labels from raw content to avoid misleading outputs
                missed_error = parsed.get('missed_error', False)
                parsed_label = compute_sentence_label(src, tgt, missed_error, edit_labels)
            elif judge == 'feedback':
                # Feedback baseline returns 4-class TP/FP1/FP2/FP3; use dedicated parser
                try:
                    parsed_label = parse_tpfp_label(content)
                except Exception:
                    parsed_label = 'Error'
            else:
                # Sentence baseline: parse 6-class directly
                from judges.sentence.advanced import parse_baseline_response
                try:
                    parsed_label, _reason = parse_baseline_response(content)
                except Exception:
                    # Fallback to simple scan preserving TN/FN
                    parsed_label = 'TP'
                    up = (content or '').upper()
                    for lbl in ['TN','FN','FP1','FP2','FP3','TP']:
                        if lbl in up:
                            parsed_label = lbl
                            break
        else:
            parsed_label = 'Error'
        
        import json as _json
        base = {
            'tp_fp_label': parsed_label,
            'reasoning': (_json.dumps(model_json_for_reason, ensure_ascii=False) if model_json_for_reason is not None else (content if content else 'No response')),
            'writing_type': 'Personal',
            'total_tokens': tokens,
            'model': backend
        }
        # Flatten pricing/token usage for downstream aggregation
        try:
            enriched = add_pricing_to_result_dict(base, pricing_info) if ok else base
        except Exception:
            enriched = base
        return enriched
        
    except Exception as e:
        return {
            'tp_fp_label': 'Error',
            'reasoning': str(e)[:100],
            'writing_type': '',
            'total_tokens': 0,
            'pricing': {}
        }


def call_single_judge_for_row_detailed(judge: str, method: str, backend: str, lang: str,
                                       row: pd.Series, moderation: str = "off",
                                       opinions: Optional[str] = None,
                                       optimization: bool = False) -> Dict[str, Any]:
    """
    PRODUCTION call_single_judge_for_row_detailed with in-process optimization.
    Falls back to subprocess if in-process fails or is unavailable.
    """
    # In-process path: driven solely by the caller's optimization flag
    use_in_process = IN_PROCESS_AVAILABLE and bool(optimization)
    # Disable in-process for all edit methods to avoid coupling to advanced.py
    if use_in_process and judge == 'edit':
        use_in_process = False
    if use_in_process:
        try:
            src = str(row.get('src', ''))
            tgt = str(row.get('tgt', ''))
            # Pass precomputed alignment when available to avoid per-row recomputation
            try:
                aligned_text = (
                    str(row.get('aligned', ''))
                    or str(row.get('aligned_sentence', ''))
                    or str(row.get('alert', ''))
                )
            except Exception:
                aligned_text = ''
            api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
            
            result = call_judge_in_process(
                judge, method, src, tgt, backend, lang, 
                moderation=(moderation == 'on'), 
                api_token=api_token,
                aligned_text=aligned_text
            )
            
            # Convert to expected format
            out = {
                'label': normalize_label(result.get('tp_fp_label', 'Error')),
                'reasoning': result.get('reasoning', ''),
                'writing_type': result.get('writing_type', ''),
                'total_tokens': result.get('total_tokens', 0),
            }
            # Pass through token/cost fields when present
            for k in ['input_tokens','output_tokens','reasoning_tokens','cached_tokens','total_cost_usd','model']:
                if k in result:
                    out[k] = result[k]
            return out
        except Exception:
            # Fall back to subprocess if in-process fails
            pass
    
    # Original subprocess implementation as fallback (baseline reliability)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_input:
        single_row = pd.DataFrame([row])
        single_row.to_csv(tmp_input.name, index=False)
        tmp_input_path = tmp_input.name
    try:
        judge_result = call_judge(
            judge=judge,
            method=method,
            backend=backend,
            lang=lang,
            input_file=tmp_input_path,
            workers=1,
            moderation=moderation,
            opinions=opinions
        )
        if len(judge_result) > 0:
            rec = judge_result.iloc[0]
            label = normalize_label(str(rec.get('tp_fp_label', 'Error')))
            reasoning = str(rec.get('reasoning', ''))
            # Prefer explicit columns if present
            writing = ''
            if 'writing_type_pred' in judge_result.columns:
                writing = str(rec.get('writing_type_pred') or '')
            elif 'writing_type' in judge_result.columns:
                writing = str(rec.get('writing_type') or '')
            if not writing:
                writing = parse_writing_type(reasoning)
            
            # Preserve all pricing and other fields from judge result
            result = {'label': label, 'reasoning': reasoning, 'writing_type': writing}
            
            # Copy pricing fields if present
            pricing_fields = ['input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_tokens', 
                            'total_cost_usd', 'model', 'input_cost_usd', 'output_cost_usd', 
                            'reasoning_cost_usd', 'cached_cost_usd']
            for field in pricing_fields:
                if field in judge_result.columns and pd.notna(rec.get(field)):
                    result[field] = rec.get(field)
                    
            return result
        else:
            return {'label': 'Error', 'reasoning': 'Judge returned empty result', 'writing_type': ''}
    except Exception as e:
        return {'label': 'Error', 'reasoning': str(e), 'writing_type': ''}
    finally:
        if os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)


def call_parallel_judges_for_row(judge: str, method: str, backends: List[str], 
                                lang: str, row: pd.Series, n_judges: int,
                                backend_offset: int = 0, moderation: str = "off",
                                opinions: Optional[str] = None) -> List[str]:
    """Call n_judges for a single row and return list of labels. Parallelized with early-exit."""
    def call_single_label(judge_idx: int) -> str:
        backend_idx = (backend_offset + judge_idx) % len(backends)
        backend = backends[backend_idx]
        return call_single_judge_for_row(judge, method, backend, lang, row, moderation, opinions)

    if n_judges <= 1:
        try:
            return [call_single_label(0)]
        except Exception:
            return ['Error']

    # Submit first two judges and check for unanimous early exit
    first_two = { _GLOBAL_JUDGE_EXECUTOR.submit(call_single_label, j): j for j in range(min(2, n_judges)) }
    partial: Dict[int, str] = {}
    for f in as_completed(first_two):
        j = first_two[f]
        try:
            partial[j] = f.result()
        except Exception:
            partial[j] = 'Error'
        if len(partial) == 2:
            labels = list(partial.values())
            if all(l != 'Error' for l in labels) and labels[0] == labels[1]:
                # Early unanimous agreement
                return labels
            break

    # Submit remaining judges
    futs = { _GLOBAL_JUDGE_EXECUTOR.submit(call_single_label, j): j for j in range(2, n_judges) }
    results: List[Optional[str]] = [None] * n_judges
    for idx, lab in partial.items():
        results[idx] = lab
    for f in as_completed(futs):
        j = futs[f]
        try:
            results[j] = f.result()
        except Exception:
            results[j] = 'Error'
    return [r or 'Error' for r in results]


def call_parallel_judges_for_row_detailed(judge: str, method: str, backends: List[str], 
                                          lang: str, row: pd.Series, n_judges: int,
                                          backend_offset: int = 0, moderation: str = "off",
                                          opinions: Optional[str] = None,
                                          optimization: bool = False) -> List[Dict[str, Any]]:
    """Call n_judges in parallel and return list of dicts. Early-exit on unanimous first-two agreement."""
    if os.getenv('DEBUG_ENSEMBLE'):
        print(f"call_parallel_judges_for_row_detailed: n_judges={n_judges}, backends={backends}", file=sys.stderr)
    
    def call_single(judge_idx):
        backend_idx = (backend_offset + judge_idx) % len(backends)
        backend = backends[backend_idx]
        if os.getenv('DEBUG_ENSEMBLE'):
            print(f"  Judge {judge_idx}: calling with backend {backend}", file=sys.stderr)
        return call_single_judge_for_row_detailed(
            judge, method, backend, lang, row,
            moderation=moderation, opinions=opinions, optimization=optimization
        )

    if n_judges <= 1:
        try:
            return [call_single(0)]
        except Exception:
            return [{'label':'Error','reasoning':'','writing_type':''}]

    # First two for potential early unanimous agreement
    first_two = { _GLOBAL_JUDGE_EXECUTOR.submit(call_single, j): j for j in range(min(2, n_judges)) }
    partial: Dict[int, Dict[str, Any]] = {}
    for f in as_completed(first_two):
        j = first_two[f]
        try:
            partial[j] = f.result()
        except Exception as e:
            partial[j] = {'label':'Error','reasoning':str(e),'writing_type':''}
        if len(partial) == 2:
            labs = [partial[0].get('label','Error'), partial[1].get('label','Error')]
            if all(l != 'Error' for l in labs) and labs[0] == labs[1]:
                return [partial[0], partial[1]]
            break

    # Submit remaining judges
    futs = { _GLOBAL_JUDGE_EXECUTOR.submit(call_single, j): j for j in range(2, n_judges) }
    results: List[Optional[Dict[str, Any]]] = [None] * n_judges
    for idx, rec in partial.items():
        results[idx] = rec
    for f in as_completed(futs):
        j = futs[f]
        try:
            results[j] = f.result()
        except Exception as e:
            results[j] = {'label':'Error','reasoning':str(e),'writing_type':''}
    return [r if r is not None else {'label':'Error','reasoning':'','writing_type':''} for r in results]


def process_rows_parallel(df: pd.DataFrame, process_row_func: Callable, 
                         workers: int = 50, desc: str = "Processing rows") -> List[Dict[str, Any]]:
    """Process DataFrame rows in parallel using ThreadPoolExecutor."""
    results = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        # Submit all rows for processing
        future_to_row = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            future = executor.submit(process_row_func, (idx, row))
            future_to_row[future] = idx
        
        # Collect results with consistent tqdm progress bar
        completed_results = [None] * len(df)
        for future in tqdm(
            as_completed(future_to_row),
            total=len(future_to_row),
            desc=desc,
            leave=True,
            disable=(os.getenv('QUIET_LOGS') == '1'),
        ):
            row_idx = future_to_row[future]
            completed_results[row_idx] = future.result()
        
        results = completed_results
    
    return results


def create_result_dict(row: pd.Series, idx: int, label: str, reasoning: str = "ENSEMBLE",
                       writing_type_pred: str = "", judge_outputs: list = None) -> Dict[str, Any]:
    """Create a standardized result dictionary, including predicted writing type and pricing if available."""
    # Coerce idx to int for optimized runners that rely on numeric indexing
    row_idx = row.get('idx', idx)
    try:
        row_idx_int = int(row_idx)
    except Exception:
        row_idx_int = idx
    result = {
        'idx': row_idx_int,
        'src': row.get('src', ''),
        'tgt': row.get('tgt', ''),
        'tp_fp_label': label,
        'reasoning': reasoning,
        'writing_type': writing_type_pred,
    }
    
    # Preserve gold labels if available in input row
    if 'tp_fp_label' in row and pd.notna(row.get('tp_fp_label')):
        result['gold_specialized'] = row.get('tp_fp_label')
    elif 'gold_specialized' in row and pd.notna(row.get('gold_specialized')):
        result['gold_specialized'] = row.get('gold_specialized')
    elif 'label' in row and pd.notna(row.get('label')):
        result['gold_specialized'] = row.get('label')
    
    # Preserve aligned_sentence if available (for alert format input)
    if 'aligned_sentence' in row and pd.notna(row.get('aligned_sentence')):
        result['aligned_sentence'] = row.get('aligned_sentence', '')
    
    # Aggregate pricing information from judge outputs if available
    if judge_outputs:
        from utils.pricing import PricingTracker
        
        # Collect pricing info from all judge outputs
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        total_cached_tokens = 0
        total_cost_usd = 0.0
        models_used = []
        
        for output in judge_outputs:
            if isinstance(output, dict):
                total_input_tokens += output.get('input_tokens', 0)
                total_output_tokens += output.get('output_tokens', 0)
                total_reasoning_tokens += output.get('reasoning_tokens', 0)
                total_cached_tokens += output.get('cached_tokens', 0)
                total_cost_usd += output.get('total_cost_usd', 0.0)
                if 'model' in output and output['model']:
                    models_used.append(output['model'])
        
        # Add aggregated pricing info if any costs were found
        if total_cost_usd > 0 or total_input_tokens > 0:
            result.update({
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'reasoning_tokens': total_reasoning_tokens,
                'cached_tokens': total_cached_tokens,
                'total_cost_usd': total_cost_usd,
                'model': models_used[0] if models_used else 'ensemble'
            })
    
    return result


def judge_row_with_filter(
    idx: int,
    row: pd.Series,
    *,
    judge: str,
    method: str,
    backends: List[str],
    lang: str,
    n_judges: int,
    moderation: str = "off",
    filter_cols: Optional[List[str]] = None,
    annotate: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Try multiple candidate target columns in order and keep the first that judges as TP/TN.

    - For each column in filter_cols, set row['tgt'] to that column's value and run the same
      judge aggregation used by ensembles.
    - If annotate=False: return a standard result dict for the first TP/TN, else None.
    - If annotate=True: record each attempted column's label into `{col}_label`, stop at
      the first TP/TN, and always return a result dict (no row is dropped). Unattempted
      columns get empty labels.

    This is designed for one-line integration inside ensembles:
        result = judge_row_with_filter(idx, row, judge=args.judge, method=args.method,
                                       backends=args.backends, lang=args.lang,
                                       n_judges=args.n_judges, moderation=args.moderation,
                                       filter_cols=args.filter, annotate=True)
    """
    if not filter_cols:
        return None

    # Accept either a proper list or a single space/comma-separated string
    try:
        if isinstance(filter_cols, list) and len(filter_cols) == 1 and isinstance(filter_cols[0], str):
            import re as _re
            if (',' in filter_cols[0]) or (' ' in filter_cols[0]):
                filter_cols = [c for c in _re.split(r'[\s,]+', filter_cols[0]) if c]
    except Exception:
        pass

    BAD = {"FP1", "FP2", "FP3", "FN", "Error"}

    if not annotate:
        for col in filter_cols:
            if col not in row or pd.isna(row.get(col)):
                continue
            candidate_tgt = str(row.get(col) or "").strip()
            if not candidate_tgt:
                continue

            temp_row = row.copy()
            temp_row['tgt'] = candidate_tgt

            judge_outputs = call_parallel_judges_for_row_detailed(
                judge=judge,
                method=method,
                backends=backends,
                lang=lang,
                row=temp_row,
                n_judges=n_judges,
                backend_offset=0,
                moderation=moderation
            )

            labels = [normalize_label(o.get('label', o.get('tp_fp_label', 'Error'))) for o in judge_outputs]
            labels = [l for l in labels if l != 'Error']
            final_label = aggregate_weighted(labels, {}) if labels else 'Error'

            if final_label in {"TP", "TN"}:
                return create_result_dict(
                    temp_row, idx, final_label,
                    judge_outputs[-1].get('reasoning', ''),
                    judge_outputs[-1].get('writing_type', ''),
                    judge_outputs
                )

        return None

    # annotate=True path: record per-column labels and always return a result dict
    labels_by_col: Dict[str, str] = {}
    reasons_by_col: Dict[str, str] = {}
    accepted_row = None
    accepted_label = None
    accepted_judge_outputs = None
    accepted_reasoning = ''
    accepted_writing_type = ''

    last_row = None
    last_label = None
    last_judge_outputs = None
    last_reasoning = ''
    last_writing_type = ''

    for col in filter_cols:
        label_key = f"{col}_label"

        if col not in row or pd.isna(row.get(col)):
            labels_by_col.setdefault(label_key, '')
            continue
        candidate_tgt = str(row.get(col) or "").strip()
        if not candidate_tgt:
            labels_by_col.setdefault(label_key, '')
            continue

        temp_row = row.copy()
        temp_row['tgt'] = candidate_tgt

        try:
            judge_outputs = call_parallel_judges_for_row_detailed(
                judge=judge,
                method=method,
                backends=backends,
                lang=lang,
                row=temp_row,
                n_judges=n_judges,
                backend_offset=0,
                moderation=moderation
            )
        except Exception:
            judge_outputs = []

        if not judge_outputs:
            labels_by_col[label_key] = 'Error'
            last_row = temp_row
            last_label = 'Error'
            last_judge_outputs = []
            last_reasoning = 'No judge outputs'
            last_writing_type = ''
            continue

        labels = [normalize_label(o.get('label', o.get('tp_fp_label', 'Error'))) for o in judge_outputs]
        labels = [l for l in labels if l != 'Error']
        final_label = aggregate_weighted(labels, {}) if labels else 'Error'

        labels_by_col[label_key] = final_label

        last_row = temp_row
        last_label = final_label
        last_judge_outputs = judge_outputs
        last = judge_outputs[-1] if judge_outputs else {'reasoning': '', 'writing_type': ''}
        last_reasoning = last.get('reasoning', '')
        last_writing_type = last.get('writing_type', '')
        # Store per-column reasoning when available
        reasons_by_col[f"{col}_reasoning"] = last_reasoning

        if final_label in {"TP", "TN"}:
            accepted_row = temp_row
            accepted_label = final_label
            accepted_judge_outputs = judge_outputs
            accepted_reasoning = last_reasoning
            accepted_writing_type = last_writing_type
            break

    # ensure all configured columns have a label key
    for col in filter_cols:
        labels_by_col.setdefault(f"{col}_label", '')

    if accepted_row is not None:
        out = create_result_dict(
            accepted_row, idx, accepted_label or 'Error',
            accepted_reasoning, accepted_writing_type,
            accepted_judge_outputs or []
        )
    elif last_row is not None:
        out = create_result_dict(
            last_row, idx, last_label or 'Error',
            last_reasoning, last_writing_type,
            last_judge_outputs or []
        )
    else:
        out = create_result_dict(row, idx, 'Error', 'No valid filter candidates', '', [])

    out.update(labels_by_col)
    # Attach per-column reasonings (best-effort)
    out.update(reasons_by_col)
    # Preserve original row columns (do not overwrite standard output fields)
    try:
        for k in row.index if hasattr(row, 'index') else list(getattr(row, 'keys', lambda: [])()):
            if k not in out:
                try:
                    out[k] = row.get(k)
                except Exception:
                    # For non-Series row types
                    out[k] = getattr(row, k, None)
    except Exception:
        pass
    return out


def print_filter_annotation_stats(df_results: pd.DataFrame, filter_cols: Optional[List[str]]) -> None:
    """Best-effort console stats for filter annotation columns.

    - Prints per-column attempted count and per-label distribution for `{col}_label`.
    - Prints distribution of the column where a TP/TN was first accepted.
    """
    try:
        if not filter_cols or df_results is None or len(df_results) == 0:
            return

        total = len(df_results)

        # Normalize filter list if passed as a single combined string
        if isinstance(filter_cols, list) and len(filter_cols) == 1 and isinstance(filter_cols[0], str):
            val = filter_cols[0]
            if (',' in val) or (' ' in val):
                import re as _re
                filter_cols = [c for c in _re.split(r'[\s,]+', val) if c]

        print("\nFilter Annotation Stats:")
        tried_cols: List[str] = []
        for col in filter_cols:
            key = f"{col}_label"
            if key in df_results.columns:
                tried_cols.append(col)
                series = df_results[key].fillna('').replace({None: ''})
                vc = series.value_counts()
                total_non_empty = int((series != '').sum())
                print(f"  {key}: attempted={total_non_empty}")
                for lbl, cnt in vc.items():
                    if lbl == '':
                        continue
                    pct = cnt / max(1, total) * 100
                    print(f"    {lbl}: {cnt} ({pct:.1f}%)")

        if tried_cols:
            accepted_counts = {col: 0 for col in tried_cols}
            for _, r in df_results.iterrows():
                for col in tried_cols:
                    v = str(r.get(f"{col}_label", '')).strip().upper()
                    if v in {'TP', 'TN'}:
                        accepted_counts[col] += 1
                        break
            print("  Accepted at column:")
            for col in tried_cols:
                cnt = accepted_counts[col]
                pct = cnt / max(1, total) * 100
                print(f"    {col}: {cnt} ({pct:.1f}%)")
    except Exception:
        # Stats printing is best-effort; avoid breaking the run
        pass

