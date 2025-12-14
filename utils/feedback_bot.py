#!/usr/bin/env python3
"""
Utility functions for feedback bot - parsing alert formats and data processing.
"""

import re
import pandas as pd
from typing import Tuple, List, Dict, Optional
import ast


def parse_alert_format(alert: str) -> Tuple[str, str]:
    """
    Parse alert format from feedback CSV to extract original and corrected text.
    
    Format examples:
    - "(My mechanical arms were {mult tasking=>multitasking}"
    - "Yes allowing a private drone company in a "socialist state" is probably not a good idea and will lead to dysfunctions like circumventing regulation and {mobing=>moving}"
    - "Th ey were so excited about it and so emotional saying they love {=>the }company"
    - "The more you prod and prick, he will make his walls{ higher=>}."
    
    Args:
        alert: Alert string with format "text {original=>corrected} more text"
        
    Returns:
        Tuple of (original_text, corrected_text)
    """
    # Find all edit patterns in curly braces
    pattern = r'\{([^}]*?)=>([^}]*?)\}'
    matches = re.findall(pattern, alert)
    
    if not matches:
        # No edit found, return the original text as is
        return alert, alert
    
    # Start with the original alert text
    original = alert
    corrected = alert
    
    # Apply edits in reverse order to maintain indices
    for match in reversed(matches):
        orig_part, corr_part = match
        
        # Replace the edit pattern in original and corrected versions
        edit_pattern = f'{{{orig_part}=>{corr_part}}}'
        
        original = original.replace(edit_pattern, orig_part)
        corrected = corrected.replace(edit_pattern, corr_part)
    
    return original.strip(), corrected.strip()


def prepare_feedback_data(csv_path: str) -> pd.DataFrame:
    """
    Prepare feedback data for processing by parsing alert format.
    
    Args:
        csv_path: Path to feedback CSV file
        
    Returns:
        DataFrame with columns: idx, alert, src, tgt
    """
    df = pd.read_csv(csv_path)
    
    # Parse each alert to extract src and tgt
    parsed_data = []
    for idx, row in df.iterrows():
        alert = str(row['alert'])
        src, tgt = parse_alert_format(alert)
        
        parsed_data.append({
            'idx': idx,
            'alert': alert,
            'src': src,
            'tgt': tgt
        })
    
    return pd.DataFrame(parsed_data)


def prepare_alert_data(csv_path: str) -> pd.DataFrame:
    """
    Prepare alert format data for direct use (keeps aligned sentences).
    
    Args:
        csv_path: Path to feedback CSV file with alert format
        
    Returns:
        DataFrame with columns: idx, alert, src, tgt, aligned_sentence
    """
    df = pd.read_csv(csv_path)

    # PREFERRED Case: sanitized_* strict path (use when available to avoid multi-edit alert noise)
    # Use sanitized_sentence as src, apply sanitized_replacements to create tgt
    required_src = 'sanitized_sentence'
    required_rep = 'sanitized_replacements'
    begin_col = next((c for c in ['begin','begins','start','starts','span_begin','span_begins'] if c in df.columns), None)
    end_col = next((c for c in ['end','ends','finish','finishes','span_end','span_ends'] if c in df.columns), None)

    if all(c in df.columns for c in [required_src, required_rep]) and begin_col and end_col:
        def _as_list(v):
            if v is None:
                return []
            if isinstance(v, list):
                return v
            if isinstance(v, (int, float)):
                return [int(v)]
            s = str(v).strip()
            if s == '' or s.lower() in ['nan','none']:
                return []
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [parsed]
            except Exception:
                return [s]

        def _build_alert_from_spans(src: str, begins: List[int], ends: List[int], reps: List[str]) -> str:
            # Deterministic reconstruction; assume spans valid
            edits = sorted(zip(begins, ends, reps), key=lambda x: (int(x[0]), int(x[1])))
            pos = 0
            parts: List[str] = []
            for b, e, r in edits:
                b = int(b); e = int(e)
                new_text = '' if r is None else str(r)
                parts.append(src[pos:b])
                orig = src[b:e]
                parts.append(f"{{{orig}=>{new_text}}}")
                pos = e
            parts.append(src[pos:])
            return ''.join(parts)
        
        def _apply_replacements_to_text(src: str, begins: List[int], ends: List[int], reps: List[str]) -> str:
            # Apply replacements to create target text
            edits = sorted(zip(begins, ends, reps), key=lambda x: (int(x[0]), int(x[1])), reverse=True)
            result = src
            for b, e, r in edits:
                b = int(b); e = int(e)
                new_text = '' if r is None else str(r)
                result = result[:b] + new_text + result[e:]
            return result

        parsed_data = []
        qualifier_col = 'pname_qualifier' if 'pname_qualifier' in df.columns else None
        for idx, row in df.iterrows():
            if pd.isna(row.get(required_src)):
                continue
            src = str(row.get(required_src, ''))
            begins = _as_list(row.get(begin_col))
            ends = _as_list(row.get(end_col))
            reps = _as_list(row.get(required_rep))

            # If dataset encodes a single span (scalar begin/end), coerce to length-1 lists
            if begins and not isinstance(begins[0], (int, float)) and str(begins[0]).isdigit():
                begins = [int(begins[0])]
            if ends and not isinstance(ends[0], (int, float)) and str(ends[0]).isdigit():
                ends = [int(ends[0])]

            # If lengths mismatch (common in noisy refs), take the first span with the first replacement
            if not (len(begins) == len(ends) == len(reps)):
                if begins and ends and reps:
                    begins, ends, reps = [begins[0]], [ends[0]], [reps[0]]
                else:
                    continue

            # If dataset marks this as SingleEdit but multiple spans present, keep only the most salient one
            if len(begins) > 1 and qualifier_col:
                q = str(row.get(qualifier_col, '')).lower()
                if 'singleedit' in q or 'single_edit' in q or 'single' in q:
                    try:
                        # Choose the span with largest combined change
                        scores = []
                        src_len = len(src)
                        for b, e, r in zip(begins, ends, reps):
                            b = int(b); e = int(e)
                            orig_len = max(0, min(e, src_len) - max(0, b))
                            new_len = len(str(r) if r is not None else '')
                            scores.append(orig_len + new_len)
                        keep_i = max(range(len(scores)), key=lambda i: scores[i])
                        begins, ends, reps = [begins[keep_i]], [ends[keep_i]], [reps[keep_i]]
                    except Exception:
                        # Fallback: first span
                        begins, ends, reps = [begins[0]], [ends[0]], [reps[0]]

            try:
                # Create target text by applying replacements
                tgt = _apply_replacements_to_text(src, begins, ends, reps)
                # Create alignment format for display
                alert = _build_alert_from_spans(src, begins, ends, reps)
            except Exception:
                continue

            parsed_data.append({
                'idx': idx,
                'alert': alert,
                'src': src,
                'tgt': tgt,
                'aligned_sentence': alert
            })
        if parsed_data:
            return pd.DataFrame(parsed_data)

    # Fallback Case: alert or pre-aligned fields
    if 'alert' in df.columns or 'aligned_sentence' in df.columns or 'aligned' in df.columns:
        parsed_data = []
        for idx, row in df.iterrows():
            alert = str(row.get('alert', '')) if 'alert' in df.columns else ''
            src_from_alert, tgt_from_alert = parse_alert_format(alert) if alert else ('','')
            src = str(row.get('src', src_from_alert))
            tgt = str(row.get('tgt', tgt_from_alert))
            aligned_text = (
                str(row.get('aligned', ''))
                or str(row.get('aligned_sentence', ''))
                or alert
            )
            parsed_data.append({
                'idx': idx,
                'alert': alert,
                'src': src,
                'tgt': tgt,
                'aligned_sentence': aligned_text
            })
        return pd.DataFrame(parsed_data)

    # Otherwise error out

    # If neither 'alert' nor strict sanitized format is available, error out
    raise ValueError("Unsupported input: expected 'alert' column or sanitized format columns ('sanitized_sentence','sanitized_text','sanitized_replacements', and begin/end spans).")


def extract_edit_changes(alert: str) -> List[Dict[str, str]]:
    """
    Extract individual edit changes from an alert.
    
    Args:
        alert: Alert string with edit patterns
        
    Returns:
        List of dicts with 'original' and 'corrected' keys
    """
    pattern = r'\{([^}]*?)=>([^}]*?)\}'
    matches = re.findall(pattern, alert)
    
    changes = []
    for orig, corr in matches:
        changes.append({
            'original': orig,
            'corrected': corr,
            'type': classify_edit_type(orig, corr)
        })
    
    return changes


def classify_edit_type(original: str, corrected: str) -> str:
    """
    Classify the type of edit based on original and corrected text.
    
    Args:
        original: Original text part
        corrected: Corrected text part
        
    Returns:
        Edit type: 'insertion', 'deletion', 'substitution'
    """
    if not original and corrected:
        return 'insertion'
    elif original and not corrected:
        return 'deletion'
    elif original != corrected:
        return 'substitution'
    else:
        return 'no_change'


def analyze_processed_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze processed feedback data to extract statistics.
    
    Args:
        df: Processed DataFrame with TP/FP classifications
        
    Returns:
        Dictionary with analysis results
    """
    # Handle different column name formats
    tp_fp_col = None
    if 'tp_fp_label' in df.columns:
        tp_fp_col = 'tp_fp_label'
    elif 'pred_specialized' in df.columns:
        tp_fp_col = 'pred_specialized'
    else:
        raise ValueError("DataFrame must contain 'tp_fp_label' or 'pred_specialized' column")
    
    # Count classifications
    label_counts = df[tp_fp_col].value_counts()
    
    # Calculate percentages
    total = len(df)
    label_percentages = (label_counts / total * 100).round(2)
    
    # Writing type distribution if available
    writing_type_dist = {}
    writing_type_col = None
    if 'type_of_writing' in df.columns:
        writing_type_col = 'type_of_writing'
    elif 'predicted_writing_type' in df.columns:
        writing_type_col = 'predicted_writing_type'
    elif 'writing_type' in df.columns:
        writing_type_col = 'writing_type'
    
    if writing_type_col:
        writing_type_dist = df[writing_type_col].value_counts().to_dict()
    
    # Most common errors by writing type
    error_by_writing = {}
    if writing_type_col:
        for writing_type in df[writing_type_col].unique():
            if pd.notna(writing_type):
                subset = df[df[writing_type_col] == writing_type]
                error_counts = subset[tp_fp_col].value_counts()
                error_by_writing[writing_type] = error_counts.to_dict()
    
    return {
        'total_samples': total,
        'label_counts': label_counts.to_dict(),
        'label_percentages': label_percentages.to_dict(),
        'writing_type_distribution': writing_type_dist,
        'errors_by_writing_type': error_by_writing,
        'problematic_areas': identify_problematic_areas(df)
    }


def identify_problematic_areas(df: pd.DataFrame) -> List[Dict[str, any]]:
    """
    Identify the most problematic areas based on FP1/FP2/FP3 rates.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        List of problematic areas with statistics
    """
    problematic_areas = []
    
    # Handle different column name formats
    tp_fp_col = 'tp_fp_label' if 'tp_fp_label' in df.columns else 'pred_specialized'
    writing_type_col = None
    if 'type_of_writing' in df.columns:
        writing_type_col = 'type_of_writing'
    elif 'predicted_writing_type' in df.columns:
        writing_type_col = 'predicted_writing_type'
    elif 'writing_type' in df.columns:
        writing_type_col = 'writing_type'
    
    if not writing_type_col:
        return problematic_areas
    
    # Calculate minimum sample count (same as chart filtering)
    dataset_total = len(df) or 1
    min_count = max(5, int(0.10 * dataset_total))
    
    # Calculate FP rates by writing type
    for writing_type in df[writing_type_col].unique():
        if pd.notna(writing_type):
            subset = df[df[writing_type_col] == writing_type]
            total_count = len(subset)
            
            if total_count < min_count:  # Skip categories with too few samples (same as charts)
                continue
            
            fp_counts = subset[subset[tp_fp_col].isin(['FP1', 'FP2', 'FP3'])][tp_fp_col].value_counts()
            fp_rate = (fp_counts.sum() / total_count * 100).round(2)
            
            # Include all writing types that meet minimum count (no 30% threshold)
            problematic_areas.append({
                'writing_type': writing_type,
                'total_samples': total_count,
                'fp_rate': fp_rate,
                'fp_breakdown': fp_counts.to_dict(),
                'most_common_fp': fp_counts.index[0] if not fp_counts.empty else 'None'
            })
    
    # Sort by FP rate descending
    problematic_areas.sort(key=lambda x: x['fp_rate'], reverse=True)
    
    return problematic_areas


def sample_examples_by_category(df: pd.DataFrame, category: str, n_samples: int = 5) -> List[Dict[str, str]]:
    """
    Sample representative examples for a given TP/FP category.
    
    Args:
        df: Processed DataFrame
        category: TP/FP category (e.g., 'FP1', 'FP2', 'FP3', 'TP')
        n_samples: Number of samples to return
        
    Returns:
        List of example dictionaries
    """
    # Handle different column name formats
    tp_fp_col = 'tp_fp_label' if 'tp_fp_label' in df.columns else 'pred_specialized'
    category_data = df[df[tp_fp_col] == category]
    
    if len(category_data) == 0:
        return []
    
    # Sample random examples
    sampled = category_data.sample(n=min(n_samples, len(category_data)))
    
    examples = []
    for _, row in sampled.iterrows():
        # Handle different column name formats
        reasoning_col = 'reasoning'
        if 'reasoning_clean' in row:
            reasoning_col = 'reasoning_clean'
        
        writing_type_col = 'type_of_writing'
        if 'predicted_writing_type' in row:
            writing_type_col = 'predicted_writing_type'
        elif 'writing_type' in row:
            writing_type_col = 'writing_type'
        
        # Handle nan values properly
        reasoning_val = row.get(reasoning_col, row.get('reasoning', 'Not specified'))
        if pd.isna(reasoning_val) or str(reasoning_val).lower() in ['nan', 'none', '']:
            reasoning_val = 'Not specified'
            
        writing_type_val = row.get(writing_type_col, 'General')
        if pd.isna(writing_type_val) or str(writing_type_val).lower() in ['nan', 'none', '']:
            writing_type_val = 'General'
        
        example = {
            'src': str(row.get('src', '')),
            'tgt': str(row.get('tgt', '')),
            'reasoning': str(reasoning_val),
            'type_of_writing': str(writing_type_val)
        }
        if 'alert' in row:
            example['alert'] = str(row['alert'])
        # Preserve aligned_sentence if available (from alert format input)
        if 'aligned_sentence' in row and pd.notna(row.get('aligned_sentence')):
            example['aligned_sentence'] = str(row['aligned_sentence'])
        examples.append(example)
    
    return examples
