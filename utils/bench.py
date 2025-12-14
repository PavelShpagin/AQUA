#!/usr/bin/env python3
"""
Benchmark metrics utilities (copied and modularized from legacy evaluator).

Functions here are intentionally self-contained to avoid importing legacy files.
They are used by gold_eval/run_gold.py to compute metrics and write reports.
"""

import os
import sys
import io
from contextlib import redirect_stdout
from collections import Counter
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from utils.pricing import PricingTracker, calculate_cost, generate_cost_report


def convert_specialized_to_binary(specialized_class: str) -> str:
    """
    Convert specialized labels to binary for GEC evaluation.
    Binary question: "Does this text need correction?"
    - TP/TN: Correct assessment → TP (correctly identified need/no-need for correction)
    - FP1/FP2/FP3/FN: Incorrect assessment → FP (incorrectly identified need for correction)
    """
    if specialized_class == 'TP':
        return 'TP'  # Correctly identified improvement needed
    elif specialized_class in ['FP1', 'FP2', 'FP3']:
        return 'FP'  # Incorrectly identified improvement needed
    elif specialized_class == 'TN':
        return 'TP'  # Correctly identified no improvement needed
    elif specialized_class == 'FN':
        return 'FP'  # Incorrectly identified no improvement needed
    else:
        return 'FP'


def convert_binary_to_specialized(binary_class: str, confidence_score: Optional[float] = None) -> str:
    if binary_class == 'TP':
        return 'TP'
    elif binary_class == 'FP':
        return 'FP2'
    else:
        return binary_class


def convert_judge_to_4class(judge_class: str) -> str:
    if judge_class == 'TP-S':
        return 'TP'
    elif judge_class == 'TP-W':
        return 'FP3'
    elif judge_class in ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']:
        return judge_class
    else:
        return 'FP2'


def clean_reasoning(text: str) -> str:
    if pd.isna(text) or not text:
        return ""
    cleaned = ' '.join(str(text).split())
    if len(cleaned) > 500:
        cleaned = cleaned[:500] + "..."
    return cleaned


def load_predictions(pred_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(pred_file)
        
        # Handle duplicate columns (e.g., reasoning, reasoning.1)
        if df.columns.duplicated().any():
            # Keep first occurrence of each column
            df = df.loc[:, ~df.columns.duplicated()]
        required_cols = ['idx', 'src', 'tgt']
        label_col = None
        for col_name in ['judge_label', 'tpfp_label', 'tp_fp_label', 'pred_specialized']:
            if col_name in df.columns:
                label_col = col_name
                break
        if not all(col in df.columns for col in required_cols) or label_col is None:
            raise ValueError(f"Prediction file must contain columns: {required_cols} and one of ['judge_label', 'tpfp_label', 'tp_fp_label', 'pred_specialized']")

        def _map_pred_label(x: str) -> str:
            sx = str(x).strip()
            lx = sx.lower()
            if lx in ('error', 'parameter_error', 'blocked_by_sensitivity', 'nan', ''):
                return 'Error'
            if sx in ['TP', 'FP']:
                return convert_binary_to_specialized(sx)
            return convert_judge_to_4class(sx)

        df['pred_specialized'] = df[label_col].apply(_map_pred_label)
        df['pred_binary'] = df['pred_specialized'].apply(convert_specialized_to_binary)

        if 'reasoning' in df.columns:
            df['reasoning_clean'] = df['reasoning'].apply(clean_reasoning)
        if 'writing_type' not in df.columns:
            # Not strictly required; keep for compatibility
            pass
        return df
    except Exception as e:
        print(f"Error loading prediction file {pred_file}: {e}")
        sys.exit(1)


def load_gold_standard(gold_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(gold_file)
        # Allow Spanish gold with 'label' column; normalize to 'tp_fp_label'
        if 'tp_fp_label' not in df.columns and 'label' in df.columns:
            df = df.rename(columns={'label': 'tp_fp_label'})
        required_cols = ['idx', 'src', 'tgt', 'tp_fp_label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Gold file must contain columns: {required_cols}")
        df['gold_specialized'] = df['tp_fp_label']
        df['gold_binary'] = df['gold_specialized'].apply(convert_specialized_to_binary)
        if 'writing_type' not in df.columns:
            # Optional
            pass
        return df
    except Exception as e:
        print(f"Error loading gold file {gold_file}: {e}")
        sys.exit(1)


def _filter_successful(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return df[
        ~df[pred_col].isin(['Error', 'error', 'nan']) & df[pred_col].notna()
    ]


def compute_specialized_metrics(merged_df: pd.DataFrame) -> Dict[str, Any]:
    successful_df = _filter_successful(merged_df, 'pred_specialized')
    print(f"\nEvaluation Summary:")
    print(f"  Total examples: {len(merged_df)}")
    print(f"  Successful evaluations: {len(successful_df)}")
    print(f"  Failed evaluations: {len(merged_df) - len(successful_df)}")
    print(f"  Success rate: {len(successful_df)/len(merged_df)*100:.1f}%")
    if len(successful_df) == 0:
        return {
            'accuracy': 0.0,
            'per_class': {},
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'micro_precision': 0.0,
            'micro_recall': 0.0,
            'micro_f1': 0.0,
            'all_labels': []
        }
    # Get all labels, filtering out NaN values and converting to strings
    gold_labels = set(str(label) for label in successful_df['gold_specialized'].unique() 
                     if pd.notna(label) and label != 'nan')
    pred_labels = set(str(label) for label in successful_df['pred_specialized'].unique() 
                     if pd.notna(label) and label != 'nan')
    all_labels = sorted(gold_labels | pred_labels)
    accuracy = (successful_df['gold_specialized'] == successful_df['pred_specialized']).mean()
    results = {}
    for label in all_labels:
        # Convert columns to strings for consistent comparison
        gold_str = successful_df['gold_specialized'].astype(str)
        pred_str = successful_df['pred_specialized'].astype(str)
        
        tp = len(successful_df[(gold_str == label) & (pred_str == label)])
        fp = len(successful_df[(gold_str != label) & (pred_str == label)])
        fn = len(successful_df[(gold_str == label) & (pred_str != label)])
        tn = len(successful_df[(gold_str != label) & (pred_str != label)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
    macro_precision = sum(results[label]['precision'] for label in all_labels) / len(all_labels)
    macro_recall = sum(results[label]['recall'] for label in all_labels) / len(all_labels)
    macro_f1 = sum(results[label]['f1'] for label in all_labels) / len(all_labels)
    total_tp = sum(results[label]['tp'] for label in all_labels)
    total_fp = sum(results[label]['fp'] for label in all_labels)
    total_fn = sum(results[label]['fn'] for label in all_labels)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    return {
        'accuracy': accuracy,
        'per_class': results,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'all_labels': all_labels,
    }


def compute_binary_metrics(merged_df: pd.DataFrame) -> Dict[str, Any]:
    successful_df = _filter_successful(merged_df, 'pred_binary')
    if len(successful_df) == 0:
        return {
            'accuracy': 0.0,
            'per_class': {},
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'micro_precision': 0.0,
            'micro_recall': 0.0,
            'micro_f1': 0.0,
            'binary_labels': []
        }
    binary_labels = sorted(set(successful_df['gold_binary'].unique()) | set(successful_df['pred_binary'].unique()))
    accuracy = (successful_df['gold_binary'] == successful_df['pred_binary']).mean()
    results = {}
    for label in binary_labels:
        tp = len(successful_df[(successful_df['gold_binary'] == label) & (successful_df['pred_binary'] == label)])
        fp = len(successful_df[(successful_df['gold_binary'] != label) & (successful_df['pred_binary'] == label)])
        fn = len(successful_df[(successful_df['gold_binary'] == label) & (successful_df['pred_binary'] != label)])
        tn = len(successful_df[(successful_df['gold_binary'] != label) & (successful_df['pred_binary'] != label)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
    macro_precision = sum(results[label]['precision'] for label in binary_labels) / len(binary_labels)
    macro_recall = sum(results[label]['recall'] for label in binary_labels) / len(binary_labels)
    macro_f1 = sum(results[label]['f1'] for label in binary_labels) / len(binary_labels)
    total_tp = sum(results[label]['tp'] for label in binary_labels)
    total_fp = sum(results[label]['fp'] for label in binary_labels)
    total_fn = sum(results[label]['fn'] for label in binary_labels)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    return {
        'accuracy': accuracy,
        'per_class': results,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'binary_labels': binary_labels,
    }


def print_specialized_results(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("4-CLASS/6-CLASS TP/FP CLASSIFICATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nOverall F1-Score: {metrics['macro_f1']:.3f}")
    print(f"Overall Accuracy: {metrics.get('accuracy', 0.0):.3f}")
    
    print(f"\nPer-Class Results:")
    
    # Filter to only show valid classification labels (remove junk descriptive strings)
    valid_labels = {'TP', 'FP', 'FP1', 'FP2', 'FP3', 'TN', 'FN'}
    filtered_labels = [label for label in metrics['all_labels'] if label in valid_labels]
    
    for label in filtered_labels:
        if label in metrics['per_class']:
            result = metrics['per_class'][label]
            # Calculate per-class accuracy
            per_class_acc = result.get('accuracy', result['f1'])  # Use F1 as fallback
            print(f"  {label}: F1={result['f1']:.3f}, Acc={per_class_acc:.3f}, Support={result['support']}")
    
    # Show warning if junk labels were filtered out
    junk_labels = [label for label in metrics['all_labels'] if label not in valid_labels]
    if junk_labels:
        print(f"\nWARN: Filtered out {len(junk_labels)} invalid labels from corrupted CSV data")


def print_binary_results(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("BINARY TP/FP CLASSIFICATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall F1-Score: {metrics['macro_f1']:.3f}")
    print(f"Overall Accuracy: {metrics.get('accuracy', 0.0):.3f}")


def save_comparison_results(merged_df: pd.DataFrame, output_file: str) -> None:
    # Ensure normalization for writing_type and reasoning
    df = merged_df.copy()
    if 'writing_type_pred' in df.columns:
        df['writing_type'] = df['writing_type_pred']
        df.drop(columns=['writing_type_pred'], inplace=True)
    if 'predicted_writing_type' in df.columns:
        df['writing_type'] = df['predicted_writing_type']
        df.drop(columns=['predicted_writing_type'], inplace=True)
    if 'reasoning_clean' in df.columns and 'reasoning' not in df.columns:
        df.rename(columns={'reasoning_clean': 'reasoning'}, inplace=True)
    if 'session_id' in df.columns:
        df.drop(columns=['session_id'], inplace=True)

    # Add pricing columns if token usage data is available
    if any(col in df.columns for col in ['input_tokens', 'output_tokens', 'total_tokens', 'tokens_num']):
        # Find the model column
        model_col = None
        for col_name in ['model', 'llm_backend', 'backend']:
            if col_name in df.columns:
                model_col = col_name
                break
        
        # Note: Pricing columns are added during judge processing now

    comparison_cols = ['idx', 'src', 'tgt', 'gold_specialized', 'pred_specialized', 'gold_binary', 'pred_binary', 'specialized_match', 'binary_match']
    if 'reasoning' in df.columns:
        comparison_cols.append('reasoning')
    if 'writing_type' in df.columns:
        comparison_cols.append('writing_type')
    if 'tags' in df.columns:
        comparison_cols.append('tags')
    
    # Add pricing columns to output if they exist
    pricing_cols = ['input_cost_usd', 'output_cost_usd', 'reasoning_cost_usd', 'cached_cost_usd', 'total_cost_usd']
    for col in pricing_cols:
        if col in df.columns:
            comparison_cols.append(col)
    
    # Also include token usage columns if available
    token_cols = ['input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_tokens', 'total_tokens']
    for col in token_cols:
        if col in df.columns:
            comparison_cols.append(col)
    
    comparison_df = df[comparison_cols]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    comparison_df.to_csv(output_file, index=False)
    print(f"\nDetailed comparison saved to: {output_file}")


def _failed_mask(merged_df: pd.DataFrame) -> pd.Series:
    rc_col = 'reasoning_clean' if 'reasoning_clean' in merged_df.columns else ('reasoning' if 'reasoning' in merged_df.columns else None)
    
    # Use numpy arrays to avoid pandas alignment issues
    import numpy as np
    
    pred_na = merged_df['pred_specialized'].isna().values
    pred_error = merged_df['pred_specialized'].isin(['Error', 'error', 'nan', '', 'parameter_error', 'blocked_by_sensitivity']).values
    
    if rc_col:
        # Access the column directly, but handle potential DataFrame/Series issues
        try:
            rc_series = merged_df[rc_col]
            # If we get a DataFrame instead of Series (due to duplicate columns), take first column
            if isinstance(rc_series, pd.DataFrame):
                rc_series = rc_series.iloc[:, 0]
            
            rc_na = rc_series.isna().values
            rc_empty = (rc_series == '').values
            rc_error = (rc_series == 'Error').values
            
            combined = pred_na | pred_error | rc_na | rc_empty | rc_error
        except Exception as e:
            print(f"Warning: Failed to process {rc_col} column: {e}")
            # Fallback to just pred operations
            combined = pred_na | pred_error
    else:
        combined = pred_na | pred_error
    
    # Return as pandas Series with proper index
    return pd.Series(combined, index=merged_df.index)


def save_error_and_correct_files(merged_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle duplicate columns - keep only the first occurrence
    if merged_df.columns.duplicated().any():
        # Get unique column names, keeping first occurrence
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Use simple boolean indexing to avoid pandas alignment issues
    failed_mask = _failed_mask(merged_df)
    failed_evaluations = merged_df[failed_mask]
    successful_df = merged_df[~failed_mask]
    
    # Create correct and incorrect predictions DataFrames using a simpler method
    correct_predictions = successful_df[successful_df['gold_specialized'] == successful_df['pred_specialized']]
    incorrect_predictions = successful_df[successful_df['gold_specialized'] != successful_df['pred_specialized']]

    correct_file = os.path.join(output_dir, 'correct.txt')
    with open(correct_file, 'w', encoding='utf-8') as f:
        f.write("CORRECT PREDICTIONS - DETAILED DEBUG\n")
        f.write("="*80 + "\n\n")
        for i, (idx, row) in enumerate(correct_predictions.iterrows()):
            f.write(f"Example {i+1} (idx: {row['idx']}):\n")
            f.write("-"*60 + "\n")
            f.write(f"  Source: {row['src']}\n")
            f.write(f"  Target: {row['tgt']}\n")
            f.write(f"  Gold Label: {row['gold_specialized']}\n")
            f.write(f"  Predicted Label: {row['pred_specialized']}\n")
            
            # Language detection info
            if 'detected_language' in row and pd.notna(row['detected_language']):
                f.write(f"  Detected Language: {row['detected_language']}\n")
            
            # Writing type
            if 'writing_type' in row and pd.notna(row['writing_type']) and str(row['writing_type']).strip():
                f.write(f"  Writing Type: {row['writing_type']}\n")
            
            # Aligned sentence (for feedback judge)
            if 'aligned_sentence' in row and pd.notna(row['aligned_sentence']) and str(row['aligned_sentence']).strip():
                f.write(f"  Aligned Sentence: {row['aligned_sentence']}\n")
            
            # Scores for modular judge
            if 'nonsense_score' in row and pd.notna(row['nonsense_score']):
                f.write(f"  Nonsense Score: {row['nonsense_score']}\n")
            if 'meaning_change_score' in row and pd.notna(row['meaning_change_score']):
                f.write(f"  Meaning Change Score: {row['meaning_change_score']}\n")
            if 'quality_score' in row and pd.notna(row['quality_score']):
                f.write(f"  Quality Score: {row['quality_score']}\n")
            if 'modular_scores' in row and pd.notna(row['modular_scores']) and str(row['modular_scores']).strip():
                f.write(f"  Modular Scores: {row['modular_scores']}\n")
            
            # Individual judges for ensemble
            if 'individual_judges' in row and pd.notna(row['individual_judges']) and str(row['individual_judges']).strip():
                f.write(f"  Individual Judges: {row['individual_judges']}\n")
            
            # Reasoning/Output
            if 'reasoning' in row and pd.notna(row['reasoning']) and str(row['reasoning']).strip():
                f.write(f"\n  Model Reasoning:\n")
                f.write("  " + "-"*40 + "\n")
                reasoning_text = str(row['reasoning']).replace('\n', '\n  ')
                f.write(f"  {reasoning_text}\n")
            
            # Debug: Show all model/tool outputs if available
            debug_fields = ['nonsense_output', 'meaning_change_output', 'quality_output', 
                           'tnfn_output', 'grammar_rag_output', 'web_search_output',
                           'model1_output', 'model2_output', 'agent_history']
            
            debug_found = False
            for field in debug_fields:
                if field in row and pd.notna(row[field]) and row[field]:
                    if not debug_found:
                        f.write(f"\n  Debug - Model/Tool Outputs:\n")
                        f.write("  " + "-"*40 + "\n")
                        debug_found = True
                    f.write(f"  {field}: {str(row[field])[:500]}\n")
            
            # Also check for reasoning_clean field
            if 'reasoning_clean' in row and pd.notna(row['reasoning_clean']) and str(row['reasoning_clean']).strip():
                f.write(f"\n  Clean Reasoning:\n")
                f.write("  " + "-"*40 + "\n")
                reasoning_text = str(row['reasoning_clean']).replace('\n', '\n  ')
                f.write(f"  {reasoning_text}\n")
            
            # Prompt (if available)
            if 'prompt' in row and pd.notna(row['prompt']) and str(row['prompt']).strip():
                f.write(f"\n  Full Prompt (first 2000 chars):\n")
                f.write("  " + "-"*40 + "\n")
                prompt_text = str(row['prompt'])[:2000].replace('\n', '\n  ')
                f.write(f"  {prompt_text}\n")
                if len(str(row['prompt'])) > 2000:
                    f.write(f"  ... [truncated {len(str(row['prompt'])) - 2000} chars]\n")
            
            # Model-specific outputs for sentence baseline
            if 'model1_output' in row and pd.notna(row['model1_output']):
                f.write(f"\n  Model 1 Output: {row['model1_output']}\n")
            if 'model2_output' in row and pd.notna(row['model2_output']):
                f.write(f"  Model 2 Output: {row['model2_output']}\n")
            
            # Pricing info
            if 'total_cost_usd' in row and pd.notna(row['total_cost_usd']):
                f.write(f"\n  Cost: ${row['total_cost_usd']:.6f}")
                if 'total_tokens' in row and pd.notna(row['total_tokens']):
                    f.write(f" ({row['total_tokens']} tokens)")
                f.write("\n")
            
            f.write("\n" + "="*60 + "\n\n")

    errors_file = os.path.join(output_dir, 'errors.txt')
    with open(errors_file, 'w', encoding='utf-8') as f:
        f.write("INCORRECT PREDICTIONS - DETAILED DEBUG\n")
        f.write("="*80 + "\n\n")
        
        # Group errors by type of mistake for analysis
        error_patterns = {}
        for idx, row in incorrect_predictions.iterrows():
            pattern = f"{row['gold_specialized']} -> {row['pred_specialized']}"
            if pattern not in error_patterns:
                error_patterns[pattern] = []
            error_patterns[pattern].append(row)
        
        # Summary of error patterns
        f.write("ERROR PATTERN SUMMARY:\n")
        f.write("-"*40 + "\n")
        for pattern, examples in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
            f.write(f"  {pattern}: {len(examples)} cases\n")
        f.write("\n" + "="*80 + "\n\n")
        
        for i, (idx, row) in enumerate(incorrect_predictions.iterrows()):
            f.write(f"Example {i+1} (idx: {row['idx']}):\n")
            f.write("-"*60 + "\n")
            f.write(f"  Source: {row['src']}\n")
            f.write(f"  Target: {row['tgt']}\n")
            f.write(f"  Gold Label: {row['gold_specialized']}\n")
            f.write(f"  Predicted Label: {row['pred_specialized']}\n")
            f.write(f"  ERROR TYPE: {row['gold_specialized']} -> {row['pred_specialized']}\n")
            
            # Language detection info
            if 'detected_language' in row and pd.notna(row['detected_language']):
                f.write(f"  Detected Language: {row['detected_language']}\n")
            
            # Writing type
            if 'writing_type' in row and pd.notna(row['writing_type']) and str(row['writing_type']).strip():
                f.write(f"  Writing Type: {row['writing_type']}\n")
            
            # Aligned sentence (for feedback judge)
            if 'aligned_sentence' in row and pd.notna(row['aligned_sentence']) and str(row['aligned_sentence']).strip():
                f.write(f"  Aligned Sentence: {row['aligned_sentence']}\n")
            
            # Scores for modular judge
            if 'nonsense_score' in row and pd.notna(row['nonsense_score']):
                f.write(f"  Nonsense Score: {row['nonsense_score']}\n")
            if 'meaning_change_score' in row and pd.notna(row['meaning_change_score']):
                f.write(f"  Meaning Change Score: {row['meaning_change_score']}\n")
            if 'quality_score' in row and pd.notna(row['quality_score']):
                f.write(f"  Quality Score: {row['quality_score']}\n")
            if 'modular_scores' in row and pd.notna(row['modular_scores']) and str(row['modular_scores']).strip():
                f.write(f"  Modular Scores: {row['modular_scores']}\n")
            
            # Individual judges for ensemble
            if 'individual_judges' in row and pd.notna(row['individual_judges']) and str(row['individual_judges']).strip():
                f.write(f"  Individual Judges: {row['individual_judges']}\n")
            
            # Reasoning/Output
            if 'reasoning' in row and pd.notna(row['reasoning']) and str(row['reasoning']).strip():
                f.write(f"\n  Model Reasoning:\n")
                f.write("  " + "-"*40 + "\n")
                reasoning_text = str(row['reasoning']).replace('\n', '\n  ')
                f.write(f"  {reasoning_text}\n")
            
            # Debug: Show all model/tool outputs if available
            debug_fields = ['nonsense_output', 'meaning_change_output', 'quality_output', 
                           'tnfn_output', 'grammar_rag_output', 'web_search_output',
                           'model1_output', 'model2_output', 'agent_history']
            
            debug_found = False
            for field in debug_fields:
                if field in row and pd.notna(row[field]) and row[field]:
                    if not debug_found:
                        f.write(f"\n  Debug - Model/Tool Outputs:\n")
                        f.write("  " + "-"*40 + "\n")
                        debug_found = True
                    f.write(f"  {field}: {str(row[field])[:500]}\n")
            
            # Also check for reasoning_clean field
            if 'reasoning_clean' in row and pd.notna(row['reasoning_clean']) and str(row['reasoning_clean']).strip():
                f.write(f"\n  Clean Reasoning:\n")
                f.write("  " + "-"*40 + "\n")
                reasoning_text = str(row['reasoning_clean']).replace('\n', '\n  ')
                f.write(f"  {reasoning_text}\n")
            
            # Prompt (if available)
            if 'prompt' in row and pd.notna(row['prompt']) and str(row['prompt']).strip():
                f.write(f"\n  Full Prompt (first 2000 chars):\n")
                f.write("  " + "-"*40 + "\n")
                prompt_text = str(row['prompt'])[:2000].replace('\n', '\n  ')
                f.write(f"  {prompt_text}\n")
                if len(str(row['prompt'])) > 2000:
                    f.write(f"  ... [truncated {len(str(row['prompt'])) - 2000} chars]\n")
            
            # Model-specific outputs for sentence baseline
            if 'model1_output' in row and pd.notna(row['model1_output']):
                f.write(f"\n  Model 1 Output: {row['model1_output']}\n")
            if 'model2_output' in row and pd.notna(row['model2_output']):
                f.write(f"  Model 2 Output: {row['model2_output']}\n")
            
            # Pricing info
            if 'total_cost_usd' in row and pd.notna(row['total_cost_usd']):
                f.write(f"\n  Cost: ${row['total_cost_usd']:.6f}")
                if 'total_tokens' in row and pd.notna(row['total_tokens']):
                    f.write(f" ({row['total_tokens']} tokens)")
                f.write("\n")
            
            f.write("\n" + "="*60 + "\n\n")

    fails_file = os.path.join(output_dir, 'fails.txt')
    with open(fails_file, 'w', encoding='utf-8') as f:
        f.write("FAILED EVALUATIONS - DETAILED ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write("FAILURE CRITERIA:\n")
        f.write("- pred_specialized is 'Error', 'error', 'nan', empty, 'parameter_error', or 'blocked_by_sensitivity'\n")
        f.write("- reasoning_clean is missing, empty, or 'Error'\n\n")
        f.write(f"Total failed evaluations: {len(failed_evaluations)}\n")
        f.write(f"Total successful evaluations: {len(merged_df) - len(failed_evaluations)}\n")
        f.write(f"Failure rate: {len(failed_evaluations)/len(merged_df)*100:.1f}%\n\n")
        # Brief list
        for i, (idx, row) in enumerate(failed_evaluations.head(50).iterrows()):
            f.write(f"Example {i+1} (idx: {row['idx']}):\n")
            f.write(f"  Source: {str(row['src'])[:200]}\n")
            f.write(f"  Target: {str(row['tgt'])[:200]}\n")
            f.write(f"  Pred: {row.get('pred_specialized', 'NA')}\n")
            if 'reasoning_clean' in row:
                f.write(f"  Error Details: {str(row['reasoning_clean'])[:500]}\n")
            f.write("\n")

    print(f"\n=== OUTPUT FILES GENERATED ===")
    print(f"Output directory: {output_dir}")
    print(f"Correct predictions: {correct_file} ({len(correct_predictions)} examples)")
    print(f"Error predictions: {errors_file} ({len(incorrect_predictions)} examples)")
    print(f"Failed evaluations: {fails_file} ({len(failed_evaluations)} examples)")
    print(f"Total: {len(correct_predictions) + len(incorrect_predictions) + len(failed_evaluations)} examples")
    return correct_file, errors_file, fails_file


def print_simplified_accuracy(merged_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SIMPLIFIED BINARY ACCURACY (FP3→TP, FP1/FP2→FP)")
    print("=" * 60)
    successful_df = _filter_successful(merged_df, 'pred_specialized')
    if len(successful_df) == 0:
        print("No successful evaluations for simplified accuracy calculation.")
        return
    # Apply consistent simplified mapping: FP3/TP/TN → TP, FP1/FP2/FN → FP
    simplified_pred = ['TP' if p in ['FP3', 'TP', 'TN'] else 'FP' for p in successful_df['pred_specialized']]
    simplified_gold = ['TP' if g in ['FP3', 'TP', 'TN'] else 'FP' for g in successful_df['gold_specialized']]
    simplified_correct = sum(1 for g, p in zip(simplified_gold, simplified_pred) if g == p)
    simplified_accuracy = simplified_correct / len(simplified_gold)
    print(f"\nSimplified Accuracy: {simplified_accuracy:.3f} ({simplified_correct}/{len(simplified_gold)})")


def extract_pricing_data_from_df(df: pd.DataFrame) -> PricingTracker:
    """Extract pricing information from a dataframe with token usage data"""
    tracker = PricingTracker()
    
    # Try to extract model information from various possible columns
    model_col = None
    for col_name in ['model', 'llm_backend', 'backend']:
        if col_name in df.columns:
            model_col = col_name
            break
    
    if model_col is None:
        return tracker
    
    # Extract token usage data
    for _, row in df.iterrows():
        model = row.get(model_col, 'unknown')
        
        # Try various token column formats
        input_tokens = (
            row.get('input_tokens', 0) or
            row.get('prompt_tokens', 0) or
            0
        )
        output_tokens = (
            row.get('output_tokens', 0) or
            row.get('completion_tokens', 0) or
            0
        )
        reasoning_tokens = row.get('reasoning_tokens', 0)
        cached_tokens = (
            row.get('cached_tokens', 0) or
            row.get('prompt_tokens_cached', 0) or
            0
        )
        
        # Fallback to total tokens if individual counts not available
        if input_tokens == 0 and output_tokens == 0:
            total_tokens = (
                row.get('total_tokens', 0) or
                row.get('tokens_num', 0) or
                0
            )
            # Rough estimation: assume 70% input, 30% output if we only have total
            if total_tokens > 0:
                input_tokens = int(total_tokens * 0.7)
                output_tokens = int(total_tokens * 0.3)
        
        if input_tokens > 0 or output_tokens > 0:
            tracker.add_usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens
            )
    
    return tracker


def generate_report(merged_df: pd.DataFrame, output_dir: str, args) -> str:
    report_file = os.path.join(output_dir, 'report.txt')
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        print("="*80)
        print("TP/FP EVALUATION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Gold file: {args.gold_file}")
        print(f"Prediction file: {args.pred_file}")
        print(f"Output directory: {output_dir}")
        print(f"Backend: {getattr(args, 'llm_backend', 'Unknown')}")
        print(f"Method: {getattr(args, 'method', 'Unknown')}")
        print(f"Judge: {getattr(args, 'judge', 'Unknown')}")
        print("="*80)

        # Use numpy arrays to avoid pandas alignment issues  
        import numpy as np
        
        pred_na = merged_df['pred_specialized'].isna().values
        pred_error = merged_df['pred_specialized'].isin(['Error', 'error', 'nan', '', 'parameter_error', 'blocked_by_sensitivity']).values
        
        # Handle reasoning column with error handling
        rc_col = 'reasoning_clean' if 'reasoning_clean' in merged_df.columns else ('reasoning' if 'reasoning' in merged_df.columns else None)
        if rc_col:
            try:
                rc = merged_df[rc_col]
                # If we get a DataFrame instead of Series (due to duplicate columns), take first column
                if isinstance(rc, pd.DataFrame):
                    rc = rc.iloc[:, 0]
                    
                rc_na = rc.isna().values
                rc_empty = (rc == '').values
                rc_error = (rc == 'Error').values
                
                failed_mask_array = pred_na | pred_error | rc_na | rc_empty | rc_error
            except Exception as e:
                print(f"Warning: Failed to process {rc_col} column in report generation: {e}")
                failed_mask_array = pred_na | pred_error
        else:
            failed_mask_array = pred_na | pred_error
            
        failed_count = np.sum(failed_mask_array)
        successful_count = len(merged_df) - failed_count
        success_rate = (successful_count / len(merged_df)) * 100 if len(merged_df) > 0 else 0

        print("PROCESSING SUMMARY")
        print("="*80)
        print(f"Total examples processed: {len(merged_df)}")
        print(f"Successful evaluations: {successful_count}")
        print(f"Failed evaluations: {failed_count}")
        print(f"Success rate: {success_rate:.1f}%")
        print("="*80)
        print("NOTE: Accuracy metrics below are calculated ONLY on successful evaluations.")
        print("Failed evaluations are excluded from accuracy calculations and saved to fails.txt.")
        print("="*80)

        # Performance metrics
        specialized_metrics = compute_specialized_metrics(merged_df)
        print_specialized_results(specialized_metrics)
        binary_metrics = compute_binary_metrics(merged_df)
        print_binary_results(binary_metrics)
        print_simplified_accuracy(merged_df)

        # Pricing analysis
        print("\n")
        pricing_tracker = extract_pricing_data_from_df(merged_df)
        if pricing_tracker.total_requests > 0:
            report_lines = generate_cost_report(pricing_tracker, detailed=True)
            print('\n'.join(report_lines))
        else:
            print("=" * 80)
            print("PRICING ANALYSIS")
            print("=" * 80)
            print("No pricing data available in the results.")
            print("Note: Pricing tracking requires token usage information in the CSV.")
            print("=" * 80)

        print("\n" + "="*60)
        print("OUTPUT FILES")
        print("="*60)
        print(f"Benchmark results: {os.path.join(output_dir, 'labeled.csv')}")
        print(f"Correct predictions: {os.path.join(output_dir, 'correct.txt')}")
        print(f"Error predictions: {os.path.join(output_dir, 'errors.txt')}")
        print(f"Failed evaluations: {os.path.join(output_dir, 'fails.txt')}")
        print(f"This report: {report_file}")
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
    print(f"\nComprehensive report saved to: {report_file}")
    return report_file


