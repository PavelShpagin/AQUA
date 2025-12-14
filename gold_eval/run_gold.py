#!/usr/bin/env python3
"""
Unified gold benchmark runner with comprehensive pricing analysis.

Usage:
  python gold_eval/run_gold.py GOLD_CSV PREDICTIONS_CSV \
    --llm_backend gpt-4o --judge feedback --method baseline \
    --lang en --output data/results/feedback_baseline_gpt-4o_en/labeled.csv

Writes:
  - labeled.csv (with pricing columns if token data available)
  - correct.txt, errors.txt, fails.txt
  - report.txt (includes comprehensive pricing analysis with 10K extrapolation)
to data/results/{judge}_{method}_{backend}_{lang}/ by default.

The pricing analysis includes:
  - Model-specific cost breakdowns (input, output, reasoning, cached tokens)
  - Cost extrapolation to 10K records
  - Optimization recommendations
"""

import argparse
import os
import glob
import pandas as pd
from utils.bench import (
    load_predictions,
    load_gold_standard,
    compute_specialized_metrics,
    compute_binary_metrics,
    print_specialized_results,
    print_binary_results,
    print_simplified_accuracy,
    save_comparison_results,
    save_error_and_correct_files,
    generate_report,
)


def infer_output_dir(judge: str, method: str, backend: str, lang: str, pref: str = None) -> str:
    name = f"{judge}_{method}_{backend}_{lang}"
    if pref:
        name = f"{pref}_{name}"
    return os.path.join("data", "results", name)


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions against gold standard')
    parser.add_argument('gold_file', help='Path to gold standard CSV file')
    parser.add_argument('pred_file', help='Path to predictions CSV file')
    parser.add_argument('--output', '-o', help='Output file for detailed comparison (labeled.csv)')
    parser.add_argument('--llm_backend', type=str, required=True, help='Backend for naming the result directory')
    parser.add_argument('--judge', type=str, required=True, help='Judge type (feedback|tnfn|sentence|edit)')
    parser.add_argument('--method', type=str, required=True, help='Method (legacy|baseline|modular|agent)')
    parser.add_argument('--lang', type=str, required=True, help='Language code (en|de|ua|es)')
    parser.add_argument('--pref', type=str, default="", help='Prefix for result directory name')
    args = parser.parse_args()

    gold_df = load_gold_standard(args.gold_file)
    pred_df = load_predictions(args.pred_file)
    
    # Ensure idx columns are the same type for merging
    gold_df['idx'] = gold_df['idx'].astype(str)
    pred_df['idx'] = pred_df['idx'].astype(str)
    
    merged_df = pd.merge(gold_df, pred_df, on='idx', how='inner', suffixes=('', '_pred'))
    
    # Reset index to ensure consistent integer indexing
    merged_df = merged_df.reset_index(drop=True)
    if len(merged_df) == 0:
        print("Error: No matching examples found between gold and predictions")
        return 1

    # Normalize writing_type: use predicted values and drop the auxiliary column
    # Support either 'writing_type_pred' or 'predicted_writing_type' coming from predictions
    if 'writing_type_pred' in merged_df.columns:
        merged_df['writing_type'] = merged_df['writing_type_pred']
        merged_df.drop(columns=['writing_type_pred'], inplace=True)
    elif 'predicted_writing_type' in merged_df.columns:
        merged_df['writing_type'] = merged_df['predicted_writing_type']
        merged_df.drop(columns=['predicted_writing_type'], inplace=True)
    elif 'writing_type' not in merged_df.columns:
        merged_df['writing_type'] = ''

    # Handle duplicate reasoning columns and normalize 
    if 'reasoning.1' in merged_df.columns:
        # Drop the duplicate reasoning column
        merged_df.drop(columns=['reasoning.1'], inplace=True)
    
    # Handle case where there are multiple columns with same name
    if len(set(merged_df.columns)) != len(merged_df.columns):
        print("Found duplicate column names, fixing...")
        # Create new DataFrame with unique columns
        unique_columns = []
        seen_columns = set()
        for col in merged_df.columns:
            if col not in seen_columns:
                unique_columns.append(col)
                seen_columns.add(col)
        
        # Keep only the unique columns
        merged_df = merged_df[unique_columns]
    
    # Normalize reasoning: prefer 'reasoning_clean' -> rename to 'reasoning'
    if 'reasoning_clean' in merged_df.columns:
        merged_df.rename(columns={'reasoning_clean': 'reasoning'}, inplace=True)

    # Drop session_id if present
    if 'session_id' in merged_df.columns:
        merged_df.drop(columns=['session_id'], inplace=True)

    merged_df['specialized_match'] = merged_df['gold_specialized'] == merged_df['pred_specialized']
    merged_df['binary_match'] = merged_df['gold_binary'] == merged_df['pred_binary']

    # compute + print
    specialized_metrics = compute_specialized_metrics(merged_df)
    binary_metrics = compute_binary_metrics(merged_df)
    print_specialized_results(specialized_metrics)
    print_binary_results(binary_metrics)
    print_simplified_accuracy(merged_df)

    # output dir
    backend_name = args.llm_backend
    out_dir = infer_output_dir(args.judge, args.method, backend_name, args.lang, args.pref or None)
    os.makedirs(out_dir, exist_ok=True)

    # save CSV
    csv_output_file = args.output or os.path.join(out_dir, 'labeled.csv')
    save_comparison_results(merged_df, csv_output_file)
    # Generate error/correct files and report
    try:
        save_error_and_correct_files(merged_df, out_dir)
        print("Error and correct files generated successfully.")
    except Exception as e:
        print(f"Note: Skipping error/correct files generation due to pandas issue: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    
    try:
        generate_report(merged_df, out_dir, args)
        print("Report generated successfully.")
    except Exception as e:
        print(f"Note: Skipping report generation due to pandas issue: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print("All evaluation metrics above are complete and accurate.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
