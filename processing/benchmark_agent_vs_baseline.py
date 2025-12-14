#!/usr/bin/env python3
"""Benchmark comparison: Baseline vs Agent on SpanishFPs.csv"""

import os
import sys
import json
import time
import pandas as pd
from typing import Dict, List, Any

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges.edit.agent import call_single_judge_for_row_detailed as agent_judge
from judges.edit.baseline import compute_sentence_label, parse_json_response
from judges.edit.prompts import EDIT_LEVEL_JUDGE_PROMPT
from utils.errant_align import get_alignment_for_language
from utils.judge import build_numbered_prompt, call_model_with_pricing


def baseline_judge(row: pd.Series) -> Dict[str, Any]:
    """Baseline judge wrapper for comparison"""
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    # Get alignment
    alignment = str(row.get('alignment', ''))
    if not alignment or alignment == 'nan':
        alignment = get_alignment_for_language(src, tgt, language='es')
    
    # Build prompt
    prompt = build_numbered_prompt(EDIT_LEVEL_JUDGE_PROMPT, 'Spanish', src, tgt, alignment)
    
    # Call model
    api_token = os.getenv('API_TOKEN', '')
    ok, content, _, pricing_info = call_model_with_pricing(
        prompt, 'gpt-4o-mini', api_token=api_token, moderation=False
    )
    
    if not ok:
        return {
            'tp_fp_label': 'Error',
            'reasoning': 'LLM call failed',
            'writing_type': 'Unknown',
            'confidence': 0.0,
            'cost_estimate': 0.0,
            'tools_used': []
        }
    
    # Parse response
    parsed = parse_json_response(content)
    edit_labels = parsed['labels']
    missed_error = parsed['missed_error']
    
    # Compute sentence label
    sentence_label = compute_sentence_label(src, tgt, missed_error, edit_labels)
    
    return {
        'tp_fp_label': sentence_label,
        'reasoning': parsed.get('reasoning', ''),
        'writing_type': parsed.get('writing_type', 'General'),
        'confidence': 0.8,
        'cost_estimate': pricing_info.get('cost_breakdown', {}).get('total_cost_usd', 0.0),
        'tools_used': []
    }


def evaluate_judge(judge_func, df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Evaluate a judge on the dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    predictions = []
    total_cost = 0.0
    tools_used_count = {}
    
    for idx, row in df.iterrows():
        if idx >= 100:  # Limit to 100 samples
            break
            
        try:
            result = judge_func(row)
            predictions.append(result['tp_fp_label'])
            total_cost += result.get('cost_estimate', 0.0)
            
            # Track tools
            for tool in result.get('tools_used', []):
                tool_type = tool.split(':')[0] if ':' in tool else tool
                tools_used_count[tool_type] = tools_used_count.get(tool_type, 0) + 1
                
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            predictions.append('FP3')  # Default
            
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/100 samples...")
            
    elapsed = time.time() - start_time
    
    # Calculate metrics
    gold_labels = df['tp_fp_label'].iloc[:len(predictions)].tolist()
    
    # Binary accuracy (TP vs FP*)
    binary_pred = ['TP' if p == 'TP' else 'FP' for p in predictions]
    binary_gold = ['TP' if g == 'TP' else 'FP' for g in gold_labels]
    binary_acc = sum(p == g for p, g in zip(binary_pred, binary_gold)) / len(binary_pred)
    
    # 6-class accuracy
    six_class_acc = sum(p == g for p, g in zip(predictions, gold_labels)) / len(predictions)
    
    # Per-class accuracy
    class_metrics = {}
    for cls in ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']:
        cls_gold = [g for g in gold_labels if g == cls]
        cls_pred = [p for p, g in zip(predictions, gold_labels) if g == cls]
        if cls_gold:
            class_metrics[cls] = {
                'count': len(cls_gold),
                'accuracy': sum(p == cls for p in cls_pred) / len(cls_gold)
            }
            
    return {
        'name': name,
        'binary_accuracy': binary_acc,
        'six_class_accuracy': six_class_acc,
        'per_class': class_metrics,
        'total_cost': total_cost,
        'time_seconds': elapsed,
        'samples_per_second': len(predictions) / elapsed,
        'cost_per_1k': (total_cost / len(predictions)) * 1000,
        'tools_used': tools_used_count
    }


def main():
    """Run benchmark comparison"""
    
    # Load dataset
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data/eval/SpanishFPs.csv'
    )
    
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset size: {len(df)} samples")
    
    # Run baseline
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    
    os.environ['AGENT_BACKEND'] = 'gpt-4o-mini'
    baseline_results = evaluate_judge(baseline_judge, df, "Edit Baseline (gpt-4o-mini)")
    
    # Run agent without RAG
    print("\n" + "="*60)
    print("AGENT WITHOUT RAG")
    print("="*60)
    
    os.environ['AGENT_USE_RAG'] = '0'
    agent_no_rag_results = evaluate_judge(agent_judge, df, "Edit Agent (gpt-4o-mini, no RAG)")
    
    # Run agent with RAG
    print("\n" + "="*60)
    print("AGENT WITH RAG")
    print("="*60)
    
    os.environ['AGENT_USE_RAG'] = '1'
    agent_rag_results = evaluate_judge(agent_judge, df, "Edit Agent (gpt-4o-mini, with RAG)")
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    results = [baseline_results, agent_no_rag_results, agent_rag_results]
    
    print(f"\n{'Method':<30} {'Binary':<10} {'6-Class':<10} {'Cost/1K':<10} {'Time(s)':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<30} {r['binary_accuracy']:.3f}      {r['six_class_accuracy']:.3f}      ${r['cost_per_1k']:.2f}      {r['time_seconds']:.1f}")
        
    # Detailed per-class
    print("\n\nPer-Class Accuracy:")
    print(f"\n{'Method':<30} {'TP':<8} {'FP1':<8} {'FP2':<8} {'FP3':<8}")
    print("-" * 62)
    
    for r in results:
        row = f"{r['name'][:29]:<30}"
        for cls in ['TP', 'FP1', 'FP2', 'FP3']:
            if cls in r['per_class']:
                acc = r['per_class'][cls]['accuracy']
                row += f" {acc:.3f}  "
            else:
                row += " ---    "
        print(row)
        
    # Tools usage
    print("\n\nTools Usage (Agent with RAG):")
    for tool, count in agent_rag_results.get('tools_used', {}).items():
        print(f"  {tool}: {count} times")
        
    # Save results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'baseline': baseline_results,
            'agent_no_rag': agent_no_rag_results,
            'agent_with_rag': agent_rag_results
        }, f, indent=2)
    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    main()


import os
import sys
import json
import time
import pandas as pd
from typing import Dict, List, Any

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges.edit.agent import call_single_judge_for_row_detailed as agent_judge
from judges.edit.baseline import compute_sentence_label, parse_json_response
from judges.edit.prompts import EDIT_LEVEL_JUDGE_PROMPT
from utils.errant_align import get_alignment_for_language
from utils.judge import build_numbered_prompt, call_model_with_pricing


def baseline_judge(row: pd.Series) -> Dict[str, Any]:
    """Baseline judge wrapper for comparison"""
    src = str(row.get('src', ''))
    tgt = str(row.get('tgt', ''))
    
    # Get alignment
    alignment = str(row.get('alignment', ''))
    if not alignment or alignment == 'nan':
        alignment = get_alignment_for_language(src, tgt, language='es')
    
    # Build prompt
    prompt = build_numbered_prompt(EDIT_LEVEL_JUDGE_PROMPT, 'Spanish', src, tgt, alignment)
    
    # Call model
    api_token = os.getenv('API_TOKEN', '')
    ok, content, _, pricing_info = call_model_with_pricing(
        prompt, 'gpt-4o-mini', api_token=api_token, moderation=False
    )
    
    if not ok:
        return {
            'tp_fp_label': 'Error',
            'reasoning': 'LLM call failed',
            'writing_type': 'Unknown',
            'confidence': 0.0,
            'cost_estimate': 0.0,
            'tools_used': []
        }
    
    # Parse response
    parsed = parse_json_response(content)
    edit_labels = parsed['labels']
    missed_error = parsed['missed_error']
    
    # Compute sentence label
    sentence_label = compute_sentence_label(src, tgt, missed_error, edit_labels)
    
    return {
        'tp_fp_label': sentence_label,
        'reasoning': parsed.get('reasoning', ''),
        'writing_type': parsed.get('writing_type', 'General'),
        'confidence': 0.8,
        'cost_estimate': pricing_info.get('cost_breakdown', {}).get('total_cost_usd', 0.0),
        'tools_used': []
    }


def evaluate_judge(judge_func, df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Evaluate a judge on the dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    predictions = []
    total_cost = 0.0
    tools_used_count = {}
    
    for idx, row in df.iterrows():
        if idx >= 100:  # Limit to 100 samples
            break
            
        try:
            result = judge_func(row)
            predictions.append(result['tp_fp_label'])
            total_cost += result.get('cost_estimate', 0.0)
            
            # Track tools
            for tool in result.get('tools_used', []):
                tool_type = tool.split(':')[0] if ':' in tool else tool
                tools_used_count[tool_type] = tools_used_count.get(tool_type, 0) + 1
                
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            predictions.append('FP3')  # Default
            
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/100 samples...")
            
    elapsed = time.time() - start_time
    
    # Calculate metrics
    gold_labels = df['tp_fp_label'].iloc[:len(predictions)].tolist()
    
    # Binary accuracy (TP vs FP*)
    binary_pred = ['TP' if p == 'TP' else 'FP' for p in predictions]
    binary_gold = ['TP' if g == 'TP' else 'FP' for g in gold_labels]
    binary_acc = sum(p == g for p, g in zip(binary_pred, binary_gold)) / len(binary_pred)
    
    # 6-class accuracy
    six_class_acc = sum(p == g for p, g in zip(predictions, gold_labels)) / len(predictions)
    
    # Per-class accuracy
    class_metrics = {}
    for cls in ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']:
        cls_gold = [g for g in gold_labels if g == cls]
        cls_pred = [p for p, g in zip(predictions, gold_labels) if g == cls]
        if cls_gold:
            class_metrics[cls] = {
                'count': len(cls_gold),
                'accuracy': sum(p == cls for p in cls_pred) / len(cls_gold)
            }
            
    return {
        'name': name,
        'binary_accuracy': binary_acc,
        'six_class_accuracy': six_class_acc,
        'per_class': class_metrics,
        'total_cost': total_cost,
        'time_seconds': elapsed,
        'samples_per_second': len(predictions) / elapsed,
        'cost_per_1k': (total_cost / len(predictions)) * 1000,
        'tools_used': tools_used_count
    }


def main():
    """Run benchmark comparison"""
    
    # Load dataset
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data/eval/SpanishFPs.csv'
    )
    
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset size: {len(df)} samples")
    
    # Run baseline
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    
    os.environ['AGENT_BACKEND'] = 'gpt-4o-mini'
    baseline_results = evaluate_judge(baseline_judge, df, "Edit Baseline (gpt-4o-mini)")
    
    # Run agent without RAG
    print("\n" + "="*60)
    print("AGENT WITHOUT RAG")
    print("="*60)
    
    os.environ['AGENT_USE_RAG'] = '0'
    agent_no_rag_results = evaluate_judge(agent_judge, df, "Edit Agent (gpt-4o-mini, no RAG)")
    
    # Run agent with RAG
    print("\n" + "="*60)
    print("AGENT WITH RAG")
    print("="*60)
    
    os.environ['AGENT_USE_RAG'] = '1'
    agent_rag_results = evaluate_judge(agent_judge, df, "Edit Agent (gpt-4o-mini, with RAG)")
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    results = [baseline_results, agent_no_rag_results, agent_rag_results]
    
    print(f"\n{'Method':<30} {'Binary':<10} {'6-Class':<10} {'Cost/1K':<10} {'Time(s)':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<30} {r['binary_accuracy']:.3f}      {r['six_class_accuracy']:.3f}      ${r['cost_per_1k']:.2f}      {r['time_seconds']:.1f}")
        
    # Detailed per-class
    print("\n\nPer-Class Accuracy:")
    print(f"\n{'Method':<30} {'TP':<8} {'FP1':<8} {'FP2':<8} {'FP3':<8}")
    print("-" * 62)
    
    for r in results:
        row = f"{r['name'][:29]:<30}"
        for cls in ['TP', 'FP1', 'FP2', 'FP3']:
            if cls in r['per_class']:
                acc = r['per_class'][cls]['accuracy']
                row += f" {acc:.3f}  "
            else:
                row += " ---    "
        print(row)
        
    # Tools usage
    print("\n\nTools Usage (Agent with RAG):")
    for tool, count in agent_rag_results.get('tools_used', {}).items():
        print(f"  {tool}: {count} times")
        
    # Save results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'baseline': baseline_results,
            'agent_no_rag': agent_no_rag_results,
            'agent_with_rag': agent_rag_results
        }, f, indent=2)
    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    main()



