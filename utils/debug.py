#!/usr/bin/env python3
"""
Debug utilities for testing judge methods with sample data.
Provides shared functionality for debug mode across all judges.
"""

import os
import sys
import random
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

def load_debug_samples(lang: str = 'en', judge_type: str = 'feedback') -> List[Dict[str, Any]]:
    """
    Load debug samples for testing. Returns one sample per label category.
    
    Args:
        lang: Language code ('en' or 'ua')
        judge_type: Type of judge ('feedback', 'sentence', 'edit', 'tnfn')
    
    Returns:
        List of sample dictionaries with idx, src, tgt, and label
    """
    samples = []
    
    # For debug mode, always use SpanishFPs.csv which has diverse examples
    if judge_type in ['edit', 'sentence', 'feedback']:
        data_file = "data/eval/SpanishFPs.csv"
        label_column = 'label'
        labels_to_sample = ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']
        
        # If SpanishFPs.csv doesn't exist, fall back to original logic
        if not os.path.exists(data_file):
            print(f"Debug dataset {data_file} not found, falling back to gold datasets")
            # Original logic for fallback
            if judge_type == 'feedback':
                lang_code = 'ua' if lang == 'ua' else lang
                data_file = f"data/eval/gold_{lang_code}.csv"
                if not os.path.exists(data_file):
                    data_file = f"data/eval/gold_tp_fp3_fp2_fp1_en.csv"
                    label_column = 'tp_fp_label'
                    labels_to_sample = ['TP', 'FP1', 'FP2', 'FP3']
            else:
                lang_code = 'ua' if lang == 'ua' else lang
                data_file = f"data/eval/gold_{lang_code}.csv"
                
    elif judge_type == 'tnfn':
        # For TNFN judge
        lang_code = 'ua' if lang == 'ua' else lang
        data_file = f"data/eval/tnfn_{lang_code}.csv"
        label_column = 'tp_fp_label'
        labels_to_sample = ['TP', 'FP', 'TN', 'FN']
    else:
        # Default to SpanishFPs for unknown types
        data_file = "data/eval/SpanishFPs.csv"
        label_column = 'label'
        labels_to_sample = ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Warning: Data file {data_file} not found. Generating synthetic samples.")
        return generate_synthetic_samples(lang, labels_to_sample)
    
    # Load the data
    try:
        df = pd.read_csv(data_file)
        
        # Sample one item per label category
        for label in labels_to_sample:
            label_df = df[df[label_column] == label]
            if len(label_df) > 0:
                sample = label_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
                samples.append({
                    'idx': sample.get('idx', len(samples)),
                    'src': sample.get('src', ''),
                    'tgt': sample.get('tgt', ''),
                    'expected_label': label,
                    'language': lang
                })
            else:
                print(f"Warning: No samples found for label '{label}' in {data_file}")
    
    except Exception as e:
        print(f"Error loading data file {data_file}: {e}")
        return generate_synthetic_samples(lang, labels_to_sample)
    
    return samples

def generate_synthetic_samples(lang: str, labels: List[str]) -> List[Dict[str, Any]]:
    """
    Generate synthetic samples for testing when real data is not available.
    """
    synthetic_data = {
        'en': {
            'TP': {
                'src': "I have went to the store yesterday.",
                'tgt': "I went to the store yesterday."
            },
            'FP1': {
                'src': "The weather is nice today.",
                'tgt': "The weather are nice today."
            },
            'FP2': {
                'src': "She likes to read books.",
                'tgt': "She like to read book."
            },
            'FP3': {
                'src': "They are going to the park.",
                'tgt': "They is going to park."
            },
            'TN': {
                'src': "This sentence is correct.",
                'tgt': "This sentence is correct."
            },
            'FN': {
                'src': "He don't know nothing about it.",
                'tgt': "He don't know nothing about it."
            },
            'FP': {
                'src': "The cat is sleeping.",
                'tgt': "The cats is sleeping."
            }
        },
        'ua': {
            'TP': {
                'src': "Я пішов до магазину вчора.",
                'tgt': "Я ходив до магазину вчора."
            },
            'FP1': {
                'src': "Погода сьогодні гарна.",
                'tgt': "Погода сьогодні гарні."
            },
            'FP2': {
                'src': "Вона любить читати книги.",
                'tgt': "Вона любити читати книга."
            },
            'FP3': {
                'src': "Вони йдуть до парку.",
                'tgt': "Вони йде до парк."
            },
            'TN': {
                'src': "Це речення правильне.",
                'tgt': "Це речення правильне."
            },
            'FN': {
                'src': "Він не знає нічого про це.",
                'tgt': "Він не знає нічого про це."
            },
            'FP': {
                'src': "Кіт спить.",
                'tgt': "Коти спить."
            }
        }
    }
    
    samples = []
    data = synthetic_data.get(lang, synthetic_data['en'])
    
    for i, label in enumerate(labels):
        if label in data:
            samples.append({
                'idx': i,
                'src': data[label]['src'],
                'tgt': data[label]['tgt'],
                'expected_label': label,
                'language': lang
            })
    
    return samples

def log_debug_output(
    sample: Dict[str, Any],
    prompt: str,
    output: str,
    reasoning: str,
    predicted_label: str,
    model_name: str,
    judge_type: str,
    method: str,
    execution_time: float = 0.0,
    additional_info: Optional[Dict] = None,
    intermediate_results: Optional[Dict] = None
):
    """
    Log debug output in a structured format.
    
    Args:
        sample: The input sample dictionary
        prompt: The full prompt sent to the model
        output: The raw model output
        reasoning: Extracted reasoning from the output
        predicted_label: The predicted label
        model_name: Name of the model used
        judge_type: Type of judge (feedback, sentence, etc.)
        method: Method used (baseline, legacy, agent, etc.)
        execution_time: Time taken for execution
        additional_info: Any additional information to log
    """
    print("\n" + "="*80)
    print(f"DEBUG OUTPUT - {judge_type.upper()} JUDGE ({method})")
    print("="*80)
    
    print(f"\n📊 SAMPLE INFO:")
    print(f"  Language: {sample.get('language', 'unknown')}")
    print(f"  Expected Label: {sample.get('expected_label', 'unknown')}")
    print(f"  Sample Index: {sample.get('idx', 'N/A')}")
    
    print(f"\n📝 INPUT:")
    print(f"  Source: {sample['src'][:100]}..." if len(sample['src']) > 100 else f"  Source: {sample['src']}")
    print(f"  Target: {sample['tgt'][:100]}..." if len(sample['tgt']) > 100 else f"  Target: {sample['tgt']}")
    
    print(f"\n🤖 MODEL: {model_name}")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    
    print(f"\n📤 PROMPT (first 1000 chars):")
    print("-"*40)
    # Properly display prompt with newlines preserved
    prompt_display = prompt[:1000] + "..." if len(prompt) > 1000 else prompt
    print(prompt_display)
    
    print(f"\n📥 RAW OUTPUT:")
    print("-"*40)
    print(output[:1000] + "..." if len(output) > 1000 else output)
    
    # Show intermediate results if available (for modular/multi-step judges)
    if intermediate_results:
        print(f"\n🔄 INTERMEDIATE RESULTS:")
        print("-"*40)
        for step_name, step_result in intermediate_results.items():
            if isinstance(step_result, dict):
                print(f"  {step_name}:")
                for key, value in step_result.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  {step_name}: {step_result}")
    
    print(f"\n🎯 PREDICTED LABEL: {predicted_label}")
    print(f"✅ EXPECTED LABEL: {sample.get('expected_label', 'unknown')}")
    print(f"{'✓ CORRECT' if predicted_label == sample.get('expected_label') else '✗ INCORRECT'}")
    
    if reasoning and reasoning != output:
        print(f"\n💭 REASONING:")
        print("-"*40)
        print(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
    
    print("\n" + "="*80)

def save_debug_log(
    logs: List[Dict],
    judge_type: str,
    method: str,
    output_dir: str = "debug_logs"
):
    """
    Save debug logs to a file for later analysis.
    
    Args:
        logs: List of debug log dictionaries
        judge_type: Type of judge
        method: Method used
        output_dir: Directory to save logs
    """
    import numpy as np
    
    def convert_to_serializable(obj):
        """Convert numpy/pandas types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/debug_{judge_type}_{method}_{timestamp}.json"
    
    # Convert logs to serializable format
    serializable_logs = convert_to_serializable(logs)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_logs, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Debug logs saved to: {filename}")

def run_debug_test(
    judge_function,
    judge_type: str,
    method: str,
    backends: List[str],
    lang: str = 'en',
    additional_args: Optional[Dict] = None
) -> List[Dict]:
    """
    Run debug test with sample data for a specific judge.
    
    Args:
        judge_function: The judge function to test
        judge_type: Type of judge
        method: Method being tested
        backends: List of backend models to test
        lang: Language to test ('en', 'ua', or 'all' for both)
        additional_args: Additional arguments for the judge function
    
    Returns:
        List of debug results
    """
    import time
    
    debug_logs = []
    
    # Load samples for both English and Ukrainian
    langs_to_test = ['en', 'ua'] if lang == 'all' else [lang]
    
    for test_lang in langs_to_test:
        print(f"\n🌍 Testing {test_lang.upper()} samples...")
        
        # For feedback judge with Ukrainian, use gold_ua.csv if available
        if judge_type == 'feedback' and test_lang == 'ua' and os.path.exists('data/eval/gold_ua.csv'):
            # Load Ukrainian-specific data for feedback judge
            df = pd.read_csv('data/eval/gold_ua.csv')
            if 'tp_fp_label' in df.columns:
                samples = []
                for label in ['TP', 'FP1', 'FP2', 'FP3']:
                    label_df = df[df['tp_fp_label'] == label]
                    if len(label_df) > 0:
                        sample = label_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
                        samples.append({
                            'idx': sample.get('idx', len(samples)),
                            'src': sample.get('src', ''),
                            'tgt': sample.get('tgt', ''),
                            'expected_label': label,
                            'language': test_lang
                        })
                if not samples:
                    samples = load_debug_samples(test_lang, judge_type)
            else:
                samples = load_debug_samples(test_lang, judge_type)
        else:
            samples = load_debug_samples(test_lang, judge_type)
        
        if not samples:
            print(f"Warning: No samples available for {test_lang}")
            continue
        
        print(f"Found {len(samples)} samples for {test_lang.upper()}")
        
        # Test ALL samples (one per label category) instead of just one random
        
        for backend in backends:
            print(f"\n🔧 Testing with backend: {backend}")
            
            # Test each sample (one per label category)
            for sample in samples:
                # Prepare arguments for judge function
                args = {
                    'src': sample['src'],
                    'tgt': sample['tgt'],
                    'llm_backend': backend,
                    'lang': test_lang
                }
                
                if additional_args:
                    args.update(additional_args)
                
                # Run the judge function
                start_time = time.time()
                try:
                    result = judge_function(**args)
                    execution_time = time.time() - start_time
                    
                    # Log the debug output
                    log_debug_output(
                        sample=sample,
                        prompt=result.get('prompt', 'N/A'),
                        output=result.get('output', 'N/A'),
                        reasoning=result.get('reasoning', 'N/A'),
                        predicted_label=result.get('label', 'N/A'),
                        model_name=backend,
                        judge_type=judge_type,
                        method=method,
                        execution_time=execution_time,
                        additional_info=None,  # Don't display additional_info
                        intermediate_results=result.get('intermediate_results', None)
                    )
                    
                    # Store for saving (but without cost info in logs)
                    debug_logs.append({
                        'language': test_lang,
                        'sample': sample,
                        'backend': backend,
                        'result': {
                            'prompt': result.get('prompt', 'N/A'),
                            'output': result.get('output', 'N/A'),
                            'reasoning': result.get('reasoning', 'N/A'),
                            'label': result.get('label', 'N/A')
                        },
                        'execution_time': execution_time,
                        'correct': result.get('label') == sample.get('expected_label')
                    })
                    
                except Exception as e:
                    print(f"❌ Error running judge: {e}")
                    debug_logs.append({
                        'language': test_lang,
                        'sample': sample,
                        'backend': backend,
                        'error': str(e),
                        'execution_time': time.time() - start_time
                    })
    
    return debug_logs

def is_debug_mode(args: Any = None) -> bool:
    """
    Check if debug mode is enabled from args or environment.
    
    Args:
        args: Parsed arguments object with potential debug attribute
    
    Returns:
        Boolean indicating if debug mode is on
    """
    # Check args first
    if args and hasattr(args, 'debug'):
        if isinstance(args.debug, bool):
            return args.debug
        return str(args.debug).lower() in ['true', 'on', '1', 'yes']
    
    # Check environment variable
    debug_env = os.environ.get('DEBUG', '').lower()
    return debug_env in ['true', 'on', '1', 'yes']
