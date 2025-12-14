#!/usr/bin/env python3
"""
Batch processor for ultra-fast ensemble integration.
Processes multiple samples per API call for 3x speedup.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from utils.llm.backends import call_model


def build_batch_prompt_for_judge(samples: List[tuple], judge: str, method: str, lang: str = 'en') -> str:
    """Build optimized batch prompt for specific judge/method."""
    lang_map = {'en': 'English', 'es': 'Spanish', 'de': 'German', 'ua': 'Ukrainian'}
    language = lang_map.get(lang, lang)
    
    if judge == 'sentence' and method == 'legacy':
        prompt = f"""You are a {language} GEC Quality Specialist. Process these {len(samples)} corrections:

"""
        for i, (idx, src, tgt) in enumerate(samples, 1):
            prompt += f"""Sample {i}:
Original: {src}
Corrected: {tgt}

"""
        
        prompt += f"""Classify each as TP/TN/FN/FP1/FP2/FP3:
- TP: Valid correction needed
- TN: Original correct, no change needed  
- FN: Error missed, should be corrected
- FP1: Critical error (nonsense/major harm)
- FP2: Grammar error (introduces issues)
- FP3: Minor/stylistic (unnecessary)

JSON array response:
[{{"sample": 1, "classifications": ["TP"], "explanation": "brief"}}, {{"sample": 2, "classifications": ["FP3"], "fp_severity": "FP3-Minor", "explanation": "brief"}}, ...]"""
    
    else:
        # Generic batch prompt for other judges
        prompt = f"""Process these {len(samples)} {language} text corrections:

"""
        for i, (idx, src, tgt) in enumerate(samples, 1):
            prompt += f"""Sample {i}: {src} → {tgt}
"""
        
        prompt += f"""Classify each as TP/FP/TN/FN. Return JSON: [{{"sample": 1, "label": "TP"}}, ...]"""
    
    return prompt


def parse_batch_response_for_judge(content: str, judge: str, method: str, batch_size: int) -> List[tuple]:
    """Parse batch response for specific judge format."""
    results = []
    
    try:
        # JSON parsing
        if '[' in content and ']' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                
                for item in data:
                    if isinstance(item, dict):
                        if judge == 'sentence' and method == 'legacy':
                            # Handle sentence/legacy format
                            classifications = item.get('classifications', [])
                            if classifications and isinstance(classifications, list):
                                label = classifications[0].upper()
                            else:
                                label = item.get('label', 'Error').upper()
                            
                            # Handle FP severity
                            if label == 'FP' and 'fp_severity' in item:
                                severity = item['fp_severity'].upper()
                                if 'FP1' in severity or 'CRITICAL' in severity:
                                    label = 'FP1'
                                elif 'FP2' in severity or 'MEDIUM' in severity:
                                    label = 'FP2'
                                elif 'FP3' in severity or 'MINOR' in severity:
                                    label = 'FP3'
                            
                            reasoning = item.get('explanation', '')[:200]
                        else:
                            # Generic format
                            label = item.get('label', item.get('classification', 'Error')).upper()
                            reasoning = item.get('reasoning', item.get('explanation', ''))[:200]
                        
                        results.append((label, reasoning))
    except:
        pass
    
    # Ensure we have results for all samples
    while len(results) < batch_size:
        results.append(('Error', 'Batch parsing failed'))
    
    return results[:batch_size]


def process_batch_for_judge(batch_args: tuple) -> List[Dict[str, Any]]:
    """Process a batch of samples for a specific judge."""
    batch_samples, judge, method, backend, lang = batch_args
    
    try:
        # Build judge-specific batch prompt
        prompt = build_batch_prompt_for_judge(batch_samples, judge, method, lang)
        
        # Single API call for entire batch
        api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
        success, content, tokens = call_model(prompt, backend, api_token, moderation=False)
        
        if success and content:
            # Parse using judge-specific logic
            parsed_results = parse_batch_response_for_judge(content, judge, method, len(batch_samples))
        else:
            parsed_results = [('Error', 'Batch API failed')] * len(batch_samples)
        
        # Convert to standard result format
        results = []
        for i, (idx, src, tgt) in enumerate(batch_samples):
            label, reasoning = parsed_results[i] if i < len(parsed_results) else ('Error', 'Incomplete')
            results.append({
                'idx': idx,
                'src': src,
                'tgt': tgt,
                'tp_fp_label': label,
                'reasoning': reasoning,
                'writing_type': 'Personal',  # Default
                'total_tokens': tokens.get('total_tokens', 0) // len(batch_samples),
                'model': backend
            })
        
        return results
        
    except Exception as e:
        return [{
            'idx': idx,
            'src': src,
            'tgt': tgt,
            'tp_fp_label': 'Error',
            'reasoning': str(e)[:100],
            'writing_type': '',
            'total_tokens': 0,
            'model': backend
        } for idx, src, tgt in batch_samples]


def process_dataframe_with_batching(df, judge: str, method: str, backend: str, lang: str, 
                                   workers: int = 100, batch_size: int = 8) -> List[Dict[str, Any]]:
    """Process entire dataframe using batch API calls for maximum speed."""
    
    # Convert to simple tuples
    rows = [(i, str(row.get('src', '')), str(row.get('tgt', ''))) 
            for i, (_, row) in enumerate(df.iterrows())]
    
    # Create batches
    batches = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        batches.append((batch, judge, method, backend, lang))
    
    print(f"Batch processing: {len(rows)} samples → {len(batches)} API calls ({batch_size} samples/call)")
    
    # Process batches in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {executor.submit(process_batch_for_judge, batch_args): i 
                          for i, batch_args in enumerate(batches)}
        
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            all_results.extend(batch_results)
    
    # Sort by original index
    all_results.sort(key=lambda x: x['idx'])
    
    return all_results







