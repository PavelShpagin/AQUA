#!/usr/bin/env python3
"""
OmniGEC dataset loader.
Clean, efficient, language-agnostic.
"""

import random
from typing import List, Dict
from .sentence_splitter import process_text_pair

# Set seed for reproducibility
random.seed(42)


def load_omnigec_data(corpus_type: str, language: str, target_samples: int = None) -> List[Dict]:
    """Load OmniGEC data for specified language."""
    try:
        from datasets import load_dataset
        import time
        
        dataset_mapping = {
            "reddit": "lang-uk/Reddit-MultiGEC",
            "wiki": "lang-uk/WikiEdits-MultiGEC",
            "ubertext": "peterua/OmniGEC-ModelTraining"  # Alternative OmniGEC dataset
        }
        
        if corpus_type not in dataset_mapping:
            return []
        
        dataset = load_dataset(dataset_mapping[corpus_type], split='train', streaming=True)
        sentences = []
        max_samples = None if target_samples is None else target_samples * 3  # Sample 3x for downstream ERRANT/filtering
        processed_count = 0
        start_time = time.time()
        max_processing_time = 300  # 5 minutes max
        
        for sample in dataset:
            processed_count += 1
            
            # Early stopping for speed or time
            if max_samples is not None and len(sentences) >= max_samples:
                break
            
            # Timeout protection
            if time.time() - start_time > max_processing_time:
                print(f"  Timeout reached after {max_processing_time}s, stopping with {len(sentences)} samples")
                break
            
            # Progress indicator for large datasets
            if processed_count % 5000 == 0:
                print(f"  Processed {processed_count} samples, found {len(sentences)} matching...")
            
            # Language filtering
            sample_lang = sample.get('language', '').lower()
            if language == 'en' and sample_lang != 'english':
                continue
            elif language == 'de' and sample_lang != 'german':
                continue
            elif language == 'uk' and sample_lang != 'ukrainian':
                continue
            
            # Extract text based on corpus type
            if corpus_type == 'wiki':
                src_text = sample.get('text_del', '').strip()
                tgt_text = sample.get('text_ins', '').strip()
            elif corpus_type == 'ubertext':
                # peterua/OmniGEC-ModelTraining format
                src_text = sample.get('text', '').strip()
                tgt_text = sample.get('correction', '').strip()
            else:  # reddit
                src_text = sample.get('text', sample.get('src', '')).strip()
                tgt_text = sample.get('correction', sample.get('tgt', '')).strip()
            
            # Split into sentences and filter by word count (10-50 words)
            if src_text and tgt_text:
                sentence_pairs = process_text_pair(src_text, tgt_text)
                for src_sent, tgt_sent in sentence_pairs:
                    sentences.append({
                        'src_text': src_sent,
                        'tgt_text': tgt_sent
                    })
        
        # Return all sampled data (3x target) - final sampling happens after ERRANT
        return sentences
        
    except Exception:
        return []