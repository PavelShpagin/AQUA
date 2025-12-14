#!/usr/bin/env python3
"""
Ukrainian GEC Dataset Processing

Processes Ukrainian grammatical error correction datasets into aligned sentence pairs.
Combines UA-GEC sentence-level data with OmniGEC Reddit data.

Usage:
    python processing/ua.py

Output:
    data/processed/ua-judge.csv - 500 Ukrainian sentence pairs with ERRANT alignment

Features:
    - Direct UA-GEC sentence file processing (source-sentences/target-sentences)
    - Multilingual ERRANT alignment (English model for Ukrainian)
    - OmniGEC streaming integration
    - Automatic sentence filtering (min 10 words)
    - Balanced dataset sampling from multiple sources
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.processing.model_setup import setup_language_models
from utils.processing.ua_gec_loader import load_ua_gec_sentences
from utils.processing.omnigec_loader import load_omnigec_data

from utils.errant_align import get_alignment_for_language
from utils.processing.dataset_writer import write_csv_dataset as write_dataset
from utils.processing.sentence_splitter import process_text_pair
from utils.processing.filters import should_keep

# Reproducible sampling
import random
random.seed(42)


def process_ua_gec_dataset(data_path: str, target_samples: int, dataset_name: str, nlp, annotator, max_dataset=True, merge_edits: bool = True):
    """Process UA-GEC sentence files into aligned sentences."""
    # Get samples based on max_dataset setting
    initial_samples = None if max_dataset else target_samples * 3
    ua_gec_data = load_ua_gec_sentences(data_path, initial_samples)
    
    print(f"Processing {len(ua_gec_data)} UA-GEC entries from {dataset_name}")
    
    # Apply ERRANT and filtering to all samples
    valid_sentences = []
    for i, data in enumerate(tqdm(ua_gec_data, desc=f"Processing {dataset_name}")):
        src_text = data['src_text']
        tgt_text = data['tgt_text']
        
        # Apply ERRANT - if it fails, sentence is filtered out
        try:
            aligned_text = get_alignment_for_language(
                src_text=src_text, tgt_text=tgt_text, language='uk',
                nlp=nlp, annotator=annotator, merge=merge_edits
            )
            
            # Only add if ERRANT succeeded (not just returning src_text) and passes filters
            if (aligned_text != src_text or src_text == tgt_text) and should_keep(src_text, tgt_text, aligned_text):
                valid_sentences.append({
                    'idx': f"{dataset_name}_{i}_{len(src_text.split())}w",
                    'src': src_text,
                    'tgt': tgt_text,
                    'aligned': aligned_text
                })
        except:
            # ERRANT failed - skip this sentence
            continue
    
    # Now sample final target_samples from valid sentences
    if len(valid_sentences) > target_samples:
        valid_sentences = random.sample(valid_sentences, target_samples)
    
    return valid_sentences


def process_omnigec_dataset(corpus_type: str, target_samples: int, dataset_name: str, nlp, annotator, max_dataset=True, merge_edits: bool = True):
    """Process Ukrainian OmniGEC data into aligned sentences."""
    # Get samples based on max_dataset setting
    initial_samples = None if max_dataset else target_samples * 3
    omnigec_data = load_omnigec_data(corpus_type, 'uk', initial_samples)
    
    print(f"Processing {len(omnigec_data)} OmniGEC entries from {dataset_name}")
    
    # Apply ERRANT and filtering to all samples
    valid_sentences = []
    for data in tqdm(omnigec_data, desc=f"Processing {dataset_name}"):
        # Use sentence splitting (already filters 10-50 words)
        sentence_pairs = process_text_pair(data['src_text'], data['tgt_text'])
        
        for i, (src_sent, tgt_sent) in enumerate(sentence_pairs):
            # Apply ERRANT - if it fails, sentence is filtered out
            try:
                aligned_text = get_alignment_for_language(
                    src_text=src_sent, tgt_text=tgt_sent, language='uk',
                    nlp=nlp, annotator=annotator, merge=merge_edits
                )
                
                # Only add if ERRANT succeeded (not just returning src_text) and passes filters
                if (aligned_text != src_sent or src_sent == tgt_sent) and should_keep(src_sent, tgt_sent, aligned_text):
                    valid_sentences.append({
                        'idx': f"{dataset_name}_{len(valid_sentences)}_{len(src_sent.split())}w",
                        'src': src_sent,
                        'tgt': tgt_sent,
                        'aligned': aligned_text
                    })
            except:
                # ERRANT failed - skip this sentence
                continue
    
    # Now sample final target_samples from valid sentences
    if len(valid_sentences) > target_samples:
        import random
        valid_sentences = random.sample(valid_sentences, target_samples)
    
    return valid_sentences


def main():
    """Main processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ukrainian GEC processing')
    parser.add_argument('--max', action='store_true', default=False, help='Use full filtered dataset (True) or 3x samples (False)')
    parser.add_argument('--no-merge', action='store_true', default=False, help='Disable merging of adjacent edits in alignment')
    
    # Individual dataset arguments
    parser.add_argument('--ua-gec', type=int, default=300, help='Number of UA-GEC samples')
    parser.add_argument('--omni', type=int, default=0, help='Number of OmniGEC samples')
    
    args = parser.parse_args()
    
    print("Ukrainian GEC Dataset Processing")
    print("=" * 40)
    print(f"Max dataset: {args.max}")
    
    nlp, annotator = setup_language_models('uk')
    
    merge_edits = not args.no_merge

    # Dataset configuration - UA-GEC + OmniGEC
    datasets = [
        {'name': 'UA_GEC_train', 'type': 'ua_gec', 'path': 'data/raw/ua-gec/data/gec-only/train', 'samples': args.ua_gec},
        {'name': 'OmniGEC_reddit', 'type': 'omnigec', 'corpus': 'reddit', 'samples': args.omni}
    ]
    
    all_sentences = []
    
    for dataset in datasets:
        if dataset['samples'] == 0:
            continue
        
        if dataset.get('type') == 'omnigec':
            sentences = process_omnigec_dataset(
                dataset['corpus'], dataset['samples'], dataset['name'], nlp, annotator, args.max, merge_edits
            )
        elif dataset.get('type') == 'ua_gec':
            sentences = process_ua_gec_dataset(
                dataset['path'], dataset['samples'], dataset['name'], nlp, annotator, args.max, merge_edits
            )
        
        all_sentences.extend(sentences)
    
    # Write output
    write_dataset(all_sentences, "data/processed/ua-judge.csv", "Ukrainian")


if __name__ == "__main__":
    main()