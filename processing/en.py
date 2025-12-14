#!/usr/bin/env python3
"""
English GEC Dataset Processing

Processes English grammatical error correction datasets into aligned sentence pairs.
Combines BEA datasets (W&I+LOCNESS, FCE, Lang-8) with OmniGEC Reddit data.

Usage:
    python processing/en.py

Output:
    data/processed/en-judge.csv - English sentence pairs with ERRANT alignment

Features:
    - Classical ERRANT alignment for research-grade quality
    - M2 format parsing for BEA datasets
    - OmniGEC streaming integration
    - Automatic sentence filtering (min 10 words)
    - Balanced dataset sampling
"""

import sys
import re
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.processing.model_setup import setup_language_models
from utils.processing.omnigec_loader import load_omnigec_data
from utils.processing.bea19_json_loader import load_wi_locness_train, load_fce_train
from utils.processing.sentence_splitter import process_text_pair
from utils.errant_align import get_alignment_for_language, detokenize_simple
from utils.processing.dataset_writer import write_dataset
from utils.processing.filters import should_keep

# Reproducible sampling
random.seed(42)


def process_bea19_json_dataset(loader_func, target_samples: int, dataset_name: str, nlp, annotator, max_dataset=True, merge_edits: bool = True):
    """Process BEA19 JSON dataset into aligned sentences."""
    # Get samples based on max_dataset setting
    initial_samples = None if max_dataset else target_samples * 3
    json_data = loader_func(initial_samples)
    
    print(f"Processing {len(json_data)} raw text entries from {dataset_name}")
    
    # Apply ERRANT and filtering to all samples
    valid_sentences = []
    for i, data in enumerate(tqdm(json_data, desc=f"Processing {dataset_name}")):
        src_text = data['src_text']
        tgt_text = data['tgt_text']
        
        # Use sentence splitting (already filters 10-50 words)
        sentence_pairs = process_text_pair(src_text, tgt_text)
        
        for j, (src_sent, tgt_sent) in enumerate(sentence_pairs):
            # Skip multiline entries (letters/essays with paragraph breaks)
            if '\n' in src_sent or '\n' in tgt_sent:
                continue
                
            # Apply ERRANT - if it fails, sentence is filtered out
            try:
                aligned_text = get_alignment_for_language(
                    src_text=src_sent, tgt_text=tgt_sent, language='en',
                    nlp=nlp, annotator=annotator, merge=merge_edits
                )
                
                if aligned_text and should_keep(src_sent, tgt_sent, aligned_text):
                    valid_sentences.append({
                        'idx': f"{dataset_name}_{i}_{j}_{len(src_sent.split())}w",
                        'src': src_sent,
                        'tgt': tgt_sent,
                        'aligned': aligned_text
                    })
            except Exception as e:
                # ERRANT failed - skip this sentence
                continue
    
    # Sample down to target if we have too many
    if len(valid_sentences) > target_samples:
        valid_sentences = random.sample(valid_sentences, target_samples)
    
    return valid_sentences




def process_omnigec_dataset(corpus_type: str, target_samples: int, dataset_name: str, nlp, annotator, use_tokenization=False, max_dataset=True, merge_edits: bool = True):
    """Process OmniGEC data into aligned sentences."""
    # Get samples based on max_dataset setting
    initial_samples = None if max_dataset else target_samples * 3
    omnigec_data = load_omnigec_data(corpus_type, 'en', initial_samples)
    
    print(f"Processing {len(omnigec_data)} OmniGEC entries from {dataset_name}")
    
    # Apply ERRANT and filtering to all samples
    valid_sentences = []
    for data in tqdm(omnigec_data, desc=f"Processing {dataset_name}"):
        # Use sentence splitting (already filters 10-50 words)
        sentence_pairs = process_text_pair(data['src_text'], data['tgt_text'])
        
        for i, (src_sent, tgt_sent) in enumerate(sentence_pairs):
            # Apply ERRANT - if it fails, sentence is filtered out
            try:
                # OmniGEC always uses tokenization (no pre-tokenized data available)
                aligned_text = get_alignment_for_language(
                    src_text=src_sent, tgt_text=tgt_sent, language='en',
                    nlp=nlp, annotator=annotator, merge=merge_edits
                )
                
                # Only add if we have a valid alignment (ERRANT or fallback succeeded)
                if aligned_text and (aligned_text != src_sent or src_sent == tgt_sent) and should_keep(src_sent, tgt_sent, aligned_text):
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
    parser = argparse.ArgumentParser(description='English GEC processing')
    parser.add_argument('--target_samples', type=int, default=500, help='Target number of samples')
    parser.add_argument('--output_file', type=str, default='data/processed/en-judge.csv', help='Output file')
    parser.add_argument('--force_tokenize', action='store_true', help='Force tokenization for all datasets (ignore pre-tokenized data)')
    parser.add_argument('--max', action='store_true', default=False, help='Use full filtered dataset (True) or 3x samples (False)')
    parser.add_argument('--no-merge', action='store_true', default=False, help='Disable merging of adjacent edits in alignment')
    
    # Individual dataset arguments
    parser.add_argument('--wi', type=int, default=100, help='Number of W&I+LOCNESS samples')
    parser.add_argument('--fce', type=int, default=100, help='Number of FCE samples')
    parser.add_argument('--omni', type=int, default=100, help='Number of OmniGEC samples')
    
    args = parser.parse_args()
    
    print("English GEC Dataset Processing")
    print("=" * 40)
    print(f"Force tokenization: {args.force_tokenize}")
    print(f"Max dataset: {args.max}")
    
    nlp, annotator = setup_language_models('en')
    
    merge_edits = not args.no_merge

    # Dataset configuration - using raw text sources instead of M2 files
    datasets = [
        {'name': 'W&I_LOCNESS_train', 'type': 'bea19_json', 'loader': load_wi_locness_train, 'samples': args.wi},
        {'name': 'FCE_train', 'type': 'bea19_json', 'loader': load_fce_train, 'samples': args.fce},
        {'name': 'OmniGEC_reddit', 'type': 'omnigec', 'corpus': 'reddit', 'samples': args.omni}
    ]
    
    all_sentences = []
    
    for dataset in datasets:
        if dataset['samples'] == 0:
            continue
        
        if dataset.get('type') == 'bea19_json':
            sentences = process_bea19_json_dataset(
                dataset['loader'], dataset['samples'], dataset['name'], nlp, annotator, args.max, merge_edits
            )
        elif dataset.get('type') == 'omnigec':
            sentences = process_omnigec_dataset(
                dataset['corpus'], dataset['samples'], dataset['name'], nlp, annotator, False, args.max, merge_edits
            )
        else:
            # No other dataset types in current pipeline
            sentences = []
        
        all_sentences.extend(sentences)
    
    # Write output
    write_dataset(all_sentences, args.output_file, "English")


if __name__ == "__main__":
    main()