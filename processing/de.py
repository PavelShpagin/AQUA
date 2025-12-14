#!/usr/bin/env python3
"""
German GEC Dataset Processing

Produces German sentence-level GEC data from FalkoMerlin TSV (raw reconstruction) and optional OmniGEC.

Usage:
    python processing/de.py

Output:
    data/processed/de-judge.csv - German sentence pairs with ERRANT alignment

Features:
    - ERRANT alignment (English model for German)
    - FalkoMerlin TSV only (no M2, no MERLIN fallback)
    - Strict sentence filtering (10–50 words) and edit cap (<5)
"""

import sys
import re
import random
from pathlib import Path
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.processing.model_setup import setup_language_models
from utils.processing.omnigec_loader import load_omnigec_data
from utils.processing.falko_merlin_loader import cleanup_german_text
from utils.processing.fm_tsv_loader import load_fm_tsv_sentence_pairs

from utils.errant_align import get_alignment_for_language
from utils.processing.dataset_writer import write_csv_dataset as write_dataset
from utils.processing.sentence_splitter import process_text_pair
from utils.processing.filters import should_keep

# Reproducible sampling
random.seed(42)


# Removed legacy MERLIN essay helpers (no longer used)


def process_falko_merlin_dataset(data_path: str, target_samples: int, dataset_name: str, nlp, annotator, max_dataset=True, merge_edits: bool = True):
    """Process FalkoMerlin using native sentence-level parallel data.
    Preference order:
      1) fm-*.m2 (apply edits, then detokenize to human text)
      2) de_falko-merlin_*.tsv fallback (already sentence-level, filtered to 1:1 in loader)
    """
    initial_samples = None if max_dataset else target_samples * 3

    valid_sentences = []

    # FalkoMerlin fm-*.tsv only (raw, sentence-level). No MERLIN or other fallbacks.
    fm_tsv = 'data/raw/multiged-2023/german/fm-train.tsv'
    if not Path(fm_tsv).exists():
        print(f"Error: missing FalkoMerlin TSV at {fm_tsv}. No fallback is used.")
        return []
            
    fm_pairs = load_fm_tsv_sentence_pairs(fm_tsv, initial_samples)
    print(f"Processing {len(fm_pairs)} FalkoMerlin fm-*.tsv sentence-level pairs")
    if fm_pairs:
        for i, item in enumerate(tqdm(fm_pairs, desc=f"Processing {dataset_name} (FalkoMerlin TSV)")):
            src_text = item['src_text']
            tgt_text = item['tgt_text']
            try:
                aligned_text = get_alignment_for_language(
                    src_text=src_text, tgt_text=tgt_text, language='de',
                    nlp=nlp, annotator=annotator, merge=merge_edits
                )
                if aligned_text and should_keep(src_text, tgt_text, aligned_text):
                    valid_sentences.append({
                        'idx': f"{dataset_name}_{item['id']}_{len(src_text.split())}w",
                        'src': src_text,
                        'tgt': tgt_text,
                        'aligned': aligned_text
                    })
            except Exception:
                pass
    
    # Sample down to target if we have too many
    if len(valid_sentences) > target_samples:
        valid_sentences = random.sample(valid_sentences, target_samples)
    
    return valid_sentences


def process_omnigec_dataset(corpus_type: str, target_samples: int, dataset_name: str, nlp, annotator, max_dataset=True, merge_edits: bool = True):
    """Process German OmniGEC data into aligned sentences."""
    # Get samples based on max_dataset setting
    initial_samples = None if max_dataset else target_samples * 3
    omnigec_data = load_omnigec_data(corpus_type, 'de', initial_samples)
    
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
                    src_text=src_sent, tgt_text=tgt_sent, language='de',
                    nlp=nlp, annotator=annotator, merge=merge_edits
                )
                
                # Only add if ERRANT succeeded (not just returning src_text)
                if aligned_text != src_sent or src_sent == tgt_sent:
                    # Filter out sentences with too many edits (>=5)
                    num_edits = len(re.findall(r"\{[^{}]*?=>[^{}]*?\}", aligned_text))
                    if num_edits >= 5:
                        continue
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
    
    parser = argparse.ArgumentParser(description='German GEC processing')
    parser.add_argument('--max', action='store_true', default=False, help='Use full filtered dataset (True) or 3x samples (False)')
    parser.add_argument('--no-merge', action='store_true', default=False, help='Disable merging of adjacent edits in alignment')
    
    # Individual dataset arguments
    parser.add_argument('--merlin', type=int, default=300, help='Number of FalkoMerlin samples')
    parser.add_argument('--omni', type=int, default=0, help='Number of OmniGEC samples')
    
    args = parser.parse_args()
    
    print("German GEC Dataset Processing")
    print("=" * 40)
    print(f"Max dataset: {args.max}")
    
    nlp, annotator = setup_language_models('de')
    
    merge_edits = not args.no_merge
    
    # Dataset configuration - using sentence-level FalkoMerlin data
    datasets = [
        {'name': 'FalkoMerlin', 'type': 'falko_merlin', 'path': 'data/raw/multiged-2023/german/de_falko-merlin_train.tsv', 'samples': args.merlin},
        {'name': 'OmniGEC_reddit', 'type': 'omnigec', 'corpus': 'reddit', 'samples': args.omni}
    ]
    
    all_sentences = []
    
    for dataset in datasets:
        if dataset['samples'] == 0:
            continue
        
        if dataset['type'] == 'falko_merlin':
            sentences = process_falko_merlin_dataset(
                dataset['path'], dataset['samples'], dataset['name'], nlp, annotator, args.max, merge_edits
            )
        elif dataset['type'] == 'omnigec':
            sentences = process_omnigec_dataset(
                dataset['corpus'], dataset['samples'], dataset['name'], nlp, annotator, args.max, merge_edits
            )
        
        all_sentences.extend(sentences)
    
    # Write output
    write_dataset(all_sentences, "data/processed/de-judge.csv", "German")


if __name__ == "__main__":
    main()