#!/usr/bin/env python3
"""
BEA English Training Data Processor

Processes English GEC training data mixture following the Pillars of GEC paper approach.
Combines multiple datasets to create a comprehensive training set.

Based on: https://arxiv.org/pdf/2404.14914 (Pillars of Grammatical Error Correction)

Available datasets:
- W&I+LOCNESS (BEA-2019)
- FCE
- Lang-8  
- NUCLE (CoNLL-14 test data)
- cLang-8 (synthetic corrections from third-party/clang8)
- Troy-1BW (One Billion Word synthetic - if available)
- Troy-Blogs (Blog Authorship synthetic - if available) 
- CoWSL2H (Spanish learner corpus - additional)

Usage:
    python processing/process_bea_training.py --output data/training/bea
    python processing/process_bea_training.py --output data/training/bea --samples 10000 --max
"""

import argparse
import random
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.processing.bea19_json_loader import load_wi_locness_train, load_fce_train
from utils.processing.lang8_loader import load_lang8_data
from utils.processing.nucle_loader import load_nucle_data
from utils.processing.cowsl2h_loader import load_cowsl2h_data
from utils.processing.clang8_loader import load_clang8_data
from utils.processing.troy_loader import load_troy_1bw_data, load_troy_blogs_data
from utils.processing.dataset_writer import write_training_dataset
from utils.processing.model_setup import setup_language_models
from utils.alignment import get_alignment_for_language
from utils.processing.filters import should_keep

# Set seed for reproducibility
random.seed(42)


def process_dataset_with_errant(dataset_data, dataset_name, nlp, annotator, merge_edits=True):
    """Process dataset with ERRANT alignment and filtering."""
    print(f"\nProcessing {dataset_name} with ERRANT...")
    
    valid_sentences = []
    failed_count = 0
    
    for i, data in enumerate(dataset_data):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(dataset_data)}, {len(valid_sentences)} valid, {failed_count} failed")
        
        src_text = data['src_text']
        tgt_text = data['tgt_text']
        
        # Skip if texts are identical (no corrections needed)
        if src_text == tgt_text:
            continue
        
        try:
            # Apply ERRANT alignment
            aligned_text = get_alignment_for_language(
                src_text=src_text, 
                tgt_text=tgt_text, 
                language='en',
                nlp=nlp, 
                annotator=annotator, 
                merge=merge_edits
            )
            
            # Filter: only keep if ERRANT succeeded and passes quality filters
            if (aligned_text != src_text or src_text == tgt_text) and should_keep(src_text, tgt_text, aligned_text):
                valid_sentences.append({
                    'src': src_text,
                    'tgt': tgt_text,
                    'aligned': aligned_text,
                    'dataset': dataset_name,
                    'id': data.get('id', f"{dataset_name}_{i}")
                })
                
        except Exception as e:
            failed_count += 1
            # Skip failed alignments
            continue
    
    print(f"  Final: {len(valid_sentences)} valid sentences from {dataset_name}")
    return valid_sentences


def main():
    parser = argparse.ArgumentParser(description='Process BEA English training data mixture')
    parser.add_argument('--output', type=str, default='data/training/bea', 
                        help='Output directory for training data')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of samples per dataset (before ERRANT filtering)')
    parser.add_argument('--max', action='store_true', default=False,
                        help='Use maximum available data (ignores --samples)')
    parser.add_argument('--no-merge', action='store_true', default=False,
                        help='Disable merging of adjacent edits in alignment')
    
    # Individual dataset control
    parser.add_argument('--wi-locness', type=int, default=None,
                        help='Number of W&I+LOCNESS samples (default: uses --samples)')
    parser.add_argument('--fce', type=int, default=None,
                        help='Number of FCE samples (default: uses --samples)')
    parser.add_argument('--lang8', type=int, default=None,
                        help='Number of Lang-8 samples (default: uses --samples)')
    parser.add_argument('--nucle', type=int, default=None,
                        help='Number of NUCLE samples (default: uses --samples)')
    parser.add_argument('--cowsl2h', type=int, default=0,
                        help='Number of CoWSL2H samples (default: 0, set >0 to include)')
    parser.add_argument('--clang8', type=int, default=None,
                        help='Number of cLang-8 English samples (default: uses --samples)')
    parser.add_argument('--troy-1bw', type=int, default=0,
                        help='Number of Troy-1BW samples (default: 0, set >0 to include)')
    parser.add_argument('--troy-blogs', type=int, default=0,
                        help='Number of Troy-Blogs samples (default: 0, set >0 to include)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("BEA English Training Data Processing")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Max dataset mode: {args.max}")
    print(f"Merge edits: {not args.no_merge}")
    
    # Setup language models for ERRANT
    print("\nSetting up language models...")
    nlp, annotator = setup_language_models('en')
    
    merge_edits = not args.no_merge
    all_sentences = []
    
    # Determine sample sizes
    default_samples = None if args.max else args.samples
    wi_samples = args.wi_locness if args.wi_locness is not None else default_samples
    fce_samples = args.fce if args.fce is not None else default_samples
    lang8_samples = args.lang8 if args.lang8 is not None else default_samples
    nucle_samples = args.nucle if args.nucle is not None else default_samples
    cowsl2h_samples = args.cowsl2h
    clang8_samples = args.clang8 if args.clang8 is not None else default_samples
    troy_1bw_samples = getattr(args, 'troy_1bw', 0)
    troy_blogs_samples = getattr(args, 'troy_blogs', 0)
    
    # Process W&I+LOCNESS
    if wi_samples != 0:
        print(f"\n1. Loading W&I+LOCNESS data (target: {wi_samples or 'all'})...")
        try:
            wi_data = load_wi_locness_train(wi_samples)
            print(f"   Loaded {len(wi_data)} W&I+LOCNESS entries")
            
            if wi_data:
                wi_sentences = process_dataset_with_errant(
                    wi_data, "WI_LOCNESS", nlp, annotator, merge_edits
                )
                all_sentences.extend(wi_sentences)
        except Exception as e:
            print(f"   Error loading W&I+LOCNESS: {e}")
    
    # Process FCE
    if fce_samples != 0:
        print(f"\n2. Loading FCE data (target: {fce_samples or 'all'})...")
        try:
            fce_data = load_fce_train(fce_samples)
            print(f"   Loaded {len(fce_data)} FCE entries")
            
            if fce_data:
                fce_sentences = process_dataset_with_errant(
                    fce_data, "FCE", nlp, annotator, merge_edits
                )
                all_sentences.extend(fce_sentences)
        except Exception as e:
            print(f"   Error loading FCE: {e}")
    
    # Process Lang-8
    if lang8_samples != 0:
        print(f"\n3. Loading Lang-8 data (target: {lang8_samples or 'all'})...")
        try:
            lang8_data = load_lang8_data(
                'data/raw/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat', 
                lang8_samples, 
                'en'
            )
            print(f"   Loaded {len(lang8_data)} Lang-8 entries")
            
            if lang8_data:
                lang8_sentences = process_dataset_with_errant(
                    lang8_data, "LANG8", nlp, annotator, merge_edits
                )
                all_sentences.extend(lang8_sentences)
        except Exception as e:
            print(f"   Error loading Lang-8: {e}")
    
    # Process NUCLE
    if nucle_samples != 0:
        print(f"\n4. Loading NUCLE data (target: {nucle_samples or 'all'})...")
        try:
            nucle_data = load_nucle_data(
                'data/raw/BEA/conll14st-test-data/noalt', 
                nucle_samples
            )
            print(f"   Loaded {len(nucle_data)} NUCLE entries")
            
            if nucle_data:
                # Note: NUCLE data doesn't have corrections, so it's mainly for fluency
                nucle_sentences = process_dataset_with_errant(
                    nucle_data, "NUCLE", nlp, annotator, merge_edits
                )
                all_sentences.extend(nucle_sentences)
        except Exception as e:
            print(f"   Error loading NUCLE: {e}")
    
    # Process cLang-8
    if clang8_samples != 0:
        print(f"\n5. Loading cLang-8 English data (target: {clang8_samples or 'all'})...")
        try:
            clang8_data = load_clang8_data(
                'third-party/clang8/output_data/clang8_source_target_en.tsv',
                clang8_samples, 'en'
            )
            print(f"   Loaded {len(clang8_data)} cLang-8 entries")
            
            if clang8_data:
                clang8_sentences = process_dataset_with_errant(
                    clang8_data, "CLANG8", nlp, annotator, merge_edits
                )
                all_sentences.extend(clang8_sentences)
        except Exception as e:
            print(f"   Error loading cLang-8: {e}")
    
    # Process Troy-1BW (optional)
    if troy_1bw_samples > 0:
        print(f"\n6. Loading Troy-1BW data (target: {troy_1bw_samples})...")
        try:
            troy_1bw_data = load_troy_1bw_data('data/raw/troy-1bw', troy_1bw_samples)
            print(f"   Loaded {len(troy_1bw_data)} Troy-1BW entries")
            
            if troy_1bw_data:
                troy_1bw_sentences = process_dataset_with_errant(
                    troy_1bw_data, "TROY_1BW", nlp, annotator, merge_edits
                )
                all_sentences.extend(troy_1bw_sentences)
        except Exception as e:
            print(f"   Error loading Troy-1BW: {e}")
    
    # Process Troy-Blogs (optional)
    if troy_blogs_samples > 0:
        print(f"\n7. Loading Troy-Blogs data (target: {troy_blogs_samples})...")
        try:
            troy_blogs_data = load_troy_blogs_data('data/raw/troy-blogs', troy_blogs_samples)
            print(f"   Loaded {len(troy_blogs_data)} Troy-Blogs entries")
            
            if troy_blogs_data:
                troy_blogs_sentences = process_dataset_with_errant(
                    troy_blogs_data, "TROY_BLOGS", nlp, annotator, merge_edits
                )
                all_sentences.extend(troy_blogs_sentences)
        except Exception as e:
            print(f"   Error loading Troy-Blogs: {e}")
    
    # Process CoWSL2H (optional, Spanish learner corpus)
    if cowsl2h_samples > 0:
        print(f"\n8. Loading CoWSL2H data (target: {cowsl2h_samples})...")
        try:
            cowsl2h_data = load_cowsl2h_data('data/raw/cowsl2h', cowsl2h_samples)
            print(f"   Loaded {len(cowsl2h_data)} CoWSL2H entries")
            
            if cowsl2h_data:
                cowsl2h_sentences = process_dataset_with_errant(
                    cowsl2h_data, "COWSL2H", nlp, annotator, merge_edits
                )
                all_sentences.extend(cowsl2h_sentences)
        except Exception as e:
            print(f"   Error loading CoWSL2H: {e}")
    
    # Final shuffle and output
    print(f"\n9. Finalizing dataset...")
    print(f"   Total sentences: {len(all_sentences)}")
    
    if all_sentences:
        # Shuffle the combined dataset
        random.shuffle(all_sentences)
        
        # Write output
        output_file = output_dir / 'bea_english_training.jsonl'
        write_training_dataset(all_sentences, output_file, "BEA English Training Mixture")
        
        # Write summary
        summary_file = output_dir / 'dataset_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("BEA English Training Data Summary\n")
            f.write("=" * 40 + "\n\n")
            
            dataset_counts = {}
            for sent in all_sentences:
                dataset = sent['dataset']
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            f.write("Dataset composition:\n")
            for dataset, count in dataset_counts.items():
                f.write(f"  {dataset}: {count} sentences\n")
            
            f.write(f"\nTotal: {len(all_sentences)} sentences\n")
            f.write(f"Output file: {output_file}\n")
            
            # Note about dataset availability
            f.write("\nDataset availability notes:\n")
            f.write("  - cLang-8: Available in third-party/clang8/\n")
            f.write("  - Troy-1BW: Requires download and processing\n")
            f.write("  - Troy-Blogs: Requires download and processing\n")
            f.write("  - Run shell/download_missing_datasets.sh to fetch additional data\n")
        
        print(f"   Written to: {output_file}")
        print(f"   Summary: {summary_file}")
        
    else:
        print("   Error: No valid sentences found!")
        return 1
    
    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
