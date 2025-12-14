#!/usr/bin/env python3
"""
MultiGEC 2025 Multilingual Training Data Processor

Processes multilingual GEC training data mixture following MultiGEC 2025 approach.
Combines datasets for English, Ukrainian, and German.

Based on: https://aclanthology.org/2025.unlp-1.17.pdf and OmniGEC methodology

Datasets by language:
- English: OmniGEC Reddit, W&I+LOCNESS, FCE, Lang-8, cLang-8
- Ukrainian: UA-GEC, OmniGEC Reddit, UberText (if available)
- German: FalkoMerlin, OmniGEC Reddit, cLang-8 DE
- Russian: cLang-8 RU (if available)
- Spanish: CoWSL2H (additional)

Usage:
    python processing/process_multigec2025_training.py --output data/training/multigec2025
    python processing/process_multigec2025_training.py --output data/training/multigec2025 --samples 5000 --max
"""

import argparse
import random
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.processing.omnigec_loader import load_omnigec_data
from utils.processing.ua_gec_loader import load_ua_gec_sentences
from utils.processing.falko_merlin_loader import load_falko_merlin_data
from utils.processing.bea19_json_loader import load_wi_locness_train, load_fce_train
from utils.processing.lang8_loader import load_lang8_data
from utils.processing.cowsl2h_loader import load_cowsl2h_data
from utils.processing.clang8_loader import load_clang8_data, load_all_clang8_languages
from utils.processing.ubertext_loader import load_ubertext_data
from utils.processing.dataset_writer import write_training_dataset
from utils.processing.model_setup import setup_language_models
from utils.alignment import get_alignment_for_language
from utils.processing.filters import should_keep

# Set seed for reproducibility
random.seed(42)


def process_multilingual_dataset(dataset_data, dataset_name, language, nlp, annotator, merge_edits=True):
    """Process dataset with language-specific ERRANT alignment and filtering."""
    print(f"\nProcessing {dataset_name} ({language}) with ERRANT...")
    
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
            # Apply ERRANT alignment with language-specific models
            aligned_text = get_alignment_for_language(
                src_text=src_text, 
                tgt_text=tgt_text, 
                language=language,
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
                    'language': language,
                    'id': data.get('id', f"{dataset_name}_{language}_{i}")
                })
                
        except Exception as e:
            failed_count += 1
            # Skip failed alignments
            continue
    
    print(f"  Final: {len(valid_sentences)} valid sentences from {dataset_name} ({language})")
    return valid_sentences


def main():
    parser = argparse.ArgumentParser(description='Process MultiGEC 2025 multilingual training data')
    parser.add_argument('--output', type=str, default='data/training/multigec2025', 
                        help='Output directory for training data')
    parser.add_argument('--samples', type=int, default=3000,
                        help='Number of samples per dataset per language (before ERRANT filtering)')
    parser.add_argument('--max', action='store_true', default=False,
                        help='Use maximum available data (ignores --samples)')
    parser.add_argument('--no-merge', action='store_true', default=False,
                        help='Disable merging of adjacent edits in alignment')
    
    # Language selection
    parser.add_argument('--languages', nargs='+', default=['en', 'ua', 'de'],
                        help='Languages to process (en, ua, de, ru, es)')
    
    # Individual dataset control
    parser.add_argument('--omnigec-samples', type=int, default=None,
                        help='Number of OmniGEC Reddit samples per language (default: uses --samples)')
    parser.add_argument('--ua-gec-samples', type=int, default=None,
                        help='Number of UA-GEC samples (default: uses --samples)')
    parser.add_argument('--merlin-samples', type=int, default=None,
                        help='Number of FalkoMerlin samples (default: uses --samples)')
    parser.add_argument('--bea-samples', type=int, default=None,
                        help='Number of BEA samples per dataset (default: uses --samples)')
    parser.add_argument('--cowsl2h-samples', type=int, default=0,
                        help='Number of CoWSL2H samples (default: 0, set >0 to include Spanish)')
    parser.add_argument('--clang8-samples', type=int, default=None,
                        help='Number of cLang-8 samples per language (default: uses --samples)')
    parser.add_argument('--ubertext-samples', type=int, default=0,
                        help='Number of UberText samples (default: 0, set >0 to include Ukrainian)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("MultiGEC 2025 Multilingual Training Data Processing")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Languages: {args.languages}")
    print(f"Max dataset mode: {args.max}")
    print(f"Merge edits: {not args.no_merge}")
    
    merge_edits = not args.no_merge
    all_sentences = []
    
    # Determine sample sizes
    default_samples = None if args.max else args.samples
    omnigec_samples = args.omnigec_samples if args.omnigec_samples is not None else default_samples
    ua_gec_samples = args.ua_gec_samples if args.ua_gec_samples is not None else default_samples
    merlin_samples = args.merlin_samples if args.merlin_samples is not None else default_samples
    bea_samples = args.bea_samples if args.bea_samples is not None else default_samples
    clang8_samples = args.clang8_samples if args.clang8_samples is not None else default_samples
    ubertext_samples = args.ubertext_samples
    
    # Process each language
    for language in args.languages:
        print(f"\n{'='*20} Processing {language.upper()} {'='*20}")
        
        # Setup language-specific models
        print(f"Setting up {language} language models...")
        try:
            nlp, annotator = setup_language_models(language)
        except Exception as e:
            print(f"Error setting up {language} models: {e}")
            continue
        
        # Process OmniGEC Reddit for this language
        if omnigec_samples != 0:
            print(f"\n1. Loading OmniGEC Reddit ({language}) (target: {omnigec_samples or 'all'})...")
            try:
                omnigec_data = load_omnigec_data('reddit', language, omnigec_samples)
                print(f"   Loaded {len(omnigec_data)} OmniGEC Reddit entries")
                
                if omnigec_data:
                    omnigec_sentences = process_multilingual_dataset(
                        omnigec_data, f"OMNIGEC_REDDIT", language, nlp, annotator, merge_edits
                    )
                    all_sentences.extend(omnigec_sentences)
            except Exception as e:
                print(f"   Error loading OmniGEC Reddit ({language}): {e}")
        
        # Process cLang-8 for this language (if available)
        if clang8_samples != 0 and language in ['en', 'de', 'ru']:
            print(f"\n2. Loading cLang-8 {language.upper()} data (target: {clang8_samples or 'all'})...")
            try:
                clang8_file_map = {
                    'en': 'third-party/clang8/output_data/clang8_source_target_en.tsv',
                    'de': 'third-party/clang8/output_data/clang8_source_target_de.tsv',
                    'ru': 'third-party/clang8/output_data/clang8_source_target_ru.tsv'
                }
                
                clang8_data = load_clang8_data(clang8_file_map[language], clang8_samples, language)
                print(f"   Loaded {len(clang8_data)} cLang-8 entries")
                
                if clang8_data:
                    clang8_sentences = process_multilingual_dataset(
                        clang8_data, f"CLANG8", language, nlp, annotator, merge_edits
                    )
                    all_sentences.extend(clang8_sentences)
            except Exception as e:
                print(f"   Error loading cLang-8 ({language}): {e}")
        
        # Language-specific datasets
        if language == 'ua':
            # UA-GEC
            if ua_gec_samples != 0:
                print(f"\n3. Loading UA-GEC data (target: {ua_gec_samples or 'all'})...")
                try:
                    ua_gec_data = load_ua_gec_sentences(
                        'data/raw/ua-gec/data/gec-only/train', ua_gec_samples
                    )
                    print(f"   Loaded {len(ua_gec_data)} UA-GEC entries")
                    
                    if ua_gec_data:
                        ua_sentences = process_multilingual_dataset(
                            ua_gec_data, "UA_GEC", language, nlp, annotator, merge_edits
                        )
                        all_sentences.extend(ua_sentences)
                except Exception as e:
                    print(f"   Error loading UA-GEC: {e}")
            
            # UberText Ukrainian
            if ubertext_samples > 0:
                print(f"\n4. Loading UberText data (target: {ubertext_samples})...")
                try:
                    ubertext_data = load_ubertext_data('data/raw/ubertext', ubertext_samples)
                    print(f"   Loaded {len(ubertext_data)} UberText entries")
                    
                    if ubertext_data:
                        ubertext_sentences = process_multilingual_dataset(
                            ubertext_data, "UBERTEXT", language, nlp, annotator, merge_edits
                        )
                        all_sentences.extend(ubertext_sentences)
                except Exception as e:
                    print(f"   Error loading UberText: {e}")
        
        elif language == 'de':
            # FalkoMerlin
            if merlin_samples != 0:
                print(f"\n3. Loading FalkoMerlin data (target: {merlin_samples or 'all'})...")
                try:
                    merlin_data = load_falko_merlin_data(
                        'data/raw/multiged-2023/german/fm-train.tsv', merlin_samples
                    )
                    print(f"   Loaded {len(merlin_data)} FalkoMerlin entries")
                    
                    if merlin_data:
                        merlin_sentences = process_multilingual_dataset(
                            merlin_data, "FALKO_MERLIN", language, nlp, annotator, merge_edits
                        )
                        all_sentences.extend(merlin_sentences)
                except Exception as e:
                    print(f"   Error loading FalkoMerlin: {e}")
        
        elif language == 'ru':
            # Russian only has cLang-8, which is handled above
            print(f"\n3. Russian datasets loaded via cLang-8")
        
        elif language == 'en':
            # W&I+LOCNESS for English
            if bea_samples != 0:
                print(f"\n3. Loading W&I+LOCNESS data (target: {bea_samples or 'all'})...")
                try:
                    wi_data = load_wi_locness_train(bea_samples)
                    print(f"   Loaded {len(wi_data)} W&I+LOCNESS entries")
                    
                    if wi_data:
                        wi_sentences = process_multilingual_dataset(
                            wi_data, "WI_LOCNESS", language, nlp, annotator, merge_edits
                        )
                        all_sentences.extend(wi_sentences)
                except Exception as e:
                    print(f"   Error loading W&I+LOCNESS: {e}")
            
            # FCE for English
            if bea_samples != 0:
                print(f"\n4. Loading FCE data (target: {bea_samples or 'all'})...")
                try:
                    fce_data = load_fce_train(bea_samples)
                    print(f"   Loaded {len(fce_data)} FCE entries")
                    
                    if fce_data:
                        fce_sentences = process_multilingual_dataset(
                            fce_data, "FCE", language, nlp, annotator, merge_edits
                        )
                        all_sentences.extend(fce_sentences)
                except Exception as e:
                    print(f"   Error loading FCE: {e}")
            
            # Lang-8 for English
            if bea_samples != 0:
                print(f"\n5. Loading Lang-8 data (target: {bea_samples or 'all'})...")
                try:
                    lang8_data = load_lang8_data(
                        'data/raw/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat', 
                        bea_samples, language
                    )
                    print(f"   Loaded {len(lang8_data)} Lang-8 entries")
                    
                    if lang8_data:
                        lang8_sentences = process_multilingual_dataset(
                            lang8_data, "LANG8", language, nlp, annotator, merge_edits
                        )
                        all_sentences.extend(lang8_sentences)
                except Exception as e:
                    print(f"   Error loading Lang-8: {e}")
    
    # Process Spanish CoWSL2H data if requested
    if 'es' in args.languages or args.cowsl2h_samples > 0:
        print(f"\n{'='*20} Processing Spanish (CoWSL2H) {'='*20}")
        
        try:
            nlp, annotator = setup_language_models('es')
            
            print(f"\nLoading CoWSL2H data (target: {args.cowsl2h_samples})...")
            cowsl2h_data = load_cowsl2h_data('data/raw/cowsl2h', args.cowsl2h_samples)
            print(f"   Loaded {len(cowsl2h_data)} CoWSL2H entries")
            
            if cowsl2h_data:
                cowsl2h_sentences = process_multilingual_dataset(
                    cowsl2h_data, "COWSL2H", 'es', nlp, annotator, merge_edits
                )
                all_sentences.extend(cowsl2h_sentences)
        except Exception as e:
            print(f"   Error processing Spanish data: {e}")
    
    # Final shuffle and output
    print(f"\n{'='*20} Finalizing Dataset {'='*20}")
    print(f"Total sentences: {len(all_sentences)}")
    
    if all_sentences:
        # Shuffle the combined dataset
        random.shuffle(all_sentences)
        
        # Write output
        output_file = output_dir / 'multigec2025_training.jsonl'
        write_training_dataset(all_sentences, output_file, "MultiGEC 2025 Training Mixture")
        
        # Write summary
        summary_file = output_dir / 'dataset_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("MultiGEC 2025 Multilingual Training Data Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Count by language and dataset
            lang_counts = {}
            dataset_counts = {}
            
            for sent in all_sentences:
                lang = sent.get('language', 'unknown')
                dataset = sent['dataset']
                
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                dataset_key = f"{dataset}_{lang}"
                dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + 1
            
            f.write("Language distribution:\n")
            for lang, count in sorted(lang_counts.items()):
                f.write(f"  {lang}: {count} sentences\n")
            
            f.write("\nDataset composition:\n")
            for dataset_key, count in sorted(dataset_counts.items()):
                f.write(f"  {dataset_key}: {count} sentences\n")
            
            f.write(f"\nTotal: {len(all_sentences)} sentences\n")
            f.write(f"Output file: {output_file}\n")
            
            # Note about dataset availability
            f.write("\nDataset availability notes:\n")
            f.write("  - cLang-8: Available for EN/DE/RU in third-party/clang8/\n")
            f.write("  - UberText: Requires download and processing\n")
            f.write("  - Run shell/download_missing_datasets.sh to fetch additional data\n")
        
        print(f"Written to: {output_file}")
        print(f"Summary: {summary_file}")
        
        # Write language-specific files
        for lang in set(sent.get('language', 'unknown') for sent in all_sentences):
            if lang != 'unknown':
                lang_sentences = [s for s in all_sentences if s.get('language') == lang]
                lang_file = output_dir / f'multigec2025_{lang}.jsonl'
                write_training_dataset(lang_sentences, lang_file, f"MultiGEC 2025 {lang.upper()}")
        
    else:
        print("Error: No valid sentences found!")
        return 1
    
    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
