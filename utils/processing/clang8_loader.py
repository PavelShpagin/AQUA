"""
cLang-8 Dataset Loader

Loads synthetic corrections from cLang-8 (cleaned Lang-8) dataset.
Processes TSV files with source-target pairs.

Usage:
    from utils.processing.clang8_loader import load_clang8_data
    pairs = load_clang8_data('third-party/clang8/output_data/clang8_source_target_en.tsv', 1000)
"""

import random
from typing import List, Dict, Optional
from pathlib import Path

# Set seed for reproducibility
random.seed(42)


def load_clang8_data(data_path: str, target_samples: int = None, language: str = 'en') -> List[Dict]:
    """
    Load cLang-8 data from TSV file.
    
    Args:
        data_path: Path to cLang-8 TSV file
        target_samples: Number of samples to load (None for all)
        language: Language code for identification
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"Error: cLang-8 file not found: {data_path}")
        return []
    
    valid_pairs = []
    
    print(f"Loading cLang-8 {language.upper()} data from {data_file.name}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  Processed {line_num} lines, found {len(valid_pairs)} valid pairs...")
            
            # Early stopping for memory efficiency
            if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                break
                
            line = line.strip()
            if not line:
                continue
            
            # Parse TSV: source \t target
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            src_text = parts[0].strip()
            tgt_text = parts[1].strip()
            
            # Quality filtering
            if src_text and tgt_text and src_text != tgt_text:
                # Filter by word count
                if 10 <= len(src_text.split()) <= 50:
                    valid_pairs.append({
                        'src_text': src_text,
                        'tgt_text': tgt_text,
                        'id': f"clang8_{language}_{line_num}"
                    })
    
    # Sample if requested
    if target_samples is not None and len(valid_pairs) > target_samples * 3:
        valid_pairs = random.sample(valid_pairs, target_samples * 3)
    
    print(f"cLang-8 {language.upper()}: Found {len(valid_pairs)} valid pairs")
    return valid_pairs


def load_all_clang8_languages(base_dir: str, target_samples: int = None) -> Dict[str, List[Dict]]:
    """Load cLang-8 data for all available languages."""
    base_path = Path(base_dir)
    
    # Language files mapping
    lang_files = {
        'en': base_path / 'clang8_source_target_en.tsv',
        'de': base_path / 'clang8_source_target_de.tsv', 
        'ru': base_path / 'clang8_source_target_ru.tsv'
    }
    
    results = {}
    for lang, file_path in lang_files.items():
        if file_path.exists():
            results[lang] = load_clang8_data(str(file_path), target_samples, lang)
        else:
            print(f"Warning: cLang-8 {lang.upper()} file not found: {file_path}")
            results[lang] = []
    
    return results


def count_clang8_entries(data_path: str) -> int:
    """Count available entries in cLang-8 TSV file."""
    data_file = Path(data_path)
    if not data_file.exists():
        return 0
    
    count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    
    return count
