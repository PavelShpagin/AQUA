"""
UA-GEC Dataset Loader

Loads sentence-level data from Grammarly's UA-GEC dataset.
Processes source-sentences and target-sentences directories efficiently.

Usage:
    from utils.ua_gec_loader import load_ua_gec_sentences
    pairs = load_ua_gec_sentences('data/raw/ua-gec/data/gec-only/train', 300)
"""

import random
from typing import List, Dict
from pathlib import Path

# Set seed for reproducibility
random.seed(42)


def load_ua_gec_sentences(data_path: str, target_samples: int = None) -> List[Dict]:
    """Load UA-GEC data from source/target sentence directories."""
    data_dir = Path(data_path)
    src_dir = data_dir / "source-sentences"
    tgt_dir = data_dir / "target-sentences"
    
    if not src_dir.exists() or not tgt_dir.exists():
        print(f"Error: UA-GEC directories not found in {data_path}")
        return []
    
    src_files = sorted(src_dir.glob("*.src.txt"))
    valid_pairs = []
    
    for src_file in src_files:
        file_id = src_file.stem.replace('.src', '')
        tgt_file = tgt_dir / f"{file_id}.a1.txt"
        
        if not tgt_file.exists():
            continue
        
        with open(src_file, 'r', encoding='utf-8') as f:
            src_sentences = [line.strip() for line in f if line.strip()]
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_sentences = [line.strip() for line in f if line.strip()]
        
        for src, tgt in zip(src_sentences, tgt_sentences):
            if len(src.split()) >= 10:
                valid_pairs.append({'src_text': src, 'tgt_text': tgt})
                if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                    break
        
        if target_samples is not None and len(valid_pairs) >= target_samples * 3:
            break
    
    print(f"UA-GEC: Processed {len(src_files)} files, found {len(valid_pairs)} valid pairs")
    
    # Return all sampled data (3x target) - final sampling happens after ERRANT
    print(f"Collected {len(valid_pairs)} pairs for downstream processing")
    
    return valid_pairs