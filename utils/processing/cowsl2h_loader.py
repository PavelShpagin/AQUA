"""
CoWSL2H Corpus Loader

Loads learner essay data from CoWSL2H (Spanish learner corpus).
Processes essay/corrected pairs for GEC training.

Usage:
    from utils.processing.cowsl2h_loader import load_cowsl2h_data
    pairs = load_cowsl2h_data('data/raw/cowsl2h', 500)
"""

import random
from typing import List, Dict, Optional
from pathlib import Path
from .sentence_splitter import process_text_pair

# Set seed for reproducibility
random.seed(42)


def load_cowsl2h_data(data_path: str, target_samples: int = None) -> List[Dict]:
    """
    Load CoWSL2H essay/corrected pairs.
    
    Args:
        data_path: Path to CoWSL2H directory
        target_samples: Number of samples to load (None for all)
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: CoWSL2H directory not found: {data_path}")
        return []
    
    valid_pairs = []
    
    # List of subcorpora to process
    subcorpora = ['famous', 'vacation', 'beautiful', 'yourself', 'special', 'terrible', 'chaplin', 'place_you_dislike']
    
    print(f"Loading CoWSL2H data...")
    
    for subcorpus in subcorpora:
        subcorpus_dir = data_dir / subcorpus
        if not subcorpus_dir.exists():
            continue
            
        # Process all quarters/semesters
        for quarter_dir in subcorpus_dir.iterdir():
            if not quarter_dir.is_dir():
                continue
                
            essays_dir = quarter_dir / 'essays'
            corrected_dir = quarter_dir / 'corrected'
            
            if not essays_dir.exists() or not corrected_dir.exists():
                continue
            
            print(f"  Processing {subcorpus}/{quarter_dir.name}...")
            quarter_pairs = process_quarter_data(essays_dir, corrected_dir)
            valid_pairs.extend(quarter_pairs)
            
            # Early stopping for memory efficiency
            if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                break
        
        if target_samples is not None and len(valid_pairs) >= target_samples * 3:
            break
    
    # Sample if requested
    if target_samples is not None and len(valid_pairs) > target_samples * 3:
        valid_pairs = random.sample(valid_pairs, target_samples * 3)
    
    print(f"CoWSL2H: Found {len(valid_pairs)} valid pairs")
    return valid_pairs


def process_quarter_data(essays_dir: Path, corrected_dir: Path) -> List[Dict]:
    """Process essay/corrected pairs from a quarter directory."""
    pairs = []
    
    # Get all essay files
    essay_files = list(essays_dir.glob('*.txt'))
    
    for essay_file in essay_files:
        # Find corresponding corrected file(s)
        essay_id = extract_essay_id(essay_file.name)
        if not essay_id:
            continue
            
        # Look for corrected files (might have multiple versions)
        corrected_files = list(corrected_dir.glob(f'{essay_id}*.corrected.txt'))
        if not corrected_files:
            continue
        
        # Use the first corrected version
        corrected_file = corrected_files[0]
        
        try:
            # Read essay and corrected text
            with open(essay_file, 'r', encoding='utf-8') as f:
                essay_text = f.read().strip()
            
            with open(corrected_file, 'r', encoding='utf-8') as f:
                corrected_text = f.read().strip()
            
            if essay_text and corrected_text:
                # Split into sentences and filter by word count
                sentence_pairs = process_text_pair(essay_text, corrected_text)
                for src_sent, tgt_sent in sentence_pairs:
                    if len(src_sent.split()) >= 10:
                        pairs.append({
                            'src_text': src_sent,
                            'tgt_text': tgt_sent,
                            'id': f"cowsl2h_{essay_id}_{len(pairs)}"
                        })
                        
        except (UnicodeDecodeError, IOError):
            continue
    
    return pairs


def extract_essay_id(filename: str) -> Optional[str]:
    """Extract essay ID from filename."""
    # Examples: 103377.F17_Famous.txt -> 103377
    # Extract the numeric ID at the start
    parts = filename.split('.')
    if parts and parts[0].isdigit():
        return parts[0]
    return None


def count_cowsl2h_essays(data_path: str) -> int:
    """Count available essay pairs in CoWSL2H corpus."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        return 0
    
    count = 0
    subcorpora = ['famous', 'vacation', 'beautiful', 'yourself', 'special', 'terrible', 'chaplin', 'place_you_dislike']
    
    for subcorpus in subcorpora:
        subcorpus_dir = data_dir / subcorpus
        if not subcorpus_dir.exists():
            continue
            
        for quarter_dir in subcorpus_dir.iterdir():
            if not quarter_dir.is_dir():
                continue
                
            essays_dir = quarter_dir / 'essays'
            corrected_dir = quarter_dir / 'corrected'
            
            if essays_dir.exists() and corrected_dir.exists():
                count += len(list(essays_dir.glob('*.txt')))
    
    return count
