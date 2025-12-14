"""
Lang-8 Dataset Loader

Loads sentence-level data from Lang-8 corpus in JSON format.
Processes learner text corrections for GEC training.

Usage:
    from utils.processing.lang8_loader import load_lang8_data
    pairs = load_lang8_data('data/raw/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat', 1000, 'en')
"""

import json
import random
from typing import List, Dict, Optional
from pathlib import Path

# Set seed for reproducibility
random.seed(42)


def load_lang8_data(data_path: str, target_samples: int = None, language: str = 'en') -> List[Dict]:
    """
    Load Lang-8 data from the .dat format file.
    
    Args:
        data_path: Path to Lang-8 .dat file
        target_samples: Number of samples to load (None for all)
        language: Target language ('en' for English, 'de' for German, etc.)
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    if not Path(data_path).exists():
        print(f"Error: Lang-8 file not found: {data_path}")
        return []
    
    lang_map = {
        'en': 'English',
        'de': 'German', 
        'ja': 'Japanese',
        'ko': 'Korean',
        'fr': 'French',
        'es': 'Spanish'
    }
    
    target_lang = lang_map.get(language, 'English')
    valid_pairs = []
    
    print(f"Loading Lang-8 data for {target_lang}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines, found {len(valid_pairs)} {target_lang} pairs...")
            
            # Early stopping for memory efficiency
            if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                break
                
            try:
                entry = json.loads(line.strip())
                
                # Check if this entry is for our target language
                if len(entry) < 4:
                    continue
                    
                # entry format: [id, user_id, native_lang, learning_lang, [title, sentences...], corrections]
                learning_lang = entry[3]
                if learning_lang != target_lang:
                    continue
                
                # Extract original text and corrections
                if len(entry) < 6:
                    continue
                    
                original_content = entry[4]  # List of sentences
                corrections = entry[5]       # Nested correction structure
                
                # Process the text and corrections
                sentence_pairs = process_lang8_corrections(original_content, corrections)
                
                for src_text, tgt_text in sentence_pairs:
                    if src_text and tgt_text and len(src_text.split()) >= 10:
                        valid_pairs.append({
                            'src_text': src_text,
                            'tgt_text': tgt_text,
                            'id': f"lang8_{entry[0]}_{len(valid_pairs)}"
                        })
                        
            except (json.JSONDecodeError, IndexError, KeyError):
                # Skip malformed lines
                continue
    
    # Sample if requested
    if target_samples is not None and len(valid_pairs) > target_samples * 3:
        valid_pairs = random.sample(valid_pairs, target_samples * 3)
    
    print(f"Lang-8: Found {len(valid_pairs)} valid pairs for {target_lang}")
    return valid_pairs


def process_lang8_corrections(original_content: List[str], corrections: List) -> List[tuple]:
    """
    Process Lang-8 corrections to extract source-target sentence pairs.
    
    Args:
        original_content: List of original sentences
        corrections: Nested correction structure
    
    Returns:
        List of (src_text, tgt_text) tuples
    """
    pairs = []
    
    if not corrections or len(corrections) == 0:
        return pairs
    
    # Skip title (first item) and process sentences
    sentences = original_content[1:] if len(original_content) > 1 else []
    
    # Process corrections - take the first corrector's suggestions
    if len(corrections) > 0 and corrections[0]:
        corrected_sentences = corrections[0][1:]  # Skip title corrections
        
        for i, (orig_sent, corr_sent) in enumerate(zip(sentences, corrected_sentences)):
            if isinstance(orig_sent, str) and isinstance(corr_sent, list):
                # Take the first correction if multiple exist
                if corr_sent and len(corr_sent) > 0:
                    corrected = corr_sent[0] if isinstance(corr_sent[0], str) else orig_sent
                else:
                    corrected = orig_sent
                
                # Clean up the text
                src_text = clean_lang8_text(orig_sent)
                tgt_text = clean_lang8_text(corrected)
                
                if src_text and tgt_text and src_text != tgt_text:
                    pairs.append((src_text, tgt_text))
    
    return pairs


def clean_lang8_text(text: str) -> str:
    """Clean up Lang-8 text by removing formatting and extra whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML-like formatting tags
    import re
    
    # Remove formatting tags like [f-blue], [/f-blue], [sline], etc.
    text = re.sub(r'\[/?[a-z-]+\]', '', text)
    text = re.sub(r'\[/?f-[a-z]+\]', '', text)
    text = re.sub(r'\[/?sline\]', '', text)
    text = re.sub(r'\[/?f-bold\]', '', text)
    text = re.sub(r'\[/?f-red\]', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def count_lang8_entries(data_path: str, language: str = 'en') -> int:
    """Count available entries in Lang-8 file for specific language."""
    if not Path(data_path).exists():
        return 0
    
    lang_map = {
        'en': 'English',
        'de': 'German',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    target_lang = lang_map.get(language, 'English')
    count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if len(entry) >= 4 and entry[3] == target_lang:
                    count += 1
            except (json.JSONDecodeError, IndexError):
                continue
    
    return count
