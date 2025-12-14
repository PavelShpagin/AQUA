"""
Simple sentence splitter for GEC research.
Handles missing spaces after punctuation, strict 10-50 word filtering.
No fallbacks, no heuristics - just clean data or skip.
"""

import re
from typing import List, Tuple


def fix_punctuation_spacing(text: str) -> str:
    """Fix missing spaces after sentence punctuation and concatenated words."""
    # Handle punctuation followed by uppercase letter
    text = re.sub(r'([.!?])([A-ZА-ЯІЇЄҐ][a-zа-яіїєґ])', r'\1 \2', text)  # sentence.Another -> sentence. Another
    text = re.sub(r'([.!?])(\d)', r'\1 \2', text)                        # end.123 -> end. 123
    
    # Handle concatenated words: lowercase+uppercase without punctuation
    # облікуЯ -> обліку. Я
    text = re.sub(r'([a-zа-яіїєґ])([A-ZА-ЯІЇЄҐ])', r'\1. \2', text)
    
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = fix_punctuation_spacing(text)
    # Split on punctuation followed by space and uppercase letter (Latin or Cyrillic)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ])', text)
    return [s.strip() for s in sentences if s.strip()]


def process_text_pair(src_text: str, tgt_text: str) -> List[Tuple[str, str]]:
    """
    Process source/target text pair into valid sentence pairs.
    Returns empty list if no valid pairs found.
    """
    if not src_text or not tgt_text:
        return []
    
    src_sentences = split_sentences(src_text)
    tgt_sentences = split_sentences(tgt_text)
    
    # Must have same number of sentences
    if len(src_sentences) != len(tgt_sentences):
        return []
    
    valid_pairs = []
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_words = len(src.split())
        tgt_words = len(tgt.split())
        
        # All filtering in one place: word count + length ratio
        if not (10 <= src_words <= 50 and 10 <= tgt_words <= 50):
            continue
        
        # 1-to-1 heuristic: reject if one sentence is >2x longer than the other
        if src_words == 0 or tgt_words == 0:
            continue
        length_ratio = max(src_words, tgt_words) / min(src_words, tgt_words)
        if length_ratio > 2.0:
            continue
            
        valid_pairs.append((src, tgt))
    
    return valid_pairs