"""
FalkoMerlin Sentence-Level Dataset Loader
Loads sentence-level German GEC data from TSV format
"""

import random
from typing import List, Dict, Tuple
from pathlib import Path

# Set seed for reproducibility
random.seed(42)


def load_falko_merlin_data(data_path: str, target_samples: int = None) -> List[Dict[str, str]]:
    """
    Load sentence-level FalkoMerlin data from TSV files.
    
    Args:
        data_path: Path to FalkoMerlin TSV file
        target_samples: Number of samples to load (None for all)
    
    Returns:
        List of dicts with 'src_text', 'tgt_text', and 'id' keys
    """
    if not Path(data_path).exists():
        print(f"Error: FalkoMerlin file not found: {data_path}")
        return []
    
    sentences = []
    current_sentence = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line:  # Empty line = sentence boundary
                if current_sentence:
                    # Keep only sentences without incorrect tokens to ensure 1-to-1 matching
                    if all(tag != 'i' for _, tag in current_sentence):
                        src_text, tgt_text = process_sentence_tokens(current_sentence)
                        if src_text and tgt_text and len(src_text.split()) >= 10:
                            sentences.append({
                                'src_text': src_text,
                                'tgt_text': tgt_text,
                                'id': f"falko_{len(sentences):06d}"
                            })
                    current_sentence = []
                    
                    # Early stopping if we have enough samples
                    if target_samples is not None and len(sentences) >= target_samples * 3:
                        break
            else:
                # Parse token and correction tag
                parts = line.split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    current_sentence.append((token, tag))
    
    # Process last sentence if file doesn't end with empty line
    if current_sentence:
        if all(tag != 'i' for _, tag in current_sentence):
            src_text, tgt_text = process_sentence_tokens(current_sentence)
            if src_text and tgt_text and len(src_text.split()) >= 10:
                sentences.append({
                    'src_text': src_text,
                    'tgt_text': tgt_text,
                    'id': f"falko_{len(sentences):06d}"
                })
    
    # Sample if requested
    if target_samples is not None and len(sentences) > target_samples * 3:
        sentences = random.sample(sentences, target_samples * 3)
    
    return sentences


def process_sentence_tokens(tokens: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Process a list of (token, tag) pairs to create source and target sentences.
    
    Args:
        tokens: List of (token, correction_tag) tuples
    
    Returns:
        Tuple of (source_text, target_text)
    """
    src_tokens = []
    tgt_tokens = []
    
    for token, tag in tokens:
        if tag == 'c':  # Correct token - appears in both src and tgt
            src_tokens.append(token)
            tgt_tokens.append(token)
        elif tag == 'i':  # Incorrect token - only in src, deleted in tgt
            src_tokens.append(token)
            # Don't add to tgt_tokens (deletion)
        else:
            # Unknown tag, treat as correct
            src_tokens.append(token)
            tgt_tokens.append(token)
    
    # Join tokens with spaces
    src_text = ' '.join(src_tokens)
    tgt_text = ' '.join(tgt_tokens)
    
    # Basic cleanup for German text
    src_text = cleanup_german_text(src_text)
    tgt_text = cleanup_german_text(tgt_text)
    
    return src_text, tgt_text


def cleanup_german_text(text: str) -> str:
    """Clean up German text spacing around punctuation."""
    # Remove spaces before punctuation
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    text = text.replace(' ;', ';')
    text = text.replace(' :', ':')
    text = text.replace(' )', ')')
    text = text.replace('( ', '(')
    text = text.replace(' "', '"')
    text = text.replace(' „', '„')
    text = text.replace(' "', '"')
    
    # Handle German quotes
    text = text.replace('„ ', '„')
    text = text.replace(' "', '"')
    
    # Clean up multiple spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text.strip()


def count_falko_merlin_sentences(data_path: str) -> int:
    """Count available sentences in FalkoMerlin file."""
    if not Path(data_path).exists():
        return 0
    
    count = 0
    current_sentence = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line = sentence boundary
                if current_sentence:
                    count += 1
                    current_sentence = []
            else:
                current_sentence.append(line)
    
    # Count last sentence if file doesn't end with empty line
    if current_sentence:
        count += 1
    
    return count
