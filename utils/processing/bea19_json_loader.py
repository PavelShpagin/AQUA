"""
BEA19 JSON Dataset Loader
Loads raw text from BEA19 JSON files (W&I+LOCNESS, FCE)
"""

import json
import random
from typing import List, Dict, Tuple


def load_bea19_json(json_path: str, target_samples: int = None) -> List[Dict[str, str]]:
    """
    Load BEA19 JSON data and convert to src/tgt pairs.
    
    Args:
        json_path: Path to JSON file
        target_samples: Number of samples to load (None for all)
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Convert to src/tgt pairs
    pairs = []
    for item in data:
        src_text = item['text']
        
        # Apply character-level edits to get target text
        tgt_text = apply_character_edits(src_text, item['edits'])
        
        pairs.append({
            'src_text': src_text,
            'tgt_text': tgt_text,
            'id': item['id']
        })
    
    # Sample if requested
    if target_samples is not None and len(pairs) > target_samples * 3:
        pairs = random.sample(pairs, target_samples * 3)
    
    return pairs


def apply_character_edits(text: str, edits_list: List) -> str:
    """
    Apply character-level edits to text.
    
    Args:
        text: Original text
        edits_list: List of edit groups in format [[annotator_id, [[edit1], [edit2], ...]]]
    
    Returns:
        Corrected text
    """
    if not edits_list or not edits_list[0]:
        return text
    
    # Get the edits from first annotator group: [annotator_id, [[edits]]]
    annotator_group = edits_list[0]
    if len(annotator_group) < 2 or not annotator_group[1]:
        return text
    
    edits = annotator_group[1]  # This should be [[edit1], [edit2], ...]
    
    # Sort edits by start position in reverse order to avoid offset issues
    edits = sorted(edits, key=lambda x: x[0], reverse=True)
    
    result = text
    for edit in edits:
        if len(edit) >= 3:
            start, end, replacement = edit[0], edit[1], edit[2]
            
            if replacement is None or replacement == "":
                # Deletion
                result = result[:start] + result[end:]
            else:
                # Replacement or insertion
                result = result[:start] + replacement + result[end:]
    
    return result


def load_wi_locness_train(target_samples: int = None) -> List[Dict[str, str]]:
    """Load W&I+LOCNESS training data from JSON files."""
    pairs = []
    
    # Load all training files
    for subset in ['A', 'B', 'C']:
        json_path = f'data/raw/BEA/wi+locness/json/{subset}.train.json'
        try:
            subset_pairs = load_bea19_json(json_path, None)  # Load all first
            pairs.extend(subset_pairs)
        except FileNotFoundError:
            print(f"Warning: {json_path} not found")
    
    # Sample if requested
    if target_samples is not None and len(pairs) > target_samples * 3:
        pairs = random.sample(pairs, target_samples * 3)
    
    return pairs


def load_fce_train(target_samples: int = None) -> List[Dict[str, str]]:
    """Load FCE training data from JSON file."""
    json_path = 'data/raw/BEA/fce/json/fce.train.json'
    return load_bea19_json(json_path, target_samples)
