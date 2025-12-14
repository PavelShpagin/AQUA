"""
NUCLE Dataset Loader

Loads learner essay data from NUCLE corpus in SGML format.
Processes CoNLL-14 test data for GEC training.

Usage:
    from utils.processing.nucle_loader import load_nucle_data
    pairs = load_nucle_data('data/raw/BEA/conll14st-test-data/noalt', 500)
"""

import random
import re
from typing import List, Dict, Optional
from pathlib import Path
from .sentence_splitter import process_text_pair

# Set seed for reproducibility
random.seed(42)


def load_nucle_data(data_path: str, target_samples: int = None) -> List[Dict]:
    """
    Load NUCLE data from SGML files.
    
    Args:
        data_path: Path to NUCLE SGML directory
        target_samples: Number of samples to load (None for all)
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: NUCLE directory not found: {data_path}")
        return []
    
    valid_pairs = []
    
    # Find all SGML files
    sgml_files = list(data_dir.glob('*.sgml'))
    
    print(f"Loading NUCLE data from {len(sgml_files)} SGML files...")
    
    for sgml_file in sgml_files:
        print(f"  Processing {sgml_file.name}...")
        try:
            file_pairs = process_nucle_sgml(sgml_file)
            valid_pairs.extend(file_pairs)
            
            # Early stopping for memory efficiency
            if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                break
                
        except Exception as e:
            print(f"    Error processing {sgml_file.name}: {e}")
            continue
    
    # Sample if requested
    if target_samples is not None and len(valid_pairs) > target_samples * 3:
        valid_pairs = random.sample(valid_pairs, target_samples * 3)
    
    print(f"NUCLE: Found {len(valid_pairs)} valid pairs")
    return valid_pairs


def process_nucle_sgml(sgml_file: Path) -> List[Dict]:
    """Process a single NUCLE SGML file."""
    pairs = []
    
    try:
        with open(sgml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse documents from SGML
        documents = parse_sgml_documents(content)
        
        for doc_id, doc_text in documents:
            # Split into sentences and filter by word count
            sentence_pairs = process_text_pair(doc_text, doc_text)  # No corrections available, use original
            
            for src_sent, tgt_sent in sentence_pairs:
                if len(src_sent.split()) >= 10:
                    pairs.append({
                        'src_text': src_sent,
                        'tgt_text': tgt_sent,  # Same as source since we don't have corrections
                        'id': f"nucle_{doc_id}_{len(pairs)}"
                    })
                    
    except Exception as e:
        print(f"    Error reading {sgml_file}: {e}")
        
    return pairs


def parse_sgml_documents(content: str) -> List[tuple]:
    """Parse documents from SGML content."""
    documents = []
    
    # Find all documents
    doc_pattern = r'<DOC\s+nid="(\d+)">(.*?)</DOC>'
    doc_matches = re.findall(doc_pattern, content, re.DOTALL)
    
    for doc_id, doc_content in doc_matches:
        # Extract text content
        text_pattern = r'<TEXT>(.*?)</TEXT>'
        text_match = re.search(text_pattern, doc_content, re.DOTALL)
        
        if text_match:
            text_content = text_match.group(1)
            
            # Extract paragraphs and title
            text_parts = []
            
            # Extract title
            title_pattern = r'<TITLE>(.*?)</TITLE>'
            title_match = re.search(title_pattern, text_content, re.DOTALL)
            if title_match:
                title_text = clean_sgml_text(title_match.group(1))
                if title_text.strip():
                    text_parts.append(title_text.strip())
            
            # Extract paragraphs
            para_pattern = r'<P>(.*?)</P>'
            para_matches = re.findall(para_pattern, text_content, re.DOTALL)
            
            for para_content in para_matches:
                para_text = clean_sgml_text(para_content)
                if para_text.strip():
                    text_parts.append(para_text.strip())
            
            # Combine all text parts
            if text_parts:
                full_text = ' '.join(text_parts)
                documents.append((doc_id, full_text))
    
    return documents


def clean_sgml_text(text: str) -> str:
    """Clean SGML text by removing tags and normalizing whitespace."""
    # Remove any remaining SGML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def count_nucle_documents(data_path: str) -> int:
    """Count available documents in NUCLE SGML files."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        return 0
    
    count = 0
    sgml_files = list(data_dir.glob('*.sgml'))
    
    for sgml_file in sgml_files:
        try:
            with open(sgml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count documents
            doc_pattern = r'<DOC\s+nid="(\d+)">'
            doc_matches = re.findall(doc_pattern, content)
            count += len(doc_matches)
            
        except Exception:
            continue
    
    return count
