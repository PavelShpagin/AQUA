"""
UberText Dataset Loader

Loader for UberText Ukrainian social media GEC dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/lang-uk/UberText-GEC
Contains Ukrainian social media text with automatic and human corrections.
"""

from typing import List, Dict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available. Install with: pip install datasets")


def load_ubertext_data(data_dir: str, max_samples: int = 1000) -> List[Dict[str, str]]:
    """
    Load UberText Ukrainian social media GEC data from Hugging Face.
    
    Args:
        data_dir: Path to UberText raw data directory (used as fallback)
        max_samples: Maximum number of samples to load
        
    Returns:
        List of dictionaries with 'src', 'tgt', and 'idx' keys
    """
    sentences = []
    
    # Try to load from Hugging Face first
    if DATASETS_AVAILABLE:
        try:
            print("Loading UberText-GEC from Hugging Face...")
            
            # The dataset has schema conflicts, so we need to load it differently
            # Load only the main data file (not the annotations with different schema)  
            dataset = load_dataset("lang-uk/UberText-GEC", data_files="uber_text_gec.csv", split="train")
            print(f"Found {len(dataset)} UberText entries on Hugging Face")
            
            # Process samples
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                
                if 'text' in item and 'correction' in item:
                    src_text = str(item['text']).strip()
                    tgt_text = str(item['correction']).strip()
                    
                    # Quality filtering
                    if src_text and tgt_text and src_text != tgt_text:
                        words_src = len(src_text.split())
                        words_tgt = len(tgt_text.split())
                        
                        # Keep reasonable length sentences
                        if 10 <= words_src <= 50 and 10 <= words_tgt <= 50:
                            sentences.append({
                                'src_text': src_text,
                                'tgt_text': tgt_text,
                                'id': f'ubertext_{count}'
                            })
                            count += 1
            
            print(f"UberText-GEC: Loaded {len(sentences)} Ukrainian social media pairs from Hugging Face")
            return sentences
            
        except Exception as e:
            print(f"Failed to load UberText-GEC from Hugging Face: {e}")
            print("Falling back to local file processing...")
    
    # Fallback to local file processing
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print("UberText not available - no local directory and HF loading failed. Skipping.")
        return []
    
    # Look for local UberText data files
    csv_files = list(data_path.glob("**/*.csv"))
    json_files = list(data_path.glob("**/*.json"))
    existing_files = [f for f in csv_files + json_files if f.exists() and f.stat().st_size > 100]
    
    if not existing_files:
        print("UberText not available - directory exists but no valid files found. Skipping.")
        return []
    
    print(f"Found {len(existing_files)} local UberText files")
    
    for file_path in existing_files[:3]:  # Process first 3 files
        try:
            if 'gec' in file_path.name.lower() or 'uber' in file_path.name.lower():
                if file_path.suffix == '.csv':
                    sentences.extend(_load_csv_file(file_path, max_samples))
                elif file_path.suffix == '.json':
                    sentences.extend(_load_json_file(file_path, max_samples))
                    
            if len(sentences) >= max_samples:
                break
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"UberText: Loaded {len(sentences)} Ukrainian social media pairs from local files")
    return sentences[:max_samples]


def _load_json_file(file_path: Path, max_samples: int) -> List[Dict[str, str]]:
    """Load UberText from JSON format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    if isinstance(data, list):
        for item in data:
            if len(sentences) >= max_samples:
                break
            if isinstance(item, dict) and 'source' in item and 'target' in item:
                sentences.append({
                    'src': item['source'].strip(),
                    'tgt': item['target'].strip(),
                    'idx': f'ubertext_{len(sentences)}'
                })
    
    return sentences


def _load_jsonl_file(file_path: Path, max_samples: int) -> List[Dict[str, str]]:
    """Load UberText from JSONL format."""
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if len(sentences) >= max_samples:
                break
                
            try:
                item = json.loads(line.strip())
                if 'source' in item and 'target' in item:
                    sentences.append({
                        'src': item['source'].strip(),
                        'tgt': item['target'].strip(),
                        'idx': f'ubertext_{line_num}'
                    })
            except json.JSONDecodeError:
                continue
    
    return sentences


def _load_tsv_file(file_path: Path, max_samples: int) -> List[Dict[str, str]]:
    """Load UberText from TSV format (source\ttarget)."""
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if len(sentences) >= max_samples:
                break
                
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                sentences.append({
                    'src': parts[0].strip(),
                    'tgt': parts[1].strip(), 
                    'idx': f'ubertext_{line_num}'
                })
    
    return sentences


def _load_csv_file(file_path: Path, max_samples: int) -> List[Dict[str, str]]:
    """Load UberText from CSV format (text,correction columns)."""
    sentences = []
    
    try:
        import pandas as pd
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Look for text and correction columns
        text_cols = [col for col in df.columns if 'text' in col.lower()]
        correction_cols = [col for col in df.columns if 'correction' in col.lower()]
        
        if text_cols and correction_cols:
            text_col = text_cols[0]
            correction_col = correction_cols[0]
            
            for i, row in df.iterrows():
                if len(sentences) >= max_samples:
                    break
                    
                if pd.notna(row[text_col]) and pd.notna(row[correction_col]):
                    src_text = str(row[text_col]).strip()
                    tgt_text = str(row[correction_col]).strip()
                    
                    if src_text and tgt_text and src_text != tgt_text:
                        words_src = len(src_text.split())
                        words_tgt = len(tgt_text.split())
                        
                        if 10 <= words_src <= 50 and 10 <= words_tgt <= 50:
                            sentences.append({
                                'src_text': src_text,
                                'tgt_text': tgt_text,
                                'id': f'ubertext_csv_{i}'
                            })
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
    
    return sentences


def _load_txt_file(file_path: Path, max_samples: int) -> List[Dict[str, str]]:
    """Load UberText from TXT format (assuming parallel files or alternating lines)."""
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Try alternating lines format (source, target, source, target...)
    if len(lines) >= 2:
        for i in range(0, len(lines) - 1, 2):
            if len(sentences) >= max_samples:
                break
                
            src = lines[i].strip()
            tgt = lines[i + 1].strip()
            
            if src != tgt:  # Only include if there's a difference
                sentences.append({
                    'src_text': src,
                    'tgt_text': tgt,
                    'id': f'ubertext_{i//2}'
                })
    
    return sentences