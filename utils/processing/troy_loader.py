"""
Troy Synthetic Datasets Loader

Loads synthetic GEC data from Troy-1BW (One Billion Word) and Troy-Blogs datasets.
These datasets are generated through ensemble knowledge distillation.

Usage:
    from utils.processing.troy_loader import load_troy_1bw_data, load_troy_blogs_data
    troy_1bw = load_troy_1bw_data('data/raw/troy-1bw', 1000)
    troy_blogs = load_troy_blogs_data('data/raw/troy-blogs', 1000)
"""

import random
import re
from typing import List, Dict, Optional
from pathlib import Path
from .sentence_splitter import process_text_pair

# Set seed for reproducibility
random.seed(42)


def load_troy_1bw_data(data_path: str, target_samples: int = None) -> List[Dict]:
    """
    Load Troy-1BW synthetic data (generated from One Billion Word Benchmark).
    
    Args:
        data_path: Path to Troy-1BW directory
        target_samples: Number of samples to load (None for all)
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Troy-1BW directory not found: {data_path}")
        return []
    
    valid_pairs = []
    
    # Look for processed Troy-1BW files first
    troy_files = list(data_dir.glob('**/troy-1bw*.tsv')) + list(data_dir.glob('**/troy_1bw*.tsv'))
    
    if not troy_files:
        # Process raw One Billion Word files directly
        print("Troy-1BW processed files not found. Processing raw One Billion Word data...")
        billion_word_dir = data_dir / "1-billion-word-language-modeling-benchmark-r13output"
        
        if billion_word_dir.exists():
            # Find training files
            training_dir = billion_word_dir / "training-monolingual.tokenized.shuffled"
            if training_dir.exists():
                en_files = list(training_dir.glob("news.en-*"))
                if en_files:
                    print(f"Found {len(en_files)} One Billion Word training files")
                    return process_one_billion_word_files(en_files, target_samples)
        
        print("No Troy-1BW data found. Please run download script first.")
        return []
    
    print(f"Loading Troy-1BW data from {len(troy_files)} files...")
    
    for troy_file in troy_files:
        try:
            file_pairs = process_troy_file(troy_file, 'troy_1bw')
            valid_pairs.extend(file_pairs)
            
            # Early stopping
            if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                break
                
        except Exception as e:
            print(f"Error processing {troy_file}: {e}")
            continue
    
    # Sample if requested
    if target_samples is not None and len(valid_pairs) > target_samples * 3:
        valid_pairs = random.sample(valid_pairs, target_samples * 3)
    
    print(f"Troy-1BW: Found {len(valid_pairs)} valid pairs")
    return valid_pairs


def load_troy_blogs_data(data_path: str, target_samples: int = None) -> List[Dict]:
    """
    Load Troy-Blogs synthetic data (generated from Blog Authorship Corpus).
    
    Args:
        data_path: Path to Troy-Blogs directory
        target_samples: Number of samples to load (None for all)
    
    Returns:
        List of dicts with 'src_text' and 'tgt_text' keys
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Troy-Blogs directory not found: {data_path}")
        return []
    
    valid_pairs = []
    
    # Look for processed Troy-Blogs files
    troy_files = list(data_dir.glob('**/troy-blogs*.tsv')) + list(data_dir.glob('**/troy_blogs*.tsv'))
    
    if not troy_files:
        # Try to find raw Blog Authorship files
        print("Troy-Blogs processed files not found. Looking for raw Blog Authorship data...")
        raw_files = list(data_dir.glob('**/blogs/**/*.xml')) + list(data_dir.glob('**/*.blog'))
        
        if raw_files:
            print(f"Found {len(raw_files)} raw blog files. These need to be processed with ensemble GEC models.")
            print("Troy-Blogs requires synthetic correction generation - not implemented in this loader.")
            return []
        else:
            print("No Troy-Blogs data found. Please run download script first.")
            return []
    
    print(f"Loading Troy-Blogs data from {len(troy_files)} files...")
    
    for troy_file in troy_files:
        try:
            file_pairs = process_troy_file(troy_file, 'troy_blogs')
            valid_pairs.extend(file_pairs)
            
            # Early stopping
            if target_samples is not None and len(valid_pairs) >= target_samples * 3:
                break
                
        except Exception as e:
            print(f"Error processing {troy_file}: {e}")
            continue
    
    # Sample if requested
    if target_samples is not None and len(valid_pairs) > target_samples * 3:
        valid_pairs = random.sample(valid_pairs, target_samples * 3)
    
    print(f"Troy-Blogs: Found {len(valid_pairs)} valid pairs")
    return valid_pairs


def process_troy_file(file_path: Path, dataset_prefix: str) -> List[Dict]:
    """Process a Troy TSV file with source-target pairs."""
    pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
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
                    pairs.append({
                        'src_text': src_text,
                        'tgt_text': tgt_text,
                        'id': f"{dataset_prefix}_{file_path.stem}_{line_num}"
                    })
    
    return pairs


def process_one_billion_word_files(en_files: List[Path], target_samples: Optional[int] = None) -> List[Dict]:
    """Process One Billion Word files to create synthetic GEC data."""
    pairs = []
    processed_files = 0
    
    for file_path in en_files:
        if processed_files >= 3:  # Limit to first 3 files to avoid overload
            break
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f"Processing {file_path.name}...")
                
                for line_num, line in enumerate(f):
                    if target_samples and len(pairs) >= target_samples:
                        break
                        
                    line = line.strip()
                    if not line or len(line.split()) < 10 or len(line.split()) > 50:
                        continue
                    
                    # Create simple synthetic errors (placeholder for ensemble approach)
                    error_text = introduce_synthetic_errors(line)
                    if error_text != line:
                        pairs.append({
                            'src_text': error_text,
                            'tgt_text': line,
                            'id': f'troy_1bw_{file_path.stem}_{line_num}'
                        })
                    
                    # Process limited lines per file
                    if line_num > 5000:
                        break
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
        processed_files += 1
    
    return pairs


def introduce_synthetic_errors(text: str) -> str:
    """Introduce synthetic grammatical errors into clean text."""
    words = text.split()
    if len(words) < 5:
        return text
    
    # Simple error patterns (placeholder for more sophisticated ensemble approach)
    modified_words = []
    
    for i, word in enumerate(words):
        if random.random() < 0.15:  # 15% chance of introducing an error
            error_type = random.choice(['article', 'plural', 'verb_form', 'preposition', 'spelling'])
            
            if error_type == 'article' and word.lower() in ['a', 'an', 'the']:
                # Remove or wrong article
                if random.random() < 0.5:
                    continue  # Remove article
                else:
                    replacements = {'a': 'an', 'an': 'a', 'the': 'a'}
                    modified_words.append(replacements.get(word.lower(), word))
                    
            elif error_type == 'plural' and word.endswith('s') and len(word) > 3:
                # Wrong plural form
                modified_words.append(word[:-1])
                
            elif error_type == 'verb_form' and (word.endswith('ed') or word.endswith('ing')):
                # Wrong verb form
                if word.endswith('ed'):
                    modified_words.append(word[:-2])
                elif word.endswith('ing'):
                    modified_words.append(word[:-3])
                else:
                    modified_words.append(word)
                    
            elif error_type == 'preposition' and word.lower() in ['in', 'on', 'at', 'by', 'with']:
                # Wrong preposition
                preps = ['in', 'on', 'at', 'by', 'with']
                preps.remove(word.lower())
                modified_words.append(random.choice(preps))
                
            elif error_type == 'spelling' and len(word) > 4:
                # Simple spelling errors
                if random.random() < 0.5:
                    # Character substitution
                    pos = random.randint(1, len(word) - 2)
                    chars = list(word)
                    chars[pos] = random.choice('aeiou')
                    modified_words.append(''.join(chars))
                else:
                    modified_words.append(word)
            else:
                modified_words.append(word)
        else:
            modified_words.append(word)
    
    return ' '.join(modified_words)


def process_raw_blog_corpus(data_path: str, output_path: str) -> bool:
    """
    Process raw Blog Authorship Corpus files.
    Note: This is a placeholder - actual Troy-Blogs requires ensemble GEC model processing.
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        return False
    
    print("Raw Blog Authorship Corpus processing placeholder.")
    print("To generate Troy-Blogs, you need:")
    print("1. Ensemble of GEC models (as described in the paper)")
    print("2. Knowledge distillation process") 
    print("3. Synthetic error generation")
    print("This loader only handles already-processed Troy data.")
    
    return False


def count_troy_entries(data_path: str, dataset_type: str = 'troy-1bw') -> int:
    """Count available entries in Troy dataset directory."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        return 0
    
    pattern = f'**/{dataset_type.replace("-", "_")}*.tsv'
    troy_files = list(data_dir.glob(pattern))
    
    count = 0
    for troy_file in troy_files:
        try:
            with open(troy_file, 'r', encoding='utf-8') as f:
                count += sum(1 for line in f if line.strip())
        except Exception:
            continue
    
    return count
