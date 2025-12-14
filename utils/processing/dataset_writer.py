"""
Training Dataset Writer

Writes processed GEC training data to various formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict


def write_training_dataset(sentences: List[Dict], output_file: Path, dataset_name: str):
    """
    Write training dataset to JSONL format for LLM training.
    
    Args:
        sentences: List of sentence dictionaries with 'src', 'tgt', 'aligned' keys
        output_file: Path to output file
        dataset_name: Name of the dataset for logging
    """
    print(f"Writing {len(sentences)} sentences to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            # Format for instruction tuning
            training_example = {
                'instruction': 'Correct the grammatical errors in the following text:',
                'input': sent['src'],
                'output': sent['tgt'],
                'aligned': sent.get('aligned', sent['tgt']),  # ERRANT alignment
                'dataset': sent.get('dataset', 'unknown'),
                'id': sent.get('id', '')
            }
            
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    
    print(f"Successfully wrote {dataset_name} to {output_file}")


def write_training_dataset_json(sentences: List[Dict], output_file: Path, dataset_name: str) -> None:
    """
    Write training dataset to a single JSON file (array of examples).

    Each item mirrors the JSONL schema used in write_training_dataset.
    """
    print(f"Writing {len(sentences)} sentences to {output_file} (JSON array)...")
    payload = []
    for sent in sentences:
        payload.append({
            'instruction': 'Correct the grammatical errors in the following text:',
            'input': sent['src'],
            'output': sent['tgt'],
            'aligned': sent.get('aligned', sent['tgt']),
            'dataset': sent.get('dataset', 'unknown'),
            'id': sent.get('id', '')
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Successfully wrote {dataset_name} (JSON) to {output_file}")


def write_csv_dataset(sentences: List[Dict], output_file: Path, dataset_name: str):
    """
    Write training dataset to CSV format.
    
    Args:
        sentences: List of sentence dictionaries
        output_file: Path to output CSV file
        dataset_name: Name of the dataset for logging
    """
    print(f"Writing {len(sentences)} sentences to CSV: {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'src', 'tgt', 'aligned', 'dataset']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for sent in sentences:
            writer.writerow({
                'id': sent.get('id', ''),
                'src': sent['src'],
                'tgt': sent['tgt'],
                'aligned': sent.get('aligned', sent['tgt']),
                'dataset': sent.get('dataset', 'unknown')
            })
    
    print(f"Successfully wrote {dataset_name} CSV to {output_file}")


def write_src_tgt_files(sentences: List[Dict], output_dir: Path, dataset_name: str):
    """
    Write training dataset as separate source and target files.
    
    Args:
        sentences: List of sentence dictionaries
        output_dir: Directory to write files
        dataset_name: Name of the dataset for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    src_file = output_dir / f"{dataset_name.lower()}.src"
    tgt_file = output_dir / f"{dataset_name.lower()}.tgt"
    
    print(f"Writing {len(sentences)} sentences to source/target files...")
    
    with open(src_file, 'w', encoding='utf-8') as src_f, \
         open(tgt_file, 'w', encoding='utf-8') as tgt_f:
        
        for sent in sentences:
            src_f.write(sent['src'] + '\n')
            tgt_f.write(sent['tgt'] + '\n')
    
    print(f"Successfully wrote {dataset_name} to {src_file} and {tgt_file}")