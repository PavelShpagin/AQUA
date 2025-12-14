#!/usr/bin/env python3
"""
Count :::ERROR tags in processed data files.

Usage:
    python test/error_tags.py --lang en
    python test/error_tags.py --lang de  
    python test/error_tags.py --lang ua
    python test/error_tags.py --lang all
"""

import argparse
import csv
import os
import re
from typing import Dict, List


def count_error_tags_in_file(file_path: str) -> Dict[str, int]:
    """Count :::ERROR tags in a CSV file's alignment columns."""
    if not os.path.exists(file_path):
        return {"file_exists": False, "total_rows": 0, "error_tags": 0, "rows_with_errors": 0}
    
    total_rows = 0
    error_tags = 0
    rows_with_errors = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                
                # Check all possible alignment columns
                alignment_columns = ['aligned', 'aligned_labels', 'aligned_human', 'llm_edits', 'llm_edits_h']
                row_error_count = 0
                
                for col in alignment_columns:
                    if col in row and row[col]:
                        # Count :::ERROR tags in this column
                        row_error_count += len(re.findall(r':::ERROR', row[col]))
                
                error_tags += row_error_count
                
                if row_error_count > 0:
                    rows_with_errors += 1
                    
    except Exception as e:
        return {"file_exists": True, "error": str(e), "total_rows": 0, "error_tags": 0, "rows_with_errors": 0}
    
    return {
        "file_exists": True,
        "total_rows": total_rows,
        "error_tags": error_tags,
        "rows_with_errors": rows_with_errors
    }


def find_processed_files(lang: str) -> List[str]:
    """Find processed CSV files for a given language."""
    processed_dir = "data/results/processed_edits"
    files = []
    
    if os.path.exists(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.endswith('.csv') and lang in filename:
                files.append(os.path.join(processed_dir, filename))

    return files


def main():
    parser = argparse.ArgumentParser(description="Count :::ERROR tags in processed data")
    parser.add_argument("--lang", choices=["en", "de", "ua", "all"], default="all",
                       help="Language to check (or 'all' for all languages)")
    args = parser.parse_args()
    
    if args.lang == "all":
        languages = ["en", "de", "ua"]
    else:
        languages = [args.lang]
    
    print("ERROR Tag Analysis")
    print("=" * 50)
    
    total_files_checked = 0
    total_error_tags = 0
    total_rows_with_errors = 0
    total_rows = 0
    
    for lang in languages:
        print(f"\nLanguage: {lang.upper()}")
        print("-" * 20)
        
        files = find_processed_files(lang)
        
        if not files:
            print(f"No processed files found for language '{lang}'")
            continue
        
        lang_error_tags = 0
        lang_rows_with_errors = 0
        lang_total_rows = 0
        
        for file_path in files:
            stats = count_error_tags_in_file(file_path)
            
            if not stats["file_exists"]:
                print(f"  {os.path.basename(file_path)}: FILE NOT FOUND")
                continue
                
            if "error" in stats:
                print(f"  {os.path.basename(file_path)}: ERROR - {stats['error']}")
                continue
            
            print(f"  {os.path.basename(file_path)}:")
            print(f"    Total rows: {stats['total_rows']}")
            print(f"    Rows with :::ERROR: {stats['rows_with_errors']}")
            print(f"    Total :::ERROR tags: {stats['error_tags']}")
            
            if stats['total_rows'] > 0:
                error_rate = (stats['rows_with_errors'] / stats['total_rows']) * 100
                print(f"    Error rate: {error_rate:.1f}%")
            
            lang_error_tags += stats['error_tags']
            lang_rows_with_errors += stats['rows_with_errors']
            lang_total_rows += stats['total_rows']
            total_files_checked += 1
        
        if lang_total_rows > 0:
            lang_error_rate = (lang_rows_with_errors / lang_total_rows) * 100
            print(f"\n  Language {lang.upper()} Summary:")
            print(f"    Total rows: {lang_total_rows}")
            print(f"    Rows with errors: {lang_rows_with_errors}")
            print(f"    Total error tags: {lang_error_tags}")
            print(f"    Error rate: {lang_error_rate:.1f}%")
            
            total_error_tags += lang_error_tags
            total_rows_with_errors += lang_rows_with_errors
            total_rows += lang_total_rows
    
    if total_files_checked > 0:
        print(f"\nOVERALL SUMMARY")
        print("=" * 50)
        print(f"Files checked: {total_files_checked}")
        print(f"Total rows: {total_rows}")
        print(f"Rows with :::ERROR tags: {total_rows_with_errors}")
        print(f"Total :::ERROR tags: {total_error_tags}")
        if total_rows > 0:
            overall_error_rate = (total_rows_with_errors / total_rows) * 100
            print(f"Overall error rate: {overall_error_rate:.1f}%")


if __name__ == "__main__":
    main()
