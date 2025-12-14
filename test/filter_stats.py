#!/usr/bin/env python3
"""
Print per-dataset and merged filter annotation statistics.

Usage:
  # From a folder containing many CSVs
  python -m test.filter_stats --folder data/eval --filter "tgt1 tgt2 tgt3"

This mirrors the stats shown after filtering runs and additionally prints a
merged summary as if all datasets were concatenated. You can also pass files
directly for convenience (fallback if --folder is omitted).
"""

import argparse
import os
import sys
import pandas as pd
from typing import List

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ensemble import print_filter_annotation_stats
except ImportError:
    # Fallback: try direct import if running from project root
    sys.path.insert(0, '.')
    from utils.ensemble import print_filter_annotation_stats


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs='*', help="Input CSV files (optional if --folder is used)")
    ap.add_argument("--folder", default=None, help="Folder with CSV files")
    ap.add_argument("--filter", nargs='*', default=None, help="Filter columns (space/comma separated)")
    args = ap.parse_args()

    # Accept comma-separated filter list as a convenience
    if args.filter and len(args.filter) == 1 and ("," in args.filter[0]):
        args.filter = [c for c in args.filter[0].replace(',', ' ').split() if c]

    # Resolve input files
    files: List[str] = []
    if args.folder:
        if not os.path.isdir(args.folder):
            raise SystemExit(f"Folder not found: {args.folder}")
        for name in sorted(os.listdir(args.folder)):
            if name.lower().endswith('.csv'):
                files.append(os.path.join(args.folder, name))
        if not files:
            raise SystemExit(f"No CSV files found in folder: {args.folder}")
    elif args.inputs:
        files = list(args.inputs)
    else:
        raise SystemExit("Provide --folder or a list of CSV files")

    dfs: List[pd.DataFrame] = []
    print("\n=== Per-dataset Filter Annotation Stats ===")
    for path in files:
        df = load_df(path)
        dfs.append(df)
        print(f"\nDataset: {os.path.basename(path)}  (rows={len(df)})")
        print_filter_annotation_stats(df, args.filter)

    if not dfs:
        print("No inputs provided")
        return

    print("\n=== Merged Filter Annotation Stats ===")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged rows: {len(merged)}")
    print_filter_annotation_stats(merged, args.filter)


if __name__ == "__main__":
    main()


