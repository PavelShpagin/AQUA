#!/usr/bin/env python3
"""
Recover missing `src` in a filtered labeled CSV by joining with the original
input CSV on `idx`.

Usage:
  ./venv/bin/python -m test.recover_src \
      --orig data/filtering/es_batch1_unified.csv \
      --filtered data/filter_results/v1/es_batch1_unified_labeled_filtered.csv \
      --out data/filter_results/v1/es_batch1_unified_labeled_filtered_with_src.csv

Options:
  --orig-text-col: name of the original text column to use as `src` if the
                   original file does not already have a `src` column.
                   Default auto-detects from common names.
  --inplace:       overwrite the filtered file instead of writing a new one.
"""

from __future__ import annotations

import argparse
from typing import List
import pandas as pd
import re


COMMON_TEXT_COLS: List[str] = [
    "src", "text", "original", "original_text", "sentence", "input", "raw",
]


def pick_text_col(df: pd.DataFrame, prefer: str | None) -> str:
    if prefer and prefer in df.columns:
        return prefer
    for c in COMMON_TEXT_COLS:
        if c in df.columns:
            return c
    # Fallback: choose the longest average-length string column
    str_cols = [c for c in df.columns if df[c].dtype == object]
    if not str_cols:
        raise ValueError("No suitable text column found in original data.")
    best_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return best_col


def _extract_numeric_index(value: str) -> int | None:
    s = str(value)
    m = re.search(r"idx[\s_:-]*(\d+)", s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    nums = re.findall(r"\d+", s)
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="Path to original input CSV (with idx)")
    ap.add_argument("--filtered", required=True, help="Path to filtered labeled CSV (with idx)")
    ap.add_argument("--out", default="", help="Output path (default: <filtered> with _with_src suffix)")
    ap.add_argument("--orig-text-col", default="", help="Explicit column in --orig to use as src")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the filtered file in place")
    ap.add_argument("--orig-key", default="idx", help="Join key column in --orig (default: idx)")
    ap.add_argument("--filtered-key", default="idx", help="Join key column in --filtered (default: idx)")
    args = ap.parse_args()

    orig = pd.read_csv(args.orig)
    flt = pd.read_csv(args.filtered)

    if "idx" not in orig.columns or "idx" not in flt.columns:
        raise SystemExit("Both --orig and --filtered must contain an 'idx' column for joining.")

    # Normalize merge key dtype to avoid int64 vs object mismatches
    if args.orig_key not in orig.columns or args.filtered_key not in flt.columns:
        raise SystemExit(f"Join keys not found (orig: {args.orig_key!r} in {list(orig.columns)[:10]}..., "
                         f"filtered: {args.filtered_key!r} in {list(flt.columns)[:10]}...) ")
    # Normalize merge key dtype and strip whitespace
    orig_key = args.orig_key
    flt_key = args.filtered_key
    orig[orig_key] = orig[orig_key].astype(str).str.strip()
    flt[flt_key] = flt[flt_key].astype(str).str.strip()

    text_col = pick_text_col(orig, args.orig_text_col or None)
    src_series = orig[[orig_key, text_col]].rename(columns={text_col: "src", orig_key: "__join_key__"})
    merged = flt.copy()
    merged["__join_key__"] = merged[flt_key]
    merged = merged.merge(src_series, on="__join_key__", how="left")
    merged = merged.drop(columns=["__join_key__"]) 

    # If many src still missing, try normalized numeric join for mismatched idx formats
    missing_after_direct = int(merged["src"].isna().sum())
    if missing_after_direct > len(merged) // 2:
        orig_num = orig[orig_key].map(_extract_numeric_index)
        flt_num = flt[flt_key].map(_extract_numeric_index)
        # Fill NA with sequential to keep join stable
        if orig_num.isna().any():
            orig_num = orig_num.fillna(pd.Series(range(len(orig_num))))
        if flt_num.isna().any():
            flt_num = flt_num.fillna(pd.Series(range(len(flt_num))))

        o_base = int(orig_num.min())
        f_base = int(flt_num.min())
        src_series2 = pd.DataFrame({
            "__norm_key__": (orig_num - o_base).astype(int),
            "src": orig[text_col].astype(str)
        })
        merged = flt.copy()
        merged["__norm_key__"] = (flt_num - f_base).astype(int)
        merged = merged.merge(src_series2, on="__norm_key__", how="left")
        merged = merged.drop(columns=["__norm_key__"]) 

    out_path = args.filtered if args.inplace else (args.out or args.filtered.replace(".csv", "_with_src.csv"))
    # Ensure `src` sits immediately after `idx` (or the filtered-key column)
    cols = list(merged.columns)
    key_col = args.filtered_key
    if "src" in cols and key_col in cols:
        cols.remove("src")
        insert_at = cols.index(key_col) + 1
        cols = cols[:insert_at] + ["src"] + cols[insert_at:]
        merged = merged[cols]

    merged.to_csv(out_path, index=False)
    # Basic report
    missing = int(merged["src"].isna().sum())
    total = len(merged)
    print(f"Wrote {out_path} with src column added; missing src rows: {missing}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


