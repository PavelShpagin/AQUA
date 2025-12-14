#!/usr/bin/env python3
"""
Verify aligned vs reasoning consistency for hardcoded columns.

Checks per pair (candidate, aligned):
- Count rows where reasoning has labels == [] and aligned contains any "{x=>y}".
- Count rows where reasoning has labels != [] and aligned contains no "{x=>y}".

Usage:
  python -m test.verify_quality --input path/to/labeled.csv

Notes:
  - Columns are hardcoded from config.yaml.
  - src/tgt are NOT used.
  - If a required column is missing, it is skipped with a warning.
"""

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
import re


# Hardcoded from config.yaml (filter / filter_aligned)
# Mapping: candidate column -> aligned column
CANDIDATE_TO_ALIGNED: List[Tuple[str, str]] = [
    ("gen_rewrite_1_gpt5_gec_conservative", "gen_annotated_1_gpt5_gec_conservative"),
    ("rerank_selected_1", "rerank_annotated_1"),
]


def safe_parse_json(obj: Any) -> Dict[str, Any] | None:
    """Parse reasoning string into JSON dict. Returns None if parsing fails.

    Handles fenced code blocks and extracts the outermost JSON object if needed.
    """
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return None
    s = str(obj).strip()
    if not s:
        return None
    # Remove markdown fences
    if s.startswith("```") and s.endswith("```") and s.count("```") >= 2:
        parts = s.split("```")
        s = parts[1].strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # Try as-is
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    # Extract outermost braces
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(s[start : end + 1])
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def labels_empty_from_text(obj: Any) -> bool:
    """Fallback detector: treat various CSV-escaped forms like ""labels"": [] as empty.

    - Returns True if we find labels: [] (any number of quotes around key), and we do
      NOT see a non-empty labels array in the same text.
    - Returns False otherwise.
    """
    # Missing/NaN means this column was not attempted → do not count as empty
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return False
    s = str(obj)
    # Non-empty first (e.g., labels: ["TP"] or labels: [ 1 ])
    if re.search(r'"+labels"+\s*:\s*\[\s*[^\]\s]', s, flags=re.IGNORECASE):
        return False
    # Empty array detection with one or more quotes around key
    return bool(re.search(r'"+labels"+\s*:\s*\[\s*\]', s, flags=re.IGNORECASE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to labeled CSV to inspect")
    ap.add_argument("--all", action="store_true", help="Scan all *_reasoning columns present")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    total_issues = 0

    # Determine column pairs to check
    pairs: List[Tuple[str, str | None]] = []
    if args.all:
        # Any column that ends with _reasoning
        for col in df.columns:
            if col.endswith("_reasoning"):
                base = col[: -len("_reasoning")]
                # Try to infer aligned column name for our known pairs
                aligned = None
                if base == "gen_rewrite_1_gpt5_gec_conservative" and "gen_annotated_1_gpt5_gec_conservative" in df.columns:
                    aligned = "gen_annotated_1_gpt5_gec_conservative"
                elif base == "rerank_selected_1" and "rerank_annotated_1" in df.columns:
                    aligned = "rerank_annotated_1"
                pairs.append((base, aligned))
    else:
        pairs = [(c, a) for (c, a) in CANDIDATE_TO_ALIGNED]

    for cand_col, aligned_col in pairs:
        reason_col = f"{cand_col}_reasoning"
        if cand_col not in df.columns:
            print(f"[verify] WARN: Missing candidate column '{cand_col}', skipping.")
            continue
        if reason_col not in df.columns:
            print(f"[verify] WARN: Missing reasoning column '{reason_col}', skipping.")
            continue
        if aligned_col and aligned_col not in df.columns:
            print(f"[verify] WARN: Missing aligned column '{aligned_col}', skipping aligned checks.")

        # Determine rows attempted for this column and whether labels == []
        reason_series = df[reason_col]
        attempted = reason_series.notna() & (reason_series.astype(str).str.strip() != "")
        labels_empty_mask = []
        for val, is_attempt in zip(reason_series.tolist(), attempted.tolist()):
            if not is_attempt:
                labels_empty_mask.append(False)
                continue
            parsed = safe_parse_json(val)
            if parsed is not None:
                labels = parsed.get("labels")
                labels_empty_mask.append(isinstance(labels, list) and len(labels) == 0)
            else:
                labels_empty_mask.append(labels_empty_from_text(val))

        # Build Series for mask
        mask = pd.Series(labels_empty_mask, index=df.index)
        sub = df[mask]
        print(f"\n[verify] Column: {cand_col}")
        print(f"[verify]   rows with labels==[]: {len(sub)}")

        # Aligned vs reasoning consistency checks (if aligned column available)
        def has_edit_spans(x: Any) -> bool:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return False
            s = str(x)
            # Accept whitespace-only edits as valid; drop exact {=>} and identical spans {x=>x}
            spans = re.findall(r"\{[^}]*=>[^}]*\}", s)
            for sp in spans:
                body = sp[1:-1]
                left, right = body.split('=>', 1)
                if left == right:
                    # drop identical spans (including exact {=>} when both empty)
                    continue
                return True
            return False

        aligned_has_edits = df[aligned_col].map(has_edit_spans) if aligned_col else pd.Series([False]*len(df), index=df.index)
        labels_empty = mask & attempted
        labels_non_empty = (~mask) & attempted

        # Case A: aligned has edits but labels are empty → suspicious
        a = df[aligned_has_edits & labels_empty]
        # Case B: aligned has no edits but labels are non-empty → suspicious
        b = df[(~aligned_has_edits) & labels_non_empty]

        if aligned_col:
            print(f"[verify]   aligned '{aligned_col}' has edits & labels==[]: {len(a)}")
        if not a.empty:
            total_issues += len(a)
            cols = ["idx", cand_col, reason_col] + ([aligned_col] if aligned_col else [])
            show_cols = [c for c in cols if c in a.columns]
            print(a.head(10)[show_cols].to_string(index=False))
            if 'idx' in a.columns:
                print("[verify]   offending idx (has edits & labels==[]):", ",".join(map(str, a['idx'].tolist())))

        if aligned_col:
            print(f"[verify]   aligned '{aligned_col}' no-edits & labels!=[]: {len(b)}")
        if not b.empty:
            total_issues += len(b)
            cols = ["idx", cand_col, reason_col] + ([aligned_col] if aligned_col else [])
            show_cols = [c for c in cols if c in b.columns]
            print(b.head(10)[show_cols].to_string(index=False))
            if 'idx' in b.columns:
                print("[verify]   offending idx (no-edits & labels!=[]):", ",".join(map(str, b['idx'].tolist())))

    rc = 1 if total_issues > 0 else 0
    print(f"\n[verify] Done. Issues found: {total_issues}. Exit code: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())


