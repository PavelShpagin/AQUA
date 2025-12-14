#!/usr/bin/env python3
"""
Simple accuracy check for diff-match-patch alignment reconstruction.

For each dataset row with columns src, tgt (and optionally aligned):
- Build a word-level diff via diff-match-patch
- Convert to inline alignment notation {old=>new}
- Apply edits to src to reconstruct tgt
- Verify reconstructed equals tgt; count failures

Also detect middle-word errors where any {old=>new} spans split a token
in the middle (alphanumeric boundaries).

Datasets:
- data/eval/gold_en.csv, gold_de.csv, gold_ua.csv
- data/eval/SpanishFPs.csv
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse our thin DMP wrapper for word-level diffs
from utils.diff_extension import diff_by_words


def diffs_to_inline_alignment(src: str, diffs: List[Tuple[int, str]]) -> str:
    """
    Convert DMP diffs to inline alignment embedded into the original src string.

    Diffs are tuples (op, text) where op in {-1, 0, 1} for delete, equal, insert.
    We produce an inline string by walking the src, emitting unchanged segments
    verbatim, and wrapping edits as {old=>new}. Consecutive delete/insert pairs
    are merged into a single replacement.
    """
    result_parts: List[str] = []
    src_cursor = 0

    # We need to consume from src for EQUAL and DELETE operations, and not advance
    # for INSERT operations (which insert relative to current src position).
    pending_insert: List[str] = []
    pending_delete: List[str] = []

    def flush_pending() -> None:
        nonlocal pending_insert, pending_delete
        if not pending_insert and not pending_delete:
            return
        old_text = ''.join(pending_delete)
        new_text = ''.join(pending_insert)
        result_parts.append(f"{{{old_text}=>{new_text}}}")
        pending_insert = []
        pending_delete = []

    # Walk through diffs while consuming src
    for op, text in diffs:
        if op == 0:  # EQUAL
            # Emit any pending replacement before unchanged text
            flush_pending()
            # Append the equal text directly (should match src at this point)
            result_parts.append(text)
            src_cursor += len(text)
        elif op == -1:  # DELETE from src
            pending_delete.append(text)
            src_cursor += len(text)
        elif op == 1:  # INSERT into tgt
            pending_insert.append(text)
        else:
            # Unexpected op; flush and append raw text safely
            flush_pending()
            result_parts.append(text)

    # Flush any tail edit
    flush_pending()
    return ''.join(result_parts)


def apply_inline_alignment(aligned: str) -> Tuple[str, List[Tuple[str, str, int, int]]]:
    """
    Apply inline alignment string to reconstruct target text.

    Returns tuple of (reconstructed_text, spans_info)
    spans_info entries: (old, new, start_index_in_result, end_index_in_result)
    """
    out_parts: List[str] = []
    spans: List[Tuple[str, str, int, int]] = []

    i = 0
    L = len(aligned)
    while i < L:
        if aligned[i] == '{':
            j = aligned.find('}', i)
            if j == -1:
                # Malformed; treat as literal
                out_parts.append(aligned[i])
                i += 1
                continue
            body = aligned[i + 1:j]
            # Split on the first occurrence of => only
            sep = body.find('=>')
            if sep == -1:
                # Malformed; include raw
                out_parts.append('{' + body + '}')
                i = j + 1
                continue
            old = body[:sep]
            new = body[sep + 2:]
            start = sum(len(p) for p in out_parts)
            out_parts.append(new)
            end = sum(len(p) for p in out_parts)
            spans.append((old, new, start, end))
            i = j + 1
        else:
            out_parts.append(aligned[i])
            i += 1

    return ''.join(out_parts), spans


word_boundary_re = re.compile(r"[A-Za-z0-92-]")


def is_middle_word_span(context_before: str, old: str, context_after: str) -> bool:
    """
    Heuristic: an edit splits a word if both sides adjacent to the edit are
    alphanumeric (no whitespace/punct) and old/new themselves don't start/end at
    whitespace boundaries. Use simple character class heuristic.
    """
    left = context_before[-1] if context_before else ''
    right = context_after[0] if context_after else ''
    left_is_word = left.isalnum()
    right_is_word = right.isalnum()
    # If both neighbors are alnum and at least one of old/new is non-empty,
    # we consider it as potentially splitting a word boundary
    return left_is_word and right_is_word


def evaluate_file(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    # Require src/tgt
    if 'src' not in df.columns or 'tgt' not in df.columns:
        return {
            'file': str(csv_path),
            'total': 0,
            'ok': 0,
            'fail': 0,
            'accuracy': 0.0,
            'middle_word_spans': 0,
        }

    ok = 0
    fail = 0
    middle_word = 0

    for _, row in df.iterrows():
        src = str(row.get('src', ''))
        tgt = str(row.get('tgt', ''))
        diffs = diff_by_words(src, tgt, cleanup_semantic=True)
        aligned = diffs_to_inline_alignment(src, diffs)
        recon, spans = apply_inline_alignment(aligned)
        if recon == tgt:
            ok += 1
        else:
            fail += 1
        # middle-word heuristic check: find positions in original src-aligned stream
        # Build the aligned stream with original unchanged contexts: we already have aligned
        # Walk through aligned to detect if a span is placed between alnum chars
        # Reconstruct context from aligned string while scanning
        # Simple pass: scan original aligned string
        i = 0
        L = len(aligned)
        out_len = 0
        while i < L:
            if aligned[i] == '{':
                j = aligned.find('}', i)
                if j == -1:
                    break
                body = aligned[i + 1:j]
                sep = body.find('=>')
                if sep == -1:
                    i = j + 1
                    continue
                old = body[:sep]
                new = body[sep + 2:]
                context_before = aligned[max(0, i - 1):i]
                context_after = aligned[j + 1:j + 2] if j + 1 < L else ''
                if is_middle_word_span(context_before, old, context_after):
                    middle_word += 1
                i = j + 1
            else:
                i += 1

    total = ok + fail
    acc = (ok / total) if total else 0.0
    return {
        'file': str(csv_path),
        'total': total,
        'ok': ok,
        'fail': fail,
        'accuracy': acc,
        'middle_word_spans': middle_word,
    }


def main():
    eval_dir = ROOT / 'data' / 'eval'
    files = [
        eval_dir / 'gold_en.csv',
        eval_dir / 'gold_de.csv',
        eval_dir / 'gold_ua.csv',
        eval_dir / 'SpanishFPs.csv',
        eval_dir / 'alignment_hard.csv',
        eval_dir / 'alignment_diverse_200.csv',
        eval_dir / 'unicode_bench.csv',
        eval_dir / 'paragraph_bench.csv',
    ]
    results = []
    for f in files:
        if f.exists():
            res = evaluate_file(f)
            results.append(res)
        else:
            results.append({'file': str(f), 'total': 0, 'ok': 0, 'fail': 0, 'accuracy': 0.0, 'middle_word_spans': 0})

    print("DMP Alignment Reconstruction Accuracy:")
    for r in results:
        print(f"- {r['file']}: acc={r['accuracy']:.3f} ok={r['ok']} fail={r['fail']} total={r['total']} middle_word_spans={r['middle_word_spans']}")


if __name__ == '__main__':
    main()


