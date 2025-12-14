#!/usr/bin/env python3
"""
ERRANT alignment accuracy check with middle-word span detection.

For each dataset row with columns src, tgt:
- Generate ERRANT-based inline alignment {old=>new}
- Apply edits to src to reconstruct tgt
- Verify equality; count failures and middle-word spans

Datasets:
- data/eval/gold_en.csv, gold_de.csv, gold_ua.csv, SpanishFPs.csv, alignment_hard.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.errant_align import get_alignment_for_language


def apply_inline_alignment(aligned: str) -> Tuple[str, List[Tuple[str, str, int, int]]]:
    out_parts: List[str] = []
    spans: List[Tuple[str, str, int, int]] = []
    i = 0
    L = len(aligned)
    while i < L:
        if aligned[i] == '{':
            j = aligned.find('}', i)
            if j == -1:
                out_parts.append(aligned[i])
                i += 1
                continue
            body = aligned[i + 1:j]
            sep = body.find('=>')
            if sep == -1:
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


def is_middle_word_span(context_before: str, old: str, context_after: str) -> bool:
    left = context_before[-1] if context_before else ''
    right = context_after[0] if context_after else ''
    return left.isalnum() and right.isalnum()


def evaluate_file(csv_path: Path, language_hint: str | None = None) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    if 'src' not in df.columns or 'tgt' not in df.columns:
        return {'file': str(csv_path), 'total': 0, 'ok': 0, 'fail': 0, 'accuracy': 0.0, 'middle_word_spans': 0}

    ok = 0
    fail = 0
    middle_word = 0

    # Heuristic language detection: use filename or provided hint
    lang = (language_hint or csv_path.stem.split('_')[-1]).lower()
    if 'spanish' in csv_path.name.lower():
        lang = 'es'
    if 'gold_en' in csv_path.name:
        lang = 'en'
    elif 'gold_de' in csv_path.name:
        lang = 'de'
    elif 'gold_ua' in csv_path.name:
        lang = 'ua'

    for _, row in df.iterrows():
        src = str(row.get('src', ''))
        tgt = str(row.get('tgt', ''))
        aligned = get_alignment_for_language(src, tgt, language=lang)
        recon, spans = apply_inline_alignment(aligned)
        if recon == tgt:
            ok += 1
        else:
            fail += 1

        i = 0
        L = len(aligned)
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
                context_before = aligned[max(0, i - 1):i]
                context_after = aligned[j + 1:j + 2] if j + 1 < L else ''
                if is_middle_word_span(context_before, old, context_after):
                    middle_word += 1
                i = j + 1
            else:
                i += 1

    total = ok + fail
    acc = (ok / total) if total else 0.0
    return {'file': str(csv_path), 'total': total, 'ok': ok, 'fail': fail, 'accuracy': acc, 'middle_word_spans': middle_word}


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

    print("ERRANT Alignment Reconstruction Accuracy:")
    for r in results:
        print(f"- {r['file']}: acc={r['accuracy']:.3f} ok={r['ok']} fail={r['fail']} total={r['total']} middle_word_spans={r['middle_word_spans']}")


if __name__ == '__main__':
    main()


