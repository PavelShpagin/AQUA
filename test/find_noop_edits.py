#!/usr/bin/env python3
"""
Scan datasets for no-op edits of the form {x=>x} produced by DMP and ERRANT.
Writes matches to data/eval/noop_edits.csv and prints summary counts.
"""

from __future__ import annotations

from pathlib import Path
import csv
import sys
from typing import List, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.diff_extension import diff_by_words
from utils.errant_align import get_alignment_for_language


def dmp_inline(src: str, tgt: str) -> str:
    diffs = diff_by_words(src, tgt, cleanup_semantic=True)
    out: List[str] = []
    del_buf: List[str] = []
    ins_buf: List[str] = []
    def flush():
        nonlocal del_buf, ins_buf
        if del_buf or ins_buf:
            out.append('{' + ''.join(del_buf) + '=>' + ''.join(ins_buf) + '}')
            del_buf[:] = []
            ins_buf[:] = []
    for op, text in diffs:
        if op == 0:
            flush()
            out.append(text)
        elif op == -1:
            del_buf.append(text)
        elif op == 1:
            ins_buf.append(text)
    flush()
    return ''.join(out)


def parse_noops(aligned: str) -> List[str]:
    noops: List[str] = []
    i = 0
    L = len(aligned)
    while i < L:
        if aligned[i] == '{':
            j = aligned.find('}', i)
            if j == -1:
                break
            body = aligned[i + 1:j]
            sep = body.find('=>')
            if sep != -1:
                old = body[:sep]
                new = body[sep + 2:]
                if old == new and old != '':
                    noops.append('{' + body + '}')
            i = j + 1
        else:
            i += 1
    return noops


def detect_lang_from_name(name: str) -> str:
    n = name.lower()
    if 'gold_en' in n:
        return 'en'
    if 'gold_de' in n:
        return 'de'
    if 'gold_ua' in n:
        return 'ua'
    if 'spanish' in n or 'es' in n:
        return 'es'
    return 'en'


def main() -> None:
    eval_dir = ROOT / 'data' / 'eval'
    files = [
        eval_dir / 'gold_en.csv',
        eval_dir / 'gold_de.csv',
        eval_dir / 'gold_ua.csv',
        eval_dir / 'SpanishFPs.csv',
        eval_dir / 'alignment_hard.csv',
        eval_dir / 'alignment_diverse_200.csv',
    ]

    out_csv = eval_dir / 'noop_edits.csv'
    rows = []
    summary = { 'dmp': 0, 'errant': 0 }

    for f in files:
        if not f.exists():
            continue
        lang = detect_lang_from_name(f.name)
        df = pd.read_csv(f, low_memory=False)
        for _, r in df.iterrows():
            src = str(r.get('src',''))
            tgt = str(r.get('tgt',''))
            # DMP
            dmp_aligned = dmp_inline(src, tgt)
            dmp_noops = parse_noops(dmp_aligned)
            for seg in dmp_noops:
                rows.append({ 'file': str(f), 'lang': lang, 'which': 'dmp', 'src': src, 'tgt': tgt, 'segment': seg, 'aligned': dmp_aligned })
                summary['dmp'] += 1
            # ERRANT
            err_aligned = get_alignment_for_language(src, tgt, language=lang)
            err_noops = parse_noops(err_aligned)
            for seg in err_noops:
                rows.append({ 'file': str(f), 'lang': lang, 'which': 'errant', 'src': src, 'tgt': tgt, 'segment': seg, 'aligned': err_aligned })
                summary['errant'] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file','lang','which','src','tgt','segment','aligned'])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"No-op edits found: DMP={summary['dmp']} ERRANT={summary['errant']} -> {out_csv}")


if __name__ == '__main__':
    main()


