#!/usr/bin/env python3
"""
Compare DMP vs ERRANT alignments on curated edge cases.
Logs cases where DMP introduces middle-word spans but ERRANT does not.
Writes results to data/eval/dmp_vs_errant_edges.csv
"""

from __future__ import annotations

from pathlib import Path
import csv
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.diff_extension import diff_by_words
from utils.errant_align import get_alignment_for_language


def dmp_inline(src: str, tgt: str) -> str:
    diffs = diff_by_words(src, tgt, cleanup_semantic=True)
    # Convert DMP diffs to inline {old=>new}
    out: List[str] = []
    del_buf: List[str] = []
    ins_buf: List[str] = []
    def flush():
        nonlocal del_buf, ins_buf
        if del_buf or ins_buf:
            out.append('{' + ''.join(del_buf) + '=>' + ''.join(ins_buf) + '}')
            del_buf = []
            ins_buf = []
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


def has_middle_word_spans(aligned: str) -> bool:
    i = 0
    L = len(aligned)
    while i < L:
        if aligned[i] == '{':
            j = aligned.find('}', i)
            if j == -1:
                break
            left = aligned[i - 1] if i - 1 >= 0 else ''
            right = aligned[j + 1] if j + 1 < L else ''
            if left.isalnum() and right.isalnum():
                return True
            i = j + 1
        else:
            i += 1
    return False


def reconstruct(aligned: str) -> str:
    parts: List[str] = []
    i = 0
    L = len(aligned)
    while i < L:
        if aligned[i] == '{':
            j = aligned.find('}', i)
            if j == -1:
                parts.append(aligned[i])
                i += 1
                continue
            body = aligned[i + 1:j]
            sep = body.find('=>')
            if sep == -1:
                parts.append('{' + body + '}')
                i = j + 1
                continue
            new = body[sep + 2:]
            parts.append(new)
            i = j + 1
        else:
            parts.append(aligned[i])
            i += 1
    return ''.join(parts)


def edge_cases() -> List[Tuple[str, str, str]]:
    # (lang, src, tgt)
    cases = [
        ('en', "reenter", "re-enter"),                # hyphen insertion
        ('en', "do not", "don't"),                   # contraction
        ('en', "Hello , world !", "Hello, world!"),   # punctuation spacing
        ('en', "Wait... now.", "Wait… now."),        # ellipsis
        ('en', "ofﬁce", "office"),                   # ligature
        ('en', "cafe\u0301", "café"),               # combining accent
        ('en', "Hello\u00A0world", "Hello world"),   # NBSP
        ('en', "x\u200Dy", "xy"),                   # ZWJ
        ('en', "co\u200Cooperate", "cooperate"),     # ZWNJ
        ('en', "Hello\u200Bworld", "Hello world"),   # zero-width space
        ('en', "Line1\nLine2", "Line1 Line2"),       # newline normalization
        ('de', "Straße", "Strasse"),                  # ß -> ss
        ('en', "A(BC)", "A (BC)"),                    # spacing before bracket
        ('en', "COVID 19", "COVID-19"),                # hyphenate number
    ]
    return cases


def main():
    out_path = ROOT / 'data' / 'eval' / 'dmp_vs_errant_edges.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for lang, src, tgt in edge_cases():
        dmp_aligned = dmp_inline(src, tgt)
        err_aligned = get_alignment_for_language(src, tgt, language=lang)
        dmp_mid = has_middle_word_spans(dmp_aligned)
        err_mid = has_middle_word_spans(err_aligned)
        dmp_recon_ok = (reconstruct(dmp_aligned) == tgt)
        err_recon_ok = (reconstruct(err_aligned) == tgt)
        if dmp_mid and not err_mid and err_recon_ok:
            rows.append({
                'lang': lang,
                'src': src,
                'tgt': tgt,
                'dmp_aligned': dmp_aligned,
                'errant_aligned': err_aligned,
                'dmp_middle_word': dmp_mid,
                'errant_middle_word': err_mid,
                'dmp_recon_ok': dmp_recon_ok,
                'errant_recon_ok': err_recon_ok,
            })

    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['lang','src','tgt','dmp_aligned','errant_aligned','dmp_middle_word','errant_middle_word','dmp_recon_ok','errant_recon_ok'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} edge cases to {out_path}")


if __name__ == '__main__':
    main()
