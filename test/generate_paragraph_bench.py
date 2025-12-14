#!/usr/bin/env python3
"""
Generate 100 multi-paragraph alignment cases with newline-edge edits.
Outputs: data/eval/paragraph_bench.csv (src,tgt)
Includes edits that cross line breaks or sit flush against them.
"""

from __future__ import annotations

from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'eval' / 'paragraph_bench.csv'


def build_cases() -> list[tuple[str, str]]:
    cases = []
    # Base patterns
    bases = [
        ("Line1\nLine2", "Line1 Line2"),               # join lines
        ("Line1\n\nLine3", "Line1\nLine3"),          # reduce blank lines
        ("Para A.\nPara B.", "Para A. Para B."),      # move period spacing across NL
        ("Hello,\nworld!", "Hello, world!"),          # punctuation spacing across NL
        ("Start\n-middle", "Start-\nmiddle"),        # hyphen alignment around NL
        ("foo\nbar", "foo-bar"),                      # hyphenate across NL
        ("A\nB C", "A B\nC"),                        # swap spacing with NL
        ("Title:\n item", "Title: item"),            # colon rule across NL
        ("Note\n( extra )", "Note\n(extra)"),        # bracket spacing near NL
        ("prefix\n\tindent", "prefix\nindent"),     # tab removal
    ]

    for s, t in bases:
        cases.append((s, t))
        # Variations
        cases.append((s + "\nEnd.", t + "\nEnd."))
        cases.append(("Intro.\n" + s, "Intro.\n" + t))
        cases.append((s.replace(" ", "  "), t.replace(" ", " ")))  # double spaces
        cases.append((s.replace("\n", "\n\n"), t.replace("\n", "\n\n")))  # extra blank lines

    # Pad to 100 with simple paragraph joins/splits
    while len(cases) < 100:
        cases.append(("A\nB\nC", "A B C"))

    return cases[:100]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs = build_cases()
    with OUT.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'tgt'])
        for s, t in pairs:
            w.writerow([s, t])
    print(f"Wrote {len(pairs)} pairs to {OUT}")


if __name__ == '__main__':
    main()


