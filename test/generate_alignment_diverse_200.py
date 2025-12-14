#!/usr/bin/env python3
"""
Generate a dataset of 200 diverse alignment cases: data/eval/alignment_diverse_200.csv

Each row has columns: src, tgt
Covers mixtures of whitespace, punctuation, unicode, tokenization edge cases,
transforms like contractions, hyphenation, diacritics, and script mixing.
"""

from __future__ import annotations

from pathlib import Path
import csv
import random
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'eval' / 'alignment_diverse_200.csv'


def nfc(s: str) -> str:
    return unicodedata.normalize('NFC', s)


def nfd(s: str) -> str:
    return unicodedata.normalize('NFD', s)


def build_seed() -> list[tuple[str, str]]:
    nbsp = '\u00A0'
    zwj = '\u200D'
    zwnj = '\u200C'
    ell = '…'
    emdash = '—'
    endash = '–'

    seed = [
        # Contractions / apostrophes
        ("do not", "don't"),
        ("I am", "I'm"),
        ("she will", "she'll"),
        # Hyphenation
        ("reenter", "re-enter"),
        ("email", "e-mail"),
        ("state of the art", "state-of-the-art"),
        ("COVID 19", "COVID-19"),
        # Punctuation spacing
        ("Hello , world !", "Hello, world!"),
        ("Space before : colon", "Space before: colon"),
        # Unicode spaces
        (f"A{nbsp}B", "A B"),
        (f"x{zwj}y", "xy"),
        (f"co{zwnj}operate", "cooperate"),
        # Ellipsis
        ("Wait... now.", f"Wait{ell} now."),
        # Diacritics
        (nfd("cafe" + "\u0301"), nfc("café")),
        (nfd("naïve"), nfc("naïve")),
        # Quotes
        ("He said, \"hi\"", "He said, “hi”"),
        ("'word'", "‘word’"),
        # Newlines
        ("Line1\nLine2", "Line1 Line2"),
        # Emojis / ZWJ sequences
        ("Family 👨‍👩‍👧", "Family 👪"),
        ("Thumb 👍🏽 up", "Thumb 👍 up"),
        # Numbers formatting
        ("$ 1 , 000 . 00", "$1,000.00"),
        # Script mixing
        ("Aтака", "Атака"),  # Latin A -> Cyrillic А
        ("Straße", "Strasse"),
        # Dashes and ranges
        ("Pages 1-3", f"Pages 1{endash}3"),
        ("Dash - spaced", f"Dash {emdash} spaced"),
        # Brackets spacing
        ("( hello )", "(hello)"),
        ("[ test ]", "[test]"),
        # Tabs
        ("a\tb\tc", "a b c"),
        # Ligatures
        ("ofﬁce", "office"),
        # Case-only with punctuation
        ("Title", "title"),
        # Zero-width space
        ("Hello\u200Bworld", "Hello world"),
    ]
    return seed


def synthesize(seed: list[tuple[str, str]]) -> list[tuple[str, str]]:
    rng = random.Random(13)
    cases: list[tuple[str, str]] = []
    # Base
    cases.extend(seed)

    # Variants
    for s, t in list(seed):
        # pad left/right spaces
        cases.append((' ' + s, ' ' + t))
        cases.append((s + ' ', t + ' '))
        # repeat punctuation
        cases.append((s.replace('...', '……'), t.replace('...', '……')))
        # swap commas/semicolons in spacing examples
        cases.append((s.replace(',', ';'), t.replace(',', ';')))
        # introduce double spaces in source
        cases.append((s.replace(' ', '  '), t.replace(' ', ' ')))

    # Shuffle and trim/extend to 200
    rng.shuffle(cases)
    if len(cases) >= 200:
        cases = cases[:200]
    else:
        while len(cases) < 200:
            cases.append(rng.choice(seed))
    return cases


def main() -> None:
    seed = build_seed()
    pairs = synthesize(seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'tgt'])
        for s, t in pairs:
            w.writerow([s, t])
    print(f"Wrote {len(pairs)} cases to {OUT}")


if __name__ == '__main__':
    main()
