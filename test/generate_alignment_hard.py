#!/usr/bin/env python3
"""
Generate a dataset of 100 tricky alignment cases: data/eval/alignment_hard.csv

Each row has columns: src, tgt
Cases include:
- Multiple consecutive spaces and tabs
- Non-breaking spaces, zero-width joiners
- Mixed quotes/apostrophes and dashes
- Ellipsis char vs three dots
- Combining diacritics vs precomposed
- Inside-word substitutions, punctuation spacing, newlines
"""

from __future__ import annotations

from pathlib import Path
import unicodedata
import random
import csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'eval' / 'alignment_hard.csv'


def nfc(s: str) -> str:
    return unicodedata.normalize('NFC', s)


def nfd(s: str) -> str:
    return unicodedata.normalize('NFD', s)


def build_cases() -> list[tuple[str, str]]:
    nbsp = '\u00A0'
    zwj = '\u200D'
    zwnj = '\u200C'
    ell = '…'
    emdash = '—'
    endash = '–'
    hyphen = '-'
    rquote = '’'
    lquote = '‘'
    rdquote = '”'
    ldquote = '“'
    lig_fi = 'ﬁ'

    base: list[tuple[str, str]] = [
        # Multiple spaces collapse/expand
        ("This  has  double  spaces.", "This has double spaces."),
        ("Multiple   spaces   between   words.", "Multiple spaces between words."),
        ("Ends with   spaces   ", "Ends with spaces "),
        ("  Starts  with  spaces", " Starts with spaces"),
        ("Tabs\tand\twords", "Tabs    and    words"),
        ("Mix\t of  tabs \t and  spaces", "Mix    of tabs    and spaces"),

        # Non-breaking/zero-width
        (f"A{nbsp}B non-breaking", "A B non-breaking"),
        (f"Keep{zwj}calm and code", "Keepcalm and code"),
        (f"Co{zwnj}operate wisely", "Cooperate wisely"),

        # Quotes and apostrophes
        ("It's a test", f"It{rquote}s a test"),
        ("He said, \"hello\"", f"He said, {ldquote}hello{rdquote}"),
        ("'Single' quotes", f"{lquote}Single{rquote} quotes"),

        # Dashes and hyphens
        ("state-of-the-art", "state of the art"),
        ("Ranges 1-3 are ok", f"Ranges 1{endash}3 are ok"),
        ("Dash - spaced", f"Dash {emdash} spaced"),

        # Ellipsis
        ("Wait...", f"Wait{ell}"),
        ("Dots.... too many", f"Dots{ell} too many"),

        # Diacritics combining vs precomposed
        (nfd("café"), nfc("café")),
        (nfd("naïve"), nfc("naïve")),
        (nfd("élève"), nfc("élève")),

        # Ligatures
        (f"of{lig_fi}ce", "office"),

        # Punctuation spacing
        ("Hello ,world !", "Hello, world!"),
        ("What ? Why , though ?", "What? Why, though?"),
        ("Space before : colon", "Space before: colon"),

        # Newlines
        ("Line1\nLine2", "Line1 Line2"),
        ("Line1\n\nParagraph", "Line1  Paragraph"),

        # Inside-word edits
        ("reenter", "re-enter"),
        ("cooperate", f"co{zwj}operate"),
        ("email", "e-mail"),
        ("foo_bar", "foo-bar"),

        # Surrounding punctuation
        ("( hello )", "(hello)"),
        ("[ test ]", "[test]"),
        ("{ code }", "{code}"),

        # Leading/trailing whitespace
        ("  lead", "lead"),
        ("trail  ", "trail "),

        # Mixed unicode spacing edge cases
        (f"A {nbsp}{nbsp} B", "A   B"),
        (f"C{nbsp} {nbsp}D", "C   D"),

        # Emojis and skin tones (multi-code points)
        ("Thumb 👍🏽 up", "Thumb 👍 up"),
        ("Family 👨‍👩‍👧", "Family 👪"),

        # Case-only changes
        ("Title", "title"),
        ("ßtraße", "Strasse"),

        # Quotes/apostrophes mix
        (f"{ldquote}Quote{rdquote}", '"Quote"'),
        (f"{lquote}apos{rquote}", "'apos'"),

        # Repeated punctuation
        ("No!!! way??", "No! way?"),

        # Directional punctuation with spaces
        ("hello « world »", "hello «world»"),

        # Currency and separators
        ("$ 1,000 . 00", "$1,000.00"),

        # Accents removal
        ("résumé", "resume"),
        ("São Paulo", "Sao Paulo"),

        # Tabs to single spaces
        ("col1\tcol2", "col1 col2"),
        ("a\tb\tc", "a b c"),

        # Zero-width spaces around punctuation
        (f"Zero\u200B width", "Zero width"),
        (f"Hello,\u200Bworld", "Hello, world"),
    ]

    # Expand with programmatic variants to reach 100
    variants: list[tuple[str, str]] = []
    for s, t in base:
        variants.append((s, t))
        # Swap a dot with ellipsis in sentences that have dots
        if '...' in s or '...' in t or '…' in s or '…' in t:
            variants.append((s.replace('...', '…'), t.replace('...', '…')))
        # Add leading/trailing spaces variations
        variants.append((' ' + s, ' ' + t))
        variants.append((s + ' ', t + ' '))
        # Duplicate punctuation spacing cases with different punctuation
        variants.append((s.replace(',', ';'), t.replace(',', ';')))
        if hyphen in s or hyphen in t:
            variants.append((s.replace('-', endash), t.replace('-', endash)))

    # Deduplicate while preserving order
    seen = set()
    uniq: list[tuple[str, str]] = []
    for pair in variants:
        if pair not in seen:
            uniq.append(pair)
            seen.add(pair)

    # Trim/extend to exactly 100
    random.seed(7)
    if len(uniq) >= 100:
        uniq = uniq[:100]
    else:
        while len(uniq) < 100:
            uniq.append(random.choice(base))

    return uniq


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs = build_cases()
    with OUT.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'tgt'])
        for s, t in pairs:
            w.writerow([s, t])
    print(f"Wrote {len(pairs)} cases to {OUT}")


if __name__ == '__main__':
    main()


