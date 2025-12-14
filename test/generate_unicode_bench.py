#!/usr/bin/env python3
"""
Download and generate a large Unicode normalization benchmark from
Unicode Consortium NormalizationTest.txt.

Outputs: data/eval/unicode_bench.csv with columns: src, tgt

Pairs include (per line):
- NFD -> NFC
- NFKD -> NFKC
- source -> NFC (where different)
"""

from __future__ import annotations

from pathlib import Path
import csv
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'eval' / 'unicode_bench.csv'
URL = 'https://www.unicode.org/Public/UCD/latest/ucd/NormalizationTest.txt'


def hex_seq_to_str(hex_seq: str) -> str:
    hex_seq = hex_seq.strip()
    if not hex_seq:
        return ''
    chars = []
    for h in hex_seq.split():
        try:
            chars.append(chr(int(h, 16)))
        except Exception:
            pass
    return ''.join(chars)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {URL} ...')
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()

    pairs: list[tuple[str, str]] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ';' not in line:
            continue
        # Format per UAX15: c1; c2; c3; c4; c5; # comment
        try:
            parts = line.split('#', 1)[0].split(';')
            if len(parts) < 5:
                continue
            c1 = hex_seq_to_str(parts[0])  # source
            c2 = hex_seq_to_str(parts[1])  # NFC
            c3 = hex_seq_to_str(parts[2])  # NFD
            c4 = hex_seq_to_str(parts[3])  # NFKC
            c5 = hex_seq_to_str(parts[4])  # NFKD
        except Exception:
            continue

        # Build pairs
        if c3 != c2:
            pairs.append((c3, c2))
        if c5 != c4:
            pairs.append((c5, c4))
        if c1 != c2:
            pairs.append((c1, c2))

    # Deduplicate while preserving order
    seen = set()
    uniq: list[tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    # Optionally cap size (keep large but tractable)
    CAP = 5000
    if len(uniq) > CAP:
        uniq = uniq[:CAP]

    with OUT.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'tgt'])
        for s, t in uniq:
            w.writerow([s, t])

    print(f'Wrote {len(uniq)} pairs to {OUT}')


if __name__ == '__main__':
    main()


