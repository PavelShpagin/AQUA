#!/usr/bin/env python3
"""
Build compact multilingual lexicon files from Wiktextract dumps.

Input: data/raw/raw-wiktextract-data.jsonl (Wiktextract JSONL, one JSON per line)
Output: data/lexicon/{lang}.jsonl with entries:
  {"lemma": str, "lang": str, "pos": str, "defs": [str], "examples": [str]}

Usage:
  PYTHONPATH=. ./venv/bin/python utils/lexicon/build_from_wiktextract.py \
    --input data/raw/raw-wiktextract-data.jsonl --langs es en de fr --max 400000

Notes:
  - Streams the input; writes per-language JSONL.
  - Keeps first gloss and first example for compactness.
  - Skips entries missing glosses.
"""
from __future__ import annotations

import os
import json
import argparse
import pathlib
from typing import List, Dict, Any, Set


def normalize_lang(name: str) -> str:
    n = (name or '').strip().lower()
    # Map common names to ISO codes used by our agent
    m = {
        'english': 'en',
        'spanish': 'es',
        'german': 'de',
        'french': 'fr',
        'italian': 'it',
        'portuguese': 'pt',
        'ukrainian': 'uk',
        'russian': 'ru',
        'dutch': 'nl',
    }
    return m.get(n, n)


def iter_entries(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build(args):
    out_dir = pathlib.Path('data/lexicon')
    out_dir.mkdir(parents=True, exist_ok=True)
    want: Set[str] = set([x.lower() for x in args.langs]) if args.langs else set()
    # One handle per lang
    handles: Dict[str, Any] = {}
    def get_handle(lang_code: str):
        if lang_code not in handles:
            handles[lang_code] = open(out_dir / f"{lang_code}.jsonl", 'w', encoding='utf-8')
        return handles[lang_code]

    totals = { }
    kept = { }
    max_all = int(args.max) if args.max else None
    written_all = 0

    for entry in iter_entries(args.input):
        # Filter by language set
        lang_name = entry.get('lang') or entry.get('lang_name') or ''
        if not lang_name:
            continue
        lang_code = normalize_lang(lang_name)
        if want and lang_code not in want:
            continue

        totals[lang_code] = totals.get(lang_code, 0) + 1

        word = entry.get('word') or entry.get('lemma') or ''
        if not word:
            continue

        senses = entry.get('senses') or []
        gloss = ''
        example = ''
        for s in senses:
            gls = s.get('glosses') or s.get('gloss') or []
            if isinstance(gls, list) and gls:
                gloss = gls[0]
            elif isinstance(gls, str) and gls:
                gloss = gls
            if not example:
                exs = s.get('examples') or []
                if exs and isinstance(exs, list):
                    # example may be dict or str
                    ex0 = exs[0]
                    if isinstance(ex0, dict):
                        example = ex0.get('text', '')
                    elif isinstance(ex0, str):
                        example = ex0
            if gloss:
                break

        if not gloss:
            continue

        pos = entry.get('pos') or ''

        rec = {
            'lemma': word,
            'lang': lang_code,
            'pos': pos,
            'defs': [gloss],
            'examples': [example] if example else []
        }

        fh = get_handle(lang_code)
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        kept[lang_code] = kept.get(lang_code, 0) + 1
        written_all += 1
        if max_all and written_all >= max_all:
            break

    for h in handles.values():
        try:
            h.close()
        except Exception:
            pass

    print('[ok] totals=', totals)
    print('[ok] kept=', kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--langs', nargs='*', default=['es'])
    ap.add_argument('--max', type=int, default=400000)
    args = ap.parse_args()
    build(args)


if __name__ == '__main__':
    main()








